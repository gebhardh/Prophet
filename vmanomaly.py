#!/usr/bin/env python3
"""
VictoriaMetrics Anomaly Detection mit Facebook Prophet
======================================================
Liest Metriken aus VictoriaMetrics, trainiert ein Prophet-Modell
und schreibt Anomalie-Scores zurück nach VictoriaMetrics.

Verwendung:
    python vmanomaly.py --config vmanomaly_config.yaml
    python vmanomaly.py --config vmanomaly_config.yaml --once
    python vmanomaly.py --config vmanomaly_config.yaml --dry-run
"""

import argparse
import json
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Logging unterdrücken (Prophet ist sehr geschwätzig)
import warnings
import logging as prophet_log
warnings.filterwarnings("ignore")
prophet_log.getLogger("prophet").setLevel(logging.WARNING)
prophet_log.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ─── Konfiguration ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Lädt die YAML-Konfigurationsdatei."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_duration(duration_str: str) -> timedelta:
    """
    Parst Dauer-Strings wie '7d', '2h', '30m', '60s'.
    """
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    unit = duration_str[-1].lower()
    value = int(duration_str[:-1])
    if unit not in units:
        raise ValueError(f"Unbekannte Zeiteinheit: {unit}")
    return timedelta(seconds=value * units[unit])


# ─── VictoriaMetrics Client ───────────────────────────────────────────────────

class VMClient:
    """Kommuniziert mit der VictoriaMetrics HTTP API."""

    def __init__(self, config: dict):
        self.base_url = config["victoriametrics"]["url"].rstrip("/")
        self.timeout = config["victoriametrics"].get("timeout", 30)

    def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step: str = "5m"
    ) -> pd.DataFrame:
        """
        Führt eine range query durch und gibt einen DataFrame zurück.
        Spalten: ds (float, Unix-Epoch UTC), y (float)
        start/end werden als naive UTC-datetimes interpretiert.
        """
        url = f"{self.base_url}/api/v1/query_range"

        # Sicher: naive datetime als UTC → Unix-Epoch (calendar.timegm statt timestamp())
        import calendar
        start_ts = calendar.timegm(start.timetuple())
        end_ts   = calendar.timegm(end.timetuple())

        params = {
            "query": promql,
            "start": start_ts,
            "end":   end_ts,
            "step":  step,
        }

        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logging.error(f"VictoriaMetrics-Abfrage fehlgeschlagen: {e}")
            return pd.DataFrame()

        if data.get("status") != "success":
            logging.error(f"API-Fehler: {data}")
            return pd.DataFrame()

        results = data.get("data", {}).get("result", [])
        if not results:
            logging.warning(f"Keine Daten für Query: {promql}")
            return pd.DataFrame()

        # Mehrere Series aggregieren (Durchschnitt)
        all_rows = []
        for series in results:
            for ts, val in series["values"]:
                try:
                    all_rows.append({
                        # Unix-Epoch direkt als Zahl speichern → kein TZ-Risiko
                        "ds_epoch": float(ts),
                        "ds": datetime(1970, 1, 1) + timedelta(seconds=float(ts)),
                        "y": float(val),
                        "labels": json.dumps(series.get("metric", {}))
                    })
                except (ValueError, TypeError):
                    continue

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # Bei mehreren Label-Kombinationen: Durchschnitt pro Zeitpunkt
        # ds_epoch (Unix-Float) als primären Zeitstempel mitführen
        df = (df.groupby("ds_epoch")
                .agg(y=("y", "mean"), ds=("ds", "first"))
                .reset_index()
                .sort_values("ds_epoch")
                .reset_index(drop=True))
        return df

    def write_metric(self, metric_name: str, labels: dict, value: float, timestamp: datetime):
        """
        Schreibt eine Metrik via Prometheus remote write (Influx Line Protocol).
        """
        url = f"{self.base_url}/api/v1/import/prometheus"

        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        if label_str:
            line = f"{metric_name}{{{label_str}}} {value} {int(timestamp.timestamp() * 1000)}"
        else:
            line = f"{metric_name} {value} {int(timestamp.timestamp() * 1000)}"

        try:
            resp = requests.post(
                url,
                data=line,
                headers={"Content-Type": "text/plain"},
                timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Schreiben nach VictoriaMetrics fehlgeschlagen: {e}")

    def write_batch(self, lines: list[str]):
        """Schreibt mehrere Metriken auf einmal."""
        url = f"{self.base_url}/api/v1/import/prometheus"
        payload = "\n".join(lines)
        try:
            resp = requests.post(
                url,
                data=payload,
                headers={"Content-Type": "text/plain"},
                timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Batch-Schreiben fehlgeschlagen: {e}")


# ─── Prophet Anomalie-Erkennung ───────────────────────────────────────────────

class ProphetAnomalyDetector:
    """
    Trainiert ein Prophet-Modell auf historischen Daten und erkennt
    Anomalien im aktuellen Zeitfenster.
    """

    def __init__(self, model_config: dict):
        self.interval_width = model_config.get("interval_width", 0.99)
        self.yearly_seasonality = model_config.get("yearly_seasonality", False)
        self.weekly_seasonality = model_config.get("weekly_seasonality", True)
        self.daily_seasonality = model_config.get("daily_seasonality", True)
        self.changepoint_prior_scale = model_config.get("changepoint_prior_scale", 0.05)
        self.seasonality_prior_scale = model_config.get("seasonality_prior_scale", 10.0)
        self.zscore_threshold = 3.0
        self.model: Optional[Prophet] = None

    def train(self, df: pd.DataFrame) -> bool:
        """
        Trainiert das Prophet-Modell auf dem übergebenen DataFrame.
        Erwartet Spalten: ds (datetime), y (float)
        """
        if df.empty or len(df) < 10:
            logging.warning("Zu wenige Datenpunkte für Prophet-Training (min. 10).")
            return False

        # NaN und Inf bereinigen
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            logging.warning("DataFrame nach Bereinigung leer.")
            return False

        try:
            self.model = Prophet(
                interval_width=self.interval_width,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
            )
            self.model.fit(df)
            logging.info(f"Prophet-Modell trainiert auf {len(df)} Datenpunkten.")
            return True

        except Exception as e:
            logging.error(f"Prophet-Training fehlgeschlagen: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt Prognosen für den übergebenen DataFrame.
        Gibt DataFrame mit Anomalie-Flags zurück.
        """
        if self.model is None:
            raise RuntimeError("Modell wurde noch nicht trainiert.")

        if df.empty:
            return pd.DataFrame()

        future = df[["ds"]].copy()
        forecast = self.model.predict(future)

        # Ergebnisse zusammenführen
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result = result.merge(df[["ds", "y"]], on="ds", how="left")

        # Anomalie-Flag: Außerhalb des Konfidenzintervalls
        result["anomaly"] = (
            (result["y"] < result["yhat_lower"]) |
            (result["y"] > result["yhat_upper"])
        )

        # Abweichung in absoluten Werten
        result["deviation"] = result["y"] - result["yhat"]

        # Z-Score der Abweichung (normalisiert)
        std = result["deviation"].std()
        if std > 0:
            result["zscore"] = (result["deviation"] / std).abs()
        else:
            result["zscore"] = 0.0

        # Anomalie-Score (0.0 = normal, 1.0 = starke Anomalie)
        result["anomaly_score"] = result["zscore"].apply(
            lambda z: min(1.0, z / 10.0)
        )

        # Zusatz: Z-Score-basierte Anomalie (zweite Methode)
        result["anomaly_zscore"] = result["zscore"] > self.zscore_threshold

        return result

    def detect(self, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
        """
        Kombinierter Aufruf: Training + Vorhersage + Anomalie-Erkennung.
        train_df: historische Daten (Training)
        eval_df:  aktuelle Daten (Auswertung)
        """
        if not self.train(train_df):
            return pd.DataFrame()
        return self.predict(eval_df)


# ─── Alert-Manager ────────────────────────────────────────────────────────────

class AlertManager:
    """Verwaltet und versendet Anomalie-Alerts."""

    def __init__(self, config: dict):
        self.alert_config = config.get("alerting", {})
        self.enabled = self.alert_config.get("enabled", True)
        self.log_anomalies = self.alert_config.get("log_anomalies", True)
        self.log_file = self.alert_config.get("log_file", "anomalies.log")
        self.webhook_config = self.alert_config.get("webhook", {})

    def process(self, query_name: str, result: pd.DataFrame):
        """Verarbeitet Anomalien aus einem Erkennungs-Ergebnis."""
        if not self.enabled or result.empty:
            return

        anomalies = result[result["anomaly"] == True]
        if anomalies.empty:
            return

        for _, row in anomalies.iterrows():
            self._handle_anomaly(query_name, row)

    def _handle_anomaly(self, query_name: str, row: pd.Series):
        """Verarbeitet eine einzelne Anomalie."""
        msg = (
            f"[ANOMALIE] {query_name} | "
            f"Zeit: {row['ds']} | "
            f"Wert: {row['y']:.4f} | "
            f"Erwartet: {row['yhat']:.4f} | "
            f"Intervall: [{row['yhat_lower']:.4f}, {row['yhat_upper']:.4f}] | "
            f"Score: {row.get('anomaly_score', 0):.3f}"
        )
        logging.warning(msg)

        if self.log_anomalies:
            self._write_log(msg)

        if self.webhook_config.get("enabled", False):
            self._send_webhook(query_name, row)

    def _write_log(self, msg: str):
        """Schreibt Anomalie in Logdatei."""
        try:
            with open(self.log_file, "a") as f:
                f.write(msg + "\n")
        except IOError as e:
            logging.error(f"Log-Schreiben fehlgeschlagen: {e}")

    def _send_webhook(self, query_name: str, row: pd.Series):
        """Sendet Alert an Webhook (z. B. Alertmanager)."""
        payload = [{
            "labels": {
                "alertname": "VMAnomaly",
                "query": query_name,
                "severity": "warning"
            },
            "annotations": {
                "summary": f"Anomalie in {query_name}",
                "description": (
                    f"Wert {row['y']:.4f} außerhalb des erwarteten Bereichs "
                    f"[{row['yhat_lower']:.4f}, {row['yhat_upper']:.4f}]"
                )
            }
        }]
        try:
            requests.post(
                self.webhook_config["url"],
                json=payload,
                timeout=self.webhook_config.get("timeout", 10)
            )
        except requests.RequestException as e:
            logging.error(f"Webhook-Versand fehlgeschlagen: {e}")


# ─── Hauptprogramm ────────────────────────────────────────────────────────────

class VMAnomalyDetection:
    """Hauptklasse – orchestriert alle Komponenten."""

    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.vm = VMClient(config)
        self.alert_manager = AlertManager(config)
        self.model_config = config.get("model", {})
        self.writer_config = config.get("writer", {})
        self.queries = config.get("queries", [])

        # Zeiträume parsen
        self.training_lookback = parse_duration(
            self.model_config.get("training_lookback", "7d")
        )

    def run_once(self):
        """Führt einen einzelnen Erkennungs-Durchlauf durch."""
        import calendar, time as _time
        now_epoch    = _time.time()                                    # UTC Unix-Epoch
        now          = datetime(1970, 1, 1) + timedelta(seconds=now_epoch)  # naive UTC
        train_start  = now - self.training_lookback
        train_end    = now - timedelta(minutes=5)
        eval_start   = train_end
        eval_end     = now

        logging.info(f"Starte Anomalie-Erkennung | "
                     f"Training: {train_start:%Y-%m-%d %H:%M} → {train_end:%H:%M} | "
                     f"Eval: {eval_start:%H:%M} → {eval_end:%H:%M}")

        output_lines = []

        for query_cfg in self.queries:
            name = query_cfg["name"]
            promql = query_cfg["promql"]
            desc = query_cfg.get("description", name)

            logging.info(f"Verarbeite: {desc} ({name})")

            # Trainingsdaten laden
            train_df = self.vm.query_range(promql, train_start, train_end)
            if train_df.empty:
                logging.warning(f"Keine Trainingsdaten für '{name}' – überspringe.")
                continue

            # Eval-Daten laden
            eval_df = self.vm.query_range(promql, eval_start, eval_end)
            if eval_df.empty:
                logging.warning(f"Keine Eval-Daten für '{name}' – überspringe.")
                continue

            # Anomalie-Erkennung
            detector = ProphetAnomalyDetector(self.model_config)
            result = detector.detect(train_df, eval_df)

            if result.empty:
                continue

            # Anomalien ausgeben
            n_anomalies = result["anomaly"].sum()
            logging.info(f"'{name}': {n_anomalies}/{len(result)} Anomalien erkannt.")

            # Alerts verarbeiten
            self.alert_manager.process(name, result)

            # Ergebnisse nach VictoriaMetrics schreiben
            if self.writer_config.get("enabled", True) and not self.dry_run:
                prefix = self.writer_config.get("metric_prefix", "vmanomaly")
                for _, row in result.iterrows():
                    # Unix-Epoch direkt aus ds_epoch – kein TZ-Risiko
                    ts_ms = int(row.get("ds_epoch", 0) * 1000)
                    labels = f'query="{name}"'

                    # Anomalie-Flag (0 oder 1)
                    output_lines.append(
                        f'{prefix}_anomaly{{{labels}}} '
                        f'{1 if row["anomaly"] else 0} {ts_ms}'
                    )
                    # Anomalie-Score (0.0 – 1.0)
                    score = row.get("anomaly_score", 0.0)
                    if not math.isnan(score):
                        output_lines.append(
                            f'{prefix}_score{{{labels}}} {score:.6f} {ts_ms}'
                        )
                    # Abweichung vom Erwartungswert
                    dev = row.get("deviation", 0.0)
                    if not math.isnan(dev):
                        output_lines.append(
                            f'{prefix}_deviation{{{labels}}} {dev:.6f} {ts_ms}'
                        )

        # Batch-Schreiben
        if output_lines and not self.dry_run:
            self.vm.write_batch(output_lines)
            logging.info(f"{len(output_lines)} Metriken nach VictoriaMetrics geschrieben.")
        elif self.dry_run:
            logging.info(f"[DRY-RUN] {len(output_lines)} Metriken würden geschrieben.")

    def run_scheduled(self, interval_minutes: int):
        """Führt die Erkennung periodisch aus."""
        logging.info(f"Starte Scheduler – Intervall: {interval_minutes} Minuten")
        while True:
            try:
                self.run_once()
            except Exception as e:
                logging.error(f"Fehler im Hauptdurchlauf: {e}", exc_info=True)

            logging.info(f"Nächste Ausführung in {interval_minutes} Minuten...")
            time.sleep(interval_minutes * 60)


# ─── Einstiegspunkt ───────────────────────────────────────────────────────────

def setup_logging(config: dict):
    """Konfiguriert das Logging."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    fmt = log_config.get("format", "%(asctime)s [%(levelname)s] %(message)s")
    log_file = log_config.get("file", None)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def main():
    parser = argparse.ArgumentParser(
        description="VictoriaMetrics Anomaly Detection mit Facebook Prophet"
    )
    parser.add_argument(
        "--config", "-c",
        default="vmanomaly_config.yaml",
        help="Pfad zur Konfigurationsdatei (default: vmanomaly_config.yaml)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Nur einmal ausführen, dann beenden"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Keine Daten nach VictoriaMetrics schreiben"
    )
    args = parser.parse_args()

    # Konfiguration laden
    config = load_config(args.config)
    setup_logging(config)

    logging.info("=" * 60)
    logging.info("VictoriaMetrics Anomaly Detection gestartet")
    logging.info(f"Konfiguration: {args.config}")
    logging.info(f"Dry-Run: {args.dry_run}")
    logging.info("=" * 60)

    detector = VMAnomalyDetection(config, dry_run=args.dry_run)

    if args.once:
        detector.run_once()
    else:
        interval = config.get("scheduler", {}).get("interval_minutes", 5)
        detector.run_scheduled(interval)


if __name__ == "__main__":
    main()
