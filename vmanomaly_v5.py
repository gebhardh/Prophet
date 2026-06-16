#!/usr/bin/env python3
"""
VictoriaMetrics Anomaly Detection mit Facebook Prophet
======================================================
Liest Metriken aus VictoriaMetrics (Cluster), trainiert je Zeitreihe ein
Prophet-Modell und schreibt Anomalie-Scores zurück nach VictoriaMetrics.

Verwendung:
    python vmanomaly_v3.py --config vmanomaly_config.yaml
    python vmanomaly_v3.py --config vmanomaly_config.yaml --once
    python vmanomaly_v3.py --config vmanomaly_config.yaml --dry-run
"""

import argparse
import calendar
import json
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml
from prophet import Prophet

import warnings
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ─── Konfiguration ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Lädt die YAML-Konfigurationsdatei."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_duration(duration_str: str) -> timedelta:
    """Parst Dauer-Strings wie '7d', '2h', '30m', '60s'."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    unit = duration_str[-1].lower()
    value = int(duration_str[:-1])
    if unit not in units:
        raise ValueError(f"Unbekannte Zeiteinheit: {unit}")
    return timedelta(seconds=value * units[unit])


def _label_fingerprint(labels: dict) -> str:
    """Eindeutiger Schlüssel für eine Label-Kombination (stabile Sortierung)."""
    return json.dumps(labels, sort_keys=True)


def _labels_to_promstr(labels: dict) -> str:
    """Label-Dict → Prometheus-Expositions-String  key="val",... (sortiert)."""
    return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))


def _build_holiday_df(holiday_config: dict) -> Optional[pd.DataFrame]:
    """
    Erstellt einen Holiday-DataFrame für Prophet aus der Konfiguration.

    Unterstützt:
      country  – ISO-3166-1-Code (z. B. "DE")
      province – Bundesland-Kürzel (z. B. "BY" für Bayern)
      custom   – Liste eigener Ereignisse mit name, dates, lower_window, upper_window
    """
    import holidays as hdays

    rows: list[dict] = []
    country  = holiday_config.get("country")
    province = holiday_config.get("province")

    if country:
        now   = datetime.utcnow()
        years = list(range(now.year - 3, now.year + 2))
        kwargs: dict = {"years": years}
        if province:
            kwargs["subdiv"] = province
        for date, name in hdays.country_holidays(country, **kwargs).items():
            rows.append({"holiday": name, "ds": pd.Timestamp(date)})

    for event in holiday_config.get("custom", []):
        for ds_str in event.get("dates", []):
            rows.append({
                "holiday":      event["name"],
                "ds":           pd.Timestamp(ds_str),
                "lower_window": event.get("lower_window", 0),
                "upper_window": event.get("upper_window", 0),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"]).dt.normalize()
    return df


# ─── VictoriaMetrics Client ───────────────────────────────────────────────────

class VMClient:
    """Kommuniziert mit der VictoriaMetrics Cluster HTTP API."""

    def __init__(self, config: dict):
        vm = config["victoriametrics"]
        self.read_url  = vm["read_url"].rstrip("/")   # vmselect
        self.write_url = vm["write_url"].rstrip("/")  # vminsert
        self.timeout   = vm.get("timeout", 30)

    def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step: str = "5m",
    ) -> list[tuple[dict, pd.DataFrame]]:
        """
        Führt eine range query durch.
        Gibt eine Liste von (labels, df) zurück – ein Eintrag pro Zeitreihe.
        df-Spalten: ds_epoch (float, Unix-UTC), ds (datetime, naive UTC), y (float)
        """
        url = f"{self.read_url}/api/v1/query_range"
        params = {
            "query": promql,
            "start": calendar.timegm(start.timetuple()),
            "end":   calendar.timegm(end.timetuple()),
            "step":  step,
        }

        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logging.error(f"VictoriaMetrics-Abfrage fehlgeschlagen: {e}")
            return []

        if data.get("status") != "success":
            logging.error(f"API-Fehler: {data}")
            return []

        results = data.get("data", {}).get("result", [])
        if not results:
            logging.warning(f"Keine Daten für Query: {promql}")
            return []

        series_list = []
        for series in results:
            # __name__ und andere interne Labels (__*) herausfiltern –
            # sie sind im Prometheus-Exposition-Format nicht als Label erlaubt
            labels = {k: v for k, v in series.get("metric", {}).items()
                      if not k.startswith("__")}
            rows = []
            for ts, val in series["values"]:
                try:
                    rows.append({
                        "ds_epoch": float(ts),
                        "ds": datetime(1970, 1, 1) + timedelta(seconds=float(ts)),
                        "y": float(val),
                    })
                except (ValueError, TypeError):
                    continue
            if rows:
                df = (pd.DataFrame(rows)
                        .sort_values("ds_epoch")
                        .reset_index(drop=True))
                series_list.append((labels, df))

        return series_list

    def write_batch(self, lines: list[str]):
        """Schreibt mehrere Metriken auf einmal via Prometheus remote write."""
        url = f"{self.write_url}/api/v1/import/prometheus"
        payload = "\n".join(lines)
        try:
            resp = requests.post(
                url,
                data=payload,
                headers={"Content-Type": "text/plain"},
                timeout=self.timeout,
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
        self.interval_width          = model_config.get("interval_width", 0.99)
        self.yearly_seasonality      = model_config.get("yearly_seasonality", False)
        self.weekly_seasonality      = model_config.get("weekly_seasonality", True)
        self.daily_seasonality       = model_config.get("daily_seasonality", True)
        self.changepoint_prior_scale = model_config.get("changepoint_prior_scale", 0.05)
        self.seasonality_prior_scale = model_config.get("seasonality_prior_scale", 10.0)
        self.holiday_config          = model_config.get("holidays", {})
        self.model: Optional[Prophet] = None

    def train(self, df: pd.DataFrame) -> bool:
        """Trainiert das Modell. Erwartet Spalten: ds (datetime), y (float)."""
        if df.empty or len(df) < 10:
            logging.warning("Zu wenige Datenpunkte für Prophet-Training (min. 10).")
            return False

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty:
            logging.warning("DataFrame nach Bereinigung leer.")
            return False

        try:
            holiday_df = _build_holiday_df(self.holiday_config) if self.holiday_config else None
            if holiday_df is not None:
                logging.info(
                    f"Feiertagsmodell aktiv: {len(holiday_df)} Einträge "
                    f"({holiday_df['holiday'].nunique()} Ereignisse)."
                )
            self.model = Prophet(
                interval_width=self.interval_width,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays=holiday_df,
            )
            self.model.fit(df)
            logging.info(f"Prophet-Modell trainiert auf {len(df)} Datenpunkten.")
            return True
        except Exception as e:
            logging.error(f"Prophet-Training fehlgeschlagen: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erstellt Prognosen und gibt DataFrame mit Anomalie-Flags zurück."""
        if self.model is None:
            raise RuntimeError("Modell wurde noch nicht trainiert.")
        if df.empty:
            return pd.DataFrame()

        forecast = self.model.predict(df[["ds"]].copy())

        # Ergebnisse zusammenführen (ds_epoch mitziehen, falls vorhanden)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        merge_cols = ["ds", "ds_epoch", "y"] if "ds_epoch" in df.columns else ["ds", "y"]
        result = result.merge(df[merge_cols], on="ds", how="left")

        # Anomalie-Flag: Messwert außerhalb des Konfidenzbands
        result["anomaly"] = (
            (result["y"] < result["yhat_lower"]) |
            (result["y"] > result["yhat_upper"])
        )

        # Abweichung vom Prognosewert
        result["deviation"] = result["y"] - result["yhat"]

        # Anomalie-Score: normalisierte Distanz von yhat zur Bandgrenze
        # 0.0 = Wert liegt auf der Prognose (yhat)
        # 1.0 = Wert liegt genau auf der Bandgrenze  ↔  anomaly = True
        half_band = ((result["yhat_upper"] - result["yhat_lower"]) / 2).clip(lower=1e-9)
        result["anomaly_score"] = (result["deviation"].abs() / half_band).clip(upper=1.0)

        return result

    def detect(self, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Training + Vorhersage in einem Schritt."""
        if not self.train(train_df):
            return pd.DataFrame()
        return self.predict(eval_df)


# ─── Alert-Manager ────────────────────────────────────────────────────────────

class AlertManager:
    """Verwaltet und versendet Anomalie-Alerts."""

    def __init__(self, config: dict):
        self.alert_config   = config.get("alerting", {})
        self.enabled        = self.alert_config.get("enabled", True)
        self.log_anomalies  = self.alert_config.get("log_anomalies", True)
        self.log_file       = self.alert_config.get("log_file", "anomalies.log")
        self.webhook_config = self.alert_config.get("webhook", {})

    def process(self, query_name: str, result: pd.DataFrame):
        """Verarbeitet Anomalien aus einem Erkennungs-Ergebnis."""
        if not self.enabled or result.empty:
            return
        for _, row in result[result["anomaly"] == True].iterrows():
            self._handle_anomaly(query_name, row)

    def _handle_anomaly(self, query_name: str, row: pd.Series):
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
        try:
            with open(self.log_file, "a") as f:
                f.write(msg + "\n")
        except IOError as e:
            logging.error(f"Log-Schreiben fehlgeschlagen: {e}")

    def _send_webhook(self, query_name: str, row: pd.Series):
        payload = [{
            "labels": {
                "alertname": "VMAnomaly",
                "query":     query_name,
                "severity":  "warning",
            },
            "annotations": {
                "summary": f"Anomalie in {query_name}",
                "description": (
                    f"Wert {row['y']:.4f} außerhalb des erwarteten Bereichs "
                    f"[{row['yhat_lower']:.4f}, {row['yhat_upper']:.4f}]"
                ),
            },
        }]
        try:
            requests.post(
                self.webhook_config["url"],
                json=payload,
                timeout=self.webhook_config.get("timeout", 10),
            )
        except requests.RequestException as e:
            logging.error(f"Webhook-Versand fehlgeschlagen: {e}")


# ─── Hauptprogramm ────────────────────────────────────────────────────────────

class VMAnomalyDetection:
    """Orchestriert VMClient, ProphetAnomalyDetector und AlertManager."""

    def __init__(self, config: dict, dry_run: bool = False):
        self.config         = config
        self.dry_run        = dry_run
        self.vm             = VMClient(config)
        self.alert_manager  = AlertManager(config)
        self.model_config   = config.get("model", {})
        self.writer_config  = config.get("writer", {})
        self.queries        = config.get("queries", [])
        self.training_lookback = parse_duration(
            self.model_config.get("training_lookback", "7d")
        )

    def run_once(self):
        """Führt einen einzelnen Erkennungs-Durchlauf durch."""
        now_epoch   = time.time()
        now         = datetime(1970, 1, 1) + timedelta(seconds=now_epoch)
        train_start = now - self.training_lookback
        train_end   = now - timedelta(minutes=5)
        eval_start  = train_end
        eval_end    = now

        logging.info(
            f"Starte Anomalie-Erkennung | "
            f"Training: {train_start:%Y-%m-%d %H:%M} → {train_end:%H:%M} | "
            f"Eval: {eval_start:%H:%M} → {eval_end:%H:%M}"
        )

        output_lines = []

        for query_cfg in self.queries:
            name   = query_cfg["name"]
            promql = query_cfg["promql"]
            desc   = query_cfg.get("description", name)
            logging.info(f"Verarbeite: {desc} ({name})")

            # Jede Query liefert eine Liste von (labels, df) – eine pro Zeitreihe
            train_series = self.vm.query_range(promql, train_start, train_end)
            if not train_series:
                logging.warning(f"Keine Trainingsdaten für '{name}' – überspringe.")
                continue

            eval_series = self.vm.query_range(promql, eval_start, eval_end)
            if not eval_series:
                logging.warning(f"Keine Eval-Daten für '{name}' – überspringe.")
                continue

            # Serien per Label-Fingerprint paaren
            train_by_fp = {_label_fingerprint(lbl): (lbl, df) for lbl, df in train_series}
            eval_by_fp  = {_label_fingerprint(lbl): (lbl, df) for lbl, df in eval_series}
            common_fps  = set(train_by_fp) & set(eval_by_fp)

            logging.info(
                f"  {len(train_series)} Train- / {len(eval_series)} Eval-Serien, "
                f"{len(common_fps)} gemeinsame."
            )

            for fp in common_fps:
                metric_labels, train_df = train_by_fp[fp]
                _,             eval_df  = eval_by_fp[fp]
                label_str = _labels_to_promstr(metric_labels)

                detector = ProphetAnomalyDetector(self.model_config)
                result   = detector.detect(train_df, eval_df)

                if result.empty:
                    continue

                n_anomalies = result["anomaly"].sum()
                logging.info(
                    f"  {{{label_str}}}: {n_anomalies}/{len(result)} Anomalien erkannt."
                )

                self.alert_manager.process(f"{name}{{{label_str}}}", result)

                if self.writer_config.get("enabled", True) and not self.dry_run:
                    prefix       = self.writer_config.get("metric_prefix", "vmanomaly")
                    write_labels = _labels_to_promstr({**metric_labels, "query": name})

                    for _, row in result.iterrows():
                        ts_ms = int(row.get("ds_epoch", 0) * 1000)

                        output_lines.append(
                            f'{prefix}_anomaly{{{write_labels}}} '
                            f'{1 if row["anomaly"] else 0} {ts_ms}'
                        )
                        for col, metric in [
                            ("anomaly_score", "score"),
                            ("deviation",     "deviation"),
                            ("yhat",          "yhat"),
                            ("yhat_lower",    "yhat_lower"),
                            ("yhat_upper",    "yhat_upper"),
                        ]:
                            val = row.get(col, float("nan"))
                            if not math.isnan(val):
                                output_lines.append(
                                    f'{prefix}_{metric}{{{write_labels}}} {val:.6f} {ts_ms}'
                                )

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
    level    = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    fmt      = log_config.get("format", "%(asctime)s [%(levelname)s] %(message)s")
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
        help="Pfad zur Konfigurationsdatei (default: vmanomaly_config.yaml)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Nur einmal ausführen, dann beenden",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Keine Daten nach VictoriaMetrics schreiben",
    )
    args = parser.parse_args()

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
