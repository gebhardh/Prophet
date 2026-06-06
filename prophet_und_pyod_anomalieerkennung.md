# Anomalieerkennung: Prophet und PyOD im Vergleich

---

## Teil 1: Methoden der Anomalieerkennung in Prophet

Prophet erkennt Anomalien **nicht direkt** – es ist primär ein Prognosemodell.
Die Anomalieerkennung ergibt sich als **Nebenprodukt der Prognose**: Datenpunkte,
die stark von der Vorhersage abweichen, gelten als Anomalien. Intern kombiniert
Prophet dabei mehrere mathematische Methoden.

---

### 1. Additives Dekompositionsmodell (Kern)

Prophet fittet ein additives Regressionsmodell auf eine Zeitreihe:

**y(t) = g(t) + s(t) + h(t) + ε_t**

| Komponente | Bedeutung |
|---|---|
| **g(t)** | Trend: nicht-periodische Langzeitveränderungen |
| **s(t)** | Saisonalität: periodische Schwankungen (täglich, wöchentlich, jährlich) |
| **h(t)** | Feiertage/Ereignisse: unregelmäßige Sondereffekte |
| **ε_t** | Fehlerterm: nicht erklärte Zufallsabweichungen |

Ein Datenpunkt gilt als Anomalie, wenn sein tatsächlicher Wert **y(t)**
signifikant vom modellierten Wert abweicht.

---

### 2. Zwei Trend-Modelle (alternativ wählbar)

Für die Trend-Komponente g(t) stehen zwei Methoden zur Verfügung:

- **Piecewise lineare Regression** – für gleichmäßiges, unbegrenztes Wachstum
- **Piecewise logistisches Wachstum** – für gesättigte Zeitreihen mit
  Kapazitätsobergrenze:

```
g(t) = C / (1 + exp(-k(t-b)))
```

Dabei ist C die Kapazität, k die Wachstumsrate und b der Zeitversatz.

---

### 3. Fourier-Reihen für Saisonalität

Periodische Schwankungen werden durch Fourier-Reihen approximiert:

```
s(t) = Σ cₙ · e^(i·2πnt/P)
```

P = Periodenlänge, N = Fourier-Ordnung (bestimmt die Glattheit der Kurve).
Höhere Fourier-Ordnungen erlauben komplexere Muster wie Morgen- und Abendspitzen.

---

### 4. Changepoint-Erkennung

Prophet erkennt automatisch Zeitpunkte, an denen sich der Trend strukturell
ändert – sogenannte Changepoints. Der Hyperparameter `changepoint_prior_scale`
steuert die Sensitivität:

- **Niedrig (0.001):** starres Modell, wenige Trendwechsel
- **Hoch (0.5):** flexibles Modell, viele Trendwechsel

Changepoints helfen dem Modell, neue Trends zu berücksichtigen, ohne sie
fälschlicherweise als Anomalien zu werten.

---

### 5. Bayes'sche Schätzung mit Stan (Konfidenzintervall)

Unter der Haube nutzt Prophet **Stan** für die Bayes'sche Schätzung. Das
Konfidenzintervall (yhat_lower, yhat_upper) entsteht durch Monte-Carlo-Sampling
der Posterior-Verteilung – nicht durch eine einfache Standardabweichung.

Das Konfidenzintervall ist dadurch **asymmetrisch und zeitabhängig**:
- Breiter bei hoher Unsicherheit (z. B. Feiertage, Wochenenden)
- Enger bei gut vorhersagbaren Mustern

---

### 6. Anomalie-Entscheidungsregel per Konfidenzintervall

Die Anomalie-Erkennung wird durch den `interval_width`-Hyperparameter gesteuert:

```
y(t) < yhat_lower  →  Anomalie (Wert zu tief)
y(t) > yhat_upper  →  Anomalie (Wert zu hoch)
```

Ein breiteres Intervall bedeutet, dass nur extreme Werte als Anomalien markiert
werden. Ein engeres Intervall ist sensitiver, erzeugt aber mehr False Positives.

---

### 7. Shewhart Control Chart als Ergänzung (kombinierter Ansatz)

Ein verbreitetes Muster kombiniert Prophet mit einem adaptiven Shewhart
Control Chart:

- Prophet liefert die Prognose ŷ(t)
- Der Control Chart überwacht die Abweichungsreihe ε(t) = y(t) − ŷ(t)
- Anomalie wenn: ε ≥ μ + h·σ (typisch: h = 3, entspricht 3-Sigma-Regel)
- Der adaptive Ansatz aktualisiert μ und σ kontinuierlich über die Zeit

---

### Zusammenfassung der Prophet-Methoden

| Methode | Zweck | Prophet-intern? |
|---|---|---|
| Additives GAM-Modell | Zerlegung in Trend + Saison + Feiertage | ✅ Kern |
| Piecewise Linear/Logistic | Trend-Modellierung | ✅ |
| Fourier-Reihen | Saisonalitäts-Approximation | ✅ |
| Changepoint-Erkennung | Automatische Trendwechsel | ✅ |
| Bayes'sche Schätzung (Stan) | Konfidenzintervall-Berechnung | ✅ |
| Konfidenzintervall-Vergleich | Anomalie-Entscheidung | ✅ Indirekt |
| Z-Score der Abweichung | Anomalie-Scoring | ❌ Zusatz |
| Shewhart Control Chart | Adaptive Schwellenwerte | ❌ Zusatz |

---

### Was Prophet nicht kann

- Keine **multivariate** Anomalieerkennung (nur eine Zeitreihe gleichzeitig)
- Keine **Online-Lernfähigkeit** (kein inkrementelles Training)
- Keine **direkte** Klassifikation von Anomalie-Typen (Spike, Level-Shift, Trend-Change)
- Keine **kausale** Erklärung der Anomalie

---

## Teil 2: PyOD – Python Outlier Detection

### Überblick

PyOD ist die meistgenutzte Open-Source-Bibliothek für Outlier Detection mit
über 8.500 GitHub-Stars und 25 Millionen Downloads. Sie enthält mehr als 60
Detektoren für tabellarische Daten, Zeitreihen, Graphen, Text und Bilder –
mit einer einheitlichen API über alle Algorithmen hinweg.

Anwendungsgebiete: Betrugserkennung, Netzwerk-Intrusion-Detection,
Clickstream-Analyse, Anomaliedetektion in Telemetrie-Daten.

**PyOD 2** (2024) erweitert die Bibliothek um 12 Deep-Learning-Modelle in
einem einheitlichen PyTorch-Framework und führt eine LLM-basierte Pipeline
zur automatischen Modellauswahl ein.

- GitHub: https://github.com/yzhao062/pyod
- Installation: `pip install pyod`

---

### Algorithmen-Kategorien

#### 1. Statistisch / Proximität-basiert

| Algorithmus | Prinzip |
|---|---|
| **Z-Score** | Abweichung vom Mittelwert in Standardabweichungen |
| **LOF** (Local Outlier Factor) | Lokale Dichte im Vergleich zu Nachbarn |
| **COPOD** | Copula-basierte Verteilungsanalyse |
| **ECOD** | Empirische kumulative Verteilungsfunktion |
| **HBOS** | Histogramm-basierte Outlier Detection |

#### 2. Ensemble / Isolation

| Algorithmus | Prinzip |
|---|---|
| **Isolation Forest** | Anomalien lassen sich leichter isolieren als normale Punkte |
| **Feature Bagging** | Mehrere LOF-Modelle auf zufälligen Feature-Subsets |
| **SUOD** | Skalierbare Ensemble-Methode mit Parallelisierung |

#### 3. Klassisch linear

| Algorithmus | Prinzip |
|---|---|
| **PCA** | Rekonstruktionsfehler im Hauptkomponentenraum |
| **One-Class SVM (OCSVM)** | Hyperebene um normale Datenpunkte |
| **MCD** | Minimum Covariance Determinant |

#### 4. Deep Learning (PyOD 2, PyTorch-basiert)

| Algorithmus | Prinzip |
|---|---|
| **Autoencoder** | Rekonstruktionsfehler als Anomalie-Score |
| **VAE** (Variational Autoencoder) | Probabilistischer Latent Space |
| **DeepSVDD** | Deep One-Class Classification |
| **MAD-GAN** | GAN-basierte multivariate Anomalieerkennung |
| **USAD** | Unsupervised Anomaly Detection für Zeitreihen |

---

### Einheitliche API – alle Algorithmen gleich bedienbar

```python
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.auto_encoder import AutoEncoder
import numpy as np

# Beispieldaten: 1000 Trainings-, 200 Testpunkte mit je 5 Features
X_train = np.random.randn(1000, 5)
X_test  = np.random.randn(200, 5)

# Alle Modelle haben dieselbe API
for ModelClass, name in [
    (IForest, "Isolation Forest"),
    (LOF,     "Local Outlier Factor"),
]:
    model = ModelClass(contamination=0.05)  # 5% Ausreißer erwartet
    model.fit(X_train)

    scores = model.decision_function(X_test)  # kontinuierlicher Anomalie-Score
    labels = model.predict(X_test)            # 0 = normal, 1 = Anomalie

    print(f"{name}: {labels.sum()} Anomalien erkannt")
```

---

### Automatische Modellauswahl mit PyOD 2 (LLM-gestützt)

```python
from pyod.models.ad_engine import ADEngine

engine = ADEngine(contamination=0.05)
engine.fit(X_train)

# LLM wählt automatisch das beste Modell für die Daten
best_model = engine.get_best_model()
labels = engine.predict(X_test)
```

---

## Vergleich: Prophet vs. PyOD

| Kriterium | Prophet | PyOD |
|---|---|---|
| **Primärzweck** | Zeitreihen-Prognose | Allgemeine Outlier Detection |
| **Datentyp** | Univariate Zeitreihe | Multivariat, tabularisch, Zeitreihen |
| **Methode** | Additives Bayes-Modell | 60+ Algorithmen |
| **Saisonalität** | ✅ eingebaut | ❌ nicht eingebaut |
| **Multivariate Daten** | ❌ | ✅ |
| **Online-Lernen** | ❌ | ⚠️ teilweise |
| **Erklärbarkeit** | ✅ hoch | ⚠️ algorithmusabhängig |
| **Einstieg** | Einfach | Mittel |
| **Automatische Modellwahl** | ❌ | ✅ PyOD 2 mit LLM |
| **Deep Learning** | ❌ | ✅ PyOD 2 |

---

## Empfehlung: Wann welches Tool?

| Szenario | Empfehlung |
|---|---|
| Zeitreihe mit Tages-/Wochensaisonalität | **Prophet** |
| Multivariate Metriken (CPU + RAM + Netz gleichzeitig) | **PyOD** |
| Unbekannte Anomalie-Art, schneller Einstieg | **PyOD Isolation Forest** |
| Netzwerk-Intrusion-Detection | **PyOD LOF oder AutoEncoder** |
| Einfache Baseline ohne ML | **PyOD Z-Score** |
| Komplexe Muster, große Datenmenge | **PyOD AutoEncoder / VAE** |
| Kombination Saisonalität + Multivariate | **Prophet + PyOD kombiniert** |

---

## Kombination Prophet + PyOD (Beispiel)

```python
from prophet import Prophet
from pyod.models.iforest import IForest
import pandas as pd
import numpy as np

# 1. Prophet: Residuen berechnen
model = Prophet(interval_width=0.99)
model.fit(train_df)
forecast = model.predict(eval_df)

residuals = eval_df["y"].values - forecast["yhat"].values

# 2. PyOD auf Residuen anwenden
detector = IForest(contamination=0.05)
detector.fit(residuals.reshape(-1, 1))
labels = detector.predict(residuals.reshape(-1, 1))

# labels: 0 = normal, 1 = Anomalie
anomalies = eval_df[labels == 1]
print(f"Anomalien: {len(anomalies)}")
```

Dieser kombinierte Ansatz nutzt Prophet für die Saisonalitätserkennung und
PyOD für die robuste Ausreißererkennung in den Residuen – ohne Annahmen über
die Verteilung der Abweichungen.

---

## Quellen

- Facebook Prophet Paper (Taylor & Letham, 2018):
  https://peerj.com/preprints/3190/
- Prophet Dokumentation:
  https://facebook.github.io/prophet/
- PyOD Paper (Zhao et al., 2019):
  https://arxiv.org/abs/1901.01588
- PyOD 2 Paper (2024):
  https://arxiv.org/abs/2412.12154
- PyOD GitHub:
  https://github.com/yzhao062/pyod
