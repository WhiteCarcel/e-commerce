import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

# Warnungen unterdrücken ohne die Funktionalität zu beeinträchtigen
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def prepare_dataset(df):
    """
    Bereitet den Datensatz für das F9Tuned-Modell vor
    
    Args:
        df (pandas.DataFrame): Der zu verarbeitende Datensatz
        
    Returns:
        pandas.DataFrame: Vorbereiteter Datensatz
    """
    # Erstelle eine Kopie des Datensatzes um die Originaldaten nicht zu verändern
    df = df.copy()
    
    # Konvertiert Zielvariable in einen Integer-Typ (0 oder 1)

    if 'Revenue' in df.columns:
        df['Revenue'] = df['Revenue'].astype(int)
    
    # Weekend-Spalte sicher umwandeln
    if 'Weekend' in df.columns:
        # Boolesche Werte direkt zu 0/1 konvertieren
        if df['Weekend'].dtype == bool:
            df['Weekend'] = df['Weekend'].astype(int)
        # Stringwerte wie 'TRUE'/'FALSE' in 0/1 umwandeln
        elif df['Weekend'].dtype == object:
            # Konvertiere mit einer flexiblen Zuordnung, die verschiedene Wahrheitswerte erkennt
            df['Weekend'] = df['Weekend'].map(
                lambda x: 1 if str(x).lower() in ('true', '1', 't', 'y', 'yes', 'wahr', 'ja') else 0
            )
    
    return df

def select_features(df):
    """
    Wählt die relevanten Features für das F9Tuned-Modell aus
    
    Args:
        df (pandas.DataFrame): Der vorbereitete Datensatz
        
    Returns:
        pandas.DataFrame: Datensatz mit ausgewählten Features
    """
    # Auswahl der 9 wichtigsten Features durch  Analysen identifiziert und führen zum höchsten F1-Score
    selected_features = [
        'Informational', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
        'Month', 'OperatingSystems', 'VisitorType', 'Weekend'
    ]
    
    # Prüft ob alle benötigten Features vorhanden sind
    # Falls einige fehlen, werden nur verfügbare Features verwendet
    available_features = [feat for feat in selected_features if feat in df.columns]
    
    # Datensatz auf die verfügbaren Features reduzieren
    X_selected = df[available_features].copy()
    
    # Kategorische Spalten identifizieren, die eine spezielle Behandlung benötigen
    # Month, VisitorType und Weekend sind die kategorischen Variablen in unserem Datensatz
    categorical_cols = [col for col in ['Month', 'VisitorType', 'Weekend'] if col in X_selected.columns]
    
    # One-Hot-Encoding für kategorische Features anwenden
    # Dies ist notwendig, da der RandomForest-Algorithmus keine kategorischen Variablen direkt verarbeiten kann
    # drop_first=True reduziert die Dimensionalität und vermeidet Multikollinearität
    if categorical_cols:
        X_selected = pd.get_dummies(X_selected, columns=categorical_cols, drop_first=True)
        
    return X_selected

def train_f9tuned_model(df):
    """
    Trainiert das F9Tuned-Modell (vereinfachte Version mit RandomForest)
    
    Args:
        df (pandas.DataFrame): Der vollständige Datensatz mit Revenue-Spalte
        
    Returns:
        dict: Dictionary mit dem trainierten Modell und relevanten Informationen
    """
    # Datensatz vorbereiten - Konvertierung der Datentypen und Formatierung
    # Die prepare_dataset Funktion wandelt boolesche Werte in 0/1 um
    df = prepare_dataset(df)
    
    # Features auswählen - nur die 9 wichtigsten Features werden verwendet
    # Dies verbessert die Modellperformance und Trainingsgeschwindigkeit
    X_selected = select_features(df)
    
    # Zielvariable extrahieren (Kaufabsicht: 0=kein Kauf, 1=Kauf)
    y = df['Revenue']
    
    # Datensatz in Trainings- und Testdaten aufteilen
    # 80/20-Split ist optimal für diesen Datensatz (basierend auf Experimenten)
    # Stratifizierung sorgt dafür, dass die Klassenverteilung in beiden Sets gleich bleibt
    # random_state=42 garantiert Reproduzierbarkeit der Ergebnisse
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Optimierter RandomForest mit sorgfältig ausgewählten Hyperparametern
    # Diese Parameter wurden durch umfangreiche Experimente ermittelt
    best_rf = RandomForestClassifier(
        n_estimators=100,      # 100 Entscheidungsbäume bieten guten Kompromiss aus Genauigkeit und Geschwindigkeit
        min_samples_leaf=5,    # Mindestens 5 Samples pro Blatt verhindert Overfitting
        max_samples=0.7,       # Verwende 70% der Daten für jeden Baum (Bootstrap-Sampling)
        max_features='sqrt',   # Anzahl Features pro Split = Wurzel der Gesamtanzahl (klassische RF-Einstellung)
        bootstrap=True,        # Bootstrap-Sampling aktiviert für Ensemble-Diversität
        random_state=42,       # Für reproduzierbare Ergebnisse
        n_jobs=-1              # Parallelisierung auf allen verfügbaren CPU-Kernen
    )
    
    # Training des Modells mit den Trainingsdaten
    best_rf.fit(X_train, y_train)
    
    # Optimaler Schwellenwert für die Klassifizierung
    # 0.4 wurde durch ROC-Analyse als bester Kompromiss zwischen Recall und Precision ermittelt
    best_threshold = 0.4
    
    # Vorhersagen auf den Testdaten zur Evaluierung
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeiten für Klasse 1 (Kauf)
    y_pred = (y_pred_proba >= best_threshold).astype(int)  # Binäre Vorhersagen basierend auf Schwellenwert
    
    # Berechnung der wichtigsten Metriken zur Modellbewertung
    accuracy = accuracy_score(y_test, y_pred)       # Genauigkeit (Anteil korrekter Vorhersagen)
    recall = recall_score(y_test, y_pred)           # Trefferquote (Anteil erkannter positiver Fälle)
    precision = precision_score(y_test, y_pred)     # Präzision (Anteil korrekter positiver Vorhersagen)
    f1 = f1_score(y_test, y_pred)                   # F1-Score (harmonisches Mittel aus Precision und Recall)
    
    # Ergebnisse in einem Dictionary zusammenfassen für einfache Weiterverarbeitung
    return {
        'model': best_rf,                          # Das trainierte Modell
        'model_name': 'RandomForest',              # Name des verwendeten Algorithmus
        'threshold': best_threshold,               # Der optimale Schwellenwert für Klassifikation
        'feature_names': X_selected.columns.tolist(),  # Namen der verwendeten Features
        'metrics': {                               # Performancemetriken
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1
        },
        'X_test': X_test,                          # Testdaten für spätere Evaluierungen
        'y_test': y_test                           # Tatsächliche Labels für Testdaten
    }

def predict_with_f9tuned(model_info, new_data):
    """
    Führt Vorhersagen mit dem F9Tuned-Modell durch
    
    Args:
        model_info (dict): Modellinformationen aus train_f9tuned_model()
        new_data (pandas.DataFrame): Neue Daten für die Vorhersage
        
    Returns:
        tuple: (Wahrscheinlichkeiten, binäre Vorhersagen)
    """
    # Neue Daten mit den gleichen Schritten wie beim Training vorbereiten
    # Dies stellt sicher, dass die Datenformate konsistent sind
    prepared_data = prepare_dataset(new_data)
    
    # Features auswählen und One-Hot-Encoding anwenden
    # Dadurch erhalten wir die gleiche Feature-Struktur wie beim Training
    X_selected = select_features(prepared_data)
    
    # Die vom Modell erwarteten Features abrufen
    # Diese stammen aus dem Training und müssen exakt übereinstimmen
    required_features = model_info['feature_names']
    
    # Fehlende Spalten mit 0 auffüllen
    # Dies kann passieren, wenn bestimmte kategorische Werte in den neuen Daten fehlen
    for feature in required_features:
        if feature not in X_selected.columns:
            X_selected[feature] = 0
    
    # Die Reihenfolge der Features ist entscheidend!
    # Wir stellen sicher, dass die Spalten exakt der Reihenfolge beim Training entsprechen
    X_selected = X_selected.reindex(columns=required_features, fill_value=0)
    
    # Modell und Schwellenwert aus dem model_info Dictionary extrahieren
    model = model_info['model']              # Das trainierte RandomForest-Modell
    threshold = model_info['threshold']      # Der optimale Klassifikationsschwellenwert (0.4)
    
    # Vorhersage durchführen
    # Wir erhalten zunächst Wahrscheinlichkeiten für die positive Klasse (Kauf)
    probabilities = model.predict_proba(X_selected)[:, 1]
    
    # Dann konvertieren wir diese zu binären Entscheidungen basierend auf dem Schwellenwert
    predictions = (probabilities >= threshold).astype(int)
    
    # Beide Werte zurückgeben: Die Wahrscheinlichkeiten sind nützlich für detaillierte Analysen,
    # die binären Vorhersagen für einfache Ja/Nein-Entscheidungen
    return probabilities, predictions

def get_f9tuned_evaluation(model_info):
    """
    Liefert Evaluationsmetriken für das F9Tuned-Modell
    
    Args:
        model_info (dict): Modellinformationen aus train_f9tuned_model()
        
    Returns:
        dict: Dictionary mit Evaluationsmetriken
    """
    # Metriken aus dem Modell-Dictionary abrufen
    metrics = model_info['metrics']
    model_name = model_info['model_name']
    
    # Zusammenfassung der Metriken
    evaluation = {
        'model_name': model_name,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score']
    }
    
    return evaluation