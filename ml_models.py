import pandas as pd         # Für Dataframe-Operationen und Datenmanipulation
import numpy as np          # Für numerische Berechnungen
import matplotlib.pyplot as plt  # Für Visualisierungen
import seaborn as sns       # Für erweiterte statistische Visualisierungen
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report  # Für Modellevaluation
from sklearn.model_selection import train_test_split  # Für die Aufteilung von Trainings- und Testdaten

from f9tuned_simplified import (
    train_f9tuned_model,     # Trainingsfunktion für das F9Tuned-Modell
    predict_with_f9tuned,    # Vorhersagefunktion
    get_f9tuned_evaluation,  # Liefert Evaluationsmetriken
    prepare_dataset,         # Bereitet Daten für das Modell vor
    select_features          # Wählt die wichtigsten Features aus
)

# Speichertt das trainierte Modell und alle relevanten Informationen
#
f9_model_info = None

def prepare_data(df, test_size=0.2, random_state=42, new_data=None):
    """
    Prepare data for machine learning models
    
    Args:
        df (pandas.DataFrame): The dataset
        test_size (float, optional): Proportion of the dataset to include in the test split
        random_state (int, optional): Random state for reproducibility
        new_data (pandas.DataFrame, optional): New data to transform using preprocessing pipeline
        
    Returns:
        If new_data is None:
            tuple: (X, y, X_train, X_test, y_train, y_test)
        Else:
            pandas.DataFrame: Transformed new_data
    """
    # Diese Funktion hat zwei verschiedene Modi:
    # 1. Verarbeitung neuer Daten für Vorhersagen (wenn new_data vorhanden ist)
    # 2. Aufbereitung des Datensatzes für das Training (wenn new_data None ist)
    
    # Modus 1: Verarbeitung neuer Daten für Vorhersagen
    if new_data is not None:
        # Bei neuen Daten wende ich die gleiche Vorverarbeitung an wie beim Training
        # Dies stellt sicher, dass die Datenformate konsistent sind
        prepared_data = prepare_dataset(new_data)  # Konvertiere Datentypen
        selected_features = select_features(prepared_data)  # Wähle relevante Features aus
        return selected_features
        
    # Modus 2: Aufbereitung des Datensatzes für das Training
    
    # Prüfe, ob die Zielvariable 'Revenue' im Datensatz vorhanden ist
    # Dies ist eine wichtige Validierung, um unerwartete Fehler zu vermeiden
    if 'Revenue' not in df.columns:
        raise ValueError("Datensatz muss eine 'Revenue' Spalte enthalten")
    
    # Trenne Features (X) und Zielvariable (y)
    # Alle Spalten außer 'Revenue' sind Features
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    
    # Teile Daten in Trainings- und Testsets mit festgelegtem Verhältnis (standardmäßig 80/20)
    # Stratifizierung stellt sicher, dass die Klassenverteilung in beiden Sets gleich bleibt
    # Der feste Random State garantiert Reproduzierbarkeit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Die eigentliche Datenvorverarbeitung wird vom F9Tuned-Modell selbst durchgeführt
    # Daher behalten wir hier die Daten unverändert - ein bewusster Architekturentscheid
    # Dies vereinfacht die Pipeline und vermeidet doppelte Transformationen
    X_processed = X
    X_train_processed = X_train
    X_test_processed = X_test
    
    # Gebe alle aufbereiteten Daten zurück für weiteres Training und Evaluation
    return X_processed, y, X_train_processed, X_test_processed, y_train, y_test

def build_and_evaluate_models(X_train, X_test, y_train, y_test, return_only_model=False):
    """
    Build and evaluate the F9Tuned model
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        y_train (pandas.Series): Training target
        y_test (pandas.Series): Testing target
        return_only_model (bool, optional): Whether to return only the best model name and model
        
    Returns:
        tuple: (model_results, best_model_name, best_model)
    """
    # Verwende nur das F9Tuned-Modell (LightGBM)
    model_name = 'F9Tuned (LightGBM)'
    
    # Erstelle einen temporären DataFrame für das Training
    train_df = X_train.copy()
    train_df['Revenue'] = y_train
    
    # Trainiere das F9Tuned-Modell
    global f9_model_info
    f9_model_info = train_f9tuned_model(train_df)
    model = f9_model_info['model']
    
    # Speichere die Metriken
    metrics = f9_model_info['metrics']
    model_results = {
        model_name: {
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1 Score': metrics.get('f1_score', 0),
            'Model': model
        }
    }
    
    # Gebe die Ergebnisse zurück
    if return_only_model:
        return model_results, model_name, model
    else:
        return model_results

def train_best_model(X_train, X_test, y_train, y_test, best_model_name=None, return_only_model=False):
    """
    Train the F9Tuned model and return evaluation metrics
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        y_train (pandas.Series): Training target
        y_test (pandas.Series): Testing target
        best_model_name (str, optional): Not used, F9Tuned is always the best model
        return_only_model (bool, optional): Whether to return only the model name and trained model
        
    Returns:
        If return_only_model is True:
            tuple: (best_model_name, best_model)
        Else:
            tuple: (best_model, confusion_matrix_fig, roc_curve_fig, classification_report_str)
    """
    model_name = 'F9Tuned'
    
    # Prüfe, ob das Modell bereits trainiert wurde
    global f9_model_info
    if f9_model_info is None or 'model' not in f9_model_info:
        # Falls nicht, trainiere es jetzt
        train_df = X_train.copy()
        train_df['Revenue'] = y_train
        f9_model_info = train_f9tuned_model(train_df)
    
    best_model = f9_model_info['model']
    
    # Modell-Typ-Information aus dem Modell-Info-Dictionary auslesen
    model_full_name = f"{model_name} ({f9_model_info.get('model_name', 'RandomForest')})"
    
    # Falls nur das Modell benötigt wird, gib es zurück
    if return_only_model:
        return model_full_name, best_model
    
    # Ansonsten fahre mit der Auswertung fort
    if X_test is not None and y_test is not None:
        # Mache Vorhersagen
        probabilities, predictions = predict_with_f9tuned(f9_model_info, X_test)
        
        # Erstelle Konfusionsmatrix-Plot
        cm = confusion_matrix(y_test, predictions)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Vorhergesagt')
        ax_cm.set_ylabel('Tatsächlich')
        ax_cm.set_title(f'Konfusionsmatrix ({model_name})')
        
        # ROC-Kurve entfernt auf Wunsch des Benutzers
        
        # Generiere Klassifikationsbericht
        report = classification_report(y_test, predictions, target_names=["Kein Kauf", "Kauf"])
        
        return best_model, fig_cm, report
    else:
        # Falls keine Testdaten bereitgestellt werden, gib nur das Modell zurück
        return best_model, None, "Keine Testdaten zur Evaluierung bereitgestellt"

def predict_purchase_intention(input_data, model_name=None):
    """
    Predict purchase intention for new data
    
    Args:
        input_data (pandas.DataFrame): Preprocessed input data
        model_name (str, optional): Not used, F9Tuned is always used
        
    Returns:
        tuple: (probabilities, predictions)
    """
    # Prüfe, ob das F9Tuned-Modell trainiert wurde
    global f9_model_info
    if f9_model_info is None or 'model' not in f9_model_info:
        raise ValueError("F9Tuned-Modell muss zuerst trainiert werden")
    
    # Mache Vorhersagen mit F9Tuned
    probabilities, predictions = predict_with_f9tuned(f9_model_info, input_data)
    
    return probabilities, predictions

def get_model_interpretation(model=None, X=None, model_name=None, for_single_prediction=False):
    """
    Get interpretation of model predictions
    
    Args:
        model: Not used, F9Tuned is always used
        X (pandas.DataFrame): Feature data
        model_name (str, optional): Not used, F9Tuned is always used
        for_single_prediction (bool, optional): Whether the interpretation is for a single prediction
        
    Returns:
        str: Model interpretation as a string
    """
    # Prüfe, ob das F9Tuned-Modell trainiert wurde
    global f9_model_info
    if f9_model_info is None:
        return """
        ## F9Tuned Modell

        Das F9Tuned-Modell wurde noch nicht trainiert. 
        Bitte starten Sie das Training, bevor Sie eine Interpretation anfordern.
        """
    
    # Basisinterpretation, da get_f9tuned_model_interpretation in der Original-Version nicht existiert
    model_name = f9_model_info.get('model_name', 'Ensemble')
    metrics = f9_model_info.get('metrics', {})
    
    # Generiere eine Interpretation basierend auf den verfügbaren Informationen
    interpretation = f"""
    ## F9Tuned-Modell Interpretation ({model_name})

    Das F9Tuned-Modell nutzt fortschrittliche Machine-Learning-Algorithmen, um Kaufabsichten 
    von E-Commerce-Website-Besuchern vorherzusagen.
    
    ### Modellleistung:
    
    - **Genauigkeit (Accuracy)**: {metrics.get('accuracy', 0):.4f}
    - **Präzision (Precision)**: {metrics.get('precision', 0):.4f}
    - **Trefferquote (Recall)**: {metrics.get('recall', 0):.4f}
    - **F1-Score**: {metrics.get('f1_score', 0):.4f}
    
    ### Wichtigste Merkmale:
    
    1. **PageValues**: Der mit Abstand wichtigste Prädiktor. Hohe PageValues (der Wert einer Webseite für den Online-Shop) 
       korrelieren stark mit Kaufabsichten.
    
    2. **ExitRates**: Niedrigere ExitRates (Prozentsatz der Website-Besuche, die mit dieser Seite enden) 
       weisen auf eine höhere Kaufwahrscheinlichkeit hin.
    
    3. **BounceRates**: Niedrigere BounceRates (Prozentsatz der Besucher, die die Website nach nur einer Seite verlassen) 
       korrelieren mit höherer Kaufwahrscheinlichkeit.
    
    4. **SpecialDay**: Der Einfluss von besonderen Tagen (z.B. nahe Valentinstag, Weihnachten) 
       auf das Kaufverhalten.
    
    5. **Month**: Saisonale Effekte beeinflussen das Kaufverhalten, wobei einige Monate 
       höhere Konversionsraten aufweisen.
    """
    
    return interpretation