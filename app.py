import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from data_loader import load_uci_shopping_dataset
from data_analysis import (
    get_summary_statistics, 
    analyze_numerical_features, 
    analyze_categorical_features
)
from data_visualization import (
    plot_feature_importance,
    plot_correlation_heatmap,
    plot_categorical_distribution,
    plot_pairplot,
    plot_revenue_by_visitor_type,
    plot_purchase_by_month
)
from ml_models import (
    prepare_data,
    build_and_evaluate_models,
    train_best_model,
    predict_purchase_intention,
    get_model_interpretation
)
from utils import (
    export_dataframe_to_csv,
    create_download_link,
    save_uploaded_file
)

# Constants
MODEL_NAMES = {
    'F9Tuned (LightGBM)': 'F9Tuned (LightGBM) - Optimiert für maximale Leistung bei E-Commerce-Daten'
}

# Set page configuration
st.set_page_config(
    page_title="E-Commerce Kaufabsicht",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for styled components (Wurde mit ChatGPT verbessert)
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 10px;
}
.metric-label {
    font-size: 0.8rem;
    color: #444;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar erstellen
    st.sidebar.title("E-Commerce Analyse")
    
    # Navigation Tabs erstellen
    selection = st.sidebar.radio(
        "Navigation",
        ["Datenübersicht mit Visualisierungen", "Vorhersagemodelle", "Interaktive Prognose"]
    )
    
    # Data loading
    with st.spinner("Datensatz wird geladen..."):
        df = load_uci_shopping_dataset()
    
    if selection == "Datenübersicht mit Visualisierungen":
        data_overview_page(df)
    elif selection == "Vorhersagemodelle":
        try:
            ml_models_page(df)
        except Exception as e:
            st.error(f"Fehler beim Laden der Vorhersagemodelle: {str(e)}")
            st.info("Die Modellierungsfunktionen stehen derzeit nicht zur Verfügung. Bitte versuchen Sie es später erneut oder nutzen Sie die anderen Funktionen der Anwendung.")
    elif selection == "Interaktive Prognose":
        try:
            interactive_prediction_page(df)
        except Exception as e:
            st.error(f"Fehler bei der interaktiven Prognose: {str(e)}")
            st.info("Die Prognosefunktionen stehen derzeit nicht zur Verfügung. Bitte versuchen Sie es später erneut oder nutzen Sie die Visualisierungsfunktionen der Anwendung.")


def data_overview_page(df):
    st.header("Online Shoppers Purchasing Intention Dataset")
    
    # Display basic info about the dataset
    st.subheader("Übersicht zum UCI-Datensatz")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Anzahl der Datensätze", df.shape[0])
    with col2:
        st.metric("Anzahl der Merkmale", df.shape[1])
    with col3:
        purchase_rate = df['Revenue'].mean() * 100
        st.metric("Kaufrate", f"{purchase_rate:.2f}%")
    
    st.markdown("""
    ### Über den UCI-Datensatz "Online Shoppers Purchasing Intention"
    
    Der "Online Shoppers Purchasing Intention" Datensatz stammt aus dem UCI Machine Learning Repository 
    und enthält Informationen über Besuch und Interaktionen mit einer E-Commerce-Webseite, die 
    Administrativen-, Informations- und Produktseiten enthält. Das Ziel ist die Vorhersage, ob ein Besucher 
    eine Kaufabsicht hat, was durch die binäre Revenue-Variable (0 = kein Kauf, 1 = Kauf) angezeigt wird.
    
    #### Merkmale im Datensatz:
    
    **Administrative Features:**
    - **Administrative**: Anzahl der besuchten Administrationsseiten
    - **Administrative_Duration**: Gesamtzeit auf Administrationsseiten (in Sekunden)
    
    **Informational Features:**
    - **Informational**: Anzahl der besuchten Informationsseiten
    - **Informational_Duration**: Gesamtzeit auf Informationsseiten (in Sekunden)
    
    **Product-Related Features:**
    - **ProductRelated**: Anzahl der besuchten Produktseiten
    - **ProductRelated_Duration**: Gesamtzeit auf Produktseiten (in Sekunden)
    
    **Andere Merkmale:**
    - **BounceRates**: Prozentsatz der Besucher, die die Website von dieser Seite aus verlassen haben, ohne weitere Aktionen auszuführen
    - **ExitRates**: Prozentsatz der Seitenansichten auf der Website, die die letzten in der Sitzung waren
    - **PageValues**: Durchschnittlicher Wert für die Webseite, basierend auf e-Commerce-Transaktionen
    - **SpecialDay**: Nähe des Webseitenbesuchs zu einem speziellen Tag (z.B. Muttertag, Valentinstag)
    - **Month**: Monat des Jahres
    - **OperatingSystems**: Identifikationsnummer des Betriebssystems des Besuchers
    - **Browser**: Identifikationsnummer des Browsers des Besuchers
    - **Region**: Identifikationsnummer der Region des Besuchers
    - **TrafficType**: Identifikationsnummer des Verkehrstyps (z.B. direkt, Suchmaschine)
    - **VisitorType**: Typ des Besuchers (Wiederkehrend, Neu, Andere)
    - **Weekend**: Ob der Tag ein Wochenende ist (Wahr/Falsch)
    
    **Zielvariable:**
    - **Revenue**: Ob der Besucher einen Kauf getätigt hat (Wahr/Falsch)
    """)
    
    # Interactive EDA
    st.subheader("Explorative Datenanalyse")
    
    # Tabs for different types of analysis
    tabs = st.tabs([
        "Allgemeine Statistiken", 
        "Korrelationsanalyse",
        "Besuchertyp-Analyse",
        "Zeitliche Muster"
    ])
    
    with tabs[0]:
        st.subheader("Deskriptive Statistiken")
        stats_df = get_summary_statistics(df)
        st.dataframe(stats_df, use_container_width=True)
        
        # Comparing features between converters and non-converters
        st.subheader("Feature-Vergleich: Käufer vs. Nicht-Käufer")
        num_features = analyze_numerical_features(df)
        st.dataframe(num_features)
        

    
    with tabs[1]:
        st.subheader("Korrelationsanalyse")
        
        try:
            # Correlation heatmap
            st.write("#### Korrelationsmatrix der numerischen Features")
            corr_fig = plot_correlation_heatmap(df)
            st.pyplot(corr_fig)
            

        except Exception as e:
            st.error(f"Fehler bei der Korrelationsanalyse: {str(e)}")
    
    with tabs[2]:
        st.subheader("Besuchertyp-Analyse")
        
        try:
            # Visitor type analysis
            st.write("#### Konversionsrate nach Besuchertyp")
            visitor_fig = plot_revenue_by_visitor_type(df)
            st.plotly_chart(visitor_fig, use_container_width=True)
            
            # Detailed visitor metrics
            st.write("#### Detaillierte Metriken nach Besuchertyp")
            
            # Calculate metrics for each visitor type
            visitor_metrics = {}
            
            for visitor_type in df['VisitorType'].unique():
                visitor_df = df[df['VisitorType'] == visitor_type]
                
                visitor_metrics[visitor_type] = {
                    'Anzahl': len(visitor_df),
                    'Anteil (%)': f"{len(visitor_df) / len(df) * 100:.2f}%",
                    'Konversionsrate (%)': f"{visitor_df['Revenue'].mean() * 100:.2f}%",
                    'Durchschnittl. PageValues': f"{visitor_df['PageValues'].mean():.2f}",
                    'Durchschnittl. Besuchsdauer (s)': f"{visitor_df['ProductRelated_Duration'].mean():.2f}"
                }
                
            # Erstelle DataFrame für die Besuchertyp-Analyse, aber transponiere ihn für bessere Darstellung
            visitor_df = pd.DataFrame(visitor_metrics).T
            # Konvertiere alle Spalten zum String-Format, um Arrow-Kompatibilität sicherzustellen
            visitor_df = visitor_df.astype(str)
            # Zeige die Besuchertyp-Analyse als Tabelle an
            st.dataframe(visitor_df, use_container_width=True)
        except Exception as e:
            st.error(f"Fehler bei der detaillierten Besuchertyp-Analyse: {str(e)}")

    with tabs[3]:
        st.subheader("Zeitliche Muster")
        
        try:
            # Monthly patterns
            st.write("#### Kaufrate nach Monaten")
            monthly_fig = plot_purchase_by_month(df)
            st.plotly_chart(monthly_fig, use_container_width=True)
            
            # Weekend vs Weekday
            st.write("#### Wochenende vs. Wochentag")
            
            # Manuelle Berechnung der Statistiken
            werktag_besuche = df[df['Weekend'] == False].shape[0]
            wochenend_besuche = df[df['Weekend'] == True].shape[0]
            werktag_konversion = df[df['Weekend'] == False]['Revenue'].mean() * 100
            wochenend_konversion = df[df['Weekend'] == True]['Revenue'].mean() * 100
            
            # Erstelle zwei Spalten für die Anzeige
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Besuche am Werktag", f"{werktag_besuche:,}".replace(",", "."))
                st.metric("Konversionsrate Werktag", f"{werktag_konversion:.2f}%")
            
            with col2:
                st.metric("Besuche am Wochenende", f"{wochenend_besuche:,}".replace(",", "."))
                st.metric("Konversionsrate Wochenende", f"{wochenend_konversion:.2f}%")
            
            # Erstellen eines Balkendiagramm für Wochenende vs. Werktag
            weekend_data = pd.DataFrame({
                'Zeitraum': ['Werktag', 'Wochenende'],
                'Konversionsrate': [werktag_konversion, wochenend_konversion]
            })
            
            fig = px.bar(
                weekend_data, 
                x='Zeitraum', 
                y='Konversionsrate',
                text=weekend_data['Konversionsrate'].round(2).astype(str) + '%',
                title='Konversionsraten: Wochenende vs. Werktag',
                color='Zeitraum',
                color_discrete_map={'Werktag': '#636EFA', 'Wochenende': '#EF553B'}
            )
            
            fig.update_layout(
                xaxis_title='Zeitraum',
                yaxis_title='Konversionsrate (%)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Fehler bei der Analyse zeitlicher Muster: {str(e)}")
        
    # Data filtering und export optionen
    st.subheader("Datenfilterung und Export")
    
    # Filter Optionen
    col1, col2 = st.columns(2)
    
    with col1:
        visitor_filter = st.multiselect(
            "Besuchertyp",
            options=sorted(df['VisitorType'].unique()),
            default=[]
        )
        
        month_filter = st.multiselect(
            "Monat",
            options=sorted(df['Month'].unique()),
            default=[]
        )
    
    with col2:
        weekend_filter = st.multiselect(
            "Wochenende",
            options=sorted(df['Weekend'].unique()),
            default=[]
        )
        
        revenue_filter = st.multiselect(
            "Kaufabsicht",
            options=sorted(df['Revenue'].unique()),
            default=[]
        )
    
    # Filter Anwendung
    filtered_df = df.copy()
    
    if visitor_filter:
        filtered_df = filtered_df[filtered_df['VisitorType'].isin(visitor_filter)]
    
    if month_filter:
        filtered_df = filtered_df[filtered_df['Month'].isin(month_filter)]
    
    if weekend_filter:
        filtered_df = filtered_df[filtered_df['Weekend'].isin(weekend_filter)]
    
    if revenue_filter:
        filtered_df = filtered_df[filtered_df['Revenue'].isin(revenue_filter)]
    
    # Anzeige der gefilterten Daten
    st.subheader("Gefilterte Daten")
    st.write(f"{len(filtered_df)} Einträge wurden ausgewählt")
    st.dataframe(filtered_df)
    
    # Gefilterte Daten als CSV exportieren
    csv_data = export_dataframe_to_csv(filtered_df)
    st.download_button(
        label="Als CSV herunterladen",
        data=csv_data,
        file_name="e_commerce_kaufabsicht_daten.csv",
        mime="text/csv"
    )

def ml_models_page(df):
    st.header("F9Tuned-Modell für Kaufabsichten")
    
    try:
        # Prepare data for ML
        X, y, X_train, X_test, y_train, y_test = prepare_data(df)
        
        # Split settings
        st.subheader("Datenaufteilung für das Training")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trainingsdaten (80%)", f"{len(X_train)} Einträge")
        with col2:
            st.metric("Testdaten (20%)", f"{len(X_test)} Einträge")
        with col3:
            st.metric("Positive Klasse (Käufe)", f"{sum(y)} ({sum(y)/len(y):.1%})")
        
        # Model evaluation
        st.subheader("F9Tuned-Modell")
        model_results = build_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Display F9Tuned metrics
        model_name = 'F9Tuned (LightGBM)'
        if isinstance(model_results, dict) and model_name in model_results:
            metrics_dict = model_results[model_name]
            
            # Sicherstellen, dass wir einen Dictionary haben
            if not isinstance(metrics_dict, dict):
                metrics_dict = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0}
                
            # Metrikwerte extrahieren
            accuracy = metrics_dict.get('Accuracy', 0)
            precision = metrics_dict.get('Precision', 0)
            recall = metrics_dict.get('Recall', 0)
            f1_score = metrics_dict.get('F1 Score', 0)
        else:
            accuracy = precision = recall = f1_score = 0
        
        # DataFrame für die Anzeige erstellen
        model_metrics = pd.DataFrame({
            'Metrik': ['Genauigkeit (Accuracy)', 'Präzision (Precision)', 'Trefferquote (Recall)', 'F1-Score'],
            'Wert': [accuracy, precision, recall, f1_score]
        })
        st.dataframe(model_metrics, use_container_width=True)
        
        # Highlight model
        st.success(f"F9Tuned-Modell (Optimierter Random Forest) mit einer Genauigkeit von {accuracy:.2%}")
        
        # Model interpretation
        st.subheader("Modellinterpretation")
        
        interpretation = get_model_interpretation()
        st.markdown(interpretation)
        
        # Confusion Matrix and ROC Curve
        st.subheader("Modellbewertung")
        
        # Train the best model
        result = train_best_model(X_train, X_test, y_train, y_test)
        
        # Prüfen, ob result ein Tupel mit 3 Elementen ist
        if isinstance(result, tuple) and len(result) == 3:
            trained_model, conf_matrix, classification_rep = result
            
            # Konfusionsmatrix anzeigen
            st.write("Konfusionsmatrix")
            if conf_matrix is not None:
                st.pyplot(conf_matrix)
            else:
                st.info("Keine Konfusionsmatrix verfügbar")
            
            st.write("Klassifikationsbericht")
            if classification_rep is not None:
                st.text(classification_rep)
            else:
                st.info("Kein Klassifikationsbericht verfügbar")
        else:
            st.warning("Modelltraining konnte nicht abgeschlossen werden. Versuchen Sie es später erneut.")
    
    except Exception as e:
        st.error(f"Fehler beim Modelltraining: {str(e)}")
        st.info("Das F9Tuned-Modell konnte nicht erstellt werden. Bitte versuchen Sie es später erneut oder kontaktieren Sie den Support.")

def interactive_prediction_page(df):
    st.header("Interaktive Kaufabsicht-Prognose")
    
    try:
        # Train a model using all data
        X, y, X_train, X_test, y_train, y_test = prepare_data(df)
        
        # Modellnamen setzen
        model_name = 'F9Tuned (LightGBM)'
        
        # Train the F9Tuned model with all data
        # F9Tuned-Modell laden
        try:
            model_result = train_best_model(X, None, y, None, return_only_model=True)
            if isinstance(model_result, tuple) and len(model_result) >= 2:
                # Wenn wir das erwartete Ergebnis haben
                model_name, best_model = model_result
            else:
                # Bei Fehler - setze best_model auf None und fahre fort
                best_model = None
        except Exception as model_error:
            st.error(f"Fehler beim Laden des Modells: {str(model_error)}")
            best_model = None
            
        # Informationstext
        st.info("""
        In diesem Bereich können Sie auf zwei verschiedene Arten Vorhersagen treffen:
        1. Durch manuelle Eingabe einzelner Werte
        2. Durch Hochladen eines CSV-Datensatzes für Massenvorhersagen
        
        Das F9Tuned-Modell ist mit dem 'Online Shoppers Purchasing Intention' Datensatz trainiert und kann auf ähnlich strukturierte Daten angewendet werden.
        """)
        
        # Tabs für Manuelle Eingabe oder CSV-Upload
        tab1, tab2 = st.tabs(["Manuelle Eingabe", "CSV-Upload für Massenvorhersagen"])
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {str(e)}")
        st.info("Die Vorhersagefunktion ist derzeit nicht verfügbar. Bitte versuchen Sie es später erneut.")
        return
    
    with tab1:
        st.markdown("""
        Hier können Sie verschiedene Merkmale anpassen, um die Kaufwahrscheinlichkeit für 
        unterschiedliche Kundensegmente zu prognostizieren.
        """)
        
        # Inputs for prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            visitor_type = st.selectbox(
                "Besuchertyp",
                options=df['VisitorType'].unique(),
                index=0
            )
            
            browser = st.selectbox(
                "Browser",
                options=sorted(df['Browser'].unique()),
                index=0
            )
            
            os = st.selectbox(
                "Betriebssystem",
                options=sorted(df['OperatingSystems'].unique()),
                index=0
            )
        
        with col2:
            month = st.selectbox(
                "Monat",
                options=sorted(df['Month'].unique()),
                index=0
            )
            
            weekend = st.checkbox(
                "Wochenende?",
                value=False
            )
            
            special_day = st.slider(
                "Nähe zu einem speziellen Tag (0 = weit entfernt, 1 = sehr nah)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
        
        with col3:
            page_values = st.slider(
                "Page Values",
                min_value=0.0,
                max_value=float(df['PageValues'].max()),
                value=0.0,
                step=10.0
            )
            
            bounce_rates = st.slider(
                "Bounce Rates",
                min_value=0.0,
                max_value=float(df['BounceRates'].max()),
                value=float(df['BounceRates'].mean()),
                step=0.01
            )
            
            exit_rates = st.slider(
                "Exit Rates",
                min_value=0.0,
                max_value=float(df['ExitRates'].max()),
                value=float(df['ExitRates'].mean()),
                step=0.01
            )
        
        # Page visit sliders
        st.subheader("Seitenbesuche und Verweildauer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            administrative = st.slider(
                "Administrative Seiten Besuche",
                min_value=0,
                max_value=int(df['Administrative'].max()),
                value=0
            )
            
            informational = st.slider(
                "Informational Seiten Besuche",
                min_value=0,
                max_value=int(df['Informational'].max()),
                value=0
            )
            
            product_related = st.slider(
                "Product Related Seiten Besuche",
                min_value=0,
                max_value=int(df['ProductRelated'].max()),
                value=0
            )
        
        with col2:
            administrative_duration = st.slider(
                "Administrative Seiten Verweildauer (s)",
                min_value=0.0,
                max_value=float(df['Administrative_Duration'].max()),
                value=0.0
            )
            
            informational_duration = st.slider(
                "Informational Seiten Verweildauer (s)",
                min_value=0.0,
                max_value=float(df['Informational_Duration'].max()),
                value=0.0
            )
            
            product_related_duration = st.slider(
                "Product Related Seiten Verweildauer (s)",
                min_value=0.0,
                max_value=float(df['ProductRelated_Duration'].max()),
                value=0.0
            )
        
        # Create input dataframe for prediction
        if st.button("Vorhersage treffen"):
            with st.spinner("Berechnung läuft..."):
                # Create input data
                input_data = pd.DataFrame({
                    'Administrative': [administrative],
                    'Administrative_Duration': [administrative_duration],
                    'Informational': [informational],
                    'Informational_Duration': [informational_duration],
                    'ProductRelated': [product_related],
                    'ProductRelated_Duration': [product_related_duration],
                    'BounceRates': [bounce_rates],
                    'ExitRates': [exit_rates],
                    'PageValues': [page_values],
                    'SpecialDay': [special_day],
                    'Month': [month],
                    'OperatingSystems': [os],
                    'Browser': [browser],
                    'Region': [df['Region'].mode()[0]],  # Use most common region
                    'TrafficType': [df['TrafficType'].mode()[0]],  # Use most common traffic type
                    'VisitorType': [visitor_type],
                    'Weekend': [weekend]
                })
                
                # Preprocess input data
                processed_input = prepare_data(df, new_data=input_data)
                
                # Make prediction
                try:
                    probabilities, prediction = predict_purchase_intention(processed_input)
                    # Überprüfen, ob probabilities ein ndarray oder eine Liste ist
                    purchase_prob = probabilities[0]
                    
                    # Display prediction
                    st.subheader("Vorhersageergebnis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction[0] == 1:
                            st.success(f"Der Besucher wird mit {purchase_prob:.1%} Wahrscheinlichkeit einen Kauf tätigen.")
                        else:
                            st.info(f"Der Besucher wird mit {1-purchase_prob:.1%} Wahrscheinlichkeit keinen Kauf tätigen.")
                    
                    with col2:
                        # Visualization
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.barh(['Kein Kauf', 'Kauf'], [1-purchase_prob, purchase_prob])
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Wahrscheinlichkeit')
                        ax.set_title('Kaufwahrscheinlichkeit')
                        
                        for i, v in enumerate([1-purchase_prob, purchase_prob]):
                            ax.text(v, i, f'{v:.1%}', va='center')
                        
                        st.pyplot(fig)
                    
                    # Feature importance interpretation
                    st.subheader("Einflussfaktoren auf die Entscheidung")
                    
                    # Get model interpretation for this specific prediction
                    interpretation = get_model_interpretation(model=best_model, X=processed_input, for_single_prediction=True)
                    
                    st.markdown(interpretation)
                    
                except Exception as e:
                    st.error(f"Fehler bei der Vorhersage: {str(e)}")
    
    with tab2:
        st.markdown("""
        Hier können Sie eine CSV-Datei mit Besucherdaten hochladen, um Massenvorhersagen zu treffen.
        Die Datei sollte die gleichen Spalten wie der Trainingsdatensatz haben.
        """)
        
        # Upload file
        uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei", type=["csv"])
        
        # Example download
        st.markdown("**Beispieldatei herunterladen:**")
        example_df = df.sample(5).drop('Revenue', axis=1)
        example_csv = export_dataframe_to_csv(example_df)
        st.download_button(
            label="Beispieldatei herunterladen",
            data=example_csv,
            file_name="kaufabsicht_beispiel.csv",
            mime="text/csv"
        )
        
        if uploaded_file is not None:
            try:
                # Load file
                file_data = save_uploaded_file(uploaded_file)
                
                # Parse CSV
                input_df = pd.read_csv(BytesIO(file_data.getvalue()))
                
                st.write("Vorschau der hochgeladenen Daten:")
                st.dataframe(input_df.head())
                
                # Check if all required columns are present
                required_columns = [col for col in df.columns if col != 'Revenue']
                missing_columns = [col for col in required_columns if col not in input_df.columns]
                
                if missing_columns:
                    st.error(f"Fehlende Spalten in der hochgeladenen Datei: {', '.join(missing_columns)}")
                else:
                    # Make predictions button
                    if st.button("Vorhersagen für alle Datensätze treffen"):
                        try:
                            with st.spinner("Bereite Daten für die Vorhersage vor..."):
                                # Preprocess data
                                processed_input = prepare_data(df, new_data=input_df)
                                
                                # Make predictions
                                probabilities, predictions = predict_purchase_intention(processed_input)
                                
                                # Add predictions to the original data
                                result_df = input_df.copy()
                                result_df['Kaufwahrscheinlichkeit'] = probabilities
                                result_df['Kaufabsicht_Vorhersage'] = predictions
                                
                                # Display results
                                st.subheader("Vorhersageergebnisse")
                                st.write(f"Vorhergesagte Kaufrate: {np.mean(predictions):.2%}")
                                st.dataframe(result_df)
                                
                                # Export results
                                csv = export_dataframe_to_csv(result_df)
                                st.download_button(
                                    "Vorhersageergebnisse als CSV herunterladen",
                                    csv,
                                    "vorhersage_ergebnisse.csv",
                                    "text/csv",
                                    key='download-results'
                                )
                            
                        except Exception as e:
                            st.error(f"Fehler bei der Vorhersage: {str(e)}")
                            st.info("Stellen Sie sicher, dass Ihre Datei das richtige Format hat. Prüfen Sie das Format anhand der Beispieldaten.")
            
            except Exception as e:
                st.error(f"Fehler beim Verarbeiten der hochgeladenen Datei: {str(e)}")
                st.info("Die Datei muss im CSV-Format sein. Laden Sie eine gültige CSV-Datei hoch.")

if __name__ == "__main__":
    main()