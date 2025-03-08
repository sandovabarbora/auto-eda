import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from eda import EDA

# OICT barevná paleta
OICT_COLORS = {
    'purple': '#574494',
    'yellow': '#FFE14F',
    'orange': '#E37222',
    'green': '#74ECA1',
    'black': '#000000',
    'white': '#FFFFFF'
}

OICT_PALETTE = [OICT_COLORS['purple'], OICT_COLORS['yellow'], OICT_COLORS['orange'], OICT_COLORS['green']]

def aplikuj_oict_styl():
    """Aplikuje vlastní OICT stylování na Streamlit aplikaci - moderní a výraznější verze"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #574494;
        --primary-dark: #463677;
        --primary-light: #6854a8;
        --secondary-color: #323232;
        --accent-yellow: #FFE14F;
        --accent-orange: #E37222;
        --accent-green: #74ECA1;
        --background-color: #f8f9fa;
        --card-color: #ffffff;
        --text-color: #333333;
        --text-light: #666666;
        --transition: all 0.3s ease;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: var(--background-color);
        background-image: 
            linear-gradient(120deg, rgba(87, 68, 148, 0.03) 0%, rgba(255, 255, 255, 0) 100%),
            radial-gradient(circle at 90% 10%, rgba(231, 114, 34, 0.02) 0%, rgba(255, 255, 255, 0) 70%),
            radial-gradient(circle at 10% 90%, rgba(116, 236, 161, 0.02) 0%, rgba(255, 255, 255, 0) 70%);
        background-attachment: fixed;
    }
    
    /* Header styling */
    .title-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .title-container::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 10%;
        width: 80%;
        height: 2px;
        background: linear-gradient(90deg, 
            rgba(87, 68, 148, 0), 
            rgba(87, 68, 148, 1) 20%, 
            rgba(87, 68, 148, 1) 80%, 
            rgba(87, 68, 148, 0) 100%);
    }
    
    .title-container h1 {
        color: var(--primary-color);
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        transform: scale(1);
        transition: var(--transition);
    }
    
    .title-container h1:hover {
        transform: scale(1.02);
    }
    
    .subtitle {
        color: var(--text-light);
        font-size: 1.3rem;
        font-weight: 400;
        margin-top: 0;
        opacity: 0.9;
    }
    
    /* Logo styling */
    .logo-container {
        display: inline-block;
        background-color: var(--primary-color);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 6px;
        font-weight: bold;
        font-size: 2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 20px rgba(87, 68, 148, 0.2);
        transform: perspective(400px) rotateX(0deg);
        transition: var(--transition);
    }
    
    .logo-container:hover {
        transform: perspective(400px) rotateX(5deg);
        box-shadow: 0 10px 25px rgba(87, 68, 148, 0.3);
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.07);
        transition: var(--transition);
        border-top: 5px solid transparent;
    }
    
    .card:hover {
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
        transform: translateY(-3px);
    }
    
    .card.primary {
        border-top-color: var(--primary-color);
    }
    
    .card.yellow {
        border-top-color: var(--accent-yellow);
    }
    
    .card.orange {
        border-top-color: var(--accent-orange);
    }
    
    .card.green {
        border-top-color: var(--accent-green);
    }
    
    /* Metric cards styling */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        flex: 1;
        min-width: 130px;
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, 
            var(--primary-color),
            var(--accent-yellow),
            var(--accent-orange),
            var(--accent-green));
        transform: translateX(-100%);
        transition: var(--transition);
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .metric-card:hover::before {
        transform: translateX(0);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: var(--text-light);
        margin-bottom: 0.8rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.3rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: var(--transition) !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(87, 68, 148, 0.2) !important;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark) !important;
        box-shadow: 0 6px 16px rgba(87, 68, 148, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton > button:active {
        box-shadow: 0 2px 6px rgba(87, 68, 148, 0.2) !important;
        transform: translateY(0) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.5rem 1rem !important;
        transition: var(--transition) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        border-top: 3px solid var(--primary-color) !important;
    }
    
    /* Sidebar styling */
    .css-1aumxhk {
        background-color: #FFFFFF;
        border-right: 1px solid #EEEEEE;
    }
    
    /* Sliders and selectbox styling */
    .stSlider [data-baseweb="slider"] {
        margin-top: 1rem;
    }
    
    .stSlider [data-baseweb="thumb"] {
        background-color: var(--primary-color) !important;
        border-color: var(--primary-color) !important;
        transition: var(--transition) !important;
    }
    
    .stSlider [data-baseweb="thumb"]:hover {
        transform: scale(1.1) !important;
    }
    
    /* Radio buttons, checkboxes and toggle */
    .stRadio [data-baseweb="radio"] {
        transition: var(--transition) !important;
    }
    
    .stRadio [data-baseweb="radio"]:checked {
        background-color: var(--primary-color) !important;
        border-color: var(--primary-color) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--primary-color);
        background-color: #F7F7F9;
        border-radius: 8px;
        padding: 0.75rem 1rem !important;
    }
    
    /* Footer styling */
    footer {
        margin-top: 4rem;
        padding-top: 1.5rem;
        text-align: center;
        color: var(--text-light);
        font-size: 0.9rem;
        position: relative;
    }
    
    footer::before {
        content: "";
        position: absolute;
        top: 0;
        left: 20%;
        width: 60%;
        height: 1px;
        background: linear-gradient(90deg, 
            rgba(87, 68, 148, 0), 
            rgba(87, 68, 148, 0.3) 20%, 
            rgba(87, 68, 148, 0.3) 80%, 
            rgba(87, 68, 148, 0) 100%);
    }
    
    .footer-brand {
        font-weight: 700;
        color: var(--primary-color);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        transition: var(--transition);
    }
    
    .chart-container:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    }
    
    /* Insight cards */
    .insight-card {
        background-color: #FCF9FF;
        border-left: 4px solid var(--primary-color);
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        transition: var(--transition);
    }
    
    .insight-card:hover {
        background-color: #F5EFFF;
        transform: translateX(3px);
    }
    
    /* Section headings */
    h2 {
        color: var(--primary-color);
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    h2::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 40px;
        height: 3px;
        background: linear-gradient(90deg, 
            var(--primary-color) 0%, 
            var(--accent-yellow) 100%);
    }
    
    /* Tables */
    .dataframe-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Alert and info boxes */
    .stAlert {
        border-radius: 8px !important;
        border: none !important;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: var(--primary-color) !important;
    }
    
    /* Animations for key elements */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fadeIn {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Branded code blocks */
    .stCodeBlock {
        background-color: #2A2139 !important;
        border-radius: 8px !important;
        border-left: 3px solid var(--primary-color) !important;
    }
    </style>
    """, unsafe_allow_html=True)

def zobraz_data_filtering():
    """
    Display data filtering section allowing users to create filtered subsets of their data
    """
    create_section_header("Filtrace dat", icon="🔍", 
                         description="Vytvořte filtrovaný pohled na vaše data pomocí uživatelsky definovaných podmínek")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    # Inicializace filtru v session state
    if "filter_conditions" not in st.session_state:
        st.session_state.filter_conditions = {}
    
    if "filtered_data" not in st.session_state:
        st.session_state.filtered_data = None
        st.session_state.filtered_eda = None
    
    # UI pro definování filtrů
    st.subheader("Definice filtračních podmínek")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        column_types = {
            "Všechny sloupce": list(data.columns),
            "Numerické sloupce": eda.numeric_cols,
            "Kategorické sloupce": eda.categorical_cols,
            "Datum sloupce": eda.datetime_cols
        }
        
        column_type = st.radio("Typ sloupce", list(column_types.keys()))
        available_columns = column_types[column_type]
        
        selected_column = st.selectbox(
            "Vyberte sloupec pro filtrování",
            available_columns,
            key="filter_column"
        )
    
    with col2:
        if selected_column in eda.numeric_cols:
            # Numerický filtr
            st.write(f"Definujte rozsah pro {selected_column}")
            
            # Zjistíme rozsah hodnot
            min_val = data[selected_column].min()
            max_val = data[selected_column].max()
            
            # Přidáme posuvníky pro definování rozsahu
            filter_range = st.slider(
                "Rozsah hodnot",
                min_value=float(min_val),
                max_value=float(max_val),
                value=(float(min_val), float(max_val))
            )
            
            # Tlačítko pro přidání filtru
            if st.button("Přidat filtr rozsahu"):
                st.session_state.filter_conditions[selected_column] = {
                    "type": "range",
                    "min": filter_range[0],
                    "max": filter_range[1]
                }
                st.success(f"Filtr pro {selected_column} přidán: rozsah {filter_range[0]} až {filter_range[1]}")
        
        elif selected_column in eda.categorical_cols:
            # Kategorický filtr
            unique_values = sorted(data[selected_column].dropna().unique())
            
            if len(unique_values) <= 10:
                # Pro menší počet kategorií použijeme checkboxy
                st.write(f"Vyberte hodnoty pro {selected_column}")
                
                selected_values = {}
                for val in unique_values:
                    selected_values[str(val)] = st.checkbox(str(val), value=True, key=f"filter_{selected_column}_{val}")
                
                if st.button("Přidat kategorický filtr"):
                    selected = [val for val, is_selected in selected_values.items() if is_selected]
                    if selected:
                        st.session_state.filter_conditions[selected_column] = {
                            "type": "categorical",
                            "values": selected
                        }
                        st.success(f"Filtr pro {selected_column} přidán: vybrané kategorie {', '.join(selected)}")
            else:
                # Pro větší počet kategorií použijeme multiselect
                st.write(f"Vyberte hodnoty pro {selected_column}")
                selected_values = st.multiselect(
                    "Vyberte hodnoty",
                    options=unique_values,
                    default=unique_values[:min(5, len(unique_values))]
                )
                
                if st.button("Přidat kategorický filtr"):
                    if selected_values:
                        st.session_state.filter_conditions[selected_column] = {
                            "type": "categorical",
                            "values": [str(val) for val in selected_values]
                        }
                        st.success(f"Filtr pro {selected_column} přidán: vybrané kategorie {', '.join(str(v) for v in selected_values)}")
        
        elif selected_column in eda.datetime_cols:
            # Filtr pro datum
            st.write(f"Definujte rozsah dat pro {selected_column}")
            
            # Zjistíme rozsah hodnot
            date_series = pd.to_datetime(data[selected_column])
            min_date = date_series.min().date()
            max_date = date_series.max().date()
            
            # Přidáme výběr data
            start_date = st.date_input("Počáteční datum", min_date)
            end_date = st.date_input("Koncové datum", max_date)
            
            if st.button("Přidat filtr data"):
                st.session_state.filter_conditions[selected_column] = {
                    "type": "date",
                    "start": start_date,
                    "end": end_date
                }
                st.success(f"Filtr pro {selected_column} přidán: od {start_date} do {end_date}")
        
        else:
            # Textový filtr
            st.write(f"Filtrujte text v {selected_column}")
            
            contains_text = st.text_input("Text obsahuje")
            
            if st.button("Přidat textový filtr"):
                if contains_text:
                    st.session_state.filter_conditions[selected_column] = {
                        "type": "text",
                        "contains": contains_text
                    }
                    st.success(f"Filtr pro {selected_column} přidán: obsahuje '{contains_text}'")
    
    # Zobrazení aktuálních filtrů
    if st.session_state.filter_conditions:
        st.subheader("Aktivní filtry")
        
        for col, condition in st.session_state.filter_conditions.items():
            if condition["type"] == "range":
                st.markdown(f"- **{col}**: rozsah od {condition['min']} do {condition['max']}")
            elif condition["type"] == "categorical":
                st.markdown(f"- **{col}**: kategorie {', '.join(condition['values'])}")
            elif condition["type"] == "date":
                st.markdown(f"- **{col}**: od {condition['start']} do {condition['end']}")
            elif condition["type"] == "text":
                st.markdown(f"- **{col}**: obsahuje '{condition['contains']}'")
        
        # Tlačítko pro odstranění všech filtrů
        if st.button("Vymazat všechny filtry", type="secondary"):
            st.session_state.filter_conditions = {}
            st.session_state.filtered_data = None
            st.session_state.filtered_eda = None
            st.rerun()
    
    # Aplikování filtrů
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.filter_conditions and st.button("Aplikovat filtry", type="primary", use_container_width=True):
            # Vytvoření filtrovacích podmínek ve formátu pro EDA.filter_data
            filters = {}
            
            for col, condition in st.session_state.filter_conditions.items():
                if condition["type"] == "range":
                    filters[col] = {"min": condition["min"], "max": condition["max"]}
                elif condition["type"] == "categorical":
                    filters[col] = {"in": condition["values"]}
                elif condition["type"] == "date":
                    filters[col] = {"min": pd.Timestamp(condition["start"]), 
                                  "max": pd.Timestamp(condition["end"])}
                elif condition["type"] == "text":
                    filters[col] = {"like": condition["contains"]}
            
            # Aplikování filtrů pomocí metody z EDA
            try:
                filtered_eda = eda.filter_data(filters)
                st.session_state.filtered_eda = filtered_eda
                st.session_state.filtered_data = filtered_eda.data
                st.success(f"Filtry úspěšně aplikovány. Výsledek: {len(filtered_eda.data)} řádků (původně {len(data)} řádků)")
                st.rerun()
            except Exception as e:
                st.error(f"Chyba při aplikování filtrů: {str(e)}")
    
    with col2:
        if st.session_state.filtered_data is not None:
            if st.button("Použít filtrovaná data jako hlavní dataset", type="primary", use_container_width=True):
                st.session_state.data = st.session_state.filtered_data
                st.session_state.eda = st.session_state.filtered_eda
                st.session_state.filtered_data = None
                st.session_state.filtered_eda = None
                st.session_state.filter_conditions = {}
                st.success("Filtrovaná data byla nastavena jako hlavní dataset")
                st.rerun()
    
    # Zobrazení filtrovaných dat, pokud existují
    if st.session_state.filtered_data is not None:
        st.subheader("Náhled filtrovaných dat")
        
        # Vytvoření záložek pro různé pohledy
        preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Tabulka", "Porovnání statistik", "Distribuce"])
        
        with preview_tab1:
            st.dataframe(st.session_state.filtered_data.head(10), use_container_width=True)
        
        with preview_tab2:
            if len(eda.numeric_cols) > 0:
                # Porovnání základních statistik před a po filtraci
                st.markdown("#### Porovnání statistik před a po filtraci")
                
                # Výběr sloupce pro porovnání
                compare_col = st.selectbox(
                    "Vyberte sloupec pro porovnání statistik",
                    eda.numeric_cols
                )
                
                # Vytvoření statistik
                orig_stats = data[compare_col].describe()
                filtered_stats = st.session_state.filtered_data[compare_col].describe()
                
                # Spojení statistik do jedné tabulky
                stats_comparison = pd.DataFrame({
                    'Původní data': orig_stats,
                    'Filtrovaná data': filtered_stats,
                    'Rozdíl': filtered_stats - orig_stats,
                    'Rozdíl (%)': (filtered_stats - orig_stats) / orig_stats * 100
                })
                
                st.dataframe(stats_comparison, use_container_width=True)
        
        with preview_tab3:
            if len(eda.numeric_cols) > 0:
                # Výběr sloupce pro vizualizaci
                viz_col = st.selectbox(
                    "Vyberte sloupec pro vizualizaci distribucí",
                    eda.numeric_cols,
                    key="viz_distrib_col"
                )
                
                # Vytvoření histogramů pro porovnání distribucí
                fig = go.Figure()
                
                # Histogram původních dat
                fig.add_trace(go.Histogram(
                    x=data[viz_col],
                    name='Původní data',
                    opacity=0.7,
                    marker_color=OICT_COLORS['purple'],
                    nbinsx=30
                ))
                
                # Histogram filtrovaných dat
                fig.add_trace(go.Histogram(
                    x=st.session_state.filtered_data[viz_col],
                    name='Filtrovaná data',
                    opacity=0.7,
                    marker_color=OICT_COLORS['orange'],
                    nbinsx=30
                ))
                
                # Úprava layoutu
                fig.update_layout(
                    title=f'Porovnání distribuce {viz_col} před a po filtraci',
                    xaxis_title=viz_col,
                    yaxis_title='Četnost',
                    barmode='overlay'
                )
                
                st.plotly_chart(fig, use_container_width=True)

def zobraz_data_comparison():
    """
    Display data comparison section allowing users to compare two datasets
    """
    create_section_header("Porovnání datasetů", icon="⚖️", 
                         description="Porovnejte aktuální dataset s jiným datasetem a identifikujte klíčové rozdíly")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    # Inicializace pro druhý dataset v session state
    if "comparison_data" not in st.session_state:
        st.session_state.comparison_data = None
        st.session_state.comparison_eda = None
        st.session_state.comparison_results = None
    
    # UI pro nahrání druhého datasetu
    st.subheader("Nahrání datasetu pro porovnání")
    
    # Tabs pro různé způsoby získání druhého datasetu
    source_tabs = st.tabs(["📁 Soubor", "💾 Ukázkové datasety", "🔄 Filtrovaná data"])
    
    with source_tabs[0]:  # Nahrání souboru
        uploaded_file = st.file_uploader(
            "Vyberte CSV nebo Excel soubor pro porovnání",
            type=["csv", "xlsx", "xls"]
        )
        
        if uploaded_file is not None:
            try:
                # Nastavení importu
                with st.expander("Nastavení importu", expanded=False):
                    if uploaded_file.name.endswith('.csv'):
                        separator = st.text_input("Oddělovač", value=",")
                        encoding = st.selectbox("Kódování", 
                                              options=["utf-8", "iso-8859-1", "windows-1250", "latin1", "latin2"],
                                              index=0)
                    else:  # Excel
                        sheet_name = st.text_input("Název listu (ponechte prázdné pro první list)", value="")
                
                # Nahrání dat
                if uploaded_file.name.endswith('.csv'):
                    comparison_data = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
                else:
                    if sheet_name.strip():
                        comparison_data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                    else:
                        comparison_data = pd.read_excel(uploaded_file)
                
                if st.button("Použít pro porovnání", key="use_uploaded"):
                    # Vytvoření EDA objektu
                    from eda import EDA
                    comparison_eda = EDA(comparison_data)
                    
                    # Uložení do session state
                    st.session_state.comparison_data = comparison_data
                    st.session_state.comparison_eda = comparison_eda
                    st.success(f"Dataset pro porovnání úspěšně nahrán: {comparison_data.shape[0]} řádků a {comparison_data.shape[1]} sloupců")
                    
                    # Spuštění porovnání datasetů
                    run_comparison()
                    st.rerun()
            except Exception as e:
                st.error(f"Chyba při nahrávání souboru: {str(e)}")
    
    with source_tabs[1]:  # Ukázkové datasety
        st.markdown("### Vyberte ukázkový dataset")
        
        # Ukázkové datasety v sloupcovém rozložení
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card primary" style="height: 100%;">
                <h4>Iris květiny 🌸</h4>
                <p>Klasický dataset pro klasifikaci.</p>
                <p><strong>150 řádků, 5 sloupců</strong></p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Nahrát Iris dataset", key="comp_load_iris", use_container_width=True):
                try:
                    from sklearn.datasets import load_iris
                    iris = load_iris()
                    comparison_data = pd.DataFrame(iris.data, columns=iris.feature_names)
                    comparison_data['target'] = iris.target
                    
                    # Vytvoření EDA objektu
                    from eda import EDA
                    comparison_eda = EDA(comparison_data)
                    
                    # Uložení do session state
                    st.session_state.comparison_data = comparison_data
                    st.session_state.comparison_eda = comparison_eda
                    st.success("Dataset Iris úspěšně nahrán pro porovnání")
                    
                    # Spuštění porovnání
                    run_comparison()
                    st.rerun()
                except Exception as e:
                    st.error(f"Chyba při nahrávání ukázkového datasetu: {str(e)}")
        
        with col2:
            st.markdown("""
            <div class="card orange" style="height: 100%;">
                <h4>Pasažéři Titanicu 🚢</h4>
                <p>Dataset pro analýzu přežití pasažérů.</p>
                <p><strong>891 řádků, 12 sloupců</strong></p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Nahrát Titanic dataset", key="comp_load_titanic", use_container_width=True):
                try:
                    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
                    comparison_data = pd.read_csv(url)
                    
                    # Vytvoření EDA objektu
                    from eda import EDA
                    comparison_eda = EDA(comparison_data)
                    
                    # Uložení do session state
                    st.session_state.comparison_data = comparison_data
                    st.session_state.comparison_eda = comparison_eda
                    st.success("Dataset Titanicu úspěšně nahrán pro porovnání")
                    
                    # Spuštění porovnání
                    run_comparison()
                    st.rerun()
                except Exception as e:
                    st.error(f"Chyba při nahrávání ukázkového datasetu: {str(e)}")
    
    with source_tabs[2]:  # Filtrovaná data
        st.markdown("### Použití filtrovaných dat pro porovnání")
        
        if "filtered_data" in st.session_state and st.session_state.filtered_data is not None:
            st.info(f"K dispozici jsou filtrovaná data s {len(st.session_state.filtered_data)} řádky.")
            
            if st.button("Použít filtrovaná data pro porovnání", key="use_filtered", use_container_width=True):
                # Uložení filtrovaných dat jako srovnávací dataset
                st.session_state.comparison_data = st.session_state.filtered_data
                st.session_state.comparison_eda = st.session_state.filtered_eda
                st.success("Filtrovaná data úspěšně nastavena pro porovnání")
                
                # Spuštění porovnání
                run_comparison()
                st.rerun()
        else:
            st.warning("Žádná filtrovaná data nejsou k dispozici. Nejprve použijte sekci 'Filtrace dat' pro vytvoření filtrovaného datasetu.")
    
    # Zobrazení výsledků porovnání, pokud existují
    if st.session_state.comparison_results is not None:
        display_comparison_results()

def run_comparison():
    """
    Run dataset comparison and store results in session state
    """
    if "data" in st.session_state and "comparison_data" in st.session_state:
        if st.session_state.data is not None and st.session_state.comparison_data is not None:
            try:
                # Získání výsledků porovnání pomocí metody EDA
                results = st.session_state.eda.compare_datasets(
                    st.session_state.comparison_eda,
                    name1="Původní dataset",
                    name2="Porovnávaný dataset"
                )
                
                # Uložení výsledků do session state
                st.session_state.comparison_results = results
                return True
            except Exception as e:
                st.error(f"Chyba při porovnávání datasetů: {str(e)}")
                return False
    return False

def display_comparison_results():
    """
    Display comparison results between two datasets
    """
    results = st.session_state.comparison_results
    
    st.header("Výsledky porovnání datasetů")
    
    # Základní informace o velikosti
    st.subheader("Základní porovnání velikosti")
    
    size_data = pd.DataFrame({
        'Metrika': ['Počet řádků', 'Počet sloupců'],
        'Původní dataset': [results['size']['Původní dataset']['rows'], 
                          results['size']['Původní dataset']['columns']],
        'Porovnávaný dataset': [results['size']['Porovnávaný dataset']['rows'], 
                              results['size']['Porovnávaný dataset']['columns']],
        'Rozdíl': [results['size']['difference']['rows'], 
                  results['size']['difference']['columns']]
    })
    
    st.dataframe(size_data, use_container_width=True)
    
    # Porovnání sloupců
    st.subheader("Porovnání sloupců")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Společné sloupce ({len(results['columns']['common'])}):**")
        if results['columns']['common']:
            st.write(", ".join(sorted(results['columns']['common'])))
        else:
            st.write("Žádné")
    
    with col2:
        st.markdown(f"**Rozdílné sloupce:**")
        st.markdown(f"- Pouze v původním datasetu ({len(results['columns']['only_in_Původní dataset'])}): "
                  f"{', '.join(sorted(results['columns']['only_in_Původní dataset'])) if results['columns']['only_in_Původní dataset'] else 'Žádné'}")
        st.markdown(f"- Pouze v porovnávaném datasetu ({len(results['columns']['only_in_Porovnávaný dataset'])}): "
                  f"{', '.join(sorted(results['columns']['only_in_Porovnávaný dataset'])) if results['columns']['only_in_Porovnávaný dataset'] else 'Žádné'}")
    
    # Porovnání distribucí číselných sloupců
    st.subheader("Porovnání distribucí společných sloupců")
    
    numeric_diffs = {k: v for k, v in results['distribution_differences'].items() 
                   if k in st.session_state.eda.numeric_cols and k in st.session_state.comparison_eda.numeric_cols}
    
    if numeric_diffs:
        # Výběr sloupce pro vizualizaci
        num_cols = list(numeric_diffs.keys())
        selected_col = st.selectbox("Vyberte sloupec pro porovnání distribucí", num_cols)
        
        if selected_col:
            # Získání dat z obou datasetů
            orig_data = st.session_state.data[selected_col].dropna()
            comp_data = st.session_state.comparison_data[selected_col].dropna()
            
            # Statistiky
            stats_data = pd.DataFrame({
                'Metrika': ['Průměr', 'Medián', 'Směr. odchylka', 'Min', 'Max'],
                'Původní dataset': [
                    orig_data.mean(),
                    orig_data.median(),
                    orig_data.std(),
                    orig_data.min(),
                    orig_data.max()
                ],
                'Porovnávaný dataset': [
                    comp_data.mean(),
                    comp_data.median(),
                    comp_data.std(),
                    comp_data.min(),
                    comp_data.max()
                ],
                'Absolutní rozdíl': [
                    abs(orig_data.mean() - comp_data.mean()),
                    abs(orig_data.median() - comp_data.median()),
                    abs(orig_data.std() - comp_data.std()),
                    abs(orig_data.min() - comp_data.min()),
                    abs(orig_data.max() - comp_data.max())
                ],
                'Relativní rozdíl (%)': [
                    abs((orig_data.mean() - comp_data.mean()) / comp_data.mean() * 100) if comp_data.mean() != 0 else 0,
                    abs((orig_data.median() - comp_data.median()) / comp_data.median() * 100) if comp_data.median() != 0 else 0,
                    abs((orig_data.std() - comp_data.std()) / comp_data.std() * 100) if comp_data.std() != 0 else 0,
                    abs((orig_data.min() - comp_data.min()) / comp_data.min() * 100) if comp_data.min() != 0 else 0,
                    abs((orig_data.max() - comp_data.max()) / comp_data.max() * 100) if comp_data.max() != 0 else 0,
                ]
            })
            
            st.dataframe(stats_data, use_container_width=True)
            
            # Vizualizace - porovnání histogramů
            fig = go.Figure()
            
            # Histogram původního datasetu
            fig.add_trace(go.Histogram(
                x=orig_data,
                name='Původní dataset',
                opacity=0.7,
                marker_color=OICT_COLORS['purple'],
                nbinsx=30
            ))
            
            # Histogram porovnávaného datasetu
            fig.add_trace(go.Histogram(
                x=comp_data,
                name='Porovnávaný dataset',
                opacity=0.7,
                marker_color=OICT_COLORS['orange'],
                nbinsx=30
            ))
            
            # Úprava layoutu
            fig.update_layout(
                title=f'Porovnání distribuce {selected_col}',
                xaxis_title=selected_col,
                yaxis_title='Četnost',
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot pro porovnání
            fig2 = go.Figure()
            
            fig2.add_trace(go.Box(
                y=orig_data,
                name='Původní dataset',
                marker_color=OICT_COLORS['purple']
            ))
            
            fig2.add_trace(go.Box(
                y=comp_data,
                name='Porovnávaný dataset',
                marker_color=OICT_COLORS['orange']
            ))
            
            fig2.update_layout(
                title=f'Boxplot porovnání {selected_col}',
                yaxis_title=selected_col
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    # Porovnání kategorických sloupců
    cat_diffs = {k: v for k, v in results['distribution_differences'].items() 
               if k in st.session_state.eda.categorical_cols and k in st.session_state.comparison_eda.categorical_cols}
    
    if cat_diffs:
        st.subheader("Porovnání kategorických sloupců")
        
        cat_cols = list(cat_diffs.keys())
        selected_cat = st.selectbox("Vyberte kategorický sloupec pro porovnání", cat_cols)
        
        if selected_cat:
            # Získání dat z obou datasetů
            orig_counts = st.session_state.data[selected_cat].value_counts(normalize=True)
            comp_counts = st.session_state.comparison_data[selected_cat].value_counts(normalize=True)
            
            # Sjednocení kategorií
            all_categories = sorted(set(orig_counts.index) | set(comp_counts.index))
            
            # Vytvoření srovnávací tabulky
            comparison_data = pd.DataFrame({
                'Kategorie': all_categories,
                'Původní dataset (%)': [orig_counts.get(cat, 0) * 100 for cat in all_categories],
                'Porovnávaný dataset (%)': [comp_counts.get(cat, 0) * 100 for cat in all_categories],
                'Rozdíl (%)': [abs(orig_counts.get(cat, 0) - comp_counts.get(cat, 0)) * 100 for cat in all_categories]
            }).sort_values('Rozdíl (%)', ascending=False)
            
            st.dataframe(comparison_data, use_container_width=True)
            
            # Vizualizace - sloupcový graf
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=comparison_data['Kategorie'],
                y=comparison_data['Původní dataset (%)'],
                name='Původní dataset',
                marker_color=OICT_COLORS['purple']
            ))
            
            fig.add_trace(go.Bar(
                x=comparison_data['Kategorie'],
                y=comparison_data['Porovnávaný dataset (%)'],
                name='Porovnávaný dataset',
                marker_color=OICT_COLORS['orange']
            ))
            
            fig.update_layout(
                title=f'Porovnání rozložení kategorií {selected_cat}',
                xaxis_title='Kategorie',
                yaxis_title='Procento (%)',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Vizualizace rozdílů
            fig2 = px.bar(
                comparison_data,
                x='Kategorie',
                y='Rozdíl (%)',
                title=f'Rozdíly v rozložení kategorií {selected_cat}',
                color='Rozdíl (%)',
                color_continuous_scale='Reds'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    # Závěrečné shrnutí
    st.subheader("Hlavní rozdíly mezi datasety")
    
    # Vytvoření seznamu hlavních rozdílů
    differences = []
    
    # Rozdíl ve velikosti
    size_diff_rows = abs(results['size']['difference']['rows'])
    size_diff_pct = abs(results['size']['difference']['rows'] / results['size']['Původní dataset']['rows'] * 100) if results['size']['Původní dataset']['rows'] > 0 else 0
    
    if size_diff_rows > 0:
        if results['size']['difference']['rows'] > 0:
            differences.append(f"Původní dataset má o **{size_diff_rows}** řádků více ({size_diff_pct:.1f}%).")
        else:
            differences.append(f"Porovnávaný dataset má o **{size_diff_rows}** řádků více ({size_diff_pct:.1f}%).")
    
    # Rozdíl ve sloupcích
    col_diff = len(results['columns']['only_in_Původní dataset']) + len(results['columns']['only_in_Porovnávaný dataset'])
    if col_diff > 0:
        differences.append(f"Datasety se liší v **{col_diff}** sloupcích.")
    
    # Největší rozdíly v číselných sloupcích
    if numeric_diffs:
        max_diff_col = max(numeric_diffs.items(), key=lambda x: abs(x[1].get('mean_diff', 0)))
        differences.append(f"Největší rozdíl průměrů je ve sloupci **{max_diff_col[0]}** "
                          f"(absolutní rozdíl: {abs(max_diff_col[1].get('mean_diff', 0)):.2f}).")
    
    # Největší rozdíly v kategorických sloupcích
    if cat_diffs:
        max_cat_diff_col = max(cat_diffs.items(), key=lambda x: x[1].get('total_variation_distance', 0))
        differences.append(f"Největší rozdíl v kategorickém rozložení je ve sloupci **{max_cat_diff_col[0]}** "
                          f"(total variation distance: {max_cat_diff_col[1].get('total_variation_distance', 0):.2f}).")
    
    # Zobrazení seznamu rozdílů
    if differences:
        for diff in differences:
            st.markdown(f"- {diff}")
    else:
        st.info("Datasety jsou velmi podobné, nebyly nalezeny výrazné rozdíly.")
    
    # Možnost exportu srovnávací zprávy
    if st.button("Exportovat srovnávací zprávu", type="primary"):
        try:
            report_html = vytvor_srovnavaci_report(st.session_state.data, st.session_state.comparison_data, results)
            b64 = base64.b64encode(report_html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="comparison_report.html" class="btn btn-primary">Stáhnout HTML zprávu</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Chyba při vytváření zprávy: {str(e)}")

def vytvor_srovnavaci_report(data1, data2, results):
    """
    Vytvoří HTML report s výsledky porovnání dvou datasetů
    """
    import datetime
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comparison Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 30px;
                line-height: 1.6;
                color: #333;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #574494;
                padding-bottom: 10px;
            }}
            .logo {{
                background-color: #574494;
                color: white;
                padding: 5px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 24px;
                display: inline-block;
                margin-bottom: 10px;
            }}
            h1, h2, h3 {{
                color: #574494;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .section {{
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
            .highlight {{
                background-color: #ffffcc;
                padding: 2px 5px;
                border-radius: 3px;
            }}
            .footer {{
                margin-top: 50px;
                text-align: center;
                color: #666;
                font-size: 14px;
                border-top: 1px solid #eee;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">oict.</div>
            <h1>Report porovnání datasetů</h1>
            <p>Vygenerováno: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
        
        <div class="section">
            <h2>Základní porovnání</h2>
            
            <h3>Velikost datasetů</h3>
            <table>
                <tr>
                    <th>Metrika</th>
                    <th>Původní dataset</th>
                    <th>Porovnávaný dataset</th>
                    <th>Rozdíl</th>
                </tr>
                <tr>
                    <td>Počet řádků</td>
                    <td>{results['size']['Původní dataset']['rows']}</td>
                    <td>{results['size']['Porovnávaný dataset']['rows']}</td>
                    <td>{results['size']['difference']['rows']}</td>
                </tr>
                <tr>
                    <td>Počet sloupců</td>
                    <td>{results['size']['Původní dataset']['columns']}</td>
                    <td>{results['size']['Porovnávaný dataset']['columns']}</td>
                    <td>{results['size']['difference']['columns']}</td>
                </tr>
            </table>
            
            <h3>Porovnání sloupců</h3>
            <p><strong>Společné sloupce ({len(results['columns']['common'])}):</strong> {', '.join(sorted(results['columns']['common']))}</p>
            <p><strong>Pouze v původním datasetu ({len(results['columns']['only_in_Původní dataset'])}):</strong> {', '.join(sorted(results['columns']['only_in_Původní dataset'])) if results['columns']['only_in_Původní dataset'] else 'Žádné'}</p>
            <p><strong>Pouze v porovnávaném datasetu ({len(results['columns']['only_in_Porovnávaný dataset'])}):</strong> {', '.join(sorted(results['columns']['only_in_Porovnávaný dataset'])) if results['columns']['only_in_Porovnávaný dataset'] else 'Žádné'}</p>
        </div>
    """
    
    # Přidání sekce s numerickými rozdíly
    numeric_diffs = {k: v for k, v in results['distribution_differences'].items() 
                   if isinstance(v, dict) and 'mean_diff' in v}
    
    if numeric_diffs:
        html += """
        <div class="section">
            <h2>Porovnání numerických sloupců</h2>
            <table>
                <tr>
                    <th>Sloupec</th>
                    <th>Rozdíl průměrů</th>
                    <th>Rozdíl v %</th>
                </tr>
        """
        
        for col, diff in numeric_diffs.items():
            if 'mean_diff' in diff and 'mean_pct_diff' in diff:
                html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{diff['mean_diff']:.4f}</td>
                    <td>{diff['mean_pct_diff']:.2f}%</td>
                </tr>
                """
        
        html += """
            </table>
        </div>
        """
    
    # Přidání sekce s kategorickými rozdíly
    cat_diffs = {k: v for k, v in results['distribution_differences'].items() 
               if isinstance(v, dict) and 'total_variation_distance' in v}
    
    if cat_diffs:
        html += """
        <div class="section">
            <h2>Porovnání kategorických sloupců</h2>
            <table>
                <tr>
                    <th>Sloupec</th>
                    <th>Kategorie s největším rozdílem</th>
                    <th>Maximální rozdíl (%)</th>
                    <th>Total Variation Distance</th>
                </tr>
        """
        
        for col, diff in cat_diffs.items():
            if 'max_diff_category' in diff and 'max_diff_pct' in diff and 'total_variation_distance' in diff:
                html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{diff['max_diff_category']}</td>
                    <td>{diff['max_diff_pct']:.2f}%</td>
                    <td>{diff['total_variation_distance']:.4f}</td>
                </tr>
                """
        
        html += """
            </table>
        </div>
        """
    
    # Přidání závěrečných shrnutí
    html += """
        <div class="section">
            <h2>Závěrečné shrnutí</h2>
            <h3>Hlavní rozdíly mezi datasety</h3>
            <ul>
    """
    
    # Výpočet a přidání hlavních rozdílů
    size_diff_rows = abs(results['size']['difference']['rows'])
    size_diff_pct = abs(results['size']['difference']['rows'] / results['size']['Původní dataset']['rows'] * 100) if results['size']['Původní dataset']['rows'] > 0 else 0
    
    if size_diff_rows > 0:
        if results['size']['difference']['rows'] > 0:
            html += f"<li>Původní dataset má o <span class='highlight'>{size_diff_rows}</span> řádků více ({size_diff_pct:.1f}%).</li>"
        else:
            html += f"<li>Porovnávaný dataset má o <span class='highlight'>{size_diff_rows}</span> řádků více ({size_diff_pct:.1f}%).</li>"
    
    col_diff = len(results['columns']['only_in_Původní dataset']) + len(results['columns']['only_in_Porovnávaný dataset'])
    if col_diff > 0:
        html += f"<li>Datasety se liší v <span class='highlight'>{col_diff}</span> sloupcích.</li>"
    
    if numeric_diffs:
        max_diff_col = max(numeric_diffs.items(), key=lambda x: abs(x[1].get('mean_diff', 0)))
        html += f"<li>Největší rozdíl průměrů je ve sloupci <span class='highlight'>{max_diff_col[0]}</span> (absolutní rozdíl: {abs(max_diff_col[1].get('mean_diff', 0)):.2f}).</li>"
    
    if cat_diffs:
        max_cat_diff_col = max(cat_diffs.items(), key=lambda x: x[1].get('total_variation_distance', 0))
        html += f"<li>Největší rozdíl v kategorickém rozložení je ve sloupci <span class='highlight'>{max_cat_diff_col[0]}</span> (total variation distance: {max_cat_diff_col[1].get('total_variation_distance', 0):.2f}).</li>"
    
    # Dokončení HTML
    html += """
            </ul>
        </div>
        
        <div class="footer">
            <p>Powered by OICT</p>
            <p>© 2023</p>
        </div>
    </body>
    </html>
    """
    
    return html

def zobraz_o_aplikaci():
    st.header("O aplikaci")
    
    st.markdown("""
    <div class="card primary">
        <h3>Vítejte v aplikaci AutoEDA</h3>
        <p>
            AuroEDA je nástroj, který vám pomůže rychle a efektivně analyzovat vaše data, 
            objevit skryté vzory a získat cenné podnikatelské poznatky. Nástroj je určen jak pro 
            datové analytiky, tak pro business uživatele, kteří potřebují rychle prozkoumat a 
            porozumět svým datům.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Jak začít")
    
    st.markdown("""
    <div class="insight-card">
        <h4>Rychlý návod pro práci s nástrojem</h4>
        <ol>
            <li><strong>Nahrajte data</strong> - začněte nahráním CSV nebo Excel souboru v sekci "Nahrání dat"</li>
            <li><strong>Prozkoumejte přehled</strong> - podívejte se na základní charakteristiky vašich dat v sekci "Přehled dat"</li>
            <li><strong>Analyzujte detaily</strong> - využijte specializované sekce pro hlubší analýzu</li>
            <li><strong>Generujte poznatky</strong> - využijte pokročilé funkce pro získání hodnotných informací</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Struktura aplikace
    st.subheader("Struktura aplikace")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card primary">
            <h4>Základní analýza</h4>
            <p>Základní průzkum a pochopení struktury dat</p>
            <ul>
                <li>Nahrání dat</li>
                <li>Přehled dat</li>
                <li>Chybějící hodnoty</li>
                <li>Distribuce</li>
                <li>Korelace</li>
                <li>Odlehlé hodnoty</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card orange">
            <h4>Pokročilá analýza</h4>
            <p>Hlubší statistická analýza a modelování</p>
            <ul>
                <li>Redukce dimenzí</li>
                <li>Clustering</li>
                <li>Statistické testy</li>
                <li>Návrhy úprav</li>
                <li>Rychlé modelování</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card green">
            <h4>Byznys analýza</h4>
            <p>Analýzy zaměřené na podnikatelské potřeby</p>
            <ul>
                <li>Časové řady</li>
                <li>Cross tabulky</li>
                <li>KPI dashboard</li>
                <li>Kohortní analýza</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Podrobný popis sekcí
    st.subheader("Podrobný popis funkcí")
    
    # Základní analýza
    with st.expander("Základní analýza", expanded=False):
        st.markdown("""
        ### Nahrání dat
        Umožňuje nahrát data ve formátu CSV nebo Excel. Můžete také využít ukázkové datasety pro testování funkcionalit.
        
        **Jak to pomáhá:** Jednoduchý vstupní bod pro vaše data s podporou různých formátů.
        
        ### Přehled dat
        Poskytuje základní informace o datasetu - počet řádků, sloupců, typech dat, a kvalitě dat.
        
        **Jak to pomáhá:**
        - Rychlý přehled o struktuře a kvalitě dat
        - Automatická detekce datových typů
        - Hodnocení celkové kvality dat
        
        ### Chybějící hodnoty
        Analýza a vizualizace chybějících hodnot v jednotlivých sloupcích.
        
        **Jak to pomáhá:**
        - Identifikace sloupců s nekompletními daty
        - Doporučení pro řešení chybějících hodnot
        - Vzorce pro imputaci chybějících hodnot
        
        ### Distribuce
        Zkoumá rozložení hodnot v jednotlivých sloupcích pomocí histogramů a statistik.
        
        **Jak to pomáhá:**
        - Pochopení typického rozsahu hodnot
        - Identifikace zešikmení a dalších vlastností distribuce
        - Vizualizace četností kategorií
        
        ### Korelace
        Analýza vztahů mezi numerickými proměnnými.
        
        **Jak to pomáhá:**
        - Identifikace silně korelovaných proměnných
        - Odhalení potenciálních kauzálních vztahů
        - Detekce multikolinearity, která může ovlivnit modely
        
        ### Odlehlé hodnoty
        Detekce a analýza odlehlých hodnot v numerických sloupcích.
        
        **Jak to pomáhá:**
        - Identifikace potenciálních chyb v datech
        - Analýza extrémních případů
        - Doporučení pro zpracování odlehlých hodnot
        """)
    
    # Pokročilá analýza
    with st.expander("Pokročilá analýza", expanded=False):
        st.markdown("""
        ### Redukce dimenzí (PCA)
        Analýza hlavních komponent pro snížení dimenzionality dat.
        
        **Jak to pomáhá:**
        - Redukce počtu proměnných se zachováním informační hodnoty
        - Vizualizace vícerozměrných dat ve 2D prostoru
        - Identifikace hlavních směrů variability v datech
        
        ### Clustering
        Shlukování podobných dat do skupin pomocí algoritmů strojového učení.
        
        **Jak to pomáhá:**
        - Segmentace dat do přirozených skupin
        - Identifikace podobných vzorů v datech
        - Analýza profilu jednotlivých shluků
        
        ### Statistické testy
        Různé statistické testy pro ověření hypotéz o datech.
        
        **Jak to pomáhá:**
        - Testování normality distribuce
        - Ověření významnosti rozdílů mezi skupinami
        - Testování nezávislosti kategorických proměnných
        
        ### Návrhy úprav
        Doporučení pro transformaci a přípravu dat pro další analýzu.
        
        **Jak to pomáhá:**
        - Identifikace potenciálních transformací pro zlepšení distribuce
        - Doporučení pro kódování kategorických proměnných
        - Strategie pro řešení problémů s kvalitou dat
        
        ### Rychlé modelování
        Automatické vytvoření prediktivního modelu na základě vybraných dat.
        
        **Jak to pomáhá:**
        - Rychlé otestování prediktivní síly proměnných
        - Identifikace nejdůležitějších prediktorů
        - Základní evaluace výkonnosti modelu
        """)
    
    # Byznys analýza
    with st.expander("Byznys analýza", expanded=False):
        st.markdown("""
        ### Časové řady
        Analýza dat v průběhu času, identifikace trendů a sezónnosti.
        
        **Jak to pomáhá:**
        - Vizualizace trendů v čase
        - Detekce sezónních vzorů
        - Analýza růstu nebo poklesu metrik
        
        ### Cross tabulky
        Analýza vztahů mezi kategorickými proměnnými pomocí kontingenčních tabulek.
        
        **Jak to pomáhá:**
        - Pochopení souvislostí mezi kategoriemi
        - Testování nezávislosti kategorických proměnných
        - Vizualizace rozložení hodnot mezi kategoriemi
        
        ### KPI dashboard
        Vytvoření přehledového dashboardu s klíčovými ukazateli výkonnosti.
        
        **Jak to pomáhá:**
        - Sledování nejdůležitějších metrik na jednom místě
        - Porovnání výkonnosti vůči cílům
        - Vizualizace klíčových ukazatelů
        
        ### Kohortní analýza
        Sledování chování skupin uživatelů v průběhu času.
        
        **Jak to pomáhá:**
        - Analýza retence uživatelů
        - Porovnání výkonnosti různých kohort
        - Identifikace dlouhodobých trendů v chování uživatelů
        """)
    
    # Tipy a triky
    st.subheader("Tipy a triky")
    
    st.markdown("""
    <div class="card yellow">
        <h4>Osvědčené postupy při analýze dat</h4>
        <ul>
            <li><strong>Vždy začněte průzkumem dat</strong> - použijte "Přehled dat" k pochopení základní struktury dat</li>
            <li><strong>Zkontrolujte kvalitu dat</strong> - věnujte pozornost chybějícím hodnotám a odlehlým hodnotám</li>
            <li><strong>Vizualizujte před modelováním</strong> - použijte distribuční a korelační analýzy před pokročilými metodami</li>
            <li><strong>Interpretujte, nejen počítejte</strong> - každá analýza by měla vést k pochopení a akci</li>
            <li><strong>Iterujte</strong> - často je potřeba vyzkoušet více přístupů a metod</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Příklady použití
    st.subheader("Příklady použití")
    
    tab1, tab2, tab3 = st.tabs(["Marketingová analýza", "Finanční analýza", "Operační analýza"])
    
    with tab1:
        st.markdown("""
        ### Marketingová analýza
        
        **Scénář:** Analýza výkonnosti marketingových kampaní
        
        **Doporučený postup:**
        1. **Nahrání dat** - importujte data o kampaních
        2. **Přehled dat** - získejte rychlý náhled na strukturu a kvalitu dat
        3. **Distribuce** - analyzujte rozložení výkonnostních metrik
        4. **Korelace** - identifikujte vztahy mezi výdaji a výsledky
        5. **Časové řady** - sledujte výkonnost kampaní v čase
        6. **KPI dashboard** - vytvořte přehled klíčových metrik
        
        **Očekávané výstupy:**
        - Které kanály mají nejvyšší ROI
        - Jak se výkonnost kampaní mění v čase
        - Jaké faktory nejvíce ovlivňují úspěšnost kampaní
        """)
    
    with tab2:
        st.markdown("""
        ### Finanční analýza
        
        **Scénář:** Analýza finančních výsledků a prognóza
        
        **Doporučený postup:**
        1. **Nahrání dat** - importujte finanční data
        2. **Přehled dat** - zkontrolujte strukturu dat
        3. **Časové řady** - analyzujte finanční ukazatele v čase
        4. **Odlehlé hodnoty** - identifikujte neobvyklé finanční výkyvy
        5. **Statistické testy** - proveďte testy pro ověření hypotéz
        6. **Rychlé modelování** - vytvořte prediktivní model pro prognózu
        
        **Očekávané výstupy:**
        - Trendy v příjmech a výdajích
        - Identifikace anomálií ve finančních datech
        - Predikce budoucích finančních výsledků
        """)
    
    with tab3:
        st.markdown("""
        ### Operační analýza
        
        **Scénář:** Optimalizace provozních procesů
        
        **Doporučený postup:**
        1. **Nahrání dat** - importujte operační data
        2. **Přehled dat** - získejte přehled o datech
        3. **Distribuce** - analyzujte rozložení operačních metrik
        4. **Odlehlé hodnoty** - identifikujte problematické procesy
        5. **Clustering** - seskupte podobné procesy
        6. **Cross tabulky** - analyzujte vztahy mezi kategoriemi
        
        **Očekávané výstupy:**
        - Identifikace úzkých míst v procesech
        - Segmentace procesů podle výkonnosti
        - Vztahy mezi různými operačními faktory
        """)
    
    # Kontakt a podpora
    st.subheader("Kontakt a podpora")
    
    st.markdown("""
    <div class="card">
        <p>Pro více informací nebo podporu kontaktujte:</p>
        <p><strong>Email:</strong> podporadata@golemio.cz</p>
        <p><strong>Dokumentace:</strong> <a href="https://golemio.cz/docs" target="_blank">https://golemio.cz/docs</a></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="OICT Auto EDA", layout="wide")
    aplikuj_oict_styl()
    
    st.markdown("""
    <div class="title-container animate-fadeIn">
        <h1>Golemio Auto EDA</h1>
        <p class="subtitle">Automatická explorativní analýza dat</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "active_section" not in st.session_state:
        st.session_state.active_section = "zakladni"
    
    # Use the modern navigation
    aktivni_sekce, stranka = create_navigation()
    
    # Initialize data session state
    if "data" not in st.session_state:
        st.session_state.data = None
    if "eda" not in st.session_state:
        st.session_state.eda = None
    
    # Main content area
    with st.container():
        # Display the appropriate page based on navigation
        if aktivni_sekce == "zakladni":
            if stranka == "Nahrání dat":
                zobraz_nahrani_dat()
            elif st.session_state.data is not None and st.session_state.eda is not None:
                if stranka == "Přehled dat":
                    zobraz_prehled_dat()
                elif stranka == "Chybějící hodnoty":
                    zobraz_chybejici_hodnoty()
                elif stranka == "Distribuce":
                    zobraz_distribuce()
                elif stranka == "Korelace":
                    zobraz_korelace()
                elif stranka == "Odlehlé hodnoty":
                    zobraz_odlehle_hodnoty()
            else:
                st.warning("Prosím, nejprve nahrajte data")
                zobraz_nahrani_dat()
        
        elif aktivni_sekce == "pokrocile":
            if st.session_state.data is not None and st.session_state.eda is not None:
                if stranka == "Redukce dimenzí":
                    zobraz_pca()
                elif stranka == "Clustering":
                    zobraz_clustering()
                elif stranka == "Statistické testy":
                    zobraz_statisticke_testy()
                elif stranka == "Návrhy úprav":
                    zobraz_navrhy_uprav()
                elif stranka == "Rychlé modelování":
                    zobraz_rychle_modelovani()
            else:
                st.warning("Prosím, nejprve nahrajte data")
                zobraz_nahrani_dat()
        
        elif aktivni_sekce == "byznys":
            if st.session_state.data is not None and st.session_state.eda is not None:
                if stranka == "Časové řady":
                    zobraz_casove_rady()
                elif stranka == "Cross tabulky":
                    zobraz_cross_tabulky()
                elif stranka == "KPI dashboard":
                    zobraz_kpi_dashboard()
                elif stranka == "Kohortní analýza":
                    zobraz_kohortni_analyza()
            else:
                st.warning("Prosím, nejprve nahrajte data")
                zobraz_nahrani_dat()
        
        elif aktivni_sekce == "porovnani":
            if stranka == "Filtrace dat":
                zobraz_data_filtering()
            elif stranka == "Porovnání datasetů":
                zobraz_data_comparison()
            else:
                zobraz_data_filtering()
        
        else:  # o aplikaci
            zobraz_o_aplikaci()
    
    # Footer with Golemio logo
    st.markdown("""
    <footer>
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
            <p style="margin-right: 15px;">Powered by <span class="footer-brand">OICT</span></p>
        </div>
        <div style="text-align: center; margin-bottom: 15px;">
           </div>
        <p>© 2025 Vytvořeno s ❤️ v Praze</p>
    </footer>
    """, unsafe_allow_html=True)

def create_navigation():
    """Create a modern navigation system in the sidebar with vertical layout"""
    
    with st.sidebar:
        # Logo header
        st.markdown("""
        <div style="text-align: center; padding: 25px 0; margin-bottom: 25px; border-bottom: 1px solid #eaeaea;">
            <h2 style="color: #574494; margin: 12px 0 0 0; font-size: 1.5rem; font-weight: 700;">
                Golemio AutoEDA
            </h2>
            <div style="width: 40px; height: 3px; background: linear-gradient(90deg, #574494 0%, #FFE14F 100%); margin: 12px auto;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Main navigation using visual selectors instead of radio buttons
        st.markdown("### Navigace")
        
        # Use session state to track the active section
        if "active_section" not in st.session_state:
            st.session_state.active_section = "zakladni"
        
        # Create visual section buttons
        sections = {
            "zakladni": {"name": "Základní analýza", "icon": "📊", "color": "#574494"},
            "pokrocile": {"name": "Pokročilá analýza", "icon": "🧠", "color": "#E37222"},
            "byznys": {"name": "Byznys analýza", "icon": "💼", "color": "#74ECA1"},
            "porovnani": {"name": "Porovnání a filtrace", "icon": "🔍", "color": "#FFE14F"},
            "o_aplikaci": {"name": "O aplikaci", "icon": "ℹ️", "color": "#FFE14F"}
        }
        
        # Create modern section selector tiles
        st.markdown("""
        <style>
        .section-tile {
            display: block;
            padding: 14px 16px;
            margin-bottom: 10px;
            border-radius: 8px;
            transition: all 0.2s ease;
            text-decoration: none;
            color: #333;
            background: white;
            border-left: 5px solid transparent;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        .section-tile:hover {
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
            transform: translateX(2px);
        }
        .section-tile.active {
            border-left-width: 5px;
            font-weight: bold;
            background: #f8f9fa;
        }
        .section-icon {
            display: inline-block;
            margin-right: 10px;
            font-size: 1.2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        for section_id, section_info in sections.items():
            is_active = st.session_state.active_section == section_id
            
            # Create the HTML for the section tile
            html = f"""
            <div class="section-tile{'  active' if is_active else ''}" 
                 style="border-left-color: {section_info['color']};">
                <span class="section-icon">{section_info['icon']}</span>
                {section_info['name']}
            </div>
            """
            
            # Use a button with a unique key
            if st.markdown(html, unsafe_allow_html=True):
                pass  # Markdown doesn't have click events, we'll use buttons below
            
            # The actual button - invisible but clickable, placed on top of the HTML
            if st.button(section_info['name'], key=f"btn_{section_id}", 
                       use_container_width=True, 
                       type="secondary" if is_active else "secondary",
            ):
                st.session_state.active_section = section_id
                # Reset subsection when changing main section
                if section_id == "zakladni":
                    st.session_state.zakladni_stranka = "Nahrání dat"
                elif section_id == "pokrocile":
                    st.session_state.pokrocile_stranka = "Redukce dimenzí"
                elif section_id == "byznys":
                    st.session_state.byznys_stranka = "Časové řady"
                elif section_id == "porovnani":
                    st.session_state.porovnani_stranka = "Filtrace dat"
                st.rerun()
        
        # Divider
        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
        
        # Get the active section for subsection navigation
        aktivni_sekce = st.session_state.active_section
        
        # Subsection navigation with vertical buttons
        if aktivni_sekce == "zakladni":
            st.markdown("""
            <div style="margin-bottom: 15px;">
                <h3 style="color: #574494; display: flex; align-items: center;">
                    <span style="margin-right: 8px;">📊</span> Základní analýza
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize subsection if not in session state
            if "zakladni_stranka" not in st.session_state:
                st.session_state.zakladni_stranka = "Nahrání dat"
            
            # Define subsections
            subsections = [
                {"id": "Nahrání dat", "icon": "📤", "name": "Nahrání dat"},
                {"id": "Přehled dat", "icon": "📋", "name": "Přehled dat"},
                {"id": "Chybějící hodnoty", "icon": "❓", "name": "Chybějící hodnoty"},
                {"id": "Distribuce", "icon": "📈", "name": "Distribuce"},
                {"id": "Korelace", "icon": "🔄", "name": "Korelace"},
                {"id": "Odlehlé hodnoty", "icon": "⚠️", "name": "Odlehlé hodnoty"}
            ]
            
            # Create vertical buttons for subsections
            for subsection in subsections:
                is_active = st.session_state.zakladni_stranka == subsection["id"]
                
                # Create button with icon
                button_text = f"{subsection['icon']} {subsection['name']}"
                
                if st.button(button_text, key=f"btn_basic_{subsection['id']}",
                           use_container_width=True,
                           type="primary" if is_active else "secondary"):
                    st.session_state.zakladni_stranka = subsection["id"]
                    st.rerun()
            
            stranka = st.session_state.zakladni_stranka
            
        elif aktivni_sekce == "pokrocile":
            st.markdown("""
            <div style="margin-bottom: 15px;">
                <h3 style="color: #E37222; display: flex; align-items: center;">
                    <span style="margin-right: 8px;">🧠</span> Pokročilá analýza
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize subsection if not in session state
            if "pokrocile_stranka" not in st.session_state:
                st.session_state.pokrocile_stranka = "Redukce dimenzí"
            
            # Define subsections
            subsections = [
                {"id": "Redukce dimenzí", "icon": "🧩", "name": "Redukce dimenzí"},
                {"id": "Clustering", "icon": "🔍", "name": "Clustering"},
                {"id": "Statistické testy", "icon": "🔬", "name": "Statistické testy"},
                {"id": "Návrhy úprav", "icon": "🛠️", "name": "Návrhy úprav"},
                {"id": "Rychlé modelování", "icon": "🤖", "name": "Rychlé modelování"}
            ]
            
            # Create vertical buttons for subsections
            for subsection in subsections:
                is_active = st.session_state.pokrocile_stranka == subsection["id"]
                
                # Create button with icon
                button_text = f"{subsection['icon']} {subsection['name']}"
                
                if st.button(button_text, key=f"btn_adv_{subsection['id']}",
                           use_container_width=True,
                           type="primary" if is_active else "secondary"):
                    st.session_state.pokrocile_stranka = subsection["id"]
                    st.rerun()
            
            stranka = st.session_state.pokrocile_stranka
            
        elif aktivni_sekce == "byznys":
            st.markdown("""
            <div style="margin-bottom: 15px;">
                <h3 style="color: #74ECA1; display: flex; align-items: center;">
                    <span style="margin-right: 8px;">💼</span> Byznys analýza
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize subsection if not in session state
            if "byznys_stranka" not in st.session_state:
                st.session_state.byznys_stranka = "Časové řady"
            
            # Define subsections
            subsections = [
                {"id": "Časové řady", "icon": "📅", "name": "Časové řady"},
                {"id": "Cross tabulky", "icon": "📊", "name": "Cross tabulky"},
                {"id": "KPI dashboard", "icon": "🎯", "name": "KPI dashboard"},
                {"id": "Kohortní analýza", "icon": "👥", "name": "Kohortní analýza"}
            ]
            
            # Create vertical buttons for subsections
            for subsection in subsections:
                is_active = st.session_state.byznys_stranka == subsection["id"]
                
                # Create button with icon
                button_text = f"{subsection['icon']} {subsection['name']}"
                
                if st.button(button_text, key=f"btn_biz_{subsection['id']}",
                           use_container_width=True,
                           type="primary" if is_active else "secondary"):
                    st.session_state.byznys_stranka = subsection["id"]
                    st.rerun()
            
            stranka = st.session_state.byznys_stranka

        elif aktivni_sekce == "porovnani":
            st.markdown("""
            <div style="margin-bottom: 15px;">
                <h3 style="color: #FFE14F; display: flex; align-items: center;">
                    <span style="margin-right: 8px;">🔍</span> Porovnání a filtrace
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize subsection if not in session state
            if "porovnani_stranka" not in st.session_state:
                st.session_state.porovnani_stranka = "Filtrace dat"
            
            # Define subsections
            subsections = [
                {"id": "Filtrace dat", "icon": "🔍", "name": "Filtrace dat"},
                {"id": "Porovnání datasetů", "icon": "⚖️", "name": "Porovnání datasetů"},
            ]
            
            # Create vertical buttons for subsections
            for subsection in subsections:
                is_active = st.session_state.porovnani_stranka == subsection["id"]
                
                # Create button with icon
                button_text = f"{subsection['icon']} {subsection['name']}"
                
                if st.button(button_text, key=f"btn_comp_{subsection['id']}",
                           use_container_width=True,
                           type="primary" if is_active else "secondary"):
                    st.session_state.porovnani_stranka = subsection["id"]
                    st.rerun()
            
            stranka = st.session_state.porovnani_stranka
            
        else:  # o aplikaci
            st.markdown("""
            <div style="margin-bottom: 15px;">
                <h3 style="color: #FFE14F; display: flex; align-items: center;">
                    <span style="margin-right: 8px;">ℹ️</span> O aplikaci
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            stranka = "O aplikaci"
               
    return aktivni_sekce, stranka

def create_section_header(section_title, icon="📊", description=None):
    """Create a modern section header with icon and optional description"""
    st.markdown(f"""
    <div style="padding: 1rem 0 0.5rem 0; margin-bottom: 1rem;">
        <h1>{icon} {section_title}</h1>
        {f'<p style="color: #666; font-size: 1.1rem; margin-top: 0.5rem;">{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)
    
def create_dashboard_card(title, value, delta=None, color="#574494", icon=None):
    """Create a modern dashboard metric card with title, value, and optional delta"""
    delta_html = ""
    if delta is not None:
        delta_color = "green" if delta >= 0 else "red"
        delta_icon = "↑" if delta >= 0 else "↓"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.9rem; margin-top: 0.25rem;">{delta_icon} {abs(delta):.1f}%</div>'
    
    icon_html = f'<div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>' if icon else ''
    
    st.markdown(f"""
    <div style="background-color: white; border-radius: 0.75rem; padding: 1.25rem; box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075); 
                border-top: 3px solid {color}; text-align: center; height: 100%;">
        {icon_html}
        <div style="font-size: 0.875rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 1.75rem; font-weight: 700; color: #333;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)
    
def create_metric_row(metrics):
    """Create a row of metric cards
    
    Parameters:
    metrics (list): List of dicts with keys: title, value, delta (optional), color (optional), icon (optional)
    """
    cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        with cols[i]:
            create_dashboard_card(
                title=metric['title'],
                value=metric['value'],
                delta=metric.get('delta'),
                color=metric.get('color', "#574494"),
                icon=metric.get('icon')
            )

def create_tab_navigation(tabs_dict, key_prefix="tab_nav"):
    """Create a modern tab navigation
    
    Parameters:
    tabs_dict (dict): Dict with tab names as keys and tab contents (callable functions) as values
    key_prefix (str): Prefix for session state keys to avoid conflicts
    
    Returns:
    str: The selected tab name
    """
    # Generate a unique key for this tab navigation
    state_key = f"{key_prefix}_selected"
    
    # Initialize the selected tab if not present
    if state_key not in st.session_state:
        st.session_state[state_key] = list(tabs_dict.keys())[0]
    
    # Create tab-style buttons
    cols = st.columns(len(tabs_dict))
    
    for i, (tab_name, _) in enumerate(tabs_dict.items()):
        # Determine if this tab is selected
        is_selected = st.session_state[state_key] == tab_name
        button_style = "primary" if is_selected else "secondary"
        
        # Create the button
        if cols[i].button(tab_name, key=f"{key_prefix}_{i}", 
                         use_container_width=True, type=button_style):
            st.session_state[state_key] = tab_name
            st.rerun()
    
    # Add a separator
    st.markdown('<hr style="margin: 0.5rem 0 1.5rem 0; border: none; height: 1px; background-color: #f0f0f0;">', 
              unsafe_allow_html=True)
    
    # Return the selected tab
    return st.session_state[state_key]

def create_modern_radio(label, options, default=None, horizontal=True, key=None):
    """Create a more modern-looking radio button alternative using buttons
    
    Parameters:
    label (str): The label for the radio group
    options (list): List of options
    default (str): Default selected option
    horizontal (bool): Whether to display horizontally
    key (str): Unique key for session state
    
    Returns:
    str: The selected option
    """
    if key not in st.session_state:
        st.session_state[key] = default if default else options[0]
    
    st.markdown(f"**{label}**")
    
    # Create columns if horizontal
    if horizontal:
        cols = st.columns(len(options))
    else:
        # For vertical, create a single column for each option
        cols = [st for _ in range(len(options))]
    
    for i, option in enumerate(options):
        is_selected = st.session_state[key] == option
        button_style = "primary" if is_selected else "secondary"
        
        if cols[i].button(option, key=f"{key}_{i}", 
                         use_container_width=True, 
                         type=button_style):
            st.session_state[key] = option
            st.rerun()
    
    return st.session_state[key]

def zobraz_nahrani_dat():
    create_section_header("Nahrání dat", icon="📤", 
                         description="Importujte data z různých zdrojů")
    
    # Create tabs for different data sources
    source_tabs = st.tabs(["📁 Soubor", "🌐 API", "💾 Ukázkové datasety"])
    
    with source_tabs[0]:  # File upload tab
        # Sub-tabs for different file types
        file_tabs = st.tabs(["CSV/Excel", "JSON", "Parquet"])
        
        with file_tabs[0]:  # CSV/Excel
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Vyberte CSV nebo Excel soubor", 
                    type=["csv", "xlsx", "xls"], 
                    label_visibility="collapsed"
                )
                
                if uploaded_file is not None:
                    try:
                        # Options for CSV/Excel
                        with st.expander("Nastavení importu", expanded=False):
                            if uploaded_file.name.endswith('.csv'):
                                separator = st.text_input("Oddělovač", value=",")
                                encoding = st.selectbox("Kódování", 
                                                      options=["utf-8", "iso-8859-1", "windows-1250", "latin1", "latin2"],
                                                      index=0)
                            else:  # Excel
                                sheet_name = st.text_input("Název listu (ponechte prázdné pro první list)", value="")
                        
                        # Load the data
                        if uploaded_file.name.endswith('.csv'):
                            data = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
                        else:
                            if sheet_name.strip():
                                data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                            else:
                                data = pd.read_excel(uploaded_file)
                        
                        # Set session state
                        st.session_state.data = data
                        st.session_state.eda = EDA(data)
                        
                        st.success(f"✅ Úspěšně nahráno {data.shape[0]} řádků a {data.shape[1]} sloupců")
                        
                    except Exception as e:
                        st.error(f"Chyba: {str(e)}")
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>CSV a Excel</h4>
                    <ul>
                        <li>CSV (.csv)</li>
                        <li>Excel (.xlsx, .xls)</li>
                    </ul>
                    <p>Podporuje různé oddělovače a kódování.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with file_tabs[1]:  # JSON
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_json = st.file_uploader(
                    "Vyberte JSON soubor", 
                    type=["json"], 
                    label_visibility="collapsed",
                    key="json_uploader"
                )
                
                if uploaded_json is not None:
                    try:
                        # Options for JSON
                        with st.expander("Nastavení importu", expanded=False):
                            orient = st.selectbox(
                                "Orientace JSON", 
                                options=["records", "split", "index", "columns", "values"],
                                index=0,
                                help="Určuje strukturu JSON dat."
                            )
                            lines = st.checkbox("JSON Lines formát", value=False, 
                                             help="Zaškrtněte, pokud každý řádek je samostatný JSON objekt.")
                        
                        # Load the data
                        if lines:
                            data = pd.read_json(uploaded_json, lines=True)
                        else:
                            data = pd.read_json(uploaded_json, orient=orient)
                        
                        # Normalizace nested JSON
                        if st.checkbox("Normalizovat vnořená JSON data", value=False,
                                    help="Rozbalí vnořené JSON objekty do samostatných sloupců"):
                            # Find columns that contain lists or dicts
                            complex_columns = []
                            for col in data.columns:
                                if data[col].dtype == 'object':
                                    if any(isinstance(x, (dict, list)) for x in data[col].dropna()):
                                        complex_columns.append(col)
                            
                            if complex_columns:
                                col_to_normalize = st.selectbox("Vyberte sloupec pro normalizaci", complex_columns)
                                try:
                                    # Normalize the selected column
                                    normalized = pd.json_normalize(data[col_to_normalize])
                                    # Create prefixed column names to avoid duplicates
                                    normalized.columns = [f"{col_to_normalize}.{c}" for c in normalized.columns]
                                    # Drop the original column and join the normalized data
                                    data = data.drop(columns=[col_to_normalize]).reset_index(drop=True)
                                    data = pd.concat([data, normalized], axis=1)
                                except Exception as e:
                                    st.warning(f"Nelze normalizovat sloupec: {str(e)}")
                        
                        # Set session state
                        st.session_state.data = data
                        st.session_state.eda = EDA(data)
                        
                        st.success(f"✅ Úspěšně nahráno {data.shape[0]} řádků a {data.shape[1]} sloupců")
                        
                    except Exception as e:
                        st.error(f"Chyba při načítání JSON: {str(e)}")
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>JSON formát</h4>
                    <p>Podporuje:</p>
                    <ul>
                        <li>Standardní JSON</li>
                        <li>JSON Lines (.jsonl)</li>
                        <li>Vnořené objekty</li>
                        <li>Různé orientace</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with file_tabs[2]:  # Parquet
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_parquet = st.file_uploader(
                    "Vyberte Parquet soubor", 
                    type=["parquet", "pq"], 
                    label_visibility="collapsed",
                    key="parquet_uploader"
                )
                
                if uploaded_parquet is not None:
                    try:
                        # Check if pyarrow is installed
                        try:
                            import pyarrow.parquet as pq
                            has_pyarrow = True
                        except ImportError:
                            has_pyarrow = False
                            st.warning("Knihovna PyArrow není nainstalována. Zkusíme načíst pomocí fastparquet.")
                        
                        # Options for Parquet
                        with st.expander("Nastavení importu", expanded=False):
                            engine = "pyarrow" if has_pyarrow else "fastparquet"
                            st.info(f"Použitý engine: {engine}")
                            
                            # If using PyArrow, allow column selection
                            if has_pyarrow:
                                # Create a temporary file to analyze
                                import tempfile
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
                                    tmp.write(uploaded_parquet.getvalue())
                                    tmp_path = tmp.name
                                
                                # Get schema
                                table = pq.read_table(tmp_path)
                                all_columns = table.column_names
                                
                                # Let user select columns
                                selected_columns = st.multiselect(
                                    "Vyberte sloupce (ponechte prázdné pro všechny)",
                                    options=all_columns,
                                    default=[]
                                )
                            else:
                                selected_columns = None
                        
                        # Load the data
                        if selected_columns:
                            data = pd.read_parquet(uploaded_parquet, engine=engine, columns=selected_columns)
                        else:
                            data = pd.read_parquet(uploaded_parquet, engine=engine)
                        
                        # Set session state
                        st.session_state.data = data
                        st.session_state.eda = EDA(data)
                        
                        st.success(f"✅ Úspěšně nahráno {data.shape[0]} řádků a {data.shape[1]} sloupců")
                        
                    except Exception as e:
                        st.error(f"Chyba při načítání Parquet: {str(e)}")
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Parquet formát</h4>
                    <p>Vysoce výkonný sloupcový formát. Podporuje:</p>
                    <ul>
                        <li>Efektivní komprese</li>
                        <li>Rychlé načítání</li>
                        <li>Výběr sloupců</li>
                    </ul>
                    <p><em>Vyžaduje PyArrow nebo fastparquet</em></p>
                </div>
                """, unsafe_allow_html=True)
    
    with source_tabs[1]:  # API Connection
        st.markdown("""
        <div class="card primary">
            <h4>Načtení dat z API</h4>
            <p>Připojte se k REST API a importujte data přímo do aplikace.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # API connection form
        with st.container():
            api_url = st.text_input("URL API", placeholder="https://api.example.com/data")
            
            # API Method & Auth
            col1, col2 = st.columns(2)
            with col1:
                api_method = st.selectbox(
                    "HTTP Metoda", 
                    options=["GET", "POST"],
                    index=0
                )
            
            with col2:
                api_auth_type = st.selectbox(
                    "Autentizace", 
                    options=["Žádná", "API Key", "Bearer Token", "Basic Auth"],
                    index=0
                )
            
            # Show auth fields based on selected type
            if api_auth_type == "API Key":
                col1, col2 = st.columns(2)
                with col1:
                    api_key_name = st.text_input("Název API klíče", value="api-key")
                with col2:
                    api_key_value = st.text_input("Hodnota API klíče", type="password")
                api_key_location = st.radio("Umístění API klíče", ["Header", "Query Parameter"], horizontal=True)
            
            elif api_auth_type == "Bearer Token":
                bearer_token = st.text_input("Bearer Token", type="password")
            
            elif api_auth_type == "Basic Auth":
                col1, col2 = st.columns(2)
                with col1:
                    basic_username = st.text_input("Uživatelské jméno")
                with col2:
                    basic_password = st.text_input("Heslo", type="password")
            
            # Advanced options with expander
            with st.expander("Pokročilé nastavení"):
                # Headers
                st.subheader("HTTP Hlavičky")
                add_headers = st.checkbox("Přidat vlastní hlavičky", value=False)
                headers = {}
                
                if add_headers:
                    header_count = st.number_input("Počet hlaviček", min_value=1, max_value=10, value=1)
                    for i in range(int(header_count)):
                        col1, col2 = st.columns(2)
                        with col1:
                            header_name = st.text_input(f"Název hlavičky {i+1}", key=f"header_name_{i}")
                        with col2:
                            header_value = st.text_input(f"Hodnota hlavičky {i+1}", key=f"header_value_{i}")
                        if header_name:
                            headers[header_name] = header_value
                
                # Request body for POST
                if api_method == "POST":
                    st.subheader("Tělo požadavku")
                    body_type = st.radio("Typ těla", ["JSON", "Form Data"], horizontal=True)
                    
                    if body_type == "JSON":
                        request_body = st.text_area("JSON tělo", value="{}")
                    else:  # Form Data
                        form_count = st.number_input("Počet parametrů", min_value=1, max_value=10, value=1)
                        form_data = {}
                        for i in range(int(form_count)):
                            col1, col2 = st.columns(2)
                            with col1:
                                form_name = st.text_input(f"Název parametru {i+1}", key=f"form_name_{i}")
                            with col2:
                                form_value = st.text_input(f"Hodnota parametru {i+1}", key=f"form_value_{i}")
                            if form_name:
                                form_data[form_name] = form_value
                
                # Response options
                st.subheader("Nastavení odpovědi")
                json_path = st.text_input("JSON cesta k datům", 
                                        placeholder="data.results", 
                                        help="Cesta k datům v JSON odpovědi, např. 'data.items'")
                flatten_nested = st.checkbox("Rozbalit vnořené objekty", value=True)
                normalize_arrays = st.checkbox("Normalizovat pole", value=True)
            
            # Execute API call
            if st.button("Načíst data z API", type="primary", use_container_width=True):
                if not api_url:
                    st.error("Zadejte URL API endpoint-u")
                else:
                    try:
                        import requests
                        import json
                        
                        # Prepare request
                        request_kwargs = {"headers": {}}
                        
                        # Add authentication
                        if api_auth_type == "API Key":
                            if api_key_location == "Header":
                                request_kwargs["headers"][api_key_name] = api_key_value
                            else:  # Query Parameter
                                if "?" in api_url:
                                    api_url += f"&{api_key_name}={api_key_value}"
                                else:
                                    api_url += f"?{api_key_name}={api_key_value}"
                        
                        elif api_auth_type == "Bearer Token":
                            request_kwargs["headers"]["Authorization"] = f"Bearer {bearer_token}"
                        
                        elif api_auth_type == "Basic Auth":
                            from requests.auth import HTTPBasicAuth
                            request_kwargs["auth"] = HTTPBasicAuth(basic_username, basic_password)
                        
                        # Add custom headers
                        if headers:
                            request_kwargs["headers"].update(headers)
                        
                        # Add request body for POST
                        if api_method == "POST":
                            if body_type == "JSON":
                                request_kwargs["json"] = json.loads(request_body)
                            else:  # Form Data
                                request_kwargs["data"] = form_data
                        
                        # Make the request
                        with st.spinner("Odesílám požadavek..."):
                            if api_method == "GET":
                                response = requests.get(api_url, **request_kwargs)
                            else:  # POST
                                response = requests.post(api_url, **request_kwargs)
                            
                            # Check status
                            response.raise_for_status()
                            
                            # Parse response
                            response_json = response.json()
                            
                            # Navigate to nested data if needed
                            if json_path:
                                try:
                                    path_parts = json_path.split('.')
                                    data_obj = response_json
                                    for part in path_parts:
                                        data_obj = data_obj[part]
                                    response_json = data_obj
                                except (KeyError, TypeError) as e:
                                    st.error(f"Chyba při navigaci JSON cestou: {str(e)}")
                                    st.json(response_json)
                                    return
                            
                            # Convert to DataFrame
                            if isinstance(response_json, list):
                                data = pd.json_normalize(response_json) if flatten_nested else pd.DataFrame(response_json)
                            elif isinstance(response_json, dict):
                                if normalize_arrays:
                                    # Find list fields to normalize
                                    array_fields = [k for k, v in response_json.items() if isinstance(v, list)]
                                    if array_fields and len(array_fields) == 1:
                                        # If there's a single array field, normalize it
                                        data = pd.json_normalize(response_json[array_fields[0]]) if flatten_nested else pd.DataFrame(response_json[array_fields[0]])
                                    else:
                                        # Otherwise treat as a single record
                                        data = pd.json_normalize([response_json]) if flatten_nested else pd.DataFrame([response_json])
                                else:
                                    data = pd.json_normalize([response_json]) if flatten_nested else pd.DataFrame([response_json])
                            else:
                                st.error("Odpověď API není ve formátu JSON listu nebo objektu")
                                st.write(response_json)
                                return
                            
                            # Set session state
                            st.session_state.data = data
                            st.session_state.eda = EDA(data)
                            
                            st.success(f"✅ Úspěšně načteno {data.shape[0]} řádků a {data.shape[1]} sloupců z API")
                    
                    except requests.exceptions.RequestException as e:
                        st.error(f"Chyba požadavku API: {str(e)}")
                    except ValueError as e:
                        st.error(f"Chyba při zpracování JSON: {str(e)}")
                    except Exception as e:
                        st.error(f"Neočekávaná chyba: {str(e)}")
    
    with source_tabs[2]:  # Sample datasets
        st.markdown("### Vyberte ukázkový dataset")
        
        # Use cards for sample datasets with a more modern layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card primary" style="height: 100%;">
                <h4>Iris květiny 🌸</h4>
                <p>Klasický dataset pro klasifikaci.</p>
                <p><strong>150 řádků, 5 sloupců</strong></p>
                <p>Obsahuje: délku a šířku okvětních lístků a kališních lístků, druh</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Nahrát Iris dataset", key="load_iris", use_container_width=True):
                try:
                    from sklearn.datasets import load_iris
                    iris = load_iris()
                    data = pd.DataFrame(iris.data, columns=iris.feature_names)
                    data['target'] = iris.target
                    
                    st.session_state.data = data
                    st.session_state.eda = EDA(data)
                    
                    st.success("✅ Nahrán dataset Iris")
                    st.rerun()
                except Exception as e:
                    st.error(f"Chyba při nahrávání ukázkového datasetu: {str(e)}")
        
        with col2:
            st.markdown("""
            <div class="card orange" style="height: 100%;">
                <h4>Pasažéři Titanicu 🚢</h4>
                <p>Dataset pro analýzu přežití pasažérů.</p>
                <p><strong>891 řádků, 12 sloupců</strong></p>
                <p>Obsahuje: věk, pohlaví, třídu, cenu lístku, přežití</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Nahrát Titanic dataset", key="load_titanic", use_container_width=True):
                try:
                    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
                    data = pd.read_csv(url)
                    
                    st.session_state.data = data
                    st.session_state.eda = EDA(data)
                    
                    st.success("✅ Nahrán dataset Titanicu")
                    st.rerun()
                except Exception as e:
                    st.error(f"Chyba při nahrávání ukázkového datasetu: {str(e)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card green" style="height: 100%;">
                <h4>Boston Housing 🏠</h4>
                <p>Dataset cen nemovitostí.</p>
                <p><strong>506 řádků, 14 sloupců</strong></p>
                <p>Obsahuje: kriminalitu, počet pokojů, stáří, vzdálenosti</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Nahrát Boston dataset", key="load_boston", use_container_width=True):
                try:
                    url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
                    data = pd.read_csv(url)
                    
                    st.session_state.data = data
                    st.session_state.eda = EDA(data)
                    
                    st.success("✅ Nahrán dataset Boston Housing")
                    st.rerun()
                except Exception as e:
                    st.error(f"Chyba při nahrávání ukázkového datasetu: {str(e)}")
        
        with col2:
            st.markdown("""
            <div class="card yellow" style="height: 100%;">
                <h4>COVID-19 Data 🦠</h4>
                <p>Časová řada COVID případů a úmrtí.</p>
                <p><strong>~200 zemí, denní data</strong></p>
                <p>Obsahuje: počty případů, úmrtí, data podle zemí</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Nahrát COVID dataset", key="load_covid", use_container_width=True):
                try:
                    url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
                    data = pd.read_csv(url)
                    
                    st.session_state.data = data
                    st.session_state.eda = EDA(data)
                    
                    st.success("✅ Nahrán dataset COVID-19")
                    st.rerun()
                except Exception as e:
                    st.error(f"Chyba při nahrávání ukázkového datasetu: {str(e)}")
    
    # Display preview if data exists
    if st.session_state.data is not None:
        st.markdown("---")
        st.subheader("Náhled dat")
        
        # Create tabs for different data views
        preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Tabulka", "Informace", "Statistiky"])
        
        with preview_tab1:
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        with preview_tab2:
            buffer = io.StringIO()
            st.session_state.data.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with preview_tab3:
            if len(st.session_state.eda.numeric_cols) > 0:
                st.dataframe(st.session_state.data[st.session_state.eda.numeric_cols].describe(), use_container_width=True)
            else:
                st.info("Nejsou k dispozici žádné numerické sloupce pro statistickou analýzu")
        
        # Quick analysis button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Spustit rychlou analýzu", type="primary", use_container_width=True):
                with st.spinner("Analyzuji data..."):
                    poznatky = st.session_state.eda.run_full_analysis()
                    st.success("Analýza dokončena!")
                    st.subheader("Klíčové poznatky")
                    
                    # Display insights in a card
                    insights_html = "<div class='insight-card'><ul>"
                    for poznatek in poznatky:
                        insights_html += f"<li>{poznatek}</li>"
                    insights_html += "</ul></div>"
                    
                    st.markdown(insights_html, unsafe_allow_html=True)
        
        with col2:
            if st.button("Pokračovat na přehled dat", type="primary", use_container_width=True):
                st.session_state.zakladni_stranka = "Přehled dat"
                st.rerun()
    
def zobraz_prehled_dat():
    st.header("Přehled dat")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    # Spustit analýzu, pokud ještě nebyla provedena
    if not hasattr(eda, 'missing_analysis'):
        eda.analyze_missing_values()
    if not hasattr(eda, 'correlation_matrix') and len(eda.numeric_cols) >= 2:
        eda.analyze_correlations()
    if not hasattr(eda, 'outlier_summary'):
        eda.detect_outliers()
    
    # Klíčové metriky
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Řádky", data.shape[0])
    with col2:
        st.metric("Sloupce", data.shape[1])
    with col3:
        missing_percent = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        st.metric("Chybějící hodnoty", f"{missing_percent:.2f}%")
    with col4:
        duplicate_count, _ = eda.get_duplicate_rows()
        st.metric("Duplicitní řádky", duplicate_count)
    
    # Kvalita dat
    st.subheader("Hodnocení kvality dat")
    try:
        # Zde můžeme implementovat funkci pro výpočet skóre kvality dat
        if hasattr(eda, 'calculate_data_quality_score'):
            quality_score = eda.calculate_data_quality_score()
            
            # Vizualizace skóre kvality
            col1, col2 = st.columns([1, 3])
            with col1:
                if quality_score['overall_score'] >= 0.9:
                    st.markdown(f"<h1 style='text-align: center; color: green;'>{quality_score['grade']}</h1>", unsafe_allow_html=True)
                elif quality_score['overall_score'] >= 0.8:
                    st.markdown(f"<h1 style='text-align: center; color: #8BC34A;'>{quality_score['grade']}</h1>", unsafe_allow_html=True)
                elif quality_score['overall_score'] >= 0.7:
                    st.markdown(f"<h1 style='text-align: center; color: #FFC107;'>{quality_score['grade']}</h1>", unsafe_allow_html=True)
                elif quality_score['overall_score'] >= 0.6:
                    st.markdown(f"<h1 style='text-align: center; color: #FF9800;'>{quality_score['grade']}</h1>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{quality_score['grade']}</h1>", unsafe_allow_html=True)
            
            with col2:
                metrics_df = pd.DataFrame({
                    'Metrika': list(quality_score['metrics'].keys()),
                    'Skóre': list(quality_score['metrics'].values())
                })
                
                fig = px.bar(
                    metrics_df,
                    x='Skóre',
                    y='Metrika',
                    orientation='h',
                    color='Skóre',
                    color_continuous_scale=['red', OICT_COLORS['purple']],
                    range_color=[0, 1],
                    title='Skóre kvality dat'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"**Celkové skóre kvality dat:** {quality_score['overall_score']:.2f} (Známka {quality_score['grade']})")
        else:
            # Jednodušší alternativa pro kvalitu dat
            completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
            duplicate_rate = duplicate_count / data.shape[0]
            overall_quality = (completeness * 0.7) + ((1 - duplicate_rate) * 0.3)
            
            st.progress(overall_quality, text=f"Celková kvalita dat: {overall_quality:.2f}")
            
    except Exception as e:
        st.warning(f"Nelze spočítat skóre kvality dat: {str(e)}")
    
    # Rozdělení typů sloupců
    st.subheader("Typy sloupců")
    
    col_type_data = pd.DataFrame({
        'Typ': ['Numerické', 'Kategorické', 'Datum/čas'],
        'Počet': [len(eda.numeric_cols), len(eda.categorical_cols), len(eda.datetime_cols)]
    })
    
    fig = px.bar(
        col_type_data, 
        x='Typ', 
        y='Počet',
        color='Typ',
        color_discrete_sequence=OICT_PALETTE,
        title="Rozdělení typů sloupců"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seznam sloupců
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Numerické sloupce")
        st.write(", ".join(eda.numeric_cols) if eda.numeric_cols else "Žádné")
        
        st.markdown("##### Datum/čas sloupce")
        st.write(", ".join(eda.datetime_cols) if eda.datetime_cols else "Žádné")
    
    with col2:
        st.markdown("##### Kategorické sloupce")
        st.write(", ".join(eda.categorical_cols) if eda.categorical_cols else "Žádné")
    
    # Klíčové poznatky
    st.subheader("Klíčové poznatky")
    insights = eda.generate_insights()
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Ukázka dat
    with st.expander("Ukázka dat", expanded=False):
        st.dataframe(data.head(10))
    
    # Info o datech
    with st.expander("Informace o datech", expanded=False):
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())
        
    # Možnost stáhnout report
    st.subheader("Stáhnout report")
    if st.button("Vygenerovat HTML report"):
        report_html = vytvor_html_report(data, eda)
        b64 = base64.b64encode(report_html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="eda_report.html" class="btn btn-primary">Stáhnout HTML report</a>'
        st.markdown(href, unsafe_allow_html=True)

def zobraz_chybejici_hodnoty():
    st.header("Analýza chybějících hodnot")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    # Spustit analýzu, pokud ještě nebyla provedena
    if not hasattr(eda, 'missing_analysis'):
        eda.analyze_missing_values()
    
    # Celkový pohled na chybějící hodnoty
    total_missing = data.isnull().sum().sum()
    total_values = data.shape[0] * data.shape[1]
    missing_percent = (total_missing / total_values) * 100
    
    st.markdown(f"**Celkové chybějící hodnoty:** {total_missing} z {total_values} ({missing_percent:.2f}%)")
    
    # Chybějící hodnoty podle sloupců
    missing_cols = eda.missing_analysis[eda.missing_analysis['Missing Values'] > 0]
    
    if len(missing_cols) > 0:
        st.subheader(f"Sloupce s chybějícími hodnotami ({len(missing_cols)})")
        
        # Graf chybějících hodnot
        missing_df = missing_cols.reset_index()
        missing_df.columns = ['Sloupec', 'Chybějící hodnoty', 'Procento chybějících']
        
        fig = px.bar(
            missing_df.sort_values('Procento chybějících', ascending=False),
            x='Procento chybějících',
            y='Sloupec',
            orientation='h',
            color='Procento chybějících',
            color_continuous_scale=['#eff1fe', OICT_COLORS['purple']],
            title='Chybějící hodnoty podle sloupců',
            labels={'Procento chybějících': 'Procento chybějících (%)', 'Sloupec': 'Název sloupce'}
        )
        fig.update_layout(xaxis_ticksuffix='%')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabulka chybějících hodnot
        st.dataframe(missing_df)
        
        # Doporučení
        st.subheader("Doporučení")
        
        high_missing = missing_df[missing_df['Procento chybějících'] > 50]
        if len(high_missing) > 0:
            st.markdown(f"- Zvažte odstranění sloupců s > 50% chybějících hodnot: {', '.join(high_missing['Sloupec'])}")
        
        medium_missing = missing_df[(missing_df['Procento chybějících'] <= 50) & (missing_df['Procento chybějících'] > 10)]
        if len(medium_missing) > 0:
            st.markdown(f"- Zvažte imputaci chybějících hodnot pro sloupce s 10-50% chybějícími hodnotami: {', '.join(medium_missing['Sloupec'])}")
        
        low_missing = missing_df[missing_df['Procento chybějících'] <= 10]
        if len(low_missing) > 0:
            st.markdown(f"- Bezpečné doplnění sloupců s < 10% chybějícími hodnotami: {', '.join(low_missing['Sloupec'])}")
        
        st.markdown("- Pro numerické sloupce použijte doplnění průměrem nebo mediánem")
        st.markdown("- Pro kategorické sloupce použijte modus nebo vytvořte kategorii 'Chybějící'")
        
        # Vzorce pro doplnění chybějících hodnot
        st.subheader("Vzorce pro doplnění chybějících hodnot")
        
        code_tab1, code_tab2 = st.tabs(["Python", "R"])
        
        with code_tab1:
            st.code("""
# Doplnění numerických sloupců mediánem
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Doplnění kategorických sloupců modem
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
            """)
            
        with code_tab2:
            st.code("""
# Doplnění numerických sloupců mediánem
df[numeric_cols] <- lapply(df[numeric_cols], function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)
})

# Doplnění kategorických sloupců modem
for (col in categorical_cols) {
  mode_val <- names(sort(table(df[[col]]), decreasing = TRUE))[1]
  df[[col]][is.na(df[[col]])] <- mode_val
}
            """)
            
    else:
        st.success("V datasetu nebyly nalezeny žádné chybějící hodnoty!")

def zobraz_distribuce():
    create_section_header("Distribuce dat", icon="📊", 
                         description="Analýza rozložení hodnot v jednotlivých sloupcích")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    # Create tabs for numeric vs categorical
    tab1, tab2 = st.tabs(["📈 Numerické sloupce", "📊 Kategorické sloupce"])
    
    with tab1:
        if len(eda.numeric_cols) > 0:
            # Souhrnné statistiky
            with st.expander("📋 Souhrnné statistiky", expanded=False):
                st.dataframe(eda.analyze_distributions(), use_container_width=True)
            
            # Visualization container
            st.markdown("""
            <div class="chart-container">
                <h3>Analýza numerické distribuce</h3>
            """, unsafe_allow_html=True)
            
            # Výběr sloupce pro vizualizaci
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_num_col = st.selectbox("Vyberte numerický sloupec", eda.numeric_cols)
            
            with col2:
                # Add visualization options
                show_boxplot = st.checkbox("Zobrazit boxplot", value=True)
            
            # Vytvoření histogramu
            fig = px.histogram(
                data,
                x=selected_num_col,
                color_discrete_sequence=[OICT_COLORS['purple']],
                marginal="box" if show_boxplot else None,
                title=f"Distribuce sloupce {selected_num_col}",
                opacity=0.7,
                histnorm="percent"
            )
            
            # Přidání vertikální čáry pro průměr
            fig.add_vline(
                x=data[selected_num_col].mean(),
                line_dash="dash",
                line_color=OICT_COLORS['orange'],
                annotation_text="Průměr",
                annotation_position="top right"
            )
            
            # Přidání vertikální čáry pro medián
            fig.add_vline(
                x=data[selected_num_col].median(),
                line_dash="dash",
                line_color=OICT_COLORS['green'],
                annotation_text="Medián",
                annotation_position="top left"
            )
            
            # Enhance layout
            fig.update_layout(
                xaxis_title=selected_num_col,
                yaxis_title="Procento výskytů (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Close container
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Create a two-column layout for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Create metric cards
            metrics = [
                {"title": "Průměr", "value": f"{data[selected_num_col].mean():.2f}", "icon": "📏", 
                 "color": OICT_COLORS['purple']},
                {"title": "Medián", "value": f"{data[selected_num_col].median():.2f}", "icon": "📊", 
                 "color": OICT_COLORS['yellow']},
                {"title": "Směr. odchylka", "value": f"{data[selected_num_col].std():.2f}", "icon": "📈", 
                 "color": OICT_COLORS['orange']},
                {"title": "IQR", "value": f"{(data[selected_num_col].quantile(0.75) - data[selected_num_col].quantile(0.25)):.2f}", 
                 "icon": "📉", "color": OICT_COLORS['green']}
            ]
            
            create_metric_row(metrics)
            
            # Advanced statistics
            with st.expander("🔬 Pokročilé statistiky", expanded=False):
                try:
                    skewness = data[selected_num_col].skew()
                    kurtosis = data[selected_num_col].kurtosis()
                    shapiro_test = stats.shapiro(data[selected_num_col].dropna())
                    
                    st.markdown(f"""
                    <div class="card">
                        <h4>Šikmost a špičatost</h4>
                        <p><strong>Šikmost (skewness):</strong> {skewness:.4f} - 
                           {'Pravostranně zešikmené' if skewness > 0.5 else 'Levostranně zešikmené' if skewness < -0.5 else 'Přibližně symetrické'}</p>
                        <p><strong>Špičatost (kurtosis):</strong> {kurtosis:.4f} - 
                           {'Více špičaté než normální rozdělení' if kurtosis > 0.5 else 
                            'Plošší než normální rozdělení' if kurtosis < -0.5 else 
                            'Podobné normálnímu rozdělení'}</p>
                        <p><strong>Test normality (Shapiro-Wilk):</strong></p>
                        <ul>
                            <li>Statistika: {shapiro_test[0]:.4f}</li>
                            <li>p-hodnota: {shapiro_test[1]:.4f}</li>
                            <li>Závěr: {'Normální rozdělení nelze vyloučit' if shapiro_test[1] >= 0.05 else 'Data pravděpodobně nejsou z normálního rozdělení'}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    st.warning("Některé statistiky nelze vypočítat pro tento sloupec")
    
    with tab2:
        if len(eda.categorical_cols) > 0:
            # Visualization container
            st.markdown("""
            <div class="chart-container">
                <h3>Analýza kategorické distribuce</h3>
            """, unsafe_allow_html=True)
            
            # Výběr sloupce a nastavení
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_cat_col = st.selectbox("Vyberte kategorický sloupec", eda.categorical_cols)
            
            with col2:
                # Add visualization options
                sort_by = st.radio("Řazení", ["Četnost", "Abecedně"], horizontal=True)
            
            # Získání počtu hodnot
            cat_summaries = eda.get_categorical_summaries()
            value_counts = cat_summaries[selected_cat_col]
            
            # Sort if requested
            if sort_by == "Četnost":
                value_counts = value_counts.sort_values("Count", ascending=False)
            else:
                value_counts = value_counts.sort_values(selected_cat_col)
            
            # Check number of categories
            if len(value_counts) > 15:
                st.info(f"Zobrazeno prvních 15 kategorií z celkových {len(value_counts)}")
                value_counts = value_counts.head(15)
            
            # Enhanced bar chart
            fig = px.bar(
                value_counts,
                x=selected_cat_col,
                y='Count',
                color=selected_cat_col,
                color_discrete_sequence=OICT_PALETTE * 3,
                title=f"Distribuce sloupce {selected_cat_col}",
                text='Percentage',
                opacity=0.8
            )
            
            # Formatting
            fig.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Počet: %{y}<br>Procento: %{text:.1f}%'
            )
            
            # Enhance layout
            fig.update_layout(
                xaxis_title=selected_cat_col,
                yaxis_title="Počet",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add pie chart as an alternative view
            show_pie = st.checkbox("Zobrazit jako koláčový graf", value=False)
            
            if show_pie:
                pie_fig = px.pie(
                    value_counts, 
                    values='Count', 
                    names=selected_cat_col,
                    title=f"Rozložení kategorií v {selected_cat_col}",
                    color=selected_cat_col,
                    color_discrete_sequence=OICT_PALETTE * 3,
                    hole=0.4
                )
                
                st.plotly_chart(pie_fig, use_container_width=True)
            
            # Close container
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Summary metrics for categorical
            total_count = value_counts['Count'].sum()
            unique_count = len(value_counts)
            max_category = value_counts.iloc[0][selected_cat_col] if not value_counts.empty else "N/A"
            max_percent = value_counts['Percentage'].max() if not value_counts.empty else 0
            
            metrics = [
                {"title": "Počet kategorií", "value": unique_count, "icon": "🔢", 
                 "color": OICT_COLORS['purple']},
                {"title": "Celkový počet", "value": total_count, "icon": "📊", 
                 "color": OICT_COLORS['yellow']},
                {"title": "Nejčastější kategorie", "value": max_category, "icon": "🏆", 
                 "color": OICT_COLORS['orange']},
                {"title": "Max. zastoupení", "value": f"{max_percent:.1f}%", "icon": "📈", 
                 "color": OICT_COLORS['green']}
            ]
            
            create_metric_row(metrics)
            
            # Detailed table
            with st.expander("📋 Detailní tabulka", expanded=False):
                st.dataframe(value_counts, use_container_width=True)

def zobraz_korelace():
    create_section_header("Analýza korelací", icon="🔄", 
                         description="Zkoumání vzájemných vztahů mezi numerickými proměnnými")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    if len(eda.numeric_cols) < 2:
        st.warning("Pro analýzu korelací jsou potřeba alespoň 2 numerické sloupce")
        return
    
    # Use tabs for different correlation views
    tab1, tab2, tab3 = st.tabs(["📊 Korelační matice", "📈 Párové korelace", "📋 Vysoké korelace"])
    
    with tab1:
        # Container for correlation matrix
        st.markdown("""
        <div class="chart-container">
            <h3>Korelační matice</h3>
        """, unsafe_allow_html=True)
        
        # Spustit analýzu, pokud ještě nebyla provedena
        if not hasattr(eda, 'correlation_matrix'):
            eda.analyze_correlations()
        
        # Enhanced controls
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            # Metoda korelace as modern radio
            corr_method = create_modern_radio(
                "Metoda korelace:",
                ["Pearson", "Spearman"],
                key="corr_method"
            )
        
        with col2:
            # Nastavení filtru prahové hodnoty
            threshold = st.slider(
                "Práh korelace (abs)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05
            )
        
        with col3:
            # Add column selector
            all_selected = st.checkbox("Vybrat všechny sloupce", value=True)
            if all_selected:
                selected_columns = eda.numeric_cols
            else:
                selected_columns = st.multiselect(
                    "Vyberte sloupce pro korelační matici",
                    eda.numeric_cols,
                    default=eda.numeric_cols[:min(8, len(eda.numeric_cols))]
                )
        
        # Přepočítat korelaci, pokud se metoda změnila
        if corr_method.lower() != "pearson" or not all_selected:
            # Generate correlation only for selected columns
            correlation_matrix = data[selected_columns].corr(method=corr_method.lower())
        else:
            correlation_matrix = eda.correlation_matrix
        
        # Filtrovat podle prahu
        if threshold > 0:
            mask = np.abs(correlation_matrix) > threshold
            filtered_corr = correlation_matrix.where(mask, 0)
        else:
            filtered_corr = correlation_matrix
        
        # Enhanced heatmap
        fig = px.imshow(
            filtered_corr,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title=f'{corr_method} korelační matice (práh: {threshold})',
            aspect="auto"
        )
        
        # Improve heatmap appearance
        fig.update_layout(
            height=600,
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add text annotations to heatmap
        annotations = []
        for i, row in enumerate(filtered_corr.index):
            for j, col in enumerate(filtered_corr.columns):
                value = filtered_corr.iloc[i, j]
                # Only show text for non-zero values
                if value != 0:
                    text_color = "white" if abs(value) > 0.4 else "black"
                    annotations.append({
                        "x": j,
                        "y": i,
                        "text": f"{value:.2f}",
                        "showarrow": False,
                        "font": {
                            "color": text_color,
                            "size": 10
                        }
                    })
        
        fig.update_layout(annotations=annotations)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Close container
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        # Container for scatter plots
        st.markdown("""
        <div class="chart-container">
            <h3>Párové korelace</h3>
        """, unsafe_allow_html=True)
        
        # Two column selection
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Proměnná X", eda.numeric_cols, key="pair_x")
        
        with col2:
            remaining_cols = [col for col in eda.numeric_cols if col != x_var]
            y_var = st.selectbox("Proměnná Y", remaining_cols, key="pair_y")
        
        # Create enhanced scatter plot
        scatter_fig = px.scatter(
            data,
            x=x_var,
            y=y_var,
            trendline="ols",
            color_discrete_sequence=[OICT_COLORS['purple']],
            opacity=0.7,
            title=f"Korelace: {x_var} vs {y_var}"
        )
        
        # Enhanced layout
        scatter_fig.update_layout(
            xaxis_title=x_var,
            yaxis_title=y_var,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Calculate correlation statistics
        valid_data = data[[x_var, y_var]].dropna()
        pearson_r, pearson_p = stats.pearsonr(valid_data[x_var], valid_data[y_var])
        spearman_r, spearman_p = stats.spearmanr(valid_data[x_var], valid_data[y_var])
        
        # Add correlation info to the plot
        scatter_fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=f"Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})<br>Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=OICT_COLORS['purple'],
            borderwidth=1,
            borderpad=4
        )
        
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Add correlation interpretation
        corr_strength = abs(pearson_r)
        if corr_strength < 0.3:
            strength = "slabá"
            color = "#FFC107"  # Yellow
        elif corr_strength < 0.6:
            strength = "střední"
            color = "#FF9800"  # Orange
        elif corr_strength < 0.8:
            strength = "silná"
            color = "#F44336"  # Red
        else:
            strength = "velmi silná"
            color = "#9C27B0"  # Purple
        
        direction = "pozitivní" if pearson_r > 0 else "negativní"
        
        st.markdown(f"""
        <div class="insight-card">
            <p>Mezi proměnnými <strong>{x_var}</strong> a <strong>{y_var}</strong> existuje 
            <span style="color: {color}; font-weight: bold;">{strength} {direction}</span> korelace 
            (r = {pearson_r:.4f}).</p>
            
            <p>Korelace je statisticky {pearson_p < 0.05 and "významná" or "nevýznamná"} 
            (p {pearson_p < 0.05 and "<" or "≥"} 0.05).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Close container
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        # Container for high correlations
        st.markdown("""
        <div class="chart-container">
            <h3>Silné korelace</h3>
        """, unsafe_allow_html=True)
        
        # High correlations
        if hasattr(eda, 'high_correlations') and len(eda.high_correlations) > 0:
            # Enhanced visualization of high correlations
            high_corr_fig = px.bar(
                eda.high_correlations.sort_values('Correlation', ascending=False),
                x='Correlation',
                y=eda.high_correlations.apply(lambda x: f"{x['Feature 1']} & {x['Feature 2']}", axis=1),
                orientation='h',
                color='Correlation',
                color_continuous_scale=['blue', 'white', 'red'],
                range_color=[-1, 1],
                title='Páry s vysokou korelací (|r| ≥ 0.7)'
            )
            
            # Enhance bar chart
            high_corr_fig.update_layout(
                yaxis_title="",
                xaxis_title="Korelační koeficient",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(high_corr_fig, use_container_width=True)
            
            # Enhanced high correlations table
            st.markdown("#### Tabulka silných korelací")
            
            # Format the table with colors
            def highlight_correlation(val):
                color = 'red' if val > 0.9 else 'orange' if val > 0.8 else 'green'
                return f'color: {color}; font-weight: bold'
            
            styled_high_corrs = eda.high_correlations.style.format({
                'Correlation': '{:.4f}'
            }).map(highlight_correlation, subset=['Correlation'])
            
            st.dataframe(styled_high_corrs, use_container_width=True)
            
            # Add recommendations in a card
            st.markdown("#### Doporučení")
            
            very_high_corr = eda.high_correlations[eda.high_correlations['Correlation'].abs() > 0.9]
            if len(very_high_corr) > 0:
                st.markdown("""
                <div class="card orange">
                    <h4>Proměnné s velmi silnou korelací (|r| > 0.9)</h4>
                    <p>Zvažte odstranění jedné z každého páru, abyste předešli multikolinearitě:</p>
                    <ul>
                """, unsafe_allow_html=True)
                
                for _, row in very_high_corr.iterrows():
                    st.markdown(f"""
                    <li><strong>{row['Feature 1']}</strong> & <strong>{row['Feature 2']}</strong> 
                    (r = {row['Correlation']:.3f})</li>
                    """, unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class="card green">
                    <h4>Nebyla nalezena žádná extrémně silná korelace</h4>
                    <p>V datech nebyly nalezeny páry s velmi silnou korelací (|r| > 0.9), 
                    což je pozitivní z hlediska předcházení multikolinearitě v modelech.</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("Nebyly nalezeny žádné vysoké korelace (|r| ≥ 0.7)")
        
        # Close container
        st.markdown("</div>", unsafe_allow_html=True)


def zobraz_odlehle_hodnoty():
    st.header("Analýza odlehlých hodnot")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    if len(eda.numeric_cols) == 0:
        st.warning("Pro analýzu odlehlých hodnot jsou potřeba numerické sloupce")
        return
    
    # Spustit analýzu, pokud ještě nebyla provedena
    if not hasattr(eda, 'outlier_summary'):
        eda.detect_outliers()
    
    # Výběr sloupce pro vizualizaci
    selected_col = st.selectbox("Vyberte numerický sloupec", eda.numeric_cols)
    
    # Vytvoření boxplotu
    fig = px.box(
        data,
        y=selected_col,
        title=f"Boxplot pro {selected_col}",
        color_discrete_sequence=[OICT_COLORS['purple']]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Informace o odlehlých hodnotách
    if selected_col in eda.outlier_summary:
        outlier_info = eda.outlier_summary[selected_col]
        
        st.markdown(f"**Zjištěné odlehlé hodnoty:** {outlier_info['count']} ({outlier_info['percent']:.2f}% hodnot)")
        st.markdown(f"**Dolní hranice:** {outlier_info['lower_bound']:.2f}")
        st.markdown(f"**Horní hranice:** {outlier_info['upper_bound']:.2f}")
        
        # Vytvořit histogram se zvýrazněním odlehlých hodnot
        clean_data = data[selected_col].dropna()
        is_outlier = (clean_data < outlier_info['lower_bound']) | (clean_data > outlier_info['upper_bound'])
        
        hist_data = pd.DataFrame({
            'value': clean_data,
            'is_outlier': is_outlier
        })
        
        fig = px.histogram(
            hist_data,
            x='value',
            color='is_outlier',
            color_discrete_map={True: OICT_COLORS['orange'], False: OICT_COLORS['purple']},
            title=f'Distribuce se zvýrazněnými odlehlými hodnotami (oranžová)',
            barmode='overlay',
            opacity=0.7
        )
        
        # Přidat vertikální čáry pro hranice
        fig.add_vline(x=outlier_info['lower_bound'], line_dash="dash", line_color="black")
        fig.add_vline(x=outlier_info['upper_bound'], line_dash="dash", line_color="black")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Doporučení
        st.subheader("Doporučení")
        
        if outlier_info['percent'] < 1:
            st.markdown("- Odlehlé hodnoty tvoří velmi malé procento dat a mohou být skutečné anomálie")
            st.markdown("- Zvažte prozkoumání jednotlivých odlehlých bodů, abyste pochopili jejich původ")
        elif outlier_info['percent'] < 5:
            st.markdown("- Odlehlé hodnoty tvoří malé procento dat")
            st.markdown("- Možnosti, jak s nimi naložit:")
            st.markdown("  1. Omezit odlehlé hodnoty na dolní/horní hranice (winsorization)")
            st.markdown("  2. Odstranit odlehlé hodnoty, pokud se jedná o chyby měření")
            st.markdown("  3. Vytvořit binární příznak indikující přítomnost odlehlých hodnot")
        else:
            st.markdown("- Distribuce je pravděpodobně zešikmená nebo má významný počet odlehlých hodnot")
            st.markdown("- Zvažte aplikaci transformací (log, sqrt) pro normalizaci distribuce")
            st.markdown("- Přehodnoťte definici odlehlých hodnot pro tuto proměnnou - metoda IQR nemusí být vhodná")
    else:
        st.success(f"V {selected_col} nebyly zjištěny žádné odlehlé hodnoty")
    
    # Souhrnný přehled odlehlých hodnot
    st.subheader("Souhrnný přehled odlehlých hodnot")
    
    if hasattr(eda, 'outlier_summary') and len(eda.outlier_summary) > 0:
        outlier_data = []
        for col, info in eda.outlier_summary.items():
            outlier_data.append({
                'Sloupec': col,
                'Počet odlehlých hodnot': info['count'],
                'Procento odlehlých hodnot': info['percent']
            })
        
        outlier_df = pd.DataFrame(outlier_data)
        outlier_df = outlier_df.sort_values('Procento odlehlých hodnot', ascending=False)
        
        fig = px.bar(
            outlier_df,
            x='Sloupec',
            y='Procento odlehlých hodnot',
            color='Procento odlehlých hodnot',
            color_continuous_scale=['#eff1fe', OICT_COLORS['purple']],
            title='Procento odlehlých hodnot podle sloupce'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(outlier_df)
    else:
        st.success("V žádném numerickém sloupci nebyly zjištěny odlehlé hodnoty")

def zobraz_pca():
    st.header("Redukce dimenzí (PCA)")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    if len(eda.numeric_cols) < 3:
        st.warning("Pro PCA jsou potřeba alespoň 3 numerické sloupce.")
        return
    
    # Nastavení PCA
    st.subheader("Nastavení PCA")
    
    # Výběr sloupců pro PCA
    selected_features = st.multiselect(
        "Vyberte sloupce pro PCA",
        eda.numeric_cols,
        default=eda.numeric_cols[:min(10, len(eda.numeric_cols))]
    )
    
    n_components = st.slider("Počet komponent", 2, min(10, len(selected_features)), 2)
    
    if len(selected_features) < 3:
        st.warning("Vyberte alespoň 3 sloupce pro provedení PCA")
        return
    
    # Tlačítko pro spuštění PCA
    run_pca = st.button("Spustit PCA")
    
    if run_pca:
        with st.spinner("Provádím PCA..."):
            # Příprava dat
            pca_data = data[selected_features].dropna()
            
            if len(pca_data) < 10:
                st.error("Nedostatek dat pro PCA po odstranění chybějících hodnot")
                return
            
            # Standardizace
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pca_data)
            
            # Aplikace PCA
            pca = PCA(n_components=n_components)
            transformed_data = pca.fit_transform(scaled_data)
            
            # Vytvoření výsledného dataframe
            result_df = pd.DataFrame(
                transformed_data,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=pca_data.index
            )
            
            # Zobrazení výsledků
            st.subheader("Výsledky PCA")
            
            # Vysvětlený rozptyl
            explained_variance = pca.explained_variance_ratio_
            total_variance = sum(explained_variance)
            
            st.markdown(f"**Celkový vysvětlený rozptyl:** {total_variance:.2%}")
            
            # Graf vysvětleného rozptylu
            variance_data = pd.DataFrame({
                'Komponenta': [f'PC{i+1}' for i in range(len(explained_variance))],
                'Vysvětlený rozptyl': explained_variance,
                'Kumulativní rozptyl': np.cumsum(explained_variance)
            })
            
            fig = px.bar(
                variance_data,
                x='Komponenta',
                y='Vysvětlený rozptyl',
                title='Vysvětlený rozptyl podle komponent',
                text=variance_data['Vysvětlený rozptyl'].apply(lambda x: f'{x:.1%}')
            )
            
            # Přidat čáru kumulativního rozptylu
            fig.add_scatter(
                x=variance_data['Komponenta'],
                y=variance_data['Kumulativní rozptyl'],
                mode='lines+markers',
                name='Kumulativní rozptyl',
                line=dict(color=OICT_COLORS['orange']),
                yaxis='y2'
            )
            
            fig.update_layout(
                yaxis2=dict(
                    title='Kumulativní rozptyl',
                    overlaying='y',
                    side='right',
                    range=[0, 1.1],
                    tickformat='.0%',
                    showgrid=False
                ),
                yaxis_tickformat='.0%'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Zobrazení PCA v prostoru prvních dvou komponent
            if n_components >= 2:
                st.subheader("Projekce PCA (první dvě komponenty)")
                
                # Obarvení bodů podle kategorie, pokud je k dispozici
                if len(eda.categorical_cols) > 0:
                    color_by = st.selectbox(
                        "Obarvit podle kategorie",
                        ["Žádné"] + eda.categorical_cols
                    )
                    
                    if color_by != "Žádné":
                        pca_df = result_df.copy()
                        pca_df[color_by] = data.loc[pca_df.index, color_by]
                        
                        fig = px.scatter(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            color=color_by,
                            title=f'PCA: První dvě komponenty (vysvětleno {(explained_variance[0] + explained_variance[1]):.1%} rozptylu)',
                            opacity=0.7
                        )
                    else:
                        fig = px.scatter(
                            result_df,
                            x='PC1',
                            y='PC2',
                            title=f'PCA: První dvě komponenty (vysvětleno {(explained_variance[0] + explained_variance[1]):.1%} rozptylu)',
                            color_discrete_sequence=[OICT_COLORS['purple']],
                            opacity=0.7
                        )
                else:
                    fig = px.scatter(
                        result_df,
                        x='PC1',
                        y='PC2',
                        title=f'PCA: První dvě komponenty (vysvětleno {(explained_variance[0] + explained_variance[1]):.1%} rozptylu)',
                        color_discrete_sequence=[OICT_COLORS['purple']],
                        opacity=0.7
                    )
                
                fig.update_layout(
                    xaxis_title=f'PC1 ({explained_variance[0]:.1%} rozptylu)',
                    yaxis_title=f'PC2 ({explained_variance[1]:.1%} rozptylu)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Váhy příznaků (loadings)
                st.subheader("Váhy příznaků (loadings)")
                
                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(n_components)],
                    index=selected_features
                )
                
                loadings_melted = pd.melt(
                    loadings.reset_index(),
                    id_vars='index',
                    var_name='Komponenta',
                    value_name='Váha'
                )
                
                fig = px.imshow(
                    loadings,
                    color_continuous_scale='RdBu_r',
                    title='Váhy příznaků'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabulka vah
                st.dataframe(loadings)

def zobraz_clustering():
    st.header("Shluková analýza (clustering)")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    if len(eda.numeric_cols) < 2:
        st.warning("Pro shlukovou analýzu jsou potřeba alespoň 2 numerické sloupce.")
        return
    
    # Nastavení shlukové analýzy
    st.subheader("Nastavení shlukové analýzy")
    
    # Výběr sloupců
    selected_features = st.multiselect(
        "Vyberte sloupce pro shlukovou analýzu",
        eda.numeric_cols,
        default=eda.numeric_cols[:min(5, len(eda.numeric_cols))]
    )
    
    # Počet shluků
    n_clusters = st.slider("Počet shluků", 2, 10, 3)
    
    # Metoda shlukování
    clustering_method = st.radio(
        "Metoda shlukování",
        ["K-means", "Hierarchické shlukování"],
        horizontal=True
    )
    
    if len(selected_features) < 2:
        st.warning("Vyberte alespoň 2 sloupce pro shlukovou analýzu")
        return
    
    # Tlačítko pro spuštění shlukové analýzy
    run_clustering = st.button("Spustit shlukovou analýzu")
    
    if run_clustering:
        with st.spinner("Provádím shlukovou analýzu..."):
            # Příprava dat
            cluster_data = data[selected_features].dropna()
            
            if len(cluster_data) < n_clusters * 3:
                st.error(f"Nedostatek dat pro {n_clusters} shluků. Potřebujete alespoň {n_clusters * 3} řádků bez chybějících hodnot.")
                return
            
            # Standardizace
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Aplikace shlukové analýzy
            if clustering_method == "K-means":
                from sklearn.cluster import KMeans
                
                # K-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Elbow method (pro zobrazení optimálního počtu shluků)
                inertia = []
                for k in range(1, min(11, len(cluster_data) // 3)):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(scaled_data)
                    inertia.append(km.inertia_)
                
                elbow_data = pd.DataFrame({
                    'Počet shluků': range(1, len(inertia) + 1),
                    'Inertia': inertia
                })
                
                fig = px.line(
                    elbow_data,
                    x='Počet shluků',
                    y='Inertia',
                    title='Elbow Method pro určení optimálního počtu shluků',
                    markers=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                from sklearn.cluster import AgglomerativeClustering
                
                # Hierarchické shlukování
                hc = AgglomerativeClustering(n_clusters=n_clusters)
                clusters = hc.fit_predict(scaled_data)
            
            # Přidání informace o shlucích k datům
            cluster_results = cluster_data.copy()
            cluster_results['Shluk'] = clusters
            
            # Zobrazení výsledků
            st.subheader("Výsledky shlukové analýzy")
            
            # Distribuce shluků
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            cluster_counts_df = cluster_counts.reset_index()
            cluster_counts_df.columns = ['Shluk', 'Počet']
            
            fig = px.bar(
                cluster_counts_df,
                x='Shluk',
                y='Počet',
                color='Shluk',
                title='Distribuce shluků'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Profily shluků
            st.subheader("Profily shluků")
            
            # Výpočet průměrů pro každý shluk
            cluster_means = cluster_results.groupby('Shluk').mean()
            
            # Standardizace průměrů pro lepší vizualizaci
            cluster_means_scaled = (cluster_means - cluster_means.mean()) / cluster_means.std()
            
            # Heatmapa profilů shluků
            fig = px.imshow(
                cluster_means_scaled,
                color_continuous_scale='RdBu_r',
                title='Profily shluků (standardizované průměry příznaků)',
                labels={'index': 'Shluk', 'variable': 'Příznak', 'value': 'Standardizovaná hodnota'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabulka průměrů pro shluky
            st.markdown("#### Průměry příznaků podle shluků")
            st.dataframe(cluster_means)
            
            # Vizualizace shluků v 2D prostoru
            if len(selected_features) >= 2:
                st.subheader("Vizualizace shluků")
                
                if len(selected_features) == 2:
                    # Přímá vizualizace, pokud máme jen 2 příznaky
                    viz_data = cluster_results.copy()
                    
                    fig = px.scatter(
                        viz_data,
                        x=selected_features[0],
                        y=selected_features[1],
                        color='Shluk',
                        title=f'Shluky v prostoru {selected_features[0]} vs {selected_features[1]}',
                        opacity=0.7
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # Použití PCA pro vizualizaci ve 2D
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_data)
                    
                    pca_df = pd.DataFrame({
                        'PC1': pca_result[:, 0],
                        'PC2': pca_result[:, 1],
                        'Shluk': clusters
                    })
                    
                    fig = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color='Shluk',
                        title='Shluky v PCA prostoru',
                        opacity=0.7
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Charakteristiky jednotlivých shluků
            st.subheader("Charakteristiky shluků")
            
            for cluster_id in range(n_clusters):
                with st.expander(f"Shluk {cluster_id}"):
                    # Počet záznamů ve shluku
                    cluster_size = (clusters == cluster_id).sum()
                    st.markdown(f"**Počet záznamů:** {cluster_size} ({cluster_size / len(clusters):.1%} z celku)")
                    
                    # Typické hodnoty
                    st.markdown("**Typické hodnoty:**")
                    
                    # Určení význačných vlastností
                    profile = cluster_means_scaled.loc[cluster_id]
                    significant_high = profile[profile > 0.5].sort_values(ascending=False)
                    significant_low = profile[profile < -0.5].sort_values()
                    
                    if not significant_high.empty:
                        st.markdown("*Nadprůměrné hodnoty:*")
                        for feature, value in significant_high.items():
                            st.markdown(f"- {feature}: {value:.2f} směr. odchylek nad průměrem (hodnota: {cluster_means.loc[cluster_id, feature]:.2f})")
                    
                    if not significant_low.empty:
                        st.markdown("*Podprůměrné hodnoty:*")
                        for feature, value in significant_low.items():
                            st.markdown(f"- {feature}: {-value:.2f} směr. odchylek pod průměrem (hodnota: {cluster_means.loc[cluster_id, feature]:.2f})")

def zobraz_statisticke_testy():
    st.header("Statistické testy")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    # Výběr typu testu
    test_type = st.selectbox(
        "Vyberte typ statistického testu",
        ["Test normality", "Testy korelace", "t-test", "ANOVA", "Chi-kvadrát"]
    )
    
    if test_type == "Test normality":
        # Test normality
        st.subheader("Test normality")
        
        if len(eda.numeric_cols) == 0:
            st.warning("Pro test normality jsou potřeba numerické sloupce")
            return
        
        col = st.selectbox("Vyberte sloupec pro test normality", eda.numeric_cols)
        
        clean_data = data[col].dropna()
        if len(clean_data) < 3:
            st.error("Nedostatek dat pro test normality")
            return
        
        # Limit počtu hodnot pro test Shapiro-Wilk (max 5000)
        if len(clean_data) > 5000:
            st.warning(f"Pro test Shapiro-Wilk bude použit vzorek 5000 hodnot z celkových {len(clean_data)}")
            sample_data = clean_data.sample(5000, random_state=42)
        else:
            sample_data = clean_data
        
        # Provedení testu Shapiro-Wilk
        shapiro_test = stats.shapiro(sample_data)
        
        # Vizualizace histogramu s překrytou normální distribucí
        fig = px.histogram(
            sample_data,
            nbins=30,
            histnorm='probability density',
            title=f"Distribuce {col} s normální křivkou"
        )
        
        # Přidání normální křivky
        x = np.linspace(min(sample_data), max(sample_data), 100)
        mean, std = sample_data.mean(), sample_data.std()
        pdf = stats.norm.pdf(x, mean, std)
        fig.add_scatter(x=x, y=pdf, mode='lines', name='Normální distribuce')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Vytvoření Q-Q plotu
        fig = px.scatter(
            x=np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(sample_data)))),
            y=np.sort(sample_data),
            title="Q-Q Plot (normální distribuce)",
            labels={"x": "Teoretické kvantily", "y": "Vzorové kvantily"}
        )
        
        # Přidání referenční čáry
        fig.add_scatter(
            x=[min(sample_data), max(sample_data)],
            y=[min(sample_data), max(sample_data)],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Výsledky testu
        st.subheader("Výsledky testu normality")
        st.markdown(f"**Test Shapiro-Wilk:**")
        st.markdown(f"- Statistika: {shapiro_test[0]:.4f}")
        st.markdown(f"- p-hodnota: {shapiro_test[1]:.4f}")
        
        if shapiro_test[1] < 0.05:
            st.markdown("**Závěr: Data NEJSOU normálně distribuována (p < 0.05)**")
        else:
            st.markdown("**Závěr: Data MOHOU BÝT normálně distribuována (p ≥ 0.05)**")
        
        # Šikmost a špičatost
        skewness = stats.skew(sample_data)
        kurtosis = stats.kurtosis(sample_data)
        
        st.markdown(f"**Šikmost (Skewness):** {skewness:.4f}")
        if abs(skewness) < 0.5:
            st.markdown("Distribuce je přibližně symetrická")
        elif abs(skewness) < 1:
            st.markdown("Distribuce je mírně zešikmená")
        else:
            st.markdown("Distribuce je silně zešikmená")
        
        st.markdown(f"**Špičatost (Kurtosis):** {kurtosis:.4f}")
        if abs(kurtosis) < 0.5:
            st.markdown("Špičatost je podobná normální distribuci")
        elif kurtosis > 0:
            st.markdown("Distribuce je špičatější než normální (leptokurtická)")
        else:
            st.markdown("Distribuce je plošší než normální (platykurtická)")
        
    elif test_type == "Testy korelace":
        # Testy korelace
        st.subheader("Testy korelace")
        
        if len(eda.numeric_cols) < 2:
            st.warning("Pro testy korelace jsou potřeba alespoň 2 numerické sloupce")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Vyberte první proměnnou", eda.numeric_cols, key="x_col")
        
        with col2:
            y_col = st.selectbox("Vyberte druhou proměnnou", [col for col in eda.numeric_cols if col != x_col], key="y_col")
        
        # Odstranění chybějících hodnot
        clean_data = data[[x_col, y_col]].dropna()
        if len(clean_data) < 3:
            st.error("Nedostatek dat pro test korelace")
            return
        
        # Vizualizace scatter plotu
        fig = px.scatter(
            clean_data,
            x=x_col,
            y=y_col,
            trendline="ols",
            title=f"Vztah mezi {x_col} a {y_col}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Provedení testů korelace
        pearson_r, pearson_p = stats.pearsonr(clean_data[x_col], clean_data[y_col])
        spearman_r, spearman_p = stats.spearmanr(clean_data[x_col], clean_data[y_col])
        
        # Výsledky testů
        st.subheader("Výsledky testů korelace")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Pearsonova korelace:**")
            st.markdown(f"- Koeficient r: {pearson_r:.4f}")
            st.markdown(f"- p-hodnota: {pearson_p:.4f}")
            if pearson_p < 0.05:
                st.markdown("- Závěr: Statisticky významná lineární korelace (p < 0.05)")
            else:
                st.markdown("- Závěr: Žádná statisticky významná lineární korelace (p ≥ 0.05)")
        
        with col2:
            st.markdown("**Spearmanova korelace:**")
            st.markdown(f"- Koeficient rho: {spearman_r:.4f}")
            st.markdown(f"- p-hodnota: {spearman_p:.4f}")
            if spearman_p < 0.05:
                st.markdown("- Závěr: Statisticky významná pořadová korelace (p < 0.05)")
            else:
                st.markdown("- Závěr: Žádná statisticky významná pořadová korelace (p ≥ 0.05)")
        
        # Interpretace síly korelace
        st.subheader("Interpretace síly korelace")
        
        corr_strength = abs(pearson_r)
        if corr_strength < 0.3:
            strength = "slabá"
        elif corr_strength < 0.6:
            strength = "střední"
        elif corr_strength < 0.8:
            strength = "silná"
        else:
            strength = "velmi silná"
        
        direction = "pozitivní" if pearson_r > 0 else "negativní"
        
        st.markdown(f"Mezi proměnnými **{x_col}** a **{y_col}** existuje **{strength} {direction}** korelace.")
        
    elif test_type == "t-test":
        # t-test
        st.subheader("t-test")
        
        # Výběr typu t-testu
        t_test_type = st.radio(
            "Vyberte typ t-testu",
            ["Jednovýběrový t-test", "Dvouvýběrový t-test (nezávislé vzorky)", "Párový t-test"],
            horizontal=True
        )
        
        if t_test_type == "Jednovýběrový t-test":
            if len(eda.numeric_cols) == 0:
                st.warning("Pro jednovýběrový t-test jsou potřeba numerické sloupce")
                return
            
            col = st.selectbox("Vyberte sloupec pro test", eda.numeric_cols)
            
            # Referenční hodnota
            ref_value = st.number_input("Zadejte referenční hodnotu pro porovnání", value=0.0)
            
            # Odstranění chybějících hodnot
            clean_data = data[col].dropna()
            if len(clean_data) < 3:
                st.error("Nedostatek dat pro t-test")
                return
            
            # Provedení t-testu
            t_stat, p_value = stats.ttest_1samp(clean_data, ref_value)
            
            # Vizualizace distribuce s referenční hodnotou
            fig = px.histogram(
                clean_data,
                title=f"Distribuce {col} s referenční hodnotou",
                histnorm='probability density'
            )
            
            # Přidání referenční hodnoty
            fig.add_vline(x=ref_value, line_dash="dash", line_color="red", annotation_text="Referenční hodnota")
            
            # Přidání průměru vzorku
            fig.add_vline(x=clean_data.mean(), line_dash="dash", line_color="green", annotation_text="Průměr vzorku")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Výsledky testu
            st.subheader("Výsledky jednovýběrového t-testu")
            st.markdown(f"**Testovaná hypotéza:** Průměr sloupce *{col}* je roven {ref_value}")
            st.markdown(f"- t-statistika: {t_stat:.4f}")
            st.markdown(f"- p-hodnota: {p_value:.4f}")
            
            if p_value < 0.05:
                st.markdown("**Závěr: Zamítáme nulovou hypotézu (p < 0.05)**")
                if t_stat > 0:
                    st.markdown(f"Průměr sloupce *{col}* je statisticky významně **větší než** {ref_value}")
                else:
                    st.markdown(f"Průměr sloupce *{col}* je statisticky významně **menší než** {ref_value}")
            else:
                st.markdown("**Závěr: Nemůžeme zamítnout nulovou hypotézu (p ≥ 0.05)**")
                st.markdown(f"Nemáme dostatek důkazů, že průměr sloupce *{col}* se liší od {ref_value}")
            
        elif t_test_type == "Dvouvýběrový t-test (nezávislé vzorky)":
            if len(eda.numeric_cols) == 0:
                st.warning("Pro dvouvýběrový t-test jsou potřeba numerické sloupce")
                return
            
            # Výběr numerického sloupce
            num_col = st.selectbox("Vyberte numerický sloupec", eda.numeric_cols)
            
            # Výběr kategorie pro rozdělení do skupin
            if len(eda.categorical_cols) == 0:
                st.warning("Pro rozdělení do skupin potřebujete kategorický sloupec")
                return
            
            cat_col = st.selectbox("Vyberte kategorický sloupec pro definici skupin", eda.categorical_cols)
            
            # Získání unikátních hodnot v kategorickém sloupci
            unique_cats = data[cat_col].dropna().unique()
            
            if len(unique_cats) < 2:
                st.error("Kategorický sloupec musí mít alespoň 2 unikátní hodnoty")
                return
            
            # Výběr dvou skupin pro porovnání
            col1, col2 = st.columns(2)
            
            with col1:
                group1 = st.selectbox("Vyberte první skupinu", unique_cats, index=0)
            
            with col2:
                remaining_cats = [cat for cat in unique_cats if cat != group1]
                group2 = st.selectbox("Vyberte druhou skupinu", remaining_cats, index=0)
            
            # Příprava dat pro test
            group1_data = data[data[cat_col] == group1][num_col].dropna()
            group2_data = data[data[cat_col] == group2][num_col].dropna()
            
            if len(group1_data) < 3 or len(group2_data) < 3:
                st.error("Obě skupiny musí mít alespoň 3 hodnoty")
                return
            
            # Provedení Leveneova testu pro rovnost rozptylů
            levene_stat, levene_p = stats.levene(group1_data, group2_data)
            
            # Provedení t-testu
            equal_var = levene_p >= 0.05  # Rovnost rozptylů, pokud p-hodnota Leveneova testu je >= 0.05
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
            
            # Vizualizace distribucí obou skupin
            hist_data = pd.DataFrame({
                num_col: pd.concat([group1_data, group2_data]),
                'Skupina': [group1] * len(group1_data) + [group2] * len(group2_data)
            })
            
            fig = px.histogram(
                hist_data,
                x=num_col,
                color='Skupina',
                barmode='overlay',
                histnorm='probability density',
                title=f"Distribuce {num_col} pro skupiny {group1} a {group2}"
            )
            
            # Přidání průměrů skupin
            fig.add_vline(x=group1_data.mean(), line_dash="dash", line_color="blue", 
                        annotation_text=f"Průměr {group1}")
            fig.add_vline(x=group2_data.mean(), line_dash="dash", line_color="red", 
                        annotation_text=f"Průměr {group2}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Výsledky testu
            st.subheader("Výsledky dvouvýběrového t-testu")
            
            # Základní statistiky
            st.markdown("**Základní statistiky:**")
            stats_df = pd.DataFrame({
                'Skupina': [group1, group2],
                'Počet': [len(group1_data), len(group2_data)],
                'Průměr': [group1_data.mean(), group2_data.mean()],
                'Směr. odchylka': [group1_data.std(), group2_data.std()],
                'Min': [group1_data.min(), group2_data.min()],
                'Max': [group1_data.max(), group2_data.max()]
            })
            
            st.dataframe(stats_df)
            
            # Výsledek Leveneova testu
            st.markdown("**Leveneův test rovnosti rozptylů:**")
            st.markdown(f"- Statistika: {levene_stat:.4f}")
            st.markdown(f"- p-hodnota: {levene_p:.4f}")
            if levene_p < 0.05:
                st.markdown("- Závěr: Rozptyly nejsou stejné (použit Welchův t-test)")
            else:
                st.markdown("- Závěr: Rozptyly jsou přibližně stejné (použit standardní t-test)")
            
            # Výsledek t-testu
            st.markdown("**T-test:**")
            st.markdown(f"- t-statistika: {t_stat:.4f}")
            st.markdown(f"- p-hodnota: {p_value:.4f}")
            
            if p_value < 0.05:
                st.markdown("**Závěr: Zamítáme nulovou hypotézu (p < 0.05)**")
                st.markdown(f"Průměry sloupce *{num_col}* se statisticky významně liší mezi skupinami *{group1}* a *{group2}*")
                
                # Velikost efektu (Cohen's d)
                pooled_std = np.sqrt((group1_data.var() * (len(group1_data) - 1) + 
                                    group2_data.var() * (len(group2_data) - 1)) / 
                                    (len(group1_data) + len(group2_data) - 2))
                cohen_d = abs(group1_data.mean() - group2_data.mean()) / pooled_std
                
                st.markdown(f"**Velikost efektu (Cohen's d):** {cohen_d:.4f}")
                if cohen_d < 0.2:
                    st.markdown("Interpretace: Velmi malý efekt")
                elif cohen_d < 0.5:
                    st.markdown("Interpretace: Malý efekt")
                elif cohen_d < 0.8:
                    st.markdown("Interpretace: Střední efekt")
                else:
                    st.markdown("Interpretace: Velký efekt")
            else:
                st.markdown("**Závěr: Nemůžeme zamítnout nulovou hypotézu (p ≥ 0.05)**")
                st.markdown(f"Nemáme dostatek důkazů, že průměry sloupce *{num_col}* se liší mezi skupinami *{group1}* a *{group2}*")
        
        elif t_test_type == "Párový t-test":
            if len(eda.numeric_cols) < 2:
                st.warning("Pro párový t-test jsou potřeba alespoň 2 numerické sloupce")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                first_col = st.selectbox("Vyberte první proměnnou (před)", eda.numeric_cols, key="paired_first")
            
            with col2:
                second_col = st.selectbox("Vyberte druhou proměnnou (po)", 
                                        [col for col in eda.numeric_cols if col != first_col], 
                                        key="paired_second")
            
            # Odstranění chybějících hodnot (musí mít oba sloupce hodnoty)
            paired_data = data[[first_col, second_col]].dropna()
            
            if len(paired_data) < 3:
                st.error("Nedostatek párovaných hodnot pro t-test")
                return
            
            # Provedení párového t-testu
            t_stat, p_value = stats.ttest_rel(paired_data[first_col], paired_data[second_col])
            
            # Vizualizace rozložení obou proměnných
            fig1 = px.histogram(
                paired_data,
                x=[first_col, second_col],
                barmode='overlay',
                histnorm='probability density',
                title=f"Distribuce {first_col} a {second_col}"
            )
            
            # Přidání průměrů
            fig1.add_vline(x=paired_data[first_col].mean(), line_dash="dash", line_color="blue", 
                         annotation_text=f"Průměr {first_col}")
            fig1.add_vline(x=paired_data[second_col].mean(), line_dash="dash", line_color="red", 
                         annotation_text=f"Průměr {second_col}")
            
            # Scatter plot pro párované hodnoty
            fig2 = px.scatter(
                paired_data,
                x=first_col,
                y=second_col,
                title=f"Párované hodnoty {first_col} vs {second_col}"
            )
            
            # Přidání diagonální čáry (x=y)
            fig2.add_trace(go.Scatter(
                x=[paired_data[[first_col, second_col]].min().min(), 
                   paired_data[[first_col, second_col]].max().max()],
                y=[paired_data[[first_col, second_col]].min().min(), 
                   paired_data[[first_col, second_col]].max().max()],
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='x=y'
            ))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
            
            # Výsledky testu
            st.subheader("Výsledky párového t-testu")
            
            # Základní statistiky
            mean_diff = paired_data[first_col].mean() - paired_data[second_col].mean()
            
            st.markdown("**Základní statistiky:**")
            stats_df = pd.DataFrame({
                'Proměnná': [first_col, second_col, "Rozdíl (A-B)"],
                'Průměr': [paired_data[first_col].mean(), 
                          paired_data[second_col].mean(),
                          mean_diff],
                'Směr. odchylka': [paired_data[first_col].std(), 
                                 paired_data[second_col].std(),
                                 (paired_data[first_col] - paired_data[second_col]).std()]
            })
            
            st.dataframe(stats_df)
            
            # Výsledek t-testu
            st.markdown("**Párový t-test:**")
            st.markdown(f"- t-statistika: {t_stat:.4f}")
            st.markdown(f"- p-hodnota: {p_value:.4f}")
            
            if p_value < 0.05:
                st.markdown("**Závěr: Zamítáme nulovou hypotézu (p < 0.05)**")
                if mean_diff > 0:
                    st.markdown(f"*{first_col}* je statisticky významně **větší než** *{second_col}*")
                else:
                    st.markdown(f"*{first_col}* je statisticky významně **menší než** *{second_col}*")
                
                # Velikost efektu (Cohen's d pro párové testy)
                diff = paired_data[first_col] - paired_data[second_col]
                cohen_d = abs(diff.mean()) / diff.std()
                
                st.markdown(f"**Velikost efektu (Cohen's d):** {cohen_d:.4f}")
                if cohen_d < 0.2:
                    st.markdown("Interpretace: Velmi malý efekt")
                elif cohen_d < 0.5:
                    st.markdown("Interpretace: Malý efekt")
                elif cohen_d < 0.8:
                    st.markdown("Interpretace: Střední efekt")
                else:
                    st.markdown("Interpretace: Velký efekt")
            else:
                st.markdown("**Závěr: Nemůžeme zamítnout nulovou hypotézu (p ≥ 0.05)**")
                st.markdown(f"Nemáme dostatek důkazů, že se *{first_col}* a *{second_col}* liší")
    
    elif test_type == "ANOVA":
        # ANOVA test
        st.subheader("Jednofaktorová ANOVA")
        
        if len(eda.numeric_cols) == 0:
            st.warning("Pro ANOVA test jsou potřeba numerické sloupce")
            return
        
        if len(eda.categorical_cols) == 0:
            st.warning("Pro ANOVA test jsou potřeba kategorické sloupce jako faktory")
            return
        
        # Výběr numeric a kategoriálního sloupce
        num_col = st.selectbox("Vyberte numerický sloupec (závislá proměnná)", eda.numeric_cols)
        cat_col = st.selectbox("Vyberte kategorický sloupec (faktor)", eda.categorical_cols)
        
        # Příprava dat pro ANOVA
        anova_data = data[[num_col, cat_col]].dropna()
        
        if len(anova_data) < 3:
            st.error("Nedostatek dat pro ANOVA test")
            return
        
        # Kontrola, zda máme alespoň dvě skupiny
        groups = anova_data[cat_col].unique()
        if len(groups) < 2:
            st.error("Pro ANOVA test potřebujete alespoň 2 skupiny")
            return
        
        # Převedení dat do formátu vhodného pro ANOVA
        anova_groups = [anova_data[anova_data[cat_col] == group][num_col].values for group in groups]
        
        # Provedení ANOVA testu
        f_stat, p_value = stats.f_oneway(*anova_groups)
        
        # Vizualizace - box plot
        fig = px.box(
            anova_data,
            x=cat_col,
            y=num_col,
            title=f"Porovnání {num_col} podle {cat_col}",
            points="all"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Výsledky testu
        st.subheader("Výsledky ANOVA testu")
        
        # Základní statistiky podle skupin
        group_stats = anova_data.groupby(cat_col)[num_col].agg(['count', 'mean', 'std']).reset_index()
        st.markdown("**Statistiky podle skupin:**")
        st.dataframe(group_stats)
        
        # Výsledek ANOVA
        st.markdown("**ANOVA:**")
        st.markdown(f"- F-statistika: {f_stat:.4f}")
        st.markdown(f"- p-hodnota: {p_value:.4f}")
        
        if p_value < 0.05:
            st.markdown("**Závěr: Zamítáme nulovou hypotézu (p < 0.05)**")
            st.markdown(f"Existuje statisticky významný rozdíl v *{num_col}* mezi skupinami *{cat_col}*")
            
            # Post-hoc test (Tukey's HSD)
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            
            posthoc = pairwise_tukeyhsd(anova_data[num_col], anova_data[cat_col], alpha=0.05)
            
            st.markdown("**Post-hoc test (Tukey HSD):**")
            # Převod objektu posthoc na DataFrame
            posthoc_df = pd.DataFrame(data=posthoc._results_table.data[1:], columns=posthoc._results_table.data[0])
            st.dataframe(posthoc_df)
            
            # Interpretace výsledků post-hoc testu
            st.markdown("**Interpretace post-hoc testu:**")
            significant_pairs = posthoc_df[posthoc_df['reject'] == True]
            if len(significant_pairs) > 0:
                for _, row in significant_pairs.iterrows():
                    st.markdown(f"- Skupiny *{row['group1']}* a *{row['group2']}* se statisticky významně liší (p < 0.05)")
            else:
                st.markdown("- Žádné páry skupin nevykazují statisticky významné rozdíly v post-hoc testu")
        else:
            st.markdown("**Závěr: Nemůžeme zamítnout nulovou hypotézu (p ≥ 0.05)**")
            st.markdown(f"Nemáme dostatek důkazů, že se *{num_col}* liší mezi skupinami *{cat_col}*")
    
    elif test_type == "Chi-kvadrát":
        # Chi-kvadrát test
        st.subheader("Chi-kvadrát test")
        
        if len(eda.categorical_cols) < 2:
            st.warning("Pro Chi-kvadrát test jsou potřeba alespoň 2 kategorické sloupce")
            return
        
        # Výběr dvou kategorických sloupců
        col1, col2 = st.columns(2)
        
        with col1:
            first_cat = st.selectbox("Vyberte první kategorický sloupec", eda.categorical_cols, key="chi_first")
        
        with col2:
            second_cat = st.selectbox("Vyberte druhý kategorický sloupec", 
                                    [col for col in eda.categorical_cols if col != first_cat], 
                                    key="chi_second")
        
        # Příprava dat pro chi-kvadrát test
        chi_data = data[[first_cat, second_cat]].dropna()
        
        if len(chi_data) < 10:
            st.error("Nedostatek dat pro Chi-kvadrát test")
            return
        
        # Vytvoření kontingenční tabulky
        contingency_table = pd.crosstab(chi_data[first_cat], chi_data[second_cat])
        
        # Kontrola četností
        if (contingency_table < 5).any().any():
            st.warning("Některé buňky mají očekávanou četnost méně než 5, což může ovlivnit výsledky testu")
        
        # Provedení chi-kvadrát testu
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Vizualizace - heatmapa kontingenční tabulky
        fig = px.imshow(
            contingency_table,
            title=f"Kontingenční tabulka: {first_cat} vs {second_cat}",
            labels=dict(x=second_cat, y=first_cat, color="Četnost")
        )
        
        # Přidat hodnoty do buněk
        for i in range(contingency_table.shape[0]):
            for j in range(contingency_table.shape[1]):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=str(contingency_table.iloc[i, j]),
                    showarrow=False,
                    font=dict(color="white" if contingency_table.iloc[i, j] > contingency_table.values.mean() else "black")
                )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Vizualizace - sloupcový graf
        proportions = pd.crosstab(
            chi_data[first_cat], 
            chi_data[second_cat], 
            normalize='index'
        ).reset_index().melt(
            id_vars=first_cat,
            var_name=second_cat,
            value_name='Proportion'
        )
        
        fig2 = px.bar(
            proportions,
            x=first_cat,
            y='Proportion',
            color=second_cat,
            title=f"Rozložení {second_cat}",
            labels={'Proportion': 'Podíl'}
        )
        
        fig2.update_layout(yaxis_tickformat='.0%')
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Výsledky testu
        st.subheader("Výsledky Chi-kvadrát testu")
        
        # Kontingenční tabulka
        st.markdown("**Kontingenční tabulka (četnosti):**")
        st.dataframe(contingency_table)
        
        # Očekávané četnosti
        st.markdown("**Očekávané četnosti za předpokladu nezávislosti:**")
        expected_df = pd.DataFrame(
            expected, 
            index=contingency_table.index, 
            columns=contingency_table.columns
        )
        st.dataframe(expected_df.round(2))
        
        # Výsledek chi-kvadrát testu
        st.markdown("**Chi-kvadrát test:**")
        st.markdown(f"- Chi-kvadrát statistika: {chi2:.4f}")
        st.markdown(f"- Stupně volnosti: {dof}")
        st.markdown(f"- p-hodnota: {p_value:.4f}")
        
        if p_value < 0.05:
            st.markdown("**Závěr: Zamítáme nulovou hypotézu (p < 0.05)**")
            st.markdown(f"Existuje statisticky významná asociace mezi *{first_cat}* a *{second_cat}*")
            
            # Míra asociace - Cramer's V
            n = chi_data.shape[0]
            min_dim = min(contingency_table.shape) - 1
            cramer_v = np.sqrt(chi2 / (n * min_dim))
            
            st.markdown(f"**Síla asociace (Cramer's V):** {cramer_v:.4f}")
            if cramer_v < 0.1:
                st.markdown("Interpretace: Zanedbatelná asociace")
            elif cramer_v < 0.3:
                st.markdown("Interpretace: Slabá asociace")
            elif cramer_v < 0.5:
                st.markdown("Interpretace: Střední asociace")
            else:
                st.markdown("Interpretace: Silná asociace")
        else:
            st.markdown("**Závěr: Nemůžeme zamítnout nulovou hypotézu (p ≥ 0.05)**")
            st.markdown(f"Nemáme dostatek důkazů, že existuje asociace mezi *{first_cat}* a *{second_cat}*")

def zobraz_navrhy_uprav():
    st.header("Návrhy úprav dat")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    # Příprava návrhů úprav
    st.subheader("Návrhy transformací dat")
    
    # Detekce zešikmení v numerických sloupcích
    skewed_columns = []
    for col in eda.numeric_cols:
        clean_data = data[col].dropna()
        if len(clean_data) >= 5:
            skewness = stats.skew(clean_data)
            if abs(skewness) > 1:
                skewed_columns.append({
                    'column': col,
                    'skewness': skewness,
                    'type': 'positive' if skewness > 0 else 'negative'
                })
    
    if skewed_columns:
        st.markdown("### Zešikmené sloupce")
        st.markdown("Následující sloupce mají značně zešikmenou distribuci, která může ovlivnit statistické analýzy:")
        
        skewed_df = pd.DataFrame(skewed_columns)
        st.dataframe(skewed_df)
        
        # Ukázka transformací
        if len(skewed_columns) > 0:
            selected_col = st.selectbox(
                "Vyberte sloupec pro náhled transformací",
                [col['column'] for col in skewed_columns]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Původní distribuce**")
                fig = px.histogram(
                    data[selected_col].dropna(),
                    title=f"Původní distribuce: {selected_col}",
                    histnorm='probability density'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Návrhy transformací
            with col2:
                # Identifikace typu zešikmení
                skewness = data[selected_col].skew()
                
                if skewness > 1:  # Pravostranně zešikmené
                    st.markdown("**Logaritmická transformace**")
                    
                    # Převod na kladné hodnoty pro logaritmickou transformaci
                    min_val = data[selected_col].min()
                    offset = 0 if min_val > 0 else abs(min_val) + 1
                    log_data = np.log(data[selected_col] + offset)
                    
                    fig = px.histogram(
                        log_data.dropna(),
                        title=f"Log transformace: log({selected_col} + {offset})",
                        histnorm='probability density'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.code(f"""
# Logaritmická transformace
min_val = data['{selected_col}'].min()
offset = 0 if min_val > 0 else abs(min_val) + 1
data['{selected_col}_log'] = np.log(data['{selected_col}'] + offset)
                    """)
                    
                    st.markdown("**Odmocninová transformace**")
                    sqrt_data = np.sqrt(data[selected_col] + offset)
                    
                    fig = px.histogram(
                        sqrt_data.dropna(),
                        title=f"Odmocninová transformace: sqrt({selected_col} + {offset})",
                        histnorm='probability density'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.code(f"""
# Odmocninová transformace
min_val = data['{selected_col}'].min()
offset = 0 if min_val > 0 else abs(min_val) + 1
data['{selected_col}_sqrt'] = np.sqrt(data['{selected_col}'] + offset)
                    """)
                    
                elif skewness < -1:  # Levostranně zešikmené
                    st.markdown("**Kvadratická transformace**")
                    
                    square_data = data[selected_col] ** 2
                    
                    fig = px.histogram(
                        square_data.dropna(),
                        title=f"Kvadratická transformace: {selected_col}^2",
                        histnorm='probability density'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.code(f"""
# Kvadratická transformace
data['{selected_col}_squared'] = data['{selected_col}'] ** 2
                    """)
    
    # Chybějící hodnoty
    st.markdown("### Řešení chybějících hodnot")
    
    if not hasattr(eda, 'missing_analysis'):
        eda.analyze_missing_values()
    
    missing_cols = eda.missing_analysis[eda.missing_analysis['Missing Values'] > 0]
    
    if len(missing_cols) > 0:
        missing_df = missing_cols.reset_index()
        missing_df.columns = ['Sloupec', 'Chybějící hodnoty', 'Procento chybějících']
        
        st.dataframe(missing_df)
        
        st.markdown("**Doporučené techniky imputace:**")
        
        for _, row in missing_df.iterrows():
            col_name = row['Sloupec']
            missing_pct = row['Procento chybějících']
            
            if missing_pct > 50:
                st.markdown(f"- **{col_name}** ({missing_pct:.1f}% chybějících): Zvažte odstranění sloupce")
            elif col_name in eda.numeric_cols:
                st.markdown(f"- **{col_name}** ({missing_pct:.1f}% chybějících): Imputace mediánem nebo průměrem")
                st.code(f"""
# Imputace mediánem
data['{col_name}'] = data['{col_name}'].fillna(data['{col_name}'].median())

# NEBO: Imputace průměrem
# data['{col_name}'] = data['{col_name}'].fillna(data['{col_name}'].mean())
                """)
            elif col_name in eda.categorical_cols:
                st.markdown(f"- **{col_name}** ({missing_pct:.1f}% chybějících): Imputace nejčastější hodnotou nebo novou kategorií 'Chybějící'")
                st.code(f"""
# Imputace nejčastější hodnotou
data['{col_name}'] = data['{col_name}'].fillna(data['{col_name}'].mode()[0])

# NEBO: Vytvoření nové kategorie pro chybějící hodnoty
# data['{col_name}'] = data['{col_name}'].fillna('Chybějící')
                """)
    else:
        st.success("V datasetu nebyly nalezeny žádné chybějící hodnoty!")
    
    # Návrh nových příznaků pro kategorické sloupce
    st.markdown("### Návrhy kódování kategorických proměnných")
    
    if len(eda.categorical_cols) > 0:
        cat_cols_info = []
        for col in eda.categorical_cols:
            unique_values = data[col].nunique()
            cat_cols_info.append({
                'column': col,
                'unique_values': unique_values
            })
        
        cat_cols_df = pd.DataFrame(cat_cols_info)
        st.dataframe(cat_cols_df)
        
        for col in eda.categorical_cols:
            unique_values = data[col].nunique()
            
            st.markdown(f"**{col}** ({unique_values} unikátních hodnot):")
            
            if unique_values == 2:
                st.markdown("Doporučení: Binární kódování (0/1)")
                st.code(f"""
# Binární kódování
data['{col}_binary'] = data['{col}'].map({{'{data[col].unique()[0]}': 0, '{data[col].unique()[1]}': 1}})
                """)
            elif 2 < unique_values <= 10:
                st.markdown("Doporučení: One-hot encoding")
                st.code(f"""
# One-hot encoding
data_encoded = pd.get_dummies(data['{col}'], prefix='{col}')
data = pd.concat([data, data_encoded], axis=1)
                """)
            else:
                st.markdown("Doporučení: Target encoding nebo Frequency encoding")
                st.code(f"""
# Frequency encoding
frequency_map = data['{col}'].value_counts(normalize=True).to_dict()
data['{col}_freq'] = data['{col}'].map(frequency_map)

# Target encoding (pokud máte cílovou proměnnou)
target_means = data.groupby('{col}')['target_variable'].mean().to_dict()
data['{col}_target'] = data['{col}'].map(target_means)
                """)
    
    # Návrhy pro odlehlé hodnoty
    st.markdown("### Řešení odlehlých hodnot")
    
    if not hasattr(eda, 'outlier_summary'):
        eda.detect_outliers()
    
    if hasattr(eda, 'outlier_summary') and len(eda.outlier_summary) > 0:
        outlier_data = []
        for col, info in eda.outlier_summary.items():
            outlier_data.append({
                'column': col,
                'outlier_count': info['count'],
                'outlier_percent': info['percent']
            })
        
        outlier_df = pd.DataFrame(outlier_data)
        st.dataframe(outlier_df)
        
        for col, info in eda.outlier_summary.items():
            if info['percent'] > 5:
                st.markdown(f"**{col}** ({info['percent']:.1f}% odlehlých hodnot): Zvažte transformaci dat")
            elif info['percent'] > 1:
                st.markdown(f"**{col}** ({info['percent']:.1f}% odlehlých hodnot): Oříznutí (winsorizing)")
                st.code(f"""
# Oříznutí (winsorizing)
lower_bound = {info['lower_bound']:.4f}
upper_bound = {info['upper_bound']:.4f}
data['{col}_winsorized'] = data['{col}'].clip(lower_bound, upper_bound)
                """)
            else:
                st.markdown(f"**{col}** ({info['percent']:.1f}% odlehlých hodnot): Odstranění nebo označení jako outlier")
                st.code(f"""
# Označení odlehlých hodnot
lower_bound = {info['lower_bound']:.4f}
upper_bound = {info['upper_bound']:.4f}
data['{col}_is_outlier'] = ((data['{col}'] < lower_bound) | (data['{col}'] > upper_bound))

# Alternativně: Odstranění odlehlých hodnot
# data_clean = data[~((data['{col}'] < lower_bound) | (data['{col}'] > upper_bound))]
                """)
    else:
        st.success("V datasetu nebyly nalezeny žádné výrazné odlehlé hodnoty!")

def zobraz_rychle_modelovani():
    st.header("Rychlé modelování")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    if len(eda.numeric_cols) < 2:
        st.warning("Pro modelování jsou potřeba alespoň 2 numerické sloupce.")
        return
    
    # Nastavení modelu
    st.subheader("Nastavení modelování")
    
    # Výběr cílové proměnné
    target_options = eda.numeric_cols + eda.categorical_cols
    target_col = st.selectbox("Vyberte cílovou proměnnou", target_options)
    
    # Výběr příznaků
    feature_cols = st.multiselect(
        "Vyberte příznaky pro model",
        [col for col in eda.numeric_cols if col != target_col],
        default=[col for col in eda.numeric_cols if col != target_col][:min(5, len(eda.numeric_cols))]
    )
    
    # Výběr typu modelu
    is_classification = target_col in eda.categorical_cols
    
    if is_classification:
        model_type = st.selectbox(
            "Vyberte typ modelu",
            ["Logistická regrese", "Random Forest", "SVM"]
        )
    else:
        model_type = st.selectbox(
            "Vyberte typ modelu",
            ["Lineární regrese", "Random Forest", "SVM"]
        )
    
    # Nastavení velikosti testovacího datasetu
    test_size = st.slider("Velikost testovacího datasetu", 0.1, 0.5, 0.2, 0.05)
    
    # Tlačítko pro trénování modelu
    train_button = st.button("Trénovat model")
    
    if train_button:
        if len(feature_cols) == 0:
            st.error("Vyberte alespoň jeden příznak pro model")
            return
        
        with st.spinner("Probíhá trénování modelu..."):
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            # Příprava dat
            X = data[feature_cols].copy()
            y = data[target_col].copy()
            
            # Odstranění chybějících hodnot
            X = X.fillna(X.mean())
            
            # Pro klasifikaci potřebujeme kategorické cíle
            if is_classification:
                # Pokud cíl obsahuje chybějící hodnoty, odstraníme je
                valid_idx = y.notna()
                X = X[valid_idx]
                y = y[valid_idx]
            else:
                # Pro regresi také odstraníme chybějící hodnoty v cíli
                valid_idx = y.notna()
                X = X[valid_idx]
                y = y[valid_idx]
            
            if len(X) < 50:
                st.error("Nedostatek dat pro trénování modelu (méně než 50 platných řádků)")
                return
            
            # Rozdělení na trénovací a testovací data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Standardizace dat
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Výběr a trénink modelu
            if is_classification:
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.svm import SVC
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                
                if model_type == "Logistická regrese":
                    model = LogisticRegression(max_iter=1000, random_state=42)
                elif model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:  # SVM
                    model = SVC(probability=True, random_state=42)
                
                # Trénování modelu
                model.fit(X_train_scaled, y_train)
                
                # Predikce
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Vyhodnocení
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                # Zobrazení výsledků
                st.subheader("Výsledky klasifikačního modelu")
                
                col1, col2 = st.columns(2)
                col1.metric("Přesnost na trénovacích datech", f"{train_accuracy:.4f}")
                col2.metric("Přesnost na testovacích datech", f"{test_accuracy:.4f}")
                
                # Matice záměn
                st.markdown("**Matice záměn**")
                cm = confusion_matrix(y_test, y_test_pred)
                cm_df = pd.DataFrame(cm, 
                                   index=[f'Skutečná {i}' for i in sorted(y.unique())], 
                                   columns=[f'Predikce {i}' for i in sorted(y.unique())])
                
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predikovaná třída", y="Skutečná třída", color="Počet"),
                    x=[f'Predikce {i}' for i in sorted(y.unique())],
                    y=[f'Skutečná {i}' for i in sorted(y.unique())],
                    title="Matice záměn"
                )
                
                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        fig.add_annotation(
                            x=j, y=i,
                            text=str(cm[i, j]),
                            showarrow=False,
                            font=dict(color="white" if cm[i, j] > cm.mean() else "black")
                        )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Klasifikační report
                st.markdown("**Klasifikační report**")
                report = classification_report(y_test, y_test_pred, output_dict=True)
                report_df = pd.DataFrame(report).T
                st.dataframe(report_df)
                
            else:
                from sklearn.linear_model import LinearRegression
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.svm import SVR
                from sklearn.metrics import mean_squared_error, r2_score
                
                if model_type == "Lineární regrese":
                    model = LinearRegression()
                elif model_type == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:  # SVM
                    model = SVR()
                
                # Trénování modelu
                model.fit(X_train_scaled, y_train)
                
                # Predikce
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Vyhodnocení
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                # Zobrazení výsledků
                st.subheader("Výsledky regresního modelu")
                
                col1, col2 = st.columns(2)
                col1.metric("RMSE na trénovacích datech", f"{train_rmse:.4f}")
                col2.metric("RMSE na testovacích datech", f"{test_rmse:.4f}")
                
                col1, col2 = st.columns(2)
                col1.metric("R² na trénovacích datech", f"{train_r2:.4f}")
                col2.metric("R² na testovacích datech", f"{test_r2:.4f}")
                
                # Scatter plot predikovaných vs. skutečných hodnot
                plot_data = pd.DataFrame({
                    'Skutečné hodnoty': y_test,
                    'Predikované hodnoty': y_test_pred
                })
                
                fig = px.scatter(
                    plot_data,
                    x='Skutečné hodnoty',
                    y='Predikované hodnoty',
                    title="Predikované vs. skutečné hodnoty"
                )
                
                # Přidání diagonální čáry (ideální predikce)
                fig.add_trace(go.Scatter(
                    x=[plot_data['Skutečné hodnoty'].min(), plot_data['Skutečné hodnoty'].max()],
                    y=[plot_data['Skutečné hodnoty'].min(), plot_data['Skutečné hodnoty'].max()],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Ideální predikce'
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Důležitost příznaků (pro modely, které to podporují)
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                st.subheader("Důležitost příznaků")
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:  # Linear model coefficients
                    importances = np.abs(model.coef_)
                    if importances.ndim > 1:  # Pro multiclass logistickou regresi
                        importances = np.mean(np.abs(importances), axis=0)
                
                importance_df = pd.DataFrame({
                    'Příznak': feature_cols,
                    'Důležitost': importances
                }).sort_values('Důležitost', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Důležitost',
                    y='Příznak',
                    orientation='h',
                    title="Důležitost příznaků"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(importance_df)
                
                # Nejdůležitější příznaky
                top_features = importance_df.head(3)['Příznak'].tolist()
                st.markdown(f"**Nejdůležitější příznaky:** {', '.join(top_features)}")
            
            # Ukázka kódu pro použití modelu
            st.subheader("Kód pro trénování podobného modelu")
            
            code_tab1, code_tab2 = st.tabs(["Python", "R"])
            
            with code_tab1:
                python_code = f"""
# Python kód pro trénování {model_type}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""
                # Přidání importu podle typu modelu
                if is_classification:
                    if model_type == "Logistická regrese":
                        python_code += "from sklearn.linear_model import LogisticRegression\n"
                        python_code += "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n\n"
                        python_code += f"# Definice modelu\nmodel = LogisticRegression(max_iter=1000, random_state=42)\n"
                    elif model_type == "Random Forest":
                        python_code += "from sklearn.ensemble import RandomForestClassifier\n"
                        python_code += "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n\n"
                        python_code += f"# Definice modelu\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\n"
                    else:  # SVM
                        python_code += "from sklearn.svm import SVC\n"
                        python_code += "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n\n"
                        python_code += f"# Definice modelu\nmodel = SVC(probability=True, random_state=42)\n"
                else:
                    if model_type == "Lineární regrese":
                        python_code += "from sklearn.linear_model import LinearRegression\n"
                        python_code += "from sklearn.metrics import mean_squared_error, r2_score\n\n"
                        python_code += f"# Definice modelu\nmodel = LinearRegression()\n"
                    elif model_type == "Random Forest":
                        python_code += "from sklearn.ensemble import RandomForestRegressor\n"
                        python_code += "from sklearn.metrics import mean_squared_error, r2_score\n\n"
                        python_code += f"# Definice modelu\nmodel = RandomForestRegressor(n_estimators=100, random_state=42)\n"
                    else:  # SVM
                        python_code += "from sklearn.svm import SVR\n"
                        python_code += "from sklearn.metrics import mean_squared_error, r2_score\n\n"
                        python_code += f"# Definice modelu\nmodel = SVR()\n"
                
                # Příprava dat
                python_code += f"""
# Příprava dat
feature_cols = {feature_cols}
X = data[feature_cols].copy()
y = data['{target_col}'].copy()

# Ošetření chybějících hodnot
X = X.fillna(X.mean())
valid_idx = y.notna()
X = X[valid_idx]
y = y[valid_idx]

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)

# Standardizace
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Trénování modelu
model.fit(X_train_scaled, y_train)

# Predikce
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
"""
                
                # Kód pro vyhodnocení
                if is_classification:
                    python_code += """
# Vyhodnocení klasifikace
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Přesnost na trénovacích datech: {train_accuracy:.4f}")
print(f"Přesnost na testovacích datech: {test_accuracy:.4f}")
print("\\nKlasifikační report:")
print(classification_report(y_test, y_test_pred))
print("\\nMatice záměn:")
print(confusion_matrix(y_test, y_test_pred))
"""
                else:
                    python_code += """
# Vyhodnocení regrese
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"RMSE na trénovacích datech: {train_rmse:.4f}")
print(f"RMSE na testovacích datech: {test_rmse:.4f}")
print(f"R² na trénovacích datech: {train_r2:.4f}")
print(f"R² na testovacích datech: {test_r2:.4f}")
"""
                
                st.code(python_code, language="python")
            
            with code_tab2:
                r_code = f"""
# R kód pro trénování {model_type}

library(tidyverse)
library(caret)
"""
                # Přidání importu podle typu modelu
                if is_classification:
                    if model_type == "Logistická regrese":
                        r_code += "# Logistická regrese\n\n"
                    elif model_type == "Random Forest":
                        r_code += "library(randomForest)\n\n"
                    else:  # SVM
                        r_code += "library(e1071)\n\n"
                else:
                    if model_type == "Lineární regrese":
                        r_code += "# Lineární regrese\n\n"
                    elif model_type == "Random Forest":
                        r_code += "library(randomForest)\n\n"
                    else:  # SVM
                        r_code += "library(e1071)\n\n"
                
                # Příprava dat
                r_code += f"""
# Příprava dat
feature_cols <- c({', '.join([f'"{col}"' for col in feature_cols])})
target_col <- "{target_col}"

# Výběr dat
model_data <- data %>% 
  select(all_of(c(feature_cols, target_col))) %>%
  na.omit()  # Odstranění řádků s chybějícími hodnotami

# Rozdělení dat
set.seed(42)
train_idx <- createDataPartition(model_data[[target_col]], p = {1-test_size}, list = FALSE)
train_data <- model_data[train_idx, ]
test_data <- model_data[-train_idx, ]

# Příprava formulace modelu
formula <- as.formula(paste(target_col, "~", paste(feature_cols, collapse = " + ")))
"""
                
                # Trénování modelu
                if is_classification:
                    if model_type == "Logistická regrese":
                        r_code += """
# Trénování logistické regrese
model <- glm(formula, family = binomial(link = 'logit'), data = train_data)

# Predikce
train_prob <- predict(model, train_data, type = "response")
test_prob <- predict(model, test_data, type = "response")
train_pred <- ifelse(train_prob > 0.5, 1, 0)
test_pred <- ifelse(test_prob > 0.5, 1, 0)
"""
                    elif model_type == "Random Forest":
                        r_code += """
# Trénování Random Forest
model <- randomForest(formula, data = train_data)

# Predikce
train_pred <- predict(model, train_data)
test_pred <- predict(model, test_data)
"""
                    else:  # SVM
                        r_code += """
# Trénování SVM
model <- svm(formula, data = train_data, probability = TRUE)

# Predikce
train_pred <- predict(model, train_data)
test_pred <- predict(model, test_data)
"""
                else:
                    if model_type == "Lineární regrese":
                        r_code += """
# Trénování lineární regrese
model <- lm(formula, data = train_data)

# Predikce
train_pred <- predict(model, train_data)
test_pred <- predict(model, test_data)
"""
                    elif model_type == "Random Forest":
                        r_code += """
# Trénování Random Forest
model <- randomForest(formula, data = train_data)

# Predikce
train_pred <- predict(model, train_data)
test_pred <- predict(model, test_data)
"""
                    else:  # SVM
                        r_code += """
# Trénování SVM
model <- svm(formula, data = train_data)

# Predikce
train_pred <- predict(model, train_data)
test_pred <- predict(model, test_data)
"""
                
                # Kód pro vyhodnocení
                if is_classification:
                    r_code += """
# Vyhodnocení klasifikace
library(caret)
train_cm <- confusionMatrix(as.factor(train_pred), as.factor(train_data[[target_col]]))
test_cm <- confusionMatrix(as.factor(test_pred), as.factor(test_data[[target_col]]))

print("Přesnost na trénovacích datech:")
print(train_cm$overall['Accuracy'])
print("Přesnost na testovacích datech:")
print(test_cm$overall['Accuracy'])
print("Matice záměn (test):")
print(test_cm$table)
"""
                else:
                    r_code += """
# Vyhodnocení regrese
library(Metrics)
train_rmse <- rmse(train_data[[target_col]], train_pred)
test_rmse <- rmse(test_data[[target_col]], test_pred)
train_r2 <- cor(train_data[[target_col]], train_pred)^2
test_r2 <- cor(test_data[[target_col]], test_pred)^2

print(paste("RMSE na trénovacích datech:", round(train_rmse, 4)))
print(paste("RMSE na testovacích datech:", round(test_rmse, 4)))
print(paste("R² na trénovacích datech:", round(train_r2, 4)))
print(paste("R² na testovacích datech:", round(test_r2, 4)))
"""
                
                st.code(r_code, language="r")

def zobraz_casove_rady():
    st.header("Analýza časových řad")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    # Kontrola, zda máme sloupec s datem
    if len(eda.datetime_cols) == 0:
        st.warning("Pro analýzu časových řad je potřeba sloupec s datem.")
        
        # Nabídnout možnost konverze sloupce na datum
        st.subheader("Konverze sloupce na datum")
        
        col_to_convert = st.selectbox(
            "Vyberte sloupec pro konverzi na datum",
            data.columns
        )
        
        if st.button("Konvertovat na datum"):
            try:
                data[col_to_convert] = pd.to_datetime(data[col_to_convert])
                st.session_state.data = data
                # Aktualizace eda objektu s novým typem sloupce
                st.session_state.eda = EDA(data)
                st.success(f"Sloupec {col_to_convert} byl úspěšně konvertován na datum")
                st.rerun()
            except:
                st.error(f"Nelze konvertovat sloupec {col_to_convert} na datum")
        
        return
    
    # Nastavení analýzy časových řad
    st.subheader("Nastavení analýzy")
    
    # Výběr datumového sloupce
    date_col = st.selectbox("Vyberte sloupec s datem", eda.datetime_cols)
    
    # Výběr hodnoty, kterou chceme analyzovat
    if len(eda.numeric_cols) == 0:
        st.warning("Pro analýzu časových řad potřebujete alespoň jeden numerický sloupec")
        return
    
    value_col = st.selectbox("Vyberte hodnotu pro analýzu", eda.numeric_cols)
    
    # Nastavení agregace
    agg_method = st.radio(
        "Metoda agregace",
        ["Suma", "Průměr", "Medián", "Min", "Max"],
        horizontal=True
    )
    
    # Nastavení periody
    time_period = st.radio(
        "Časová perioda",
        ["Den", "Týden", "Měsíc", "Kvartál", "Rok"],
        horizontal=True
    )
    
    # Mapování metody agregace na pandas funkce
    agg_map = {
        "Suma": "sum",
        "Průměr": "mean",
        "Medián": "median",
        "Min": "min",
        "Max": "max"
    }
    
    # Mapování časové periody na pandas frekvence
    period_map = {
        "Den": "D",
        "Týden": "W",
        "Měsíc": "M",
        "Kvartál": "Q",
        "Rok": "Y"
    }
    
    # Tlačítko pro spuštění analýzy
    run_analysis = st.button("Spustit analýzu časových řad")
    
    if run_analysis:
        with st.spinner("Probíhá analýza..."):
            # Příprava dat
            time_series_data = data[[date_col, value_col]].copy()
            time_series_data[date_col] = pd.to_datetime(time_series_data[date_col])
            time_series_data = time_series_data.dropna()
            
            if len(time_series_data) < 3:
                st.error("Nedostatek dat pro analýzu časových řad")
                return
            
            # Setřídění podle data
            time_series_data = time_series_data.sort_values(date_col)
            
            # Agregace dat podle zvolené periody
            aggregated_data = time_series_data.set_index(date_col).resample(period_map[time_period])[value_col].agg(agg_map[agg_method])
            
            # Vytvoření dataframe pro vizualizaci
            ts_df = aggregated_data.reset_index()
            ts_df.columns = ['Datum', 'Hodnota']
            
            # Zobrazení výsledků
            st.subheader("Analýza časových řad")
            
            # Časová řada
            fig = px.line(
                ts_df,
                x='Datum',
                y='Hodnota',
                title=f"{value_col} podle {time_period.lower()}ů ({agg_method.lower()})",
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Výpočet změn (meziměsíční, meziroční)
            if len(ts_df) > 1:
                # Přidání sloupce s procentuální změnou
                ts_df['Změna (%)'] = ts_df['Hodnota'].pct_change() * 100
                
                # Vizualizace změn
                fig = px.bar(
                    ts_df.dropna(),
                    x='Datum',
                    y='Změna (%)',
                    title=f"Změna {value_col} ({agg_method.lower()}) v %",
                    color='Změna (%)',
                    color_continuous_scale=['red', 'white', 'green'],
                    range_color=[-max(abs(ts_df['Změna (%)'].dropna())), max(abs(ts_df['Změna (%)'].dropna()))]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Analýza sezónnosti, pokud máme dostatek dat
            if len(ts_df) >= 12:  # Alespoň rok dat pro sezónnost
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                try:
                    # Převedení zpět na časovou řadu
                    ts = ts_df.set_index('Datum')['Hodnota']
                    
                    # Dekompozice časové řady
                    result = seasonal_decompose(ts, model='additive')
                    
                    # Vizualizace dekompozice
                    trend = result.trend.reset_index()
                    trend.columns = ['Datum', 'Hodnota']
                    trend['Komponenta'] = 'Trend'
                    
                    seasonal = result.seasonal.reset_index()
                    seasonal.columns = ['Datum', 'Hodnota']
                    seasonal['Komponenta'] = 'Sezónnost'
                    
                    residual = result.resid.reset_index()
                    residual.columns = ['Datum', 'Hodnota']
                    residual['Komponenta'] = 'Reziduum'
                    
                    observed = ts.reset_index()
                    observed.columns = ['Datum', 'Hodnota']
                    observed['Komponenta'] = 'Pozorované'
                    
                    decomp_df = pd.concat([observed, trend, seasonal, residual])
                    
                    st.subheader("Dekompozice časové řady")
                    
                    # Vizualizace dekompozice v samostatných grafech
                    for component in ['Pozorované', 'Trend', 'Sezónnost', 'Reziduum']:
                        fig = px.line(
                            decomp_df[decomp_df['Komponenta'] == component],
                            x='Datum',
                            y='Hodnota',
                            title=f"Komponenta: {component}",
                            markers=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Síla trendu a sezónnosti
                    trend_strength = 1 - np.var(result.resid) / np.var(result.observed - result.seasonal)
                    seasonal_strength = 1 - np.var(result.resid) / np.var(result.observed - result.trend)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Síla trendu", f"{trend_strength:.4f}")
                    col2.metric("Síla sezónnosti", f"{seasonal_strength:.4f}")
                    
                    # Interpretace
                    st.markdown("**Interpretace:**")
                    if trend_strength > 0.7:
                        st.markdown("- Data vykazují **silný trend**")
                    elif trend_strength > 0.4:
                        st.markdown("- Data vykazují **střední trend**")
                    else:
                        st.markdown("- Data vykazují **slabý nebo žádný trend**")
                        
                    if seasonal_strength > 0.6:
                        st.markdown("- Data vykazují **silnou sezónní složku**")
                    elif seasonal_strength > 0.3:
                        st.markdown("- Data vykazují **střední sezónní složku**")
                    else:
                        st.markdown("- Data vykazují **slabou nebo žádnou sezónní složku**")
                    
                except Exception as e:
                    st.error(f"Chyba při dekompozici časové řady: {str(e)}")
            
            # Zobrazení tabulky s daty
            st.subheader("Agregovaná data")
            st.dataframe(ts_df)
            
            # Možnost stáhnout data
            csv = ts_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Stáhnout jako CSV",
                data=csv,
                file_name=f'time_series_{value_col}_{time_period.lower()}.csv',
                mime='text/csv',
            )

def zobraz_cross_tabulky():
    st.header("Cross tabulky")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    if len(eda.categorical_cols) < 2:
        st.warning("Pro cross tabulky jsou potřeba alespoň 2 kategorické sloupce")
        return
    
    # Nastavení cross tabulky
    st.subheader("Nastavení cross tabulky")
    
    # Výběr kategorických sloupců
    col1, col2 = st.columns(2)
    
    with col1:
        row_var = st.selectbox("Vyberte proměnnou pro řádky", eda.categorical_cols, key="row_var")
    
    with col2:
        col_var = st.selectbox("Vyberte proměnnou pro sloupce", 
                             [col for col in eda.categorical_cols if col != row_var], 
                             key="col_var")
    
    # Výběr typu cross tabulky
    cross_type = st.radio(
        "Typ cross tabulky",
        ["Četnosti", "Řádková %", "Sloupcová %", "Celková %"],
        horizontal=True
    )
    
    # Mapování typu cross tabulky na normalize parametr pandas crosstab
    normalize_map = {
        "Četnosti": None,
        "Řádková %": "index",
        "Sloupcová %": "columns",
        "Celková %": "all"
    }
    
    # Tlačítko pro vytvoření cross tabulky
    create_cross = st.button("Vytvořit cross tabulku")
    
    if create_cross:
        try:
            # Příprava dat
            cross_data = data[[row_var, col_var]].dropna()
            
            if len(cross_data) < 1:
                st.error("Nedostatek dat pro vytvoření cross tabulky")
                return
            
            # Vytvoření cross tabulky - opravené pro normalizaci
            if cross_type == "Četnosti":
                # Pro četnosti nepoužívat normalize parametr vůbec
                cross_tab = pd.crosstab(
                    cross_data[row_var], 
                    cross_data[col_var],
                    margins=True,
                    margins_name="Celkem"
                )
            else:
                # Pro procentuální tabulky použít správný normalize parametr
                cross_tab = pd.crosstab(
                    cross_data[row_var], 
                    cross_data[col_var],
                    normalize=normalize_map[cross_type],
                    margins=True,
                    margins_name="Celkem"
                )
            
            # Převod na procenta u procentuálních tabulek
            if cross_type != "Četnosti":
                cross_tab = cross_tab * 100
            
            # Vizualizace cross tabulky
            st.subheader(f"Cross tabulka: {row_var} vs {col_var} ({cross_type})")
            
            # Tabulka
            if cross_type == "Četnosti":
                st.dataframe(cross_tab.style.format(precision=0))
            else:
                st.dataframe(cross_tab.style.format("{:.1f}%"))
            
            # Heatmapa - bezpečné odstranění Celkem s kontrolou existence
            try:
                # Vytvoření kopie pro vizualizaci bez Celkem
                if 'Celkem' in cross_tab.index and 'Celkem' in cross_tab.columns:
                    cross_tab_viz = cross_tab.drop('Celkem', axis=0).drop('Celkem', axis=1)
                elif 'Celkem' in cross_tab.index:
                    cross_tab_viz = cross_tab.drop('Celkem', axis=0)
                elif 'Celkem' in cross_tab.columns:
                    cross_tab_viz = cross_tab.drop('Celkem', axis=1)
                else:
                    cross_tab_viz = cross_tab.copy()
                
                # Kontrola, zda máme data pro vizualizaci
                if cross_tab_viz.empty:
                    st.warning("Nedostatek dat pro vizualizaci")
                    return
                
                fig = px.imshow(
                    cross_tab_viz,
                    labels=dict(x=col_var, y=row_var, color="Hodnota"),
                    title=f"Cross tabulka: {row_var} vs {col_var}",
                    color_continuous_scale='blues'
                )
                
                # Přidání textových popisků do buněk
                for i in range(len(cross_tab_viz.index)):
                    for j in range(len(cross_tab_viz.columns)):
                        value = cross_tab_viz.iloc[i, j]
                        text = f"{value:.1f}%" if cross_type != "Četnosti" else f"{value:.0f}"
                        fig.add_annotation(
                            x=j,
                            y=i,
                            text=text,
                            showarrow=False,
                            font=dict(color="white" if value > cross_tab_viz.values.mean() else "black")
                        )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chyba při vytváření heatmapy: {str(e)}")
            
            # Zobrazení grafu - sloupcový
            try:
                if cross_type == "Četnosti":
                    # Převedení tabulky na long formát pro Plotly
                    cross_tab_long = cross_tab.reset_index().melt(
                        id_vars=row_var,
                        value_vars=[col for col in cross_tab.columns if col != 'Celkem'],
                        var_name=col_var,
                        value_name='Četnost'
                    )
                    
                    fig = px.bar(
                        cross_tab_long,
                        x=row_var,
                        y='Četnost',
                        color=col_var,
                        title=f"Sloupcový graf: {row_var} vs {col_var}",
                        barmode='group'
                    )
                else:
                    # Pro procentuální tabulky
                    cross_tab_long = cross_tab.reset_index().melt(
                        id_vars=row_var,
                        value_vars=[col for col in cross_tab.columns if col != 'Celkem'],
                        var_name=col_var,
                        value_name='Procento'
                    )
                    
                    fig = px.bar(
                        cross_tab_long,
                        x=row_var,
                        y='Procento',
                        color=col_var,
                        title=f"Sloupcový graf: {row_var} vs {col_var}",
                        barmode='group'
                    )
                    
                    fig.update_layout(yaxis_ticksuffix='%')
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chyba při vytváření sloupcového grafu: {str(e)}")
            
            # Chi-kvadrát test (pouze pro tabulky četností)
            if cross_type == "Četnosti":
                try:
                    # Bezpečné odstranění Celkem pro statistické testy
                    chi2_tab = cross_tab.copy()
                    if 'Celkem' in chi2_tab.index:
                        chi2_tab = chi2_tab.drop('Celkem', axis=0)
                    if 'Celkem' in chi2_tab.columns:
                        chi2_tab = chi2_tab.drop('Celkem', axis=1)
                    
                    # Chi-kvadrát test
                    from scipy.stats import chi2_contingency
                    
                    chi2, p, dof, expected = chi2_contingency(chi2_tab)
                    
                    st.subheader("Test závislosti (Chi-kvadrát)")
                    
                    if (expected < 5).any().any():
                        st.warning("Očekávané četnosti jsou v některých buňkách menší než 5, což může ovlivnit výsledky testu")
                    
                    st.markdown(f"**Chi-kvadrát statistika:** {chi2:.4f}")
                    st.markdown(f"**Stupně volnosti:** {dof}")
                    st.markdown(f"**p-hodnota:** {p:.4f}")
                    
                    if p < 0.05:
                        st.markdown(f"**Závěr:** Existuje statisticky významná souvislost mezi proměnnými {row_var} a {col_var} (p < 0.05)")
                        
                        # Cramerovo V (síla asociace)
                        n = chi2_tab.values.sum()
                        min_dim = min(chi2_tab.shape) - 1
                        cramer_v = np.sqrt(chi2 / (n * min_dim))
                        
                        st.markdown(f"**Síla asociace (Cramerovo V):** {cramer_v:.4f}")
                        
                        if cramer_v < 0.1:
                            st.markdown("Interpretace: Zanedbatelná asociace")
                        elif cramer_v < 0.3:
                            st.markdown("Interpretace: Slabá asociace")
                        elif cramer_v < 0.5:
                            st.markdown("Interpretace: Střední asociace")
                        else:
                            st.markdown("Interpretace: Silná asociace")
                    else:
                        st.markdown(f"**Závěr:** Neexistuje statisticky významná souvislost mezi proměnnými {row_var} a {col_var} (p ≥ 0.05)")
                except Exception as e:
                    st.error(f"Chyba při provádění Chi-kvadrát testu: {str(e)}")
        
        except Exception as e:
            st.error(f"Chyba při vytváření cross tabulky: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

def zobraz_kpi_dashboard():
    st.header("KPI Dashboard")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    # Úvodní informace
    st.markdown("""
    Tento dashboard umožňuje vizualizovat klíčové ukazatele výkonnosti (KPI) z vašich dat. 
    Můžete definovat vlastní metriky a sledovat je v přehledném dashboardu.
    """)
    
    # Nastavení KPI
    st.subheader("Definice KPI")
    
    with st.expander("Přidat nové KPI", expanded=True):
        # Výběr sloupce
        kpi_col = st.selectbox("Vyberte sloupec pro KPI", eda.numeric_cols)
        
        # Výběr agregační funkce
        kpi_agg = st.selectbox(
            "Vyberte agregační funkci",
            ["Suma", "Průměr", "Medián", "Min", "Max", "Počet"]
        )
        
        # Název KPI
        kpi_name = st.text_input("Název KPI", value=f"{kpi_agg} {kpi_col}")
        
        # Barva KPI
        kpi_color = st.selectbox(
            "Barva KPI",
            ["Fialová", "Žlutá", "Oranžová", "Zelená"]
        )
        
        # Mapování barvy na HTML kód
        color_map = {
            "Fialová": OICT_COLORS['purple'],
            "Žlutá": OICT_COLORS['yellow'],
            "Oranžová": OICT_COLORS['orange'],
            "Zelená": OICT_COLORS['green']
        }
        
        # Cílová hodnota
        target_value = st.number_input("Cílová hodnota (nepovinné)", value=0.0)
        
        # Přidání KPI do session_state
        if st.button("Přidat KPI"):
            if 'kpis' not in st.session_state:
                st.session_state.kpis = []
            
            # Výpočet hodnoty KPI
            if kpi_agg == "Suma":
                kpi_value = data[kpi_col].sum()
            elif kpi_agg == "Průměr":
                kpi_value = data[kpi_col].mean()
            elif kpi_agg == "Medián":
                kpi_value = data[kpi_col].median()
            elif kpi_agg == "Min":
                kpi_value = data[kpi_col].min()
            elif kpi_agg == "Max":
                kpi_value = data[kpi_col].max()
            elif kpi_agg == "Počet":
                kpi_value = data[kpi_col].count()
            
            # Přidání KPI do seznamu
            st.session_state.kpis.append({
                'name': kpi_name,
                'column': kpi_col,
                'aggregation': kpi_agg,
                'value': kpi_value,
                'target': target_value,
                'color': color_map[kpi_color]
            })
            
            st.success(f"KPI '{kpi_name}' úspěšně přidáno")
            st.rerun()
    
    # Zobrazení KPI v dashboardu
    if 'kpis' in st.session_state and len(st.session_state.kpis) > 0:
        st.subheader("KPI Dashboard")
        
        # Rozdělení do řádků po 4 KPI
        kpi_per_row = 4
        kpis = st.session_state.kpis
        
        for i in range(0, len(kpis), kpi_per_row):
            cols = st.columns(min(kpi_per_row, len(kpis) - i))
            
            for j, col in enumerate(cols):
                if i + j < len(kpis):
                    kpi = kpis[i + j]
                    
                    # Formátování hodnoty
                    if isinstance(kpi['value'], (int, np.integer)):
                        value_text = f"{kpi['value']:,}"
                    else:
                        value_text = f"{kpi['value']:.2f}"
                    
                    # Výpočet procentuálního rozdílu od cíle
                    if kpi['target'] != 0:
                        diff_pct = (kpi['value'] - kpi['target']) / abs(kpi['target']) * 100
                        diff_text = f"{diff_pct:+.1f}%"
                    else:
                        diff_text = "N/A"
                    
                    col.markdown(f"""
                    <div style="background-color: white; border-radius: 10px; padding: 15px; text-align: center; border-top: 5px solid {kpi['color']}; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <h3 style="margin-bottom: 5px; color: #333;">{kpi['name']}</h3>
                        <div style="font-size: 24px; font-weight: bold; margin: 10px 0; color: {kpi['color']};">{value_text}</div>
                        <div style="color: #666; font-size: 14px;">Cíl: {kpi['target']}</div>
                        <div style="color: {'green' if kpi['value'] >= kpi['target'] else 'red'}; font-size: 14px;">{diff_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Graf 
        if len(kpis) > 1:
            st.subheader("Srovnání KPI")
            
            # Příprava dat pro graf
            kpi_df = pd.DataFrame([{
                'KPI': kpi['name'],
                'Hodnota': kpi['value'],
                'Cíl': kpi['target'],
                'Rozdíl': kpi['value'] - kpi['target'],
                'Procento cíle': (kpi['value'] / kpi['target'] * 100) if kpi['target'] != 0 else 0
            } for kpi in kpis])
            
            # Sloupcový graf
            fig = px.bar(
                kpi_df,
                x='KPI',
                y='Hodnota',
                title="Hodnoty KPI",
                color='KPI',
                color_discrete_sequence=[kpi['color'] for kpi in kpis]
            )
            
            # Přidání cílových hodnot jako horizontální čáry
            for i, kpi in enumerate(kpis):
                if kpi['target'] != 0:
                    fig.add_shape(
                        type="line",
                        xref="x",
                        yref="y",
                        x0=i - 0.4,
                        x1=i + 0.4,
                        y0=kpi['target'],
                        y1=kpi['target'],
                        line=dict(color="red", width=2, dash="dash")
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Procento plnění cílů
            if any(kpi['target'] != 0 for kpi in kpis):
                fig = px.bar(
                    kpi_df[kpi_df['Cíl'] != 0],
                    x='KPI',
                    y='Procento cíle',
                    title="Procento plnění cílů",
                    color='KPI',
                    color_discrete_sequence=[kpi['color'] for kpi in kpis if kpi['target'] != 0]
                )
                
                # Přidání referenční čáry pro 100%
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(kpis) - 0.5,
                    y0=100,
                    y1=100,
                    line=dict(color="black", width=2, dash="dash")
                )
                
                fig.update_layout(yaxis_ticksuffix='%')
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Odstranění KPI
        if st.button("Vymazat všechny KPI"):
            st.session_state.kpis = []
            st.success("Všechny KPI byly odstraněny")
            st.rerun()
    else:
        st.info("Zatím nejsou definovány žádné KPI. Použijte sekci 'Přidat nové KPI' pro vytvoření dashboardu.")

def zobraz_kohortni_analyza():
    st.header("Kohortní analýza")
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    # Bezpečnostní kontrola
    if data is None or eda is None:
        st.warning("Žádná data nebyla nahrána. Prosím, nejprve nahrajte data.")
        return
    
    if len(eda.datetime_cols) == 0:
        st.warning("Pro kohortní analýzu je potřeba alespoň jeden sloupec s datem.")
        return
    
    # Nastavení kohortní analýzy
    st.subheader("Nastavení kohortní analýzy")
    
    # Výběr datumového sloupce pro kohortní definici
    cohort_date_col = st.selectbox(
        "Vyberte sloupec s datem pro definici kohorty (např. datum registrace)",
        eda.datetime_cols,
        key="cohort_date_col"
    )
    
    # Výběr datumového sloupce pro událost
    event_date_col = st.selectbox(
        "Vyberte sloupec s datem události",
        [col for col in eda.datetime_cols if col != cohort_date_col],
        key="event_date_col"
    )
    
    # Výběr ID sloupce
    id_options = ["Vyberte ID sloupec"] + list(data.columns)
    id_col = st.selectbox("Vyberte sloupec s unikátním ID uživatele/zákazníka", id_options)
    
    if id_col == "Vyberte ID sloupec":
        st.warning("Pro kohortní analýzu je potřeba sloupec s ID uživatele/zákazníka")
        return
    
    # Výběr hodnoty pro kohortní analýzu
    value_options = ["Počet unikátních ID"] + eda.numeric_cols
    value_col = st.selectbox("Vyberte hodnotu pro analýzu", value_options)
    
    # Časová perioda pro skupiny kohort
    cohort_period = st.radio(
        "Časová perioda pro kohortní skupiny",
        ["Denní", "Týdenní", "Měsíční", "Kvartální"],
        horizontal=True,
        index=2  # Výchozí je měsíční
    )
    
    # Časová perioda pro zobrazení retence
    retention_period = st.radio(
        "Časová perioda pro zobrazení retence",
        ["Denní", "Týdenní", "Měsíční", "Kvartální"],
        horizontal=True,
        index=2  # Výchozí je měsíční
    )
    
    # Tlačítko pro spuštění analýzy
    run_cohort = st.button("Spustit kohortní analýzu")
    
    if run_cohort:
        with st.spinner("Probíhá kohortní analýza..."):
            try:
                # Kopie dat a převod datumů
                cohort_data = data[[id_col, cohort_date_col, event_date_col]].copy()
                
                if value_col != "Počet unikátních ID":
                    cohort_data[value_col] = data[value_col]
                
                # Převod na datetime
                cohort_data[cohort_date_col] = pd.to_datetime(cohort_data[cohort_date_col])
                cohort_data[event_date_col] = pd.to_datetime(cohort_data[event_date_col])
                
                # Odstranění chybějících hodnot
                cohort_data = cohort_data.dropna(subset=[id_col, cohort_date_col, event_date_col])
                
                # Mapování period na pandas frekvence
                period_map = {
                    "Denní": "D",
                    "Týdenní": "W",
                    "Měsíční": "M",
                    "Kvartální": "Q"
                }
                
                # Vytvoření kohortní periody pro první událost
                if cohort_period == "Denní":
                    cohort_data['Cohort'] = cohort_data[cohort_date_col].dt.date
                elif cohort_period == "Týdenní":
                    cohort_data['Cohort'] = cohort_data[cohort_date_col].dt.to_period('W').dt.start_time.dt.date
                elif cohort_period == "Měsíční":
                    cohort_data['Cohort'] = cohort_data[cohort_date_col].dt.to_period('M').dt.start_time.dt.date
                elif cohort_period == "Kvartální":
                    cohort_data['Cohort'] = cohort_data[cohort_date_col].dt.to_period('Q').dt.start_time.dt.date
                
                # Výpočet období mezi datem kohorty a datem události
                if retention_period == "Denní":
                    cohort_data['Period'] = ((cohort_data[event_date_col] - cohort_data[cohort_date_col]).dt.days).astype(int)
                elif retention_period == "Týdenní":
                    cohort_data['Period'] = ((cohort_data[event_date_col] - cohort_data[cohort_date_col]).dt.days // 7).astype(int)
                elif retention_period == "Měsíční":
                    cohort_data['Period'] = ((cohort_data[event_date_col].dt.year - cohort_data[cohort_date_col].dt.year) * 12 +
                                           (cohort_data[event_date_col].dt.month - cohort_data[cohort_date_col].dt.month)).astype(int)
                elif retention_period == "Kvartální":
                    cohort_data['Period'] = ((cohort_data[event_date_col].dt.year - cohort_data[cohort_date_col].dt.year) * 4 +
                                           (cohort_data[event_date_col].dt.quarter - cohort_data[cohort_date_col].dt.quarter)).astype(int)
                
                # Filtrace period větších nebo rovných 0 (událost nemůže nastat před prvním datem)
                cohort_data = cohort_data[cohort_data['Period'] >= 0]
                
                # Vytvoření tabulky kohort
                if value_col == "Počet unikátních ID":
                    # Kohortní tabulka bude počítat unikátní ID
                    cohort_counts = cohort_data.groupby(['Cohort', 'Period'])[id_col].nunique().reset_index()
                    cohort_counts.columns = ['Cohort', 'Period', 'Count']
                else:
                    # Kohortní tabulka bude agregovat hodnotu
                    cohort_counts = cohort_data.groupby(['Cohort', 'Period'])[value_col].sum().reset_index()
                    cohort_counts.columns = ['Cohort', 'Period', 'Count']
                
                # Vytvoření pivotní tabulky
                cohort_pivot = cohort_counts.pivot_table(index='Cohort', columns='Period', values='Count')
                
                # Výpočet kohortních velikostí (počet v prvním období)
                cohort_sizes = cohort_pivot[0].copy()
                
                # Výpočet retence jako procenta původní kohorty
                retention_pivot = cohort_pivot.divide(cohort_sizes, axis=0) * 100
                
                # Zobrazení výsledků
                st.subheader("Výsledky kohortní analýzy")
                
                # Heatmapa retence
                fig = px.imshow(
                    retention_pivot,
                    labels=dict(x="Období", y="Kohorta", color="Retence (%)"),
                    color_continuous_scale='blues',
                    title=f"Retence podle kohorty (v %)"
                )
                
                # Přidání hodnot do buněk
                for i in range(len(retention_pivot.index)):
                    for j in range(len(retention_pivot.columns)):
                        if not pd.isna(retention_pivot.iloc[i, j]):
                            fig.add_annotation(
                                x=j,
                                y=i,
                                text=f"{retention_pivot.iloc[i, j]:.1f}%",
                                showarrow=False,
                                font=dict(color="white" if retention_pivot.iloc[i, j] > 50 else "black")
                            )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Graf průměrné retence podle období
                avg_retention = retention_pivot.mean(axis=0)
                
                fig = px.line(
                    x=avg_retention.index,
                    y=avg_retention.values,
                    markers=True,
                    labels={'x': 'Období', 'y': 'Průměrná retence (%)'},
                    title="Průměrná retence podle období"
                )
                
                fig.update_layout(yaxis_ticksuffix='%')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabulka s velikostmi kohort
                st.subheader("Velikosti kohort")
                
                cohort_sizes_df = pd.DataFrame({
                    'Kohorta': cohort_sizes.index,
                    'Počet': cohort_sizes.values
                })
                
                fig = px.bar(
                    cohort_sizes_df,
                    x='Kohorta',
                    y='Počet',
                    title="Velikosti kohort",
                    color_discrete_sequence=[OICT_COLORS['purple']]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabulky s daty
                st.subheader("Tabulka retence (%)")
                st.dataframe(retention_pivot.style.format("{:.1f}%"))
                
                st.subheader("Tabulka absolutních hodnot")
                st.dataframe(cohort_pivot.style.format("{:.0f}"))
                
                # Možnost stáhnout data
                csv = retention_pivot.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Stáhnout tabulku retence jako CSV",
                    data=csv,
                    file_name=f'cohort_retention.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"Chyba při kohortní analýze: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

def vytvor_html_report(data, eda):
    """
    Vytvoří HTML report s výsledky analýzy
    """
    import datetime
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EDA Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 30px;
                line-height: 1.6;
                color: #333;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #574494;
                padding-bottom: 10px;
            }}
            .logo {{
                background-color: #574494;
                color: white;
                padding: 5px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 24px;
                display: inline-block;
                margin-bottom: 10px;
            }}
            h1, h2, h3 {{
                color: #574494;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .section {{
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
            .insights {{
                background-color: #f8f8f8;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .footer {{
                margin-top: 50px;
                text-align: center;
                color: #666;
                font-size: 14px;
                border-top: 1px solid #eee;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">oict.</div>
            <h1>Report průzkumné analýzy dat</h1>
            <p>Vygenerováno: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
        
        <div class="section">
            <h2>Přehled datasetu</h2>
            <p>Tento dataset obsahuje {data.shape[0]} řádků a {data.shape[1]} sloupců.</p>
            <p>Typy sloupců:</p>
            <ul>
                <li>Numerické sloupce: {len(eda.numeric_cols)}</li>
                <li>Kategorické sloupce: {len(eda.categorical_cols)}</li>
                <li>Datum/čas sloupce: {len(eda.datetime_cols)}</li>
            </ul>
            
            <h3>Ukázka dat</h3>
            {data.head().to_html()}
        </div>
        
        <div class="section">
            <h2>Klíčové poznatky</h2>
            <div class="insights">
                <ul>
    """
    
    # Přidání poznatků
    for insight in eda.generate_insights():
        html += f"<li>{insight}</li>\n"
    
    html += """
                </ul>
            </div>
        </div>
    """
    
    # Sekce chybějících hodnot
    if hasattr(eda, 'missing_analysis'):
        missing_cols = eda.missing_analysis[eda.missing_analysis['Missing Values'] > 0]
        html += f"""
        <div class="section">
            <h2>Analýza chybějících hodnot</h2>
            <p>Celkové chybějící hodnoty: {eda.data.isnull().sum().sum()} z {eda.rows * eda.cols} buněk ({(eda.data.isnull().sum().sum() / (eda.rows * eda.cols) * 100):.2f}%)</p>
        """
        
        if len(missing_cols) > 0:
            html += f"""
            <h3>Sloupce s chybějícími hodnotami ({len(missing_cols)})</h3>
            {missing_cols.to_html()}
            """
        else:
            html += "<p>Nebyly zjištěny žádné chybějící hodnoty.</p>"
        
        html += "</div>"
    
    # Sekce korelací
    if hasattr(eda, 'correlation_matrix'):
        html += f"""
        <div class="section">
            <h2>Analýza korelací</h2>
        """
        
        if hasattr(eda, 'high_correlations') and len(eda.high_correlations) > 0:
            html += f"""
            <h3>Vysoké korelace (|r| ≥ 0.7)</h3>
            {eda.high_correlations.to_html()}
            """
        else:
            html += "<p>Nebyly zjištěny žádné vysoké korelace.</p>"
        
        html += "</div>"
    
    # Sekce odlehlých hodnot
    if hasattr(eda, 'outlier_summary') and len(eda.outlier_summary) > 0:
        html += f"""
        <div class="section">
            <h2>Analýza odlehlých hodnot</h2>
            <h3>Sloupce s odlehlými hodnotami</h3>
            <table>
                <tr>
                    <th>Sloupec</th>
                    <th>Počet odlehlých hodnot</th>
                    <th>Procento</th>
                </tr>
        """
        
        for col, info in eda.outlier_summary.items():
            html += f"""
            <tr>
                <td>{col}</td>
                <td>{info['count']}</td>
                <td>{info['percent']:.2f}%</td>
            </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    # Patička
    html += """
        <div class="footer">
            <p>Powered by OICT</p>
            <p>© 2023 Vytvořeno s ❤️</p>
        </div>
    </body>
    </html>
    """
    
    return html

if __name__ == "__main__":
    main()