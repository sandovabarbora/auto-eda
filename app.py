import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from eda import EDA
from components.data_loader import load_data_component
from components.api_loader import api_loader_component
from components.overview import overview_component
from components.missing_values import missing_values_component
from components.distributions import distributions_component
from components.correlations import correlations_component
from components.outliers import outliers_component
from utils.ui_components import OICT_COLORS

def apply_custom_style():
    """Apply custom styling to the app using OICT color palette"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary-color: #574494;  /* OICT Purple */
            --secondary-color: #E37222; /* OICT Orange */
            --accent-color: #FFE14F; /* OICT Yellow */
            --success-color: #74ECA1; /* OICT Green */
            --background-color: #f8f9fa;
            --card-color: #ffffff;
            --text-color: #333333;
            --text-light: #666666;
            --border-radius: 10px;
            --box-shadow: 0 4px 12px rgba(87, 68, 148, 0.1);
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
        
        /* Headings */
        h1, h2, h3 {
            color: var(--primary-color);
            font-weight: 600;
        }
        
        h1 {
            font-size: 2.5rem;
            letter-spacing: -0.02em;
        }
        
        h2 {
            font-size: 1.8rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f0f0f5;
        }
        
        h3 {
            font-size: 1.3rem;
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
        }
        
        /* Tabs styling */
        .stTabs {
            background-color: var(--card-color);
            padding: 1rem;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            margin-bottom: 1.5rem;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f5f5fa;
            padding: 5px;
            border-radius: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            transition: var(--transition);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color) !important;
            color: white !important;
            border: none !important;
        }
        
        /* Metric cards */
        .metric-container {
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            padding: 1.2rem;
            box-shadow: var(--box-shadow);
            transition: var(--transition);
            border-top: 3px solid var(--primary-color);
            height: 100%;
        }
        
        .metric-container:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 15px rgba(87, 68, 148, 0.15);
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 0.3rem;
        }
        
        /* Button styling */
        button[kind="primary"] {
            background-color: var(--primary-color) !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            padding: 0.5rem 1rem !important;
            border: none !important;
            transition: var(--transition) !important;
            box-shadow: 0 2px 5px rgba(87, 68, 148, 0.2) !important;
        }
        
        button[kind="primary"]:hover {
            background-color: #483A7A !important;
            box-shadow: 0 4px 8px rgba(87, 68, 148, 0.3) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Secondary button */
        button[kind="secondary"] {
            background-color: white !important;
            color: var(--primary-color) !important;
            border: 1px solid var(--primary-color) !important;
        }
        
        button[kind="secondary"]:hover {
            background-color: #f5f5fa !important;
        }
        
        /* For backward compatibility with older Streamlit versions */
        .stButton > button {
            background-color: var(--primary-color) !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            padding: 0.5rem 1rem !important;
            border: none !important;
        }
        
        /* Card styling for content */
        .card {
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--box-shadow);
            margin-bottom: 1.5rem;
            transition: var(--transition);
            border-left: 4px solid var(--primary-color);
        }
        
        .card:hover {
            box-shadow: 0 6px 15px rgba(87, 68, 148, 0.15);
        }
        
        .card.warning {
            border-left-color: var(--secondary-color);
        }
        
        .card.success {
            border-left-color: var(--success-color);
        }
        
        .card.info {
            border-left-color: var(--accent-color);
        }
        
        /* Dataframe styling */
        .dataframe-container {
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            overflow: hidden;
            margin-bottom: 1.5rem;
        }
        
        /* Code blocks */
        pre {
            background-color: #2A2139 !important;
            border-radius: 8px !important;
            border-left: 3px solid var(--primary-color) !important;
            padding: 1rem !important;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: var(--text-light);
            font-size: 0.9rem;
            border-top: 1px solid #eaeaea;
            margin-top: 3rem;
        }
        
        .logo-text {
            font-weight: 800;
            color: var(--primary-color);
            letter-spacing: -0.05em;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-good {
            background-color: var(--success-color);
        }
        
        .status-warning {
            background-color: var(--accent-color);
        }
        
        .status-bad {
            background-color: var(--secondary-color);
        }
        
        /* Insights card */
        .insight-card {
            background-color: rgba(87, 68, 148, 0.05);
            border-left: 4px solid var(--primary-color);
            border-radius: 0 8px 8px 0;
            padding: 1rem;
            margin-bottom: 1rem;
            transition: var(--transition);
        }
        
        .insight-card:hover {
            background-color: rgba(87, 68, 148, 0.08);
            transform: translateX(3px);
        }
        
        /* Custom navigation styles */
        .nav-item {
            display: flex;
            align-items: center;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .nav-item:hover {
            background-color: rgba(87, 68, 148, 0.05);
        }
        
        .nav-item.active {
            background-color: rgba(87, 68, 148, 0.1);
            border-left: 3px solid #574494;
        }
        
        .nav-icon {
            font-size: 1.2rem;
            margin-right: 10px;
            color: #574494;
        }
        
        .nav-text {
            display: flex;
            flex-direction: column;
        }
        
        .nav-title {
            font-weight: 500;
            color: #333;
        }
        
        .nav-desc {
            font-size: 0.8rem;
            color: #666;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(page_title="OICT Exploratory Data Analysis Tool", layout="wide")
    apply_custom_style()
    
    # Modern header with OICT branding
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
        <h1 style="margin-top: 0.5rem; margin-bottom: 0.5rem; font-size: 2.8rem; font-weight: 800;">Exploratory Data Analysis</h1>
        <p style="color: #666; font-size: 1.2rem; max-width: 800px; margin: 0 auto;">
            A powerful tool to help you explore, visualize, and understand your data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "data" not in st.session_state:
        st.session_state.data = None
    if "eda" not in st.session_state:
        st.session_state.eda = None
    
    # Create enhanced sidebar for navigation - SIMPLIFIED VERSION
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem; border-bottom: 1px solid #eaeaea;">
            <h3 style="margin-top: 0.8rem; margin-bottom: 0.2rem;">Data Explorer</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple navigation options with icons - compatible with all Streamlit versions
        st.markdown("<p style='font-weight: 600; font-size: 0.9rem; color: #666; margin-bottom: 0.5rem; padding-left: 0.5rem;'>NAVIGATION</p>", unsafe_allow_html=True)
        
        # Define the navigation items with icons and descriptions
        nav_items = [
            {"icon": "📥", "name": "Data Loading", "desc": "Import your dataset"},
            {"icon": "📊", "name": "Overview", "desc": "Data summary & statistics"}, 
            {"icon": "❓", "name": "Missing Values", "desc": "Find & analyze gaps"},
            {"icon": "📈", "name": "Distributions", "desc": "Explore data patterns"},
            {"icon": "🔄", "name": "Correlations", "desc": "Discover relationships"},
            {"icon": "⚠️", "name": "Outliers", "desc": "Identify unusual points"}
        ]
        
        # Use radio buttons for navigation (more compatible than custom buttons)
        selected_page = st.radio(
            "Select a section",
            [item["name"] for item in nav_items],
            label_visibility="collapsed",
            format_func=lambda x: f"{nav_items[[item['name'] for item in nav_items].index(x)]['icon']} {x}"
        )
        
        # Add info about the selected page
        st.markdown(f"""
        <div style="margin: 1rem 0.5rem; padding: 1rem; background-color: rgba(87, 68, 148, 0.05); border-radius: 8px;">
            <h4 style="color: #574494; margin-bottom: 0.5rem;">{selected_page}</h4>
            <p style="color: #666; font-size: 0.9rem;">
                {next((item["desc"] for item in nav_items if item["name"] == selected_page), "")}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add some space
        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        
        # Add info section
        with st.expander("ℹ️ About this tool"):
            st.markdown("""
            This EDA tool helps you analyze your data quickly and effectively. It provides:
            
            - Basic data exploration
            - Missing value analysis
            - Distribution visualization
            - Correlation analysis
            - Outlier detection
            
            Made with ❤️ using Python, Streamlit, and Plotly.
            """)
            
        # Add OICT branding at the bottom
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; color: #666; font-size: 0.8rem; margin-top: 2rem;">
            <div style="font-size: 0.7rem;">Powered by Streamlit</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display the appropriate page based on navigation
    if selected_page == "Data Loading":
        load_data_component()
    elif st.session_state.data is not None and st.session_state.eda is not None:
        if selected_page == "Overview":
            overview_component()
        elif selected_page == "Missing Values":
            missing_values_component()
        elif selected_page == "Distributions":
            distributions_component()
        elif selected_page == "Correlations":
            correlations_component()
        elif selected_page == "Outliers":
            outliers_component()
    else:
        st.warning("Please load data first")
        load_data_component()
    
    # Enhanced footer with OICT branding
    st.markdown("""
    <div class="footer">
        <div style="margin-bottom: 1rem;">
        </div>
        <p>Exploratory Data Analysis Tool</p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">
            Made with ❤️ using Python, Streamlit, and Plotly
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()