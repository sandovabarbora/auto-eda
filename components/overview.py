import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from utils.ui_components import create_metrics_row, create_section_header, create_insight_cards, create_card, create_tabs, OICT_COLORS

def overview_component():
    """Component for showing data overview"""
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first")
        return
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    create_section_header("Data Overview", 
                        "Get a quick summary of your dataset's structure and content",
                        "📊")
    
    # Key metrics in cards with enhanced styling
    missing_percent = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
    duplicate_count, _ = eda.get_duplicate_rows()
    duplicate_percent = (duplicate_count / data.shape[0]) * 100 if data.shape[0] > 0 else 0
    
    metrics_data = [
        {
            "title": "Rows",
            "value": f"{data.shape[0]:,}",
            "icon": "🧮",
            "color": OICT_COLORS['purple']
        },
        {
            "title": "Columns",
            "value": f"{data.shape[1]}",
            "icon": "📋",
            "color": OICT_COLORS['orange']
        },
        {
            "title": "Missing Values",
            "value": f"{missing_percent:.2f}%",
            "icon": "❓",
            "color": OICT_COLORS['yellow']
        },
        {
            "title": "Duplicate Rows",
            "value": f"{duplicate_percent:.2f}%",
            "icon": "🔄",
            "color": OICT_COLORS['green']
        }
    ]
    
    create_metrics_row(metrics_data)
    
    # Column types breakdown with enhanced styling
    create_section_header("Column Types Distribution", "Breakdown of your data's column types", "📊")
    
    column_types = {
        'Numeric': len(eda.numeric_cols),
        'Categorical': len(eda.categorical_cols),
        'DateTime': len(eda.datetime_cols)
    }
    
    # Create a card to hold the chart
    st.markdown("""
    <div style="
        background-color: white; 
        border-radius: 12px; 
        padding: 1.5rem; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
        margin-bottom: 2rem;
    ">
    """, unsafe_allow_html=True)
    
    # Column 1: Bar chart
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig = px.bar(
            x=list(column_types.keys()),
            y=list(column_types.values()),
            title="Column Types Distribution",
            labels={'x': 'Column Type', 'y': 'Count'},
            color=list(column_types.keys()),
            color_discrete_sequence=[OICT_COLORS['purple'], OICT_COLORS['orange'], OICT_COLORS['green']],
            text=list(column_types.values())
        )
        
        fig.update_traces(
            textposition='outside',
            textfont=dict(size=14),
            marker=dict(line=dict(width=1, color='white'))
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=18, color='#333'),
            title_x=0.5,
            xaxis=dict(
                showgrid=False,
                title_font=dict(size=14),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.07)',
                title_font=dict(size=14),
                tickfont=dict(size=12)
            ),
            height=350,
            margin=dict(t=50, b=50, l=50, r=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Column 2: Pie chart
    with col2:
        # Calculate percentages for pie chart
        total_cols = sum(column_types.values())
        percentages = {k: v/total_cols*100 for k, v in column_types.items()}
        
        # Prepare data for pie chart
        pie_data = pd.DataFrame({
            'Type': list(column_types.keys()),
            'Count': list(column_types.values()),
            'Percentage': [percentages['Numeric'], percentages['Categorical'], percentages['DateTime']]
        })
        
        fig = px.pie(
            pie_data,
            values='Count',
            names='Type',
            title="Column Types by Percentage",
            hole=0.5,
            color='Type',
            color_discrete_map={
                'Numeric': OICT_COLORS['purple'],
                'Categorical': OICT_COLORS['orange'],
                'DateTime': OICT_COLORS['green']
            }
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='white', width=2))
        )
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=18, color='#333'),
            title_x=0.5,
            height=350,
            margin=dict(t=50, b=50, l=30, r=30)
        )
        
        # Add center text
        fig.add_annotation(
            text=f"{total_cols}<br>columns",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Close the card div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create enhanced tabs for different views
    selected_tab = create_tabs(["Column Details", "Data Preview", "Statistics"], "overview_tabs")
    
    # Card to contain the tab content
    st.markdown("""
    <div style="
        background-color: white; 
        border-radius: 12px; 
        padding: 1.5rem; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
        margin-bottom: 2rem;
    ">
    """, unsafe_allow_html=True)
    
    if selected_tab == "Column Details":
        # Enhanced column details with better formatting
        st.markdown("<h3 style='color: #574494; margin-bottom: 1rem; font-size: 1.3rem;'>Column Details</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div style='background-color: rgba(87, 68, 148, 0.05); padding: 1rem; border-radius: 10px; height: 100%;'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color: #574494; margin-bottom: 0.8rem;'>Numeric Columns ({len(eda.numeric_cols)})</h4>", unsafe_allow_html=True)
            
            if eda.numeric_cols:
                for col in eda.numeric_cols:
                    st.markdown(f"<div style='margin-bottom: 0.5rem; padding: 0.3rem 0.6rem; background-color: white; border-radius: 5px; border-left: 3px solid #574494;'>{col}</div>", unsafe_allow_html=True)
            else:
                st.info("No numeric columns found")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div style='background-color: rgba(116, 236, 161, 0.05); padding: 1rem; border-radius: 10px; margin-top: 1rem; height: 100%;'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color: #574494; margin-bottom: 0.8rem;'>DateTime Columns ({len(eda.datetime_cols)})</h4>", unsafe_allow_html=True)
            
            if eda.datetime_cols:
                for col in eda.datetime_cols:
                    st.markdown(f"<div style='margin-bottom: 0.5rem; padding: 0.3rem 0.6rem; background-color: white; border-radius: 5px; border-left: 3px solid #74ECA1;'>{col}</div>", unsafe_allow_html=True)
            else:
                st.info("No datetime columns found")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div style='background-color: rgba(231, 114, 34, 0.05); padding: 1rem; border-radius: 10px; height: 100%;'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color: #574494; margin-bottom: 0.8rem;'>Categorical Columns ({len(eda.categorical_cols)})</h4>", unsafe_allow_html=True)
            
            if eda.categorical_cols:
                for col in eda.categorical_cols:
                    st.markdown(f"<div style='margin-bottom: 0.5rem; padding: 0.3rem 0.6rem; background-color: white; border-radius: 5px; border-left: 3px solid #E37222;'>{col}</div>", unsafe_allow_html=True)
            else:
                st.info("No categorical columns found")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif selected_tab == "Data Preview":
        # Enhanced data preview with styling
        st.markdown("<h3 style='color: #574494; margin-bottom: 1rem; font-size: 1.3rem;'>Data Preview</h3>", unsafe_allow_html=True)
        
        # Create a styled container for the dataframe
        st.markdown("<div style='background-color: #f8f9fa; padding: 0.5rem; border-radius: 8px;'>", unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add shape information
        st.markdown(f"""
        <div style="margin: 1rem 0; display: flex; gap: 2rem; font-size: 0.9rem;">
            <div>
                <span style="font-weight: 600; color: #574494;">Rows:</span> {data.shape[0]:,}
            </div>
            <div>
                <span style="font-weight: 600; color: #574494;">Columns:</span> {data.shape[1]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Data info in an expander
        with st.expander("Detailed Data Information", expanded=False):
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.code(buffer.getvalue(), language="text")
    
    else:  # Statistics tab
        st.markdown("<h3 style='color: #574494; margin-bottom: 1rem; font-size: 1.3rem;'>Basic Statistics</h3>", unsafe_allow_html=True)
        
        if len(eda.numeric_cols) > 0:
            # Create a styled container for the statistics
            st.markdown("<div style='background-color: #f8f9fa; padding: 0.5rem; border-radius: 8px;'>", unsafe_allow_html=True)
            stats_df = eda.get_basic_stats()
            st.dataframe(stats_df.style.format(precision=2), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a description of the statistics
            st.markdown("""
            <div style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
                <ul>
                    <li><strong>count</strong>: Number of non-null values</li>
                    <li><strong>mean</strong>: Average value</li>
                    <li><strong>std</strong>: Standard deviation (measure of dispersion)</li>
                    <li><strong>min/max</strong>: Minimum and maximum values</li>
                    <li><strong>25%, 50%, 75%</strong>: Quartiles (50% is the median)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            create_card(
                "No numeric columns found in this dataset. Statistics are only available for numeric data.",
                title="No Numeric Data",
                is_info=True
            )
    
    # Close the card div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Key insights with enhanced styling
    create_section_header("Key Insights", "Automatically generated observations about your data", "💡")
    
    insights = eda.generate_insights()
    create_insight_cards(insights)
    
    # Memory usage with enhanced styling
    create_section_header("Memory Usage", "How much memory your dataset is using", "💾")
    
    # Create a nice card for memory usage
    memory_usage = data.memory_usage(deep=True).sum()
    
    if memory_usage < 1024:
        memory_str = f"{memory_usage} bytes"
    elif memory_usage < 1024**2:
        memory_str = f"{memory_usage/1024:.2f} KB"
    elif memory_usage < 1024**3:
        memory_str = f"{memory_usage/1024**2:.2f} MB"
    else:
        memory_str = f"{memory_usage/1024**3:.2f} GB"
    
    # Memory usage card
    st.markdown(f"""
    <div style="
        background-color: white; 
        border-radius: 12px; 
        padding: 1.5rem; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
        margin-bottom: 2rem;
        text-align: center;
    ">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Total Memory Usage</div>
        <div style="font-size: 2.5rem; font-weight: 700; color: #574494; margin-bottom: 1rem;">{memory_str}</div>
        <div style="width: 100%; height: 10px; background-color: #f0f0f5; border-radius: 5px; overflow: hidden;">
            <div style="width: 100%; height: 100%; background-color: #574494; border-radius: 5px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Per-column memory usage with enhanced styling
    st.markdown("""
    <div style="
        background-color: white; 
        border-radius: 12px; 
        padding: 1.5rem; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
        margin-bottom: 2rem;
    ">
        <h3 style="color: #574494; margin-bottom: 1rem; font-size: 1.3rem;">Column Memory Distribution</h3>
    """, unsafe_allow_html=True)
    
    col_memory = data.memory_usage(deep=True)
    col_memory_df = pd.DataFrame({
        'Column': list(data.columns) + ['Index'],
        'Memory (MB)': [m/1024**2 for m in col_memory]
    })
    
    fig = px.bar(
        col_memory_df.sort_values('Memory (MB)', ascending=False).head(10),
        x='Column',
        y='Memory (MB)',
        title="Top 10 Columns by Memory Usage",
        color='Memory (MB)',
        color_continuous_scale=[
            [0, "rgba(87, 68, 148, 0.2)"], 
            [0.5, "rgba(87, 68, 148, 0.6)"], 
            [1, "rgba(87, 68, 148, 1)"]
        ],
        text='Memory (MB)'
    )
    
    fig.update_traces(
        texttemplate='%{text:.2f} MB',
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=16, color='#333'),
        title_x=0.5,
        xaxis_title="",
        yaxis_title="Memory Usage (MB)",
        height=400,
        margin=dict(t=50, b=50, l=50, r=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Close the card div
    st.markdown("</div>", unsafe_allow_html=True)