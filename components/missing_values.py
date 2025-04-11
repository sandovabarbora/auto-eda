import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def missing_values_component():
    """Component for analyzing missing values"""
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first")
        return
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    st.header("Missing Values Analysis")
    
    # Overall missing values summary
    total_missing = data.isnull().sum().sum()
    total_cells = data.shape[0] * data.shape[1]
    missing_percent = (total_missing / total_cells) * 100
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3>Overall Missing Values: {missing_percent:.2f}%</h3>
        <p>{total_missing:,} missing cells out of {total_cells:,} total cells</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get missing value analysis
    missing_analysis = eda.analyze_missing_values()
    missing_cols = missing_analysis[missing_analysis['Missing Values'] > 0]
    
    if len(missing_cols) > 0:
        st.subheader(f"Columns with Missing Values ({len(missing_cols)})")
        
        # Bar chart of missing values
        missing_df = missing_cols.reset_index()
        missing_df.columns = ['Column', 'Missing Values', 'Percent Missing']
        
        fig = px.bar(
            missing_df.sort_values('Percent Missing', ascending=False),
            x='Column',
            y='Percent Missing',
            title='Missing Values by Column',
            color='Percent Missing',
            color_continuous_scale='reds',
            text='Percent Missing'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        fig.update_layout(
            xaxis={'categoryorder':'total descending'},
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="",
            yaxis_title="Percent Missing (%)",
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table of missing values
        st.dataframe(missing_df.set_index('Column').style.format({
            'Missing Values': '{:,}',
            'Percent Missing': '{:.2f}%'
        }), use_container_width=True)
        
        # Missing value patterns
        st.subheader("Missing Value Patterns")
        
        # For reasonable-sized datasets, show missing patterns visualization
        if data.shape[0] <= 5000 and data.shape[1] <= 50:
            # Get columns with highest missing percentages (up to 10)
            top_missing_cols = missing_df.sort_values(
                'Percent Missing', ascending=False
            ).head(10)['Column'].tolist()
            
            # Create a binary missing data mask (True if missing)
            missing_mask = data[top_missing_cols].isnull()
            
            # Count occurrences of each missing pattern
            pattern_counts = missing_mask.value_counts().reset_index()
            pattern_counts.columns = top_missing_cols + ['Count']
            
            # Convert to long format for visualization (only top patterns)
            pattern_viz = pattern_counts.head(10).melt(
                id_vars=['Count'], 
                value_vars=top_missing_cols,
                var_name='Column', 
                value_name='Is Missing'
            )
            
            fig = px.imshow(
                pattern_counts.head(10)[top_missing_cols],
                labels=dict(x="Column", y="Pattern", color="Is Missing"),
                y=[f"Pattern {i+1} (n={row['Count']})" for i, row in pattern_counts.head(10).iterrows()],
                color_continuous_scale=['#cfedfb', '#ff5757'],
                aspect="auto"
            )
            
            fig.update_layout(
                title="Top Missing Value Patterns",
                xaxis_title="",
                yaxis_title="",
                yaxis_nticks=len(pattern_counts.head(10))
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"This chart shows the top {min(10, len(pattern_counts))} missing value patterns. Each row represents a pattern, and each column represents a feature. Red cells indicate missing values.")
        else:
            st.info("Dataset too large for pattern visualization. Analysis limited to summary statistics.")
        
        # Recommendations for handling missing values
        st.subheader("Recommendations")
        
        high_missing = missing_df[missing_df['Percent Missing'] > 50]
        medium_missing = missing_df[(missing_df['Percent Missing'] <= 50) & (missing_df['Percent Missing'] > 10)]
        low_missing = missing_df[missing_df['Percent Missing'] <= 10]
        
        if len(high_missing) > 0:
            st.markdown("#### Columns with >50% missing values")
            st.markdown("Consider dropping these columns as they contain mostly missing data:")
            for col in high_missing['Column']:
                st.markdown(f"- **{col}** ({missing_analysis.loc[col, 'Percent Missing']:.1f}%)")
        
        if len(medium_missing) > 0:
            st.markdown("#### Columns with 10-50% missing values")
            st.markdown("Consider carefully whether to impute or drop these columns:")
            for col in medium_missing['Column']:
                st.markdown(f"- **{col}** ({missing_analysis.loc[col, 'Percent Missing']:.1f}%)")
        
        if len(low_missing) > 0:
            st.markdown("#### Columns with <10% missing values")
            st.markdown("These columns can likely be imputed safely:")
            for col in low_missing['Column']:
                st.markdown(f"- **{col}** ({missing_analysis.loc[col, 'Percent Missing']:.1f}%)")
        
        # Imputation suggestions
        st.subheader("Imputation Strategies")
        
        st.markdown("Here are some recommended imputation strategies based on your data:")
        
        imputation_code = """
# Python code for imputation
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# For numeric columns
numeric_imputer = SimpleImputer(strategy='median')  # or 'mean'
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# For categorical columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# For advanced imputation, consider KNN imputation
# knn_imputer = KNNImputer(n_neighbors=5)
# df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
"""
        st.code(imputation_code, language="python")
        
        # R code version
        r_imputation_code = """
# R code for imputation
library(tidyverse)
library(mice)

# For basic imputation
df <- df %>%
  mutate_if(is.numeric, ~replace_na(., median(., na.rm = TRUE))) %>%
  mutate_if(is.character, ~replace_na(., as.character(names(which.max(table(.))))))

# For more advanced imputation using mice
# imputed_data <- mice(df, m=5, method='pmm')
# complete_data <- complete(imputed_data)
"""
        st.code(r_imputation_code, language="r")
        
    else:
        st.success("No missing values found in the dataset!")
        st.balloons()