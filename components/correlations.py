import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def correlations_component():
    """Component for analyzing correlations between variables"""
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first")
        return
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    st.header("Correlation Analysis")
    
    if len(eda.numeric_cols) < 2:
        st.warning("At least two numeric columns are required for correlation analysis.")
        return
    
    # Correlation settings
    st.subheader("Correlation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        corr_method = st.selectbox(
            "Correlation Method",
            ["pearson", "spearman", "kendall"],
            help="Pearson measures linear relationships. Spearman and Kendall measure monotonic relationships."
        )
    
    with col2:
        threshold = st.slider(
            "Correlation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Only show correlations with absolute value above this threshold"
        )
    
    # Column selection
    show_all_cols = st.checkbox("Show all numeric columns", value=True)
    
    if not show_all_cols:
        selected_columns = st.multiselect(
            "Select columns for correlation analysis",
            eda.numeric_cols,
            default=eda.numeric_cols[:min(5, len(eda.numeric_cols))]
        )
        
        if not selected_columns:
            st.warning("Please select at least two columns")
            return
        
        # Calculate correlation for selected columns
        correlation_matrix, high_correlations = eda.analyze_correlations(method=corr_method)
        correlation_matrix = correlation_matrix.loc[selected_columns, selected_columns]
    else:
        # Calculate correlation for all numeric columns
        correlation_matrix, high_correlations = eda.analyze_correlations(method=corr_method)
    
    # Apply threshold filtering if needed
    if threshold > 0:
        # Create a mask for the threshold
        mask = np.abs(correlation_matrix) > threshold
        filtered_corr = correlation_matrix.where(mask, 0)
    else:
        filtered_corr = correlation_matrix
    
    # Correlation Heatmap tab
    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Pairwise Scatter Plots", "High Correlations"])
    
    with tab1:
        st.subheader("Correlation Heatmap")
        
        # Create correlation heatmap
        fig = px.imshow(
            filtered_corr,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"{corr_method.capitalize()} Correlation Matrix"
        )
        
        fig.update_layout(
            height=600,
            width=800,
            xaxis_title="",
            yaxis_title=""
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guide
        with st.expander("Correlation Interpretation Guide", expanded=False):
            st.markdown("""
            ### Interpreting Correlation Values
            
            Correlation measures the strength and direction of the relationship between two variables.
            
            - **Values close to 1**: Strong positive correlation. As one variable increases, the other tends to increase.
            - **Values close to -1**: Strong negative correlation. As one variable increases, the other tends to decrease.
            - **Values close to 0**: Little to no correlation. The variables don't appear to be related.
            
            ### Correlation Strength Guide
            
            | Correlation Coefficient | Interpretation |
            | --- | --- |
            | 0.00 to 0.19 | Very weak correlation |
            | 0.20 to 0.39 | Weak correlation |
            | 0.40 to 0.59 | Moderate correlation |
            | 0.60 to 0.79 | Strong correlation |
            | 0.80 to 1.00 | Very strong correlation |
            
            ### Correlation ≠ Causation
            
            Remember that correlation does not imply causation. Two variables can be correlated without one causing the other.
            """)
    
    with tab2:
        st.subheader("Pairwise Scatter Plots")
        
        # Let user select columns for detailed examination
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis variable", eda.numeric_cols, key="x_axis")
        
        with col2:
            y_col = st.selectbox(
                "Y-axis variable", 
                [col for col in eda.numeric_cols if col != x_col],
                key="y_axis"
            )
        
        # Allow for coloring by a third variable
        color_by = st.selectbox(
            "Color by",
            ["None"] + eda.categorical_cols + eda.numeric_cols,
            index=0
        )
        
        # Create scatter plot
        if color_by == "None":
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                trendline="ols",
                title=f"Scatter Plot: {x_col} vs {y_col}",
                opacity=0.7
            )
        else:
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                color=color_by,
                trendline="ols",
                title=f"Scatter Plot: {x_col} vs {y_col}, colored by {color_by}",
                opacity=0.7
            )
        
        # Update layout
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation and display statistics
        valid_data = data[[x_col, y_col]].dropna()
        
        # Calculate correlation based on method
        if corr_method == "pearson":
            corr_coef, p_value = stats.pearsonr(valid_data[x_col], valid_data[y_col])
            method_name = "Pearson"
        elif corr_method == "spearman":
            corr_coef, p_value = stats.spearmanr(valid_data[x_col], valid_data[y_col])
            method_name = "Spearman"
        else:  # kendall
            corr_coef, p_value = stats.kendalltau(valid_data[x_col], valid_data[y_col])
            method_name = "Kendall's Tau"
        
        # Display statistics
        st.subheader("Correlation Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(f"{method_name} Correlation", f"{corr_coef:.4f}")
        
        with col2:
            st.metric("p-value", f"{p_value:.4f}")
        
        # Interpretation
        st.subheader("Interpretation")
        
        # Determine correlation strength
        corr_strength = abs(corr_coef)
        if corr_strength < 0.2:
            strength = "very weak"
        elif corr_strength < 0.4:
            strength = "weak"
        elif corr_strength < 0.6:
            strength = "moderate"
        elif corr_strength < 0.8:
            strength = "strong"
        else:
            strength = "very strong"
        
        # Determine direction
        direction = "positive" if corr_coef > 0 else "negative"
        
        # Statistical significance
        if p_value < 0.05:
            significance = "statistically significant (p < 0.05)"
        else:
            significance = "not statistically significant (p ≥ 0.05)"
        
        st.write(f"There is a **{strength} {direction}** correlation between **{x_col}** and **{y_col}** ({corr_coef:.4f}).")
        st.write(f"This correlation is {significance}.")
        
        if p_value < 0.05:
            if corr_coef > 0:
                st.write(f"As **{x_col}** increases, **{y_col}** tends to increase.")
            else:
                st.write(f"As **{x_col}** increases, **{y_col}** tends to decrease.")
        else:
            st.write(f"There is insufficient evidence to conclude that **{x_col}** and **{y_col}** are correlated.")
    
    with tab3:
        st.subheader("High Correlations")
        
        # Filter high correlations
        if len(high_correlations) > 0:
            # Sort by absolute correlation value
            high_correlations['Abs Correlation'] = abs(high_correlations['Correlation'])
            high_correlations = high_correlations.sort_values('Abs Correlation', ascending=False)
            
            # Create bar chart for high correlations
            fig = px.bar(
                high_correlations,
                x='Correlation',
                y=high_correlations.apply(lambda x: f"{x['Feature 1']} ↔ {x['Feature 2']}", axis=1),
                orientation='h',
                title='Feature Pairs with High Correlation (|r| ≥ 0.7)',
                color='Correlation',
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1]
            )
            
            fig.update_layout(
                yaxis_title="",
                xaxis_title="Correlation Coefficient",
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table of high correlations
            st.dataframe(
                high_correlations[['Feature 1', 'Feature 2', 'Correlation']].style.format({
                    'Correlation': '{:.4f}'
                }).background_gradient(cmap='RdBu_r', subset=['Correlation'], vmin=-1, vmax=1),
                use_container_width=True
            )
            
            # Provide recommendations
            st.subheader("Recommendations")
            
            # Check for multicollinearity
            very_high_corr = high_correlations[abs(high_correlations['Correlation']) > 0.9]
            if len(very_high_corr) > 0:
                st.warning("⚠️ Potential multicollinearity detected!")
                st.markdown("""
                **Multicollinearity** exists when two or more predictor variables are highly correlated. 
                This can cause problems when building predictive models, as it can lead to:
                
                - Unstable coefficient estimates
                - Reduced statistical power
                - Difficulty in interpreting which variable is affecting the target
                
                **Recommendation**: Consider removing one variable from each highly correlated pair 
                (|r| > 0.9) before building machine learning models.
                """)
                
                st.markdown("#### Variables to consider removing:")
                for _, row in very_high_corr.iterrows():
                    st.markdown(f"- Either **{row['Feature 1']}** or **{row['Feature 2']}** (r = {row['Correlation']:.4f})")
            
            # Code for handling correlations
            st.subheader("Code Snippet")
            
            code = """
# Python code to handle correlated features
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Find highly correlated features
def find_correlated_features(data, threshold=0.9):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

# Remove highly correlated features
high_corr_features = find_correlated_features(X, threshold=0.9)
X_reduced = X.drop(columns=high_corr_features)
"""
            st.code(code, language="python")
            
        else:
            st.info("No high correlations (|r| ≥ 0.7) found in the dataset.")