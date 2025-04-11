import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def distributions_component():
    """Component for visualizing distributions"""
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first")
        return
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    st.header("Data Distributions")
    
    # Create tabs for numeric vs categorical
    tab1, tab2 = st.tabs(["Numeric Columns", "Categorical Columns"])
    
    with tab1:
        if len(eda.numeric_cols) > 0:
            # Column selection with metrics
            selected_num_col = st.selectbox("Select numeric column", eda.numeric_cols)
            
            # Get clean data (no NaN values)
            clean_data = data[selected_num_col].dropna()
            
            # Show metrics for the selected column
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{clean_data.mean():.2f}")
            
            with col2:
                st.metric("Median", f"{clean_data.median():.2f}")
            
            with col3:
                st.metric("Std Dev", f"{clean_data.std():.2f}")
            
            with col4:
                iqr = clean_data.quantile(0.75) - clean_data.quantile(0.25)
                st.metric("IQR", f"{iqr:.2f}")
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Min", f"{clean_data.min():.2f}")
            
            with col2:
                st.metric("Max", f"{clean_data.max():.2f}")
            
            with col3:
                st.metric("Skewness", f"{clean_data.skew():.2f}")
            
            with col4:
                st.metric("Kurtosis", f"{clean_data.kurtosis():.2f}")
            
            # Visualization options
            st.subheader("Visualization")
            
            viz_type = st.radio(
                "Plot type",
                ["Histogram", "Box Plot", "Violin Plot", "QQ Plot"],
                horizontal=True
            )
            
            if viz_type == "Histogram":
                # Histogram settings
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    bins = st.slider("Number of bins", 5, 100, 30)
                
                with col2:
                    kde = st.checkbox("Show KDE", value=True)
                
                with col3:
                    show_rug = st.checkbox("Show rug", value=False)
                
                # Create histogram
                fig = px.histogram(
                    data,
                    x=selected_num_col,
                    nbins=bins,
                    title=f"Distribution of {selected_num_col}",
                    histnorm="probability density" if kde else None,
                    marginal="rug" if show_rug else None,
                )
                
                if kde:
                    # Add KDE plot
                    kde_x = np.linspace(clean_data.min(), clean_data.max(), 1000)
                    kde_y = stats.gaussian_kde(clean_data)(kde_x)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=kde_x, 
                            y=kde_y, 
                            mode='lines', 
                            name='KDE',
                            line=dict(color='red', width=2)
                        )
                    )
                
                # Add mean and median lines
                fig.add_vline(x=clean_data.mean(), line_dash="dash", line_color="red", 
                             annotation_text="Mean", annotation_position="top right")
                fig.add_vline(x=clean_data.median(), line_dash="dash", line_color="green", 
                             annotation_text="Median", annotation_position="top left")
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Box Plot":
                # Box plot settings
                show_points = st.checkbox("Show all points", value=False)
                
                # Create box plot
                fig = px.box(
                    data,
                    y=selected_num_col,
                    title=f"Box Plot of {selected_num_col}",
                    points="all" if show_points else "outliers"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Violin Plot":
                # Create violin plot
                fig = px.violin(
                    data,
                    y=selected_num_col,
                    title=f"Violin Plot of {selected_num_col}",
                    box=True,
                    points="all"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # QQ Plot
                # Create QQ plot
                quantiles = np.linspace(0.01, 0.99, 100)
                theoretical_quantiles = stats.norm.ppf(quantiles)
                empirical_quantiles = np.nanquantile(clean_data, quantiles)
                
                fig = px.scatter(
                    x=theoretical_quantiles,
                    y=empirical_quantiles,
                    title=f"Q-Q Plot of {selected_num_col} vs Normal Distribution",
                    labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'}
                )
                
                # Add reference line
                max_val = max(abs(fig.data[0].x.min()), abs(fig.data[0].x.max()))
                fig.add_trace(
                    go.Scatter(
                        x=[-max_val, max_val],
                        y=[-max_val, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Reference Line'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Normality test
                if len(clean_data) > 3:  # Need at least 3 points for Shapiro-Wilk
                    shapiro_test = stats.shapiro(clean_data)
                    
                    st.subheader("Normality Test (Shapiro-Wilk)")
                    st.write(f"Test statistic: {shapiro_test[0]:.4f}")
                    st.write(f"p-value: {shapiro_test[1]:.4f}")
                    
                    if shapiro_test[1] < 0.05:
                        st.write("**Conclusion**: Distribution is likely not normal (p < 0.05)")
                    else:
                        st.write("**Conclusion**: Cannot reject normality (p ≥ 0.05)")
            
            # Distribution interpretation
            st.subheader("Distribution Interpretation")
            
            skewness = clean_data.skew()
            kurtosis = clean_data.kurtosis()
            
            st.write("**Skewness interpretation:**")
            if abs(skewness) < 0.5:
                st.write("- Distribution is approximately symmetric")
            elif skewness < 0:
                st.write("- Distribution is left-skewed (negatively skewed)")
                st.write("- The left tail is longer; there are more extreme values on the left")
            else:
                st.write("- Distribution is right-skewed (positively skewed)")
                st.write("- The right tail is longer; there are more extreme values on the right")
            
            st.write("**Kurtosis interpretation:**")
            if abs(kurtosis) < 0.5:
                st.write("- Distribution has kurtosis similar to normal distribution (mesokurtic)")
            elif kurtosis < 0:
                st.write("- Distribution has lighter tails and flatter peak than normal (platykurtic)")
                st.write("- Data is more uniformly distributed with fewer outliers")
            else:
                st.write("- Distribution has heavier tails and sharper peak than normal (leptokurtic)")
                st.write("- Data has more outliers than a normal distribution would")
            
            # Optional transformations
            if abs(skewness) > 1:
                st.subheader("Suggested Transformations")
                
                transform_options = ["None", "Log", "Square Root", "Box-Cox"]
                selected_transform = st.selectbox("Apply transformation", transform_options)
                
                if selected_transform != "None":
                    if selected_transform == "Log":
                        # Check if we need to shift for non-positive values
                        min_val = clean_data.min()
                        shift = abs(min_val) + 1 if min_val <= 0 else 0
                        transformed = np.log(clean_data + shift)
                        transform_name = f"Log(x + {shift})" if shift > 0 else "Log(x)"
                    
                    elif selected_transform == "Square Root":
                        # Check if we need to shift for negative values
                        min_val = clean_data.min()
                        shift = abs(min_val) + 0.01 if min_val < 0 else 0
                        transformed = np.sqrt(clean_data + shift)
                        transform_name = f"√(x + {shift})" if shift > 0 else "√x"
                    
                    elif selected_transform == "Box-Cox":
                        # Box-Cox requires positive values
                        min_val = clean_data.min()
                        shift = abs(min_val) + 0.01 if min_val <= 0 else 0
                        transformed, lmbda = stats.boxcox(clean_data + shift)
                        transform_name = f"Box-Cox (λ={lmbda:.4f})"
                    
                    # Show transformed distribution
                    fig = px.histogram(
                        transformed,
                        title=f"{transform_name} Transform of {selected_num_col}",
                        histnorm="probability density",
                        labels={"value": transform_name}
                    )
                    
                    # Add KDE to transformed data
                    kde_x = np.linspace(transformed.min(), transformed.max(), 1000)
                    kde_y = stats.gaussian_kde(transformed)(kde_x)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=kde_x, 
                            y=kde_y, 
                            mode='lines', 
                            name='KDE',
                            line=dict(color='red', width=2)
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show transformation stats
                    st.write(f"Skewness after transformation: {stats.skew(transformed):.4f}")
                    st.write(f"Kurtosis after transformation: {stats.kurtosis(transformed):.4f}")
                    
                    # Code for transformation
                    if selected_transform == "Log":
                        code = f"""
# Python code for Log transformation
{'shift = ' + str(shift) if shift > 0 else '# No shift needed for positive data'}
{selected_num_col}_log = np.log({selected_num_col}{' + shift' if shift > 0 else ''})
"""
                    elif selected_transform == "Square Root":
                        code = f"""
# Python code for Square Root transformation
{'shift = ' + str(shift) if shift > 0 else '# No shift needed for non-negative data'}
{selected_num_col}_sqrt = np.sqrt({selected_num_col}{' + shift' if shift > 0 else ''})
"""
                    elif selected_transform == "Box-Cox":
                        code = f"""
# Python code for Box-Cox transformation
from scipy import stats
{'shift = ' + str(shift) if shift > 0 else '# No shift needed for positive data'}
{selected_num_col}_boxcox, lmbda = stats.boxcox({selected_num_col}{' + shift' if shift > 0 else ''})
# Lambda value: {lmbda:.4f}
"""
                    st.code(code, language="python")
        else:
            st.info("No numeric columns found in the dataset")
    
    with tab2:
        if len(eda.categorical_cols) > 0:
            # Column selection
            selected_cat_col = st.selectbox("Select categorical column", eda.categorical_cols)
            
            # Get value counts
            value_counts = data[selected_cat_col].value_counts(normalize=True, dropna=False)
            value_counts_df = value_counts.reset_index()
            value_counts_df.columns = [selected_cat_col, 'Percentage']
            value_counts_df['Percentage'] = value_counts_df['Percentage'] * 100
            value_counts_df['Count'] = data[selected_cat_col].value_counts(dropna=False).values
            
            # Handle large number of categories
            max_categories = 20
            if len(value_counts_df) > max_categories:
                st.warning(f"Column has {len(value_counts_df)} categories. Showing top {max_categories} for visualization.")
                # Keep top categories and group others
                top_categories = value_counts_df.head(max_categories-1)
                other_categories = value_counts_df.iloc[max_categories-1:]
                
                other_row = pd.DataFrame({
                    selected_cat_col: ['Other'],
                    'Percentage': [other_categories['Percentage'].sum()],
                    'Count': [other_categories['Count'].sum()]
                })
                
                value_counts_df = pd.concat([top_categories, other_row])
            
            # Visualization options
            viz_type = st.radio(
                "Plot type",
                ["Bar Chart", "Pie Chart", "Treemap"],
                horizontal=True,
                key="cat_viz_type"
            )
            
            if viz_type == "Bar Chart":
                # Create bar chart
                fig = px.bar(
                    value_counts_df,
                    x=selected_cat_col,
                    y='Percentage',
                    title=f"Distribution of {selected_cat_col}",
                    text='Percentage',
                    color='Percentage',
                    color_continuous_scale='blues'
                )
                
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(yaxis_title="Percentage (%)")
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Pie Chart":
                # Create pie chart
                fig = px.pie(
                    value_counts_df,
                    names=selected_cat_col,
                    values='Percentage',
                    title=f"Distribution of {selected_cat_col}",
                    hole=0.4,
                    hover_data=['Count']
                )
                
                fig.update_traces(textinfo='label+percent parent')  # Shows percentage relative to parent
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:  # Treemap
                # Create treemap
                fig = px.treemap(
                    value_counts_df,
                    path=[selected_cat_col],
                    values='Percentage',
                    title=f"Distribution of {selected_cat_col}",
                    hover_data=['Count'],
                    color='Percentage',
                    color_continuous_scale='blues'
                )
                
                fig.update_traces(textinfo='label+percent parent')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display table of value counts
            st.subheader("Category Frequencies")
            
            # Show full table without the 'Other' grouping
            full_counts = data[selected_cat_col].value_counts(dropna=False)
            full_counts_df = full_counts.reset_index()
            full_counts_df.columns = [selected_cat_col, 'Count']
            full_counts_df['Percentage'] = 100 * full_counts / full_counts.sum()
            
            st.dataframe(
                full_counts_df.style.format({'Percentage': '{:.2f}%'}),
                use_container_width=True
            )
            
            # Show category statistics
            st.subheader("Category Statistics")
            
            # Number of unique values
            unique_count = data[selected_cat_col].nunique()
            
            # Most common category
            most_common = data[selected_cat_col].value_counts().index[0]
            most_common_pct = data[selected_cat_col].value_counts(normalize=True).iloc[0] * 100
            
            # Missing values
            missing_count = data[selected_cat_col].isna().sum()
            missing_pct = missing_count / len(data) * 100
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Unique Categories", unique_count)
            
            with col2:
                st.metric("Most Common", str(most_common))
            
            with col3:
                st.metric("Most Common %", f"{most_common_pct:.2f}%")
            
            with col4:
                st.metric("Missing Values", f"{missing_pct:.2f}%")
            
            # Entropy (measure of uniformity)
            if unique_count > 1:
                from scipy.stats import entropy
                category_probabilities = data[selected_cat_col].value_counts(normalize=True)
                category_entropy = entropy(category_probabilities)
                max_entropy = entropy([1/unique_count] * unique_count)
                normalized_entropy = category_entropy / max_entropy if max_entropy > 0 else 0
                
                st.subheader("Entropy Analysis")
                st.write(f"Entropy: {category_entropy:.4f}")
                st.write(f"Maximum possible entropy: {max_entropy:.4f}")
                st.write(f"Normalized entropy: {normalized_entropy:.4f}")
                
                st.write("**Interpretation:**")
                if normalized_entropy < 0.25:
                    st.write("- Distribution is highly imbalanced (low entropy)")
                    st.write("- One or a few categories dominate the distribution")
                elif normalized_entropy < 0.5:
                    st.write("- Distribution is moderately imbalanced")
                elif normalized_entropy < 0.75:
                    st.write("- Distribution is relatively balanced")
                else:
                    st.write("- Distribution is very uniform (high entropy)")
                    st.write("- All categories have similar frequencies")
        else:
            st.info("No categorical columns found in the dataset")