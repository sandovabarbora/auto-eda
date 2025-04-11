import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def outliers_component():
    """Component for analyzing outliers in the data"""
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("Please load data first")
        return
    
    data = st.session_state.data
    eda = st.session_state.eda
    
    st.header("Outlier Analysis")
    
    if len(eda.numeric_cols) == 0:
        st.warning("No numeric columns found in the dataset.")
        return
    
    # Outlier detection settings
    st.subheader("Outlier Detection Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        detection_method = st.selectbox(
            "Detection Method",
            ["IQR (Interquartile Range)", "Z-Score"],
            help="IQR method is robust to non-normal distributions. Z-Score assumes normal distribution."
        )
    
    with col2:
        if detection_method == "IQR (Interquartile Range)":
            threshold = st.slider(
                "IQR Factor",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Multiplier for IQR. Standard is 1.5. Higher values detect fewer outliers."
            )
        else:  # Z-Score
            threshold = st.slider(
                "Z-Score Threshold",
                min_value=2.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                help="Number of standard deviations. Standard is 3. Lower values detect more outliers."
            )
    
    # Select column for analysis
    selected_column = st.selectbox("Select column for outlier analysis", eda.numeric_cols)
    
    # Get clean data (no NaN values)
    clean_data = data[selected_column].dropna()
    
    # Detect outliers
    if detection_method == "IQR (Interquartile Range)":
        # IQR method
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        is_outlier = (clean_data < lower_bound) | (clean_data > upper_bound)
        outliers = clean_data[is_outlier]
        
        method_name = f"IQR (factor: {threshold})"
    else:
        # Z-Score method
        z_scores = np.abs((clean_data - clean_data.mean()) / clean_data.std())
        is_outlier = z_scores > threshold
        outliers = clean_data[is_outlier]
        
        # Calculate bounds for visualization
        lower_bound = clean_data.mean() - threshold * clean_data.std()
        upper_bound = clean_data.mean() + threshold * clean_data.std()
        
        method_name = f"Z-Score (threshold: {threshold})"
    
    # Display outlier summary
    outlier_count = len(outliers)
    outlier_percent = (outlier_count / len(clean_data)) * 100
    
    st.subheader("Outlier Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Outlier Count", outlier_count)
    
    with col2:
        st.metric("Outlier Percentage", f"{outlier_percent:.2f}%")
    
    with col3:
        st.metric("Detection Method", method_name)
    
    # Display outlier bounds
    st.write(f"**Lower bound:** {lower_bound:.4f}")
    st.write(f"**Upper bound:** {upper_bound:.4f}")
    
    # Visualization
    st.subheader("Outlier Visualization")
    
    viz_type = st.radio(
        "Visualization Type",
        ["Box Plot", "Histogram", "Scatter Plot"],
        horizontal=True
    )
    
    if viz_type == "Box Plot":
        # Create box plot
        fig = px.box(
            clean_data,
            title=f"Box Plot of {selected_column} with Outliers",
            points="outliers"
        )
        
        # Add outlier points in a different color
        if outlier_count > 0:
            fig.add_trace(
                go.Scatter(
                    x=[0] * outlier_count,
                    y=outliers,
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='circle-open',
                        line=dict(width=2)
                    ),
                    name='Outliers'
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Histogram":
        # Create histogram with highlighted outliers
        fig = px.histogram(
            clean_data,
            title=f"Distribution of {selected_column} with Outliers Highlighted",
            opacity=0.7,
            color=is_outlier,
            color_discrete_map={True: 'red', False: 'blue'},
            labels={
                'color': 'Is Outlier',
                selected_column: selected_column
            }
        )
        
        # Add vertical lines for bounds
        fig.add_vline(x=lower_bound, line_dash="dash", line_color="black",
                     annotation_text="Lower Bound", annotation_position="top right")
        fig.add_vline(x=upper_bound, line_dash="dash", line_color="black",
                     annotation_text="Upper Bound", annotation_position="top left")
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Scatter Plot
        # Create a scatter plot (useful for seeing the distribution)
        # Generate indices for x-axis
        indices = np.arange(len(clean_data))
        
        # Create DataFrame for the plot
        plot_df = pd.DataFrame({
            'Index': indices,
            selected_column: clean_data.values,
            'Is Outlier': is_outlier
        })
        
        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x='Index',
            y=selected_column,
            color='Is Outlier',
            color_discrete_map={True: 'red', False: 'blue'},
            title=f"Scatter Plot of {selected_column} with Outliers Highlighted"
        )
        
        # Add horizontal lines for bounds
        fig.add_hline(y=lower_bound, line_dash="dash", line_color="black",
                     annotation_text="Lower Bound", annotation_position="left")
        fig.add_hline(y=upper_bound, line_dash="dash", line_color="black",
                     annotation_text="Upper Bound", annotation_position="left")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show actual outlier values
    if outlier_count > 0:
        with st.expander("View Outlier Values", expanded=False):
            # Create DataFrame with outlier values and their indices
            outlier_indices = np.where(is_outlier)[0]
            outlier_df = pd.DataFrame({
                'Index': outlier_indices,
                selected_column: outliers.values,
                'Z-Score': (outliers - clean_data.mean()) / clean_data.std()
            })
            
            # Sort by absolute Z-score
            outlier_df['Abs Z-Score'] = abs(outlier_df['Z-Score'])
            outlier_df = outlier_df.sort_values('Abs Z-Score', ascending=False).drop(columns=['Abs Z-Score'])
            
            st.dataframe(outlier_df.style.format({
                'Z-Score': '{:.2f}'
            }), use_container_width=True)
    
    # Outlier handling recommendations
    st.subheader("Recommendations for Handling Outliers")
    
    if outlier_percent > 10:
        st.warning("""
        **High percentage of outliers detected!**
        
        This could indicate one of the following:
        - The data naturally has a wide range or non-normal distribution
        - The threshold may be too strict for this dataset
        - The column may need transformation (e.g., log transformation)
        """)
    
    # Common outlier handling methods
    st.markdown("""
    ### Common Methods for Handling Outliers:
    
    1. **Investigation**: Examine outliers to determine if they're valid data points or errors
    2. **Winsorization**: Cap outliers at a specified percentile (e.g., 5th and 95th)
    3. **Removal**: Delete outlier rows (only if you're certain they're errors)
    4. **Transformation**: Apply log, square root, or Box-Cox transformation to reduce outlier impact
    5. **Robust Methods**: Use statistical methods that are less sensitive to outliers
    """)
    
    # Code examples for handling outliers
    with st.expander("Code Examples", expanded=False):
        st.code("""
# Python code for handling outliers

# 1. Winsorization (capping)
def cap_outliers(df, column, lower_percentile=0.05, upper_percentile=0.95):
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    df[f"{column}_capped"] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# 2. IQR method - removing outliers
def remove_outliers_iqr(df, column, factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 3. Z-score method - removing outliers
def remove_outliers_zscore(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[abs(z_scores) <= threshold]

# 4. Log transformation
import numpy as np
def log_transform(df, column):
    # Handle zero or negative values if present
    min_val = df[column].min()
    offset = 0 if min_val > 0 else abs(min_val) + 1
    df[f"{column}_log"] = np.log(df[column] + offset)
    return df
""", language="python")

        # R code example
        st.code("""
# R code for handling outliers

# 1. Winsorization (capping)
cap_outliers <- function(data, column, lower_percentile=0.05, upper_percentile=0.95) {
  lower_bound <- quantile(data[[column]], lower_percentile, na.rm = TRUE)
  upper_bound <- quantile(data[[column]], upper_percentile, na.rm = TRUE)
  data[[paste0(column, "_capped")]] <- pmin(pmax(data[[column]], lower_bound), upper_bound)
  return(data)
}

# 2. IQR method - removing outliers
remove_outliers_iqr <- function(data, column, factor=1.5) {
  Q1 <- quantile(data[[column]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[column]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - factor * IQR
  upper_bound <- Q3 + factor * IQR
  return(data[data[[column]] >= lower_bound & data[[column]] <= upper_bound, ])
}

# 3. Z-score method - removing outliers
remove_outliers_zscore <- function(data, column, threshold=3) {
  z_scores <- abs((data[[column]] - mean(data[[column]], na.rm = TRUE)) / 
                  sd(data[[column]], na.rm = TRUE))
  return(data[z_scores <= threshold, ])
}

# 4. Log transformation
log_transform <- function(data, column) {
  # Handle zero or negative values if present
  min_val <- min(data[[column]], na.rm = TRUE)
  offset <- if(min_val > 0) 0 else abs(min_val) + 1
  data[[paste0(column, "_log")]] <- log(data[[column]] + offset)
  return(data)
}
""", language="r")
    
    # Compare outlier detection methods
    st.subheader("Compare Outlier Detection Methods")
    
    # Only show this if we have enough data
    if len(clean_data) >= 30:
        # Calculate outliers using different methods
        # IQR with different factors
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        
        outliers_iqr_1 = ((clean_data < Q1 - 1.0 * IQR) | (clean_data > Q3 + 1.0 * IQR)).sum()
        outliers_iqr_15 = ((clean_data < Q1 - 1.5 * IQR) | (clean_data > Q3 + 1.5 * IQR)).sum()
        outliers_iqr_2 = ((clean_data < Q1 - 2.0 * IQR) | (clean_data > Q3 + 2.0 * IQR)).sum()
        outliers_iqr_3 = ((clean_data < Q1 - 3.0 * IQR) | (clean_data > Q3 + 3.0 * IQR)).sum()
        
        # Z-Score with different thresholds
        z_scores = np.abs((clean_data - clean_data.mean()) / clean_data.std())
        outliers_z_2 = (z_scores > 2.0).sum()
        outliers_z_25 = (z_scores > 2.5).sum()
        outliers_z_3 = (z_scores > 3.0).sum()
        outliers_z_4 = (z_scores > 4.0).sum()
        
        # Create comparison data
        methods = [
            "IQR (factor=1.0)", "IQR (factor=1.5)", "IQR (factor=2.0)", "IQR (factor=3.0)",
            "Z-Score (threshold=2.0)", "Z-Score (threshold=2.5)", "Z-Score (threshold=3.0)", "Z-Score (threshold=4.0)"
        ]
        
        outlier_counts = [
            outliers_iqr_1, outliers_iqr_15, outliers_iqr_2, outliers_iqr_3,
            outliers_z_2, outliers_z_25, outliers_z_3, outliers_z_4
        ]
        
        outlier_percentages = [count / len(clean_data) * 100 for count in outlier_counts]
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Method': methods,
            'Outlier Count': outlier_counts,
            'Percentage': outlier_percentages
        })
        
        # Plot comparison
        fig = px.bar(
            comparison_df,
            x='Method',
            y='Percentage',
            title="Comparison of Outlier Detection Methods",
            text='Outlier Count',
            color='Percentage',
            color_continuous_scale='blues'
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(yaxis_title="Percentage of Outliers (%)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Multivariate outlier detection
    if len(eda.numeric_cols) >= 2:
        st.subheader("Multivariate Outlier Detection")
        
        st.markdown("""
        Univariate outlier detection looks at each variable separately, but multivariate outliers might 
        be normal in each individual dimension while being unusual in combination.
        """)
        
        # Let user select two columns for 2D visualization
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis variable", 
                               [col for col in eda.numeric_cols if col != selected_column], 
                               key="outlier_x")
        
        with col2:
            y_col = st.selectbox("Y-axis variable", 
                               [selected_column] + [col for col in eda.numeric_cols if col != selected_column and col != x_col],
                               index=0,
                               key="outlier_y")
        
        # Get data for the two selected columns
        bivariate_data = data[[x_col, y_col]].dropna()
        
        if len(bivariate_data) > 10:
            # Calculate Mahalanobis distance for multivariate outlier detection
            try:
                from scipy.stats import chi2
                
                # Calculate mean and covariance
                X = bivariate_data.values
                mean = np.mean(X, axis=0)
                cov = np.cov(X, rowvar=False)
                
                # Calculate Mahalanobis distance
                inv_cov = np.linalg.inv(cov)
                mahalanobis_dist = np.sqrt(np.sum(np.dot(X - mean, inv_cov) * (X - mean), axis=1))
                
                # Determine threshold (chi-square with 2 degrees of freedom)
                threshold = np.sqrt(chi2.ppf(0.975, df=2))
                
                # Identify multivariate outliers
                is_mv_outlier = mahalanobis_dist > threshold
                
                # Create plot with outliers
                bivariate_data['Mahalanobis'] = mahalanobis_dist
                bivariate_data['Is Outlier'] = is_mv_outlier
                
                fig = px.scatter(
                    bivariate_data,
                    x=x_col,
                    y=y_col,
                    color='Is Outlier',
                    color_discrete_map={True: 'red', False: 'blue'},
                    title=f"Multivariate Outliers in {x_col} vs {y_col}",
                    hover_data=['Mahalanobis']
                )
                
                # Add confidence ellipse
                from matplotlib.patches import Ellipse
                import matplotlib.pyplot as plt
                
                def get_corr_ellipse(mean, cov, confidence=0.975, **kwargs):
                    """Get correlation ellipse coordinates"""
                    # Get the eigenvalues and eigenvectors
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    
                    # Get the indices of the largest and smallest eigenvalues
                    idx = eigenvals.argsort()[::-1]
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
                    
                    # Calculate the angle
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    
                    # Width and height are "full" widths, not radius
                    width, height = 2 * np.sqrt(chi2.ppf(confidence, df=2)) * np.sqrt(eigenvals)
                    
                    # Get the ellipse
                    ellipse = Ellipse(
                        xy=mean, width=width, height=height,
                        angle=angle, **kwargs
                    )
                    
                    # Get the coordinates
                    theta = np.linspace(0, 2*np.pi, 100)
                    ellipse_x = mean[0] + width/2 * np.cos(theta)
                    ellipse_y = mean[1] + height/2 * np.sin(theta)
                    
                    # Rotate the coordinates
                    angle_rad = np.radians(angle)
                    rot_matrix = np.array([
                        [np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]
                    ])
                    
                    rotated_coords = np.dot(
                        np.column_stack([ellipse_x - mean[0], ellipse_y - mean[1]]),
                        rot_matrix.T
                    )
                    
                    # Shift coordinates back
                    ellipse_x = rotated_coords[:, 0] + mean[0]
                    ellipse_y = rotated_coords[:, 1] + mean[1]
                    
                    return ellipse_x, ellipse_y
                
                # Get ellipse coordinates
                mean = bivariate_data[[x_col, y_col]].mean().values
                cov = bivariate_data[[x_col, y_col]].cov().values
                ellipse_x, ellipse_y = get_corr_ellipse(mean, cov)
                
                # Add ellipse trace
                fig.add_trace(
                    go.Scatter(
                        x=ellipse_x,
                        y=ellipse_y,
                        mode='lines',
                        line=dict(color='black', dash='dash'),
                        name='97.5% Confidence Ellipse'
                    )
                )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary
                mv_outlier_count = is_mv_outlier.sum()
                mv_outlier_percent = mv_outlier_count / len(bivariate_data) * 100
                
                st.write(f"**Multivariate outliers detected:** {mv_outlier_count} ({mv_outlier_percent:.2f}%)")
                st.write(f"**Method:** Mahalanobis distance with Chi-square threshold at 97.5% confidence")
                
            except Exception as e:
                st.error(f"Error calculating multivariate outliers: {str(e)}")
        else:
            st.warning("Not enough data points for multivariate outlier detection")