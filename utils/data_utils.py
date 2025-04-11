"""
Data utility functions for EDA app
"""
import pandas as pd
import numpy as np
from scipy import stats

def detect_column_types(df):
    """
    Detect column types (numeric, categorical, datetime)
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        tuple: (numeric_cols, categorical_cols, datetime_cols)
    """
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    
    for col in df.columns:
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 10:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        
        # Check if datetime
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        
        # Try to convert to datetime
        else:
            try:
                # Check if it might be a date column
                if any(word in col.lower() for word in ['date', 'time', 'day', 'month', 'year']):
                    pd.to_datetime(df[col].iloc[:5])  # Try first few rows
                    datetime_cols.append(col)
                else:
                    # If not a date, check if categorical
                    if df[col].nunique() <= 20:
                        categorical_cols.append(col)
                    else:
                        # High cardinality text column
                        categorical_cols.append(col)
            except:
                # If conversion fails, check if it might be categorical
                if df[col].nunique() <= 20:
                    categorical_cols.append(col)
                else:
                    # High cardinality text column
                    categorical_cols.append(col)
    
    return numeric_cols, categorical_cols, datetime_cols

def get_basic_stats(df, numeric_cols=None):
    """
    Get basic statistics for numeric columns
    
    Args:
        df: DataFrame to analyze
        numeric_cols: List of numeric columns (if None, all numeric columns are used)
        
    Returns:
        DataFrame with statistics
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 0:
        return df[numeric_cols].describe()
    
    return pd.DataFrame()

def get_missing_stats(df):
    """
    Get missing value statistics
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with missing value counts and percentages
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percent Missing': missing_percent
    }).sort_values('Percent Missing', ascending=False)
    
    return missing_df[missing_df['Missing Values'] > 0]

def calculate_outliers(series, method='iqr', threshold=1.5):
    """
    Detect outliers in a series
    
    Args:
        series: Series to analyze
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        tuple: (is_outlier, lower_bound, upper_bound)
    """
    clean_series = series.dropna()
    
    if method == 'iqr':
        # IQR method
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        is_outlier = (clean_series < lower_bound) | (clean_series > upper_bound)
    else:
        # Z-Score method
        z_scores = np.abs((clean_series - clean_series.mean()) / clean_series.std())
        is_outlier = z_scores > threshold
        
        lower_bound = clean_series.mean() - threshold * clean_series.std()
        upper_bound = clean_series.mean() + threshold * clean_series.std()
    
    return is_outlier, lower_bound, upper_bound

def calculate_correlations(df, method='pearson', threshold=None):
    """
    Calculate correlations between numeric columns
    
    Args:
        df: DataFrame to analyze
        method: 'pearson', 'spearman', or 'kendall'
        threshold: Optional threshold for filtering correlations
        
    Returns:
        tuple: (correlation_matrix, high_correlations_df)
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate correlation matrix
    correlation_matrix = df[numeric_cols].corr(method=method)
    
    # Find high correlations
    high_corr = []
    threshold = threshold or 0.7
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) >= threshold:
                high_corr.append({
                    'Feature 1': correlation_matrix.columns[i],
                    'Feature 2': correlation_matrix.columns[j],
                    'Correlation': correlation_matrix.iloc[i, j]
                })
    
    high_correlations = pd.DataFrame(high_corr)
    
    return correlation_matrix, high_correlations

def impute_missing_values(df, strategy='simple'):
    """
    Impute missing values in a DataFrame
    
    Args:
        df: DataFrame to impute
        strategy: 'simple', 'knn', or 'iterative'
        
    Returns:
        DataFrame with imputed values
    """
    imputed_df = df.copy()
    
    if strategy == 'simple':
        # Simple imputation - mean for numeric, mode for categorical
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                imputed_df[col] = df[col].fillna(df[col].mean())
            else:
                # For categorical, use mode
                mode_value = df[col].mode()[0] if not df[col].mode().empty else None
                imputed_df[col] = df[col].fillna(mode_value)
    
    elif strategy == 'knn':
        try:
            from sklearn.impute import KNNImputer
            
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                # Apply KNN imputation
                imputer = KNNImputer(n_neighbors=5)
                imputed_values = imputer.fit_transform(df[numeric_cols])
                
                # Update DataFrame
                imputed_df[numeric_cols] = imputed_values
                
                # For non-numeric columns, use simple imputation
                for col in df.columns:
                    if col not in numeric_cols:
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else None
                        imputed_df[col] = df[col].fillna(mode_value)
        except ImportError:
            # Fall back to simple imputation if sklearn not available
            return impute_missing_values(df, strategy='simple')
    
    elif strategy == 'iterative':
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                # Apply iterative imputation
                imputer = IterativeImputer(max_iter=10, random_state=0)
                imputed_values = imputer.fit_transform(df[numeric_cols])
                
                # Update DataFrame
                imputed_df[numeric_cols] = imputed_values
                
                # For non-numeric columns, use simple imputation
                for col in df.columns:
                    if col not in numeric_cols:
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else None
                        imputed_df[col] = df[col].fillna(mode_value)
        except ImportError:
            # Fall back to simple imputation if sklearn not available
            return impute_missing_values(df, strategy='simple')
    
    return imputed_df

def transform_column(series, method='log'):
    """
    Apply transformation to a series
    
    Args:
        series: Series to transform
        method: 'log', 'sqrt', 'boxcox', 'yeo-johnson'
        
    Returns:
        tuple: (transformed_series, params)
    """
    # Make a copy of the series
    clean_series = series.dropna()
    
    if method == 'log':
        # Handle negative or zero values
        min_val = clean_series.min()
        shift = 0 if min_val > 0 else abs(min_val) + 1
        
        transformed = np.log(clean_series + shift)
        params = {'shift': shift}
    
    elif method == 'sqrt':
        # Handle negative values
        min_val = clean_series.min()
        shift = 0 if min_val >= 0 else abs(min_val) + 0.01
        
        transformed = np.sqrt(clean_series + shift)
        params = {'shift': shift}
    
    elif method == 'boxcox':
        # Box-Cox requires positive values
        min_val = clean_series.min()
        shift = 0 if min_val > 0 else abs(min_val) + 0.01
        
        transformed, lmbda = stats.boxcox(clean_series + shift)
        params = {'shift': shift, 'lambda': lmbda}
    
    elif method == 'yeo-johnson':
        transformed, lmbda = stats.yeojohnson(clean_series)
        params = {'lambda': lmbda}
    
    else:
        raise ValueError(f"Unknown transformation method: {method}")
    
    # Create a new series with the same index
    transformed_series = pd.Series(transformed, index=clean_series.index)
    
    return transformed_series, params