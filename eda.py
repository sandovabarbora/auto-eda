import pandas as pd
import numpy as np

class EDA:
    """
    Core Exploratory Data Analysis class.
    Provides basic functionality for quick exploration of datasets.
    """
    
    def __init__(self, data):
        """Initialize with a DataFrame"""
        self.data = data.copy()
        self.rows, self.cols = self.data.shape
        self._identify_column_types()
    
    def _identify_column_types(self):
        """Identifies column types as numeric, categorical, or datetime"""
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                # Check if numeric column might actually be categorical
                if self.data[col].nunique() <= 10:
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.datetime_cols.append(col)
            else:
                # Try to convert to datetime
                try:
                    # Check if it might be a date column
                    if any(word in col.lower() for word in ['date', 'time', 'day', 'month', 'year']):
                        pd.to_datetime(self.data[col].iloc[:5])  # Try first few rows
                        self.datetime_cols.append(col)
                    else:
                        # If not a date, check if categorical
                        if self.data[col].nunique() <= 20:
                            self.categorical_cols.append(col)
                        else:
                            # High cardinality text column
                            self.categorical_cols.append(col)
                except:
                    # If conversion fails, check if it might be categorical
                    if self.data[col].nunique() <= 20:
                        self.categorical_cols.append(col)
                    else:
                        # High cardinality text column
                        self.categorical_cols.append(col)
    
    def analyze_missing_values(self):
        """
        Analyzes missing values in the dataset
        
        Returns:
            DataFrame: Missing value counts and percentages by column
        """
        missing = self.data.isnull().sum()
        missing_percent = (missing / len(self.data)) * 100
        
        missing_analysis = pd.DataFrame({
            'Missing Values': missing,
            'Percent Missing': missing_percent
        }).sort_values('Percent Missing', ascending=False)
        
        return missing_analysis
    
    def get_basic_stats(self):
        """
        Returns basic statistics for numeric columns
        
        Returns:
            DataFrame: Summary statistics for numeric columns
        """
        if len(self.numeric_cols) > 0:
            return self.data[self.numeric_cols].describe()
        return pd.DataFrame()
    
    def get_categorical_summaries(self):
        """
        Returns value counts for categorical columns
        
        Returns:
            dict: Dictionary of DataFrames with value counts for each categorical column
        """
        summaries = {}
        
        for col in self.categorical_cols:
            # Get value counts
            value_counts = self.data[col].value_counts(dropna=False).reset_index()
            value_counts.columns = [col, 'Count']
            
            # Calculate percentages
            value_counts['Percentage'] = 100 * value_counts['Count'] / value_counts['Count'].sum()
            
            summaries[col] = value_counts
        
        return summaries
    
    def analyze_correlations(self, method='pearson'):
        """
        Analyzes correlations between numeric variables
        
        Args:
            method (str): Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            tuple: (correlation_matrix, high_correlations_df)
        """
        if len(self.numeric_cols) < 2:
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = self.data[self.numeric_cols].corr(method=method)
        
        # Find high correlations (|r| >= 0.7)
        high_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) >= 0.7:
                    high_corr.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
        
        high_correlations = pd.DataFrame(high_corr)
        
        return correlation_matrix, high_correlations
    
    def detect_outliers(self, method='iqr'):
        """
        Detects outliers in numeric variables
        
        Args:
            method (str): Method to use for outlier detection ('iqr' or 'zscore')
            
        Returns:
            dict: Summary of outliers for each numeric column
        """
        outlier_summary = {}
        
        for col in self.numeric_cols:
            data = self.data[col].dropna()
            
            if method == 'iqr':
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
            else:
                # Z-score method
                z_scores = (data - data.mean()) / data.std()
                outliers = data[abs(z_scores) > 3]
                
                lower_bound = data.mean() - 3 * data.std()
                upper_bound = data.mean() + 3 * data.std()
            
            if len(outliers) > 0:
                outlier_summary[col] = {
                    'count': len(outliers),
                    'percent': len(outliers) / len(data) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        return outlier_summary
    
    def generate_insights(self):
        """
        Generates key insights from the analysis
        
        Returns:
            list: List of insight strings
        """
        insights = []
        
        # Data size insight
        insights.append(f"Dataset contains {self.rows} rows and {self.cols} columns.")
        
        # Column type breakdown
        insights.append(f"Found {len(self.numeric_cols)} numeric, {len(self.categorical_cols)} categorical, and {len(self.datetime_cols)} datetime columns.")
        
        # Missing values
        missing_analysis = self.analyze_missing_values()
        missing_cols = sum(missing_analysis['Missing Values'] > 0)
        if missing_cols > 0:
            missing_percent = (missing_analysis['Missing Values'].sum() / (self.rows * self.cols)) * 100
            insights.append(f"Found missing values in {missing_cols} columns. Overall, {missing_percent:.2f}% of data is missing.")
        else:
            insights.append("No missing values detected in the dataset.")
        
        # Outliers
        outlier_summary = self.detect_outliers()
        if outlier_summary:
            outlier_cols = len(outlier_summary)
            insights.append(f"Detected potential outliers in {outlier_cols} numeric columns.")
        
        # Correlations
        if len(self.numeric_cols) >= 2:
            _, high_correlations = self.analyze_correlations()
            if len(high_correlations) > 0:
                insights.append(f"Found {len(high_correlations)} pairs of highly correlated features (|r| ≥ 0.7).")
        
        return insights
    
    def get_duplicate_rows(self):
        """
        Finds duplicate rows in the dataset
        
        Returns:
            tuple: (count, DataFrame with duplicates)
        """
        duplicates = self.data[self.data.duplicated()]
        return len(duplicates), duplicates
    
    def filter_data(self, filters):
        """
        Apply filters to create a subset of the data
        
        Args:
            filters (dict): Dictionary of column name to filter conditions
            
        Returns:
            DataFrame: Filtered data
        """
        filtered_data = self.data.copy()
        
        for column, condition in filters.items():
            if column not in filtered_data.columns:
                continue
                
            if isinstance(condition, dict):
                if 'min' in condition and condition['min'] is not None:
                    filtered_data = filtered_data[filtered_data[column] >= condition['min']]
                if 'max' in condition and condition['max'] is not None:
                    filtered_data = filtered_data[filtered_data[column] <= condition['max']]
                if 'equal' in condition and condition['equal'] is not None:
                    filtered_data = filtered_data[filtered_data[column] == condition['equal']]
                if 'in' in condition and condition['in'] is not None:
                    filtered_data = filtered_data[filtered_data[column].isin(condition['in'])]
                if 'not_in' in condition and condition['not_in'] is not None:
                    filtered_data = filtered_data[~filtered_data[column].isin(condition['not_in'])]
                if 'like' in condition and condition['like'] is not None:
                    filtered_data = filtered_data[filtered_data[column].astype(str).str.contains(condition['like'])]
        
        return filtered_data