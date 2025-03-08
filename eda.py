import pandas as pd
import numpy as np
import re

class EDA:
    """
    Exploratory Data Analysis tool for data analysts.
    Provides core functionality for quick exploration of datasets.
    """
    
    def __init__(self, data):
        """Initialize with a DataFrame"""
        self.data = data.copy()
        self.rows, self.cols = self.data.shape
        self.summary = {}  # Initialize summary dictionary
        self._identify_column_types()
    
    def _identify_column_types(self):
        """Identifies column types as numeric, categorical, or datetime"""
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                if self.data[col].nunique() <= 10:
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.datetime_cols.append(col)
            else:
                # Try to convert to datetime - use try/except to avoid warnings
                try:
                    # First check if it's a standard date format - use a more robust approach
                    sample = self.data[col].dropna().iloc[:100] if len(self.data[col].dropna()) > 100 else self.data[col].dropna()
                    
                    # Check if it looks like a date before trying conversion (to avoid excessive warnings)
                    date_indicators = ['date', 'time', 'day', 'month', 'year']
                    if (any(indicator in col.lower() for indicator in date_indicators) or 
                        sample.astype(str).str.contains(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').any() or
                        sample.astype(str).str.contains(r'\d{4}-\d{2}-\d{2}').any()):
                        
                        # Parse with explicit error handling
                        try:
                            pd.to_datetime(sample, errors='raise')
                            self.datetime_cols.append(col)
                        except:
                            # Not a datetime, continue with categorical check
                            if self.data[col].nunique() <= 20:
                                self.categorical_cols.append(col)
                    else:
                        # Not a likely date column, check if categorical
                        if self.data[col].nunique() <= 20:
                            self.categorical_cols.append(col)
                except:
                    # If conversion fails, check if it might be categorical
                    if self.data[col].nunique() <= 20:
                        self.categorical_cols.append(col)
        
        # Save column types to summary
        self.summary['column_types'] = {
            'numeric': len(self.numeric_cols),
            'categorical': len(self.categorical_cols),
            'datetime': len(self.datetime_cols)
        }
    
    def analyze_missing_values(self):
        """
        Analyzes missing values in the dataset
        
        Returns:
            DataFrame: Missing value counts and percentages by column
        """
        missing = self.data.isnull().sum()
        missing_percent = (missing / len(self.data)) * 100
        self.missing_analysis = pd.DataFrame({
            'Missing Values': missing,
            'Percent Missing': missing_percent
        }).sort_values('Percent Missing', ascending=False)
        
        self.summary['missing'] = {
            'total_missing': missing.sum(),
            'total_missing_percent': (missing.sum() / (self.rows * self.cols)) * 100,
            'columns_with_missing': sum(missing > 0)
        }
        
        return self.missing_analysis

    def detect_datetime_columns(self):
        """
        Detects columns containing date/time information.
        Returns a list of column names with datetime data.
        """
        datetime_cols = []
        
        for col in self.data.columns:
            # Skip if column is numeric
            if pd.api.types.is_numeric_dtype(self.data[col]):
                continue
                
            # Check if column is already datetime
            if pd.api.types.is_datetime64_dtype(self.data[col]):
                datetime_cols.append(col)
                continue
                
            # Try to infer if the column contains datetime data
            try:
                # Get non-null values to check
                sample = self.data[col].dropna().sample(min(5, len(self.data[col].dropna()))).astype(str)
                
                # Try to identify common date formats before parsing
                # This helps avoid the warning about inferring format
                if len(sample) > 0:
                    first_val = sample.iloc[0]
                    date_format = None
                    
                    # Check for common date formats
                    if re.match(r'\d{4}-\d{2}-\d{2}', first_val):
                        date_format = '%Y-%m-%d'
                    elif re.match(r'\d{2}/\d{2}/\d{4}', first_val):
                        date_format = '%m/%d/%Y'
                    elif re.match(r'\d{2}\.\d{2}\.\d{4}', first_val):
                        date_format = '%d.%m.%Y'
                    elif re.match(r'\d{4}/\d{2}/\d{2}', first_val):
                        date_format = '%Y/%m/%d'
                        
                    # Try to parse with the detected format
                    if date_format:
                        pd.to_datetime(sample, format=date_format, errors='raise')
                        datetime_cols.append(col)
                    else:
                        # Fall back to let pandas detect the format
                        pd.to_datetime(sample, errors='raise')
                        datetime_cols.append(col)
            except:
                # Not a datetime column
                pass
        
        return datetime_cols
    
    def analyze_distributions(self):
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
            value_counts = self.data[col].value_counts().reset_index()
            value_counts.columns = [col, 'Count']
            value_counts['Percentage'] = 100 * value_counts['Count'] / value_counts['Count'].sum()
            summaries[col] = value_counts
        
        return summaries
    
    def analyze_correlations(self, method='pearson'):
        """
        Analyzes correlations between numeric variables
        
        Args:
            method (str): Correlation method ('pearson' or 'spearman')
            
        Returns:
            DataFrame: Correlation matrix
        """
        if len(self.numeric_cols) < 2:
            return pd.DataFrame()
        
        self.correlation_matrix = self.data[self.numeric_cols].corr(method=method)
        
        # Find high correlations
        high_corr = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                if abs(self.correlation_matrix.iloc[i, j]) >= 0.7:
                    high_corr.append({
                        'Feature 1': self.correlation_matrix.columns[i],
                        'Feature 2': self.correlation_matrix.columns[j],
                        'Correlation': self.correlation_matrix.iloc[i, j]
                    })
        
        self.high_correlations = pd.DataFrame(high_corr)
        self.summary['correlations'] = {
            'high_correlations_count': len(high_corr)
        }
        
        return self.correlation_matrix
    
    def detect_outliers(self):
        """
        Detects outliers in numeric variables using IQR method
        
        Returns:
            dict: Summary of outliers for each numeric column
        """
        outlier_summary = {}
        
        for col in self.numeric_cols:
            data = self.data[col].dropna()
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            if len(outliers) > 0:
                outlier_summary[col] = {
                    'count': len(outliers),
                    'percent': len(outliers) / len(data) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        self.outlier_summary = outlier_summary
        self.summary['outliers'] = {
            'columns_with_outliers': len(outlier_summary),
            'total_outlier_columns': sum([info['count'] > 0 for col, info in outlier_summary.items()])
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
        if hasattr(self, 'missing_analysis'):
            missing_cols = sum(self.missing_analysis['Missing Values'] > 0)
            if missing_cols > 0:
                insights.append(f"Found missing values in {missing_cols} columns. "
                               f"Overall, {self.summary['missing']['total_missing_percent']:.2f}% of data is missing.")
            else:
                insights.append("No missing values detected in the dataset.")
        
        # Correlations
        if hasattr(self, 'high_correlations') and len(self.high_correlations) > 0:
            insights.append(f"Detected {len(self.high_correlations)} pairs of highly correlated features (|r| ≥ 0.7).")
        
        # Outliers
        if hasattr(self, 'outlier_summary') and len(self.outlier_summary) > 0:
            outlier_cols = len(self.outlier_summary)
            insights.append(f"Detected potential outliers in {outlier_cols} numeric columns.")
        
        return insights
    
    def get_duplicate_rows(self):
        """
        Finds duplicate rows in the dataset
        
        Returns:
            int: Number of duplicate rows
            DataFrame: The duplicate rows
        """
        duplicates = self.data[self.data.duplicated()]
        return len(duplicates), duplicates
    
    def get_column_completeness(self):
        """
        Calculates completeness percentage for each column
        
        Returns:
            Series: Percentage of non-null values for each column
        """
        return (1 - self.data.isnull().mean()) * 100
    
    def run_full_analysis(self):
        """
        Runs all analysis methods at once
        """
        self.analyze_missing_values()
        self.analyze_distributions()
        self.analyze_correlations()
        self.detect_outliers()
        return self.generate_insights()
    # Business-Focused Enhancements

    ## 1. Time Series Analysis
    def analyze_time_series(self, date_column, metric_column):
        """
        Analyze trends over time and seasonality patterns
        
        Args:
            date_column: Column containing dates
            metric_column: Metric to analyze over time
        """
        # Ensure date column is datetime type
        data = self.data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Group by different time periods
        daily = data.groupby(data[date_column].dt.date)[metric_column].mean()
        weekly = data.groupby(pd.Grouper(key=date_column, freq='W'))[metric_column].mean()
        monthly = data.groupby(pd.Grouper(key=date_column, freq='M'))[metric_column].mean()
        
        # Calculate growth rates
        pct_change = monthly.pct_change() * 100
        
        # Detect seasonality
        if len(monthly) >= 24:  # Need at least 2 years for annual seasonality
            monthly_decompose = seasonal_decompose(monthly, model='additive', period=12)
            seasonal_strength = 1 - np.var(monthly_decompose.resid) / np.var(monthly_decompose.observed - monthly_decompose.trend)
        else:
            seasonal_strength = None
        
        return {
            'daily': daily,
            'weekly': weekly,
            'monthly': monthly,
            'growth_rates': pct_change,
            'seasonal_strength': seasonal_strength
        }

    ## 2. Cross-Tabulation Analysis
    def cross_tabulation(self, column1, column2, normalize=None):
        """
        Create a cross-tabulation between two categorical variables
        
        Args:
            column1: First categorical column
            column2: Second categorical column
            normalize: None, 'index', 'columns', or 'all'
        
        Returns:
            DataFrame with cross-tabulation
        """
        cross_tab = pd.crosstab(
            self.data[column1], 
            self.data[column2], 
            normalize=normalize,
            margins=True, 
            margins_name='Total'
        )
        
        # Calculate chi-square test for independence
        chi2, p, dof, expected = chi2_contingency(cross_tab.iloc[:-1, :-1])
        
        result = {
            'cross_tab': cross_tab,
            'chi2': chi2,
            'p_value': p,
            'dof': dof,
            'relationship': 'Significant relationship' if p < 0.05 else 'No significant relationship'
        }
        
        return result

    ## 3. KPI Dashboard Generator
    def generate_kpi_dashboard(self, metrics_config):
        """
        Generate a KPI dashboard based on configuration
        
        Args:
            metrics_config: Dict with metric name, column, and aggregation method
        
        Returns:
            Dict of calculated KPIs
        """
        results = {}
        
        for metric_name, config in metrics_config.items():
            column = config['column']
            method = config['method']
            
            if method == 'sum':
                value = self.data[column].sum()
            elif method == 'mean':
                value = self.data[column].mean()
            elif method == 'count':
                value = self.data[column].count()
            elif method == 'count_distinct':
                value = self.data[column].nunique()
            elif method == 'conversion_rate' and 'target_column' in config:
                value = (self.data[config['target_column']] == config['target_value']).mean() * 100
            
            results[metric_name] = value
        
        return results

    ## 4. Cohort Analysis
    def cohort_analysis(self, date_column, cohort_column, metric_column):
        """
        Perform cohort analysis to track retention over time
        
        Args:
            date_column: Column with event date
            cohort_column: Column to define cohorts (e.g., signup date)
            metric_column: Metric to aggregate
        
        Returns:
            DataFrame with cohort analysis
        """
        # Convert to datetime
        data = self.data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        data[cohort_column] = pd.to_datetime(data[cohort_column])
        
        # Extract cohort period and period number
        data['cohort_period'] = data[cohort_column].dt.to_period('M')
        data['period_num'] = ((data[date_column].dt.year - data[cohort_column].dt.year) * 12 + 
                            (data[date_column].dt.month - data[cohort_column].dt.month))
        
        # Group by cohort and period number
        cohort_data = data.groupby(['cohort_period', 'period_num'])[metric_column].agg(['count', 'sum'])
        
        # Create pivot table for retention analysis
        cohort_counts = cohort_data['count'].unstack(1).fillna(0)
        retention_matrix = cohort_counts.div(cohort_counts[0], axis=0) * 100
        
        return {
            'cohort_counts': cohort_counts,
            'retention_matrix': retention_matrix
        }

    # Data Science Enhancements

    ## 1. Feature Engineering Suggestions
    def suggest_feature_transformations(self):
        """
        Suggest potential feature transformations based on data characteristics
        
        Returns:
            Dict with transformation suggestions for columns
        """
        suggestions = {}
        
        for col in self.numeric_cols:
            column_data = self.data[col].dropna()
            
            # Check for skewness
            skewness = column_data.skew()
            if abs(skewness) > 1:
                if skewness > 0:
                    suggestions[col] = {
                        'issue': f'Right-skewed distribution (skewness = {skewness:.2f})',
                        'transformations': ['Log transform', 'Square root transform', 'Box-Cox transform']
                    }
                else:
                    suggestions[col] = {
                        'issue': f'Left-skewed distribution (skewness = {skewness:.2f})',
                        'transformations': ['Square transform', 'Cube transform', 'Exponential transform']
                    }
            
            # Check for outliers
            if hasattr(self, 'outlier_summary') and col in self.outlier_summary:
                if self.outlier_summary[col]['percent'] > 5:
                    suggestions.setdefault(col, {'issue': [], 'transformations': []})
                    suggestions[col]['issue'] += [f'Contains significant outliers ({self.outlier_summary[col]["percent"]:.2f}%)']
                    suggestions[col]['transformations'] += ['Winsorization', 'Z-score normalization']
                    
            # Check for high cardinality categorical
            if col in self.categorical_cols and self.data[col].nunique() > 10:
                suggestions[col] = {
                    'issue': f'High cardinality categorical ({self.data[col].nunique()} unique values)',
                    'transformations': ['Target encoding', 'Frequency encoding', 'Binary encoding', 'Grouping categories']
                }
        
        return suggestions

    ## 2. Quick Model Building
    def build_quick_model(self, target_column, feature_columns=None):
        """
        Build a simple predictive model with minimal configuration
        
        Args:
            target_column: Target variable to predict
            feature_columns: List of feature columns (uses all numeric if None)
        
        Returns:
            Dict with model results
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        
        # Prepare data
        if feature_columns is None:
            feature_columns = [col for col in self.numeric_cols if col != target_column]
        
        X = self.data[feature_columns].copy()
        y = self.data[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 50:
            return {"error": "Insufficient data for modeling (less than 50 valid rows)"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Determine problem type and build model
        is_classification = target_column in self.categorical_cols
        
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            train_f1 = f1_score(y_train, train_preds, average='weighted')
            test_f1 = f1_score(y_test, test_preds, average='weighted')
            
            metrics = {
                "task_type": "Classification",
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_f1": train_f1,
                "test_f1": test_f1
            }
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            
            metrics = {
                "task_type": "Regression",
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2
            }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": feature_importance
        }

    ## 3. Statistical Testing
    def statistical_testing(self, column1, column2=None, test_type=None):
        """
        Perform statistical tests on the data
        
        Args:
            column1: First column to test
            column2: Second column for comparison tests
            test_type: Type of test to perform (auto-detect if None)
        
        Returns:
            Dict with test results
        """
        from scipy import stats
        
        results = {}
        
        # Auto-detect test type if not specified
        if test_type is None:
            if column2 is None:
                # Single sample tests
                if column1 in self.numeric_cols:
                    test_type = 'normality'
                else:
                    test_type = 'chi_square_goodness_of_fit'
            else:
                # Two sample tests
                if column1 in self.numeric_cols and column2 in self.numeric_cols:
                    test_type = 'correlation'
                elif column1 in self.numeric_cols and column2 in self.categorical_cols:
                    test_type = 'anova'
                elif column1 in self.categorical_cols and column2 in self.categorical_cols:
                    test_type = 'chi_square_independence'
        
        # Perform the test
        if test_type == 'normality':
            # Shapiro-Wilk test for normality
            data = self.data[column1].dropna()
            if len(data) > 5000:
                data = data.sample(5000, random_state=42)
            
            stat, p = stats.shapiro(data)
            results = {
                'test': 'Shapiro-Wilk test for normality',
                'column': column1,
                'statistic': stat,
                'p_value': p,
                'interpretation': 'Data is normally distributed' if p > 0.05 else 'Data is not normally distributed'
            }
        
        elif test_type == 'correlation':
            # Correlation test
            data = self.data[[column1, column2]].dropna()
            r, p = stats.pearsonr(data[column1], data[column2])
            results = {
                'test': 'Pearson correlation',
                'columns': [column1, column2],
                'correlation': r,
                'p_value': p,
                'interpretation': f"{'Significant' if p < 0.05 else 'No significant'} correlation (r={r:.4f})"
            }
        
        elif test_type == 'anova':
            # One-way ANOVA
            groups = [self.data[self.data[column2] == val][column1].dropna() for val in self.data[column2].unique()]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                f_stat, p = stats.f_oneway(*groups)
                results = {
                    'test': 'One-way ANOVA',
                    'numeric_column': column1,
                    'group_column': column2,
                    'f_statistic': f_stat,
                    'p_value': p,
                    'interpretation': f"{'Significant' if p < 0.05 else 'No significant'} difference between groups"
                }
            else:
                results = {'error': 'Insufficient data for ANOVA'}
        
        elif test_type == 'chi_square_independence':
            # Chi-square test of independence
            contingency_table = pd.crosstab(self.data[column1], self.data[column2])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            results = {
                'test': 'Chi-square test of independence',
                'columns': [column1, column2],
                'chi2': chi2,
                'p_value': p,
                'dof': dof,
                'interpretation': f"{'Significant' if p < 0.05 else 'No significant'} association between variables"
            }
        
        return results

    ## 4. Dimensionality Reduction and Clustering
    def reduce_dimensions(self, n_components=2):
        """
        Perform dimensionality reduction using PCA
        
        Args:
            n_components: Number of components to keep
        
        Returns:
            Dict with PCA results
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        numeric_data = self.data[self.numeric_cols].dropna()
        if len(numeric_data) < 10 or len(self.numeric_cols) < 3:
            return {"error": "Insufficient data for PCA"}
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(self.numeric_cols)))
        transformed_data = pca.fit_transform(scaled_data)
        
        # Create result dataframe
        result_df = pd.DataFrame(
            transformed_data,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=numeric_data.index
        )
        
        # Calculate feature loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=numeric_data.columns
        )
        
        return {
            'pca': pca,
            'transformed_data': result_df,
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'loadings': loadings
        }

    def cluster_data(self, n_clusters=3, method='kmeans'):
        """
        Cluster the data using various algorithms
        
        Args:
            n_clusters: Number of clusters to create
            method: Clustering method ('kmeans' or 'hierarchical')
        
        Returns:
            Dict with clustering results
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        
        # Prepare data
        numeric_data = self.data[self.numeric_cols].dropna()
        if len(numeric_data) < n_clusters * 3 or len(self.numeric_cols) < 2:
            return {"error": "Insufficient data for clustering"}
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Apply clustering
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        
        clusters = clustering.fit_predict(scaled_data)
        
        # Evaluate clustering
        if len(set(clusters)) > 1:  # Ensure we have more than one cluster
            silhouette = silhouette_score(scaled_data, clusters)
        else:
            silhouette = 0
        
        # Create result with cluster assignments
        result_df = numeric_data.copy()
        result_df['Cluster'] = clusters
        
        # Calculate cluster profiles
        cluster_means = result_df.groupby('Cluster').mean()
        
        # Standardize means for visualization
        cluster_profiles = (cluster_means - cluster_means.mean()) / cluster_means.std()
        
        return {
            'clustering': clustering,
            'cluster_assignments': clusters,
            'silhouette_score': silhouette,
            'data_with_clusters': result_df,
            'cluster_profiles': cluster_profiles
        }

    # Shared Enhancements

    ## 1. Data Quality Score
    def calculate_data_quality_score(self):
        """
        Calculate overall data quality score
        
        Returns:
            Dict with quality scores and metrics
        """
        metrics = {}
        
        # Completeness (1 - % missing)
        missing_pct = self.data.isnull().sum().sum() / (self.rows * self.cols)
        completeness = 1 - missing_pct
        metrics['completeness'] = completeness
        
        # Validity (% of data passing validation rules)
        validity_scores = []
        
        # Numeric columns: check for values in reasonable ranges
        for col in self.numeric_cols:
            data = self.data[col].dropna()
            if len(data) > 0:
                # Check for extreme Z-scores
                z_scores = np.abs((data - data.mean()) / data.std())
                validity = 1 - (z_scores > 5).mean()  # % of data with Z-score <= 5
                validity_scores.append(validity)
        
        # Categorical columns: check for unexpected values
        for col in self.categorical_cols:
            # High cardinality might indicate issues
            n_unique = self.data[col].nunique()
            n_total = len(self.data[col].dropna())
            if n_total > 0:
                # Ratio of unique values should not be too high
                validity = 1 - min(n_unique / n_total, 0.5) / 0.5
                validity_scores.append(validity)
        
        if validity_scores:
            metrics['validity'] = np.mean(validity_scores)
        else:
            metrics['validity'] = 1.0
        
        # Consistency (correlation structure makes sense)
        if hasattr(self, 'high_correlations') and len(self.numeric_cols) > 1:
            correlation_matrix = self.data[self.numeric_cols].corr().abs()
            # Too many high correlations could indicate data issues
            high_corr_pairs = sum(correlation_matrix.unstack() > 0.95) - len(self.numeric_cols)
            consistency = 1 - min(high_corr_pairs / (len(self.numeric_cols) * (len(self.numeric_cols) - 1)), 1)
            metrics['consistency'] = consistency
        else:
            metrics['consistency'] = 1.0
        
        # Uniformity (distribution uniformity)
        uniformity_scores = []
        for col in self.numeric_cols:
            data = self.data[col].dropna()
            if len(data) > 0:
                # Skewness should not be too high
                skewness = abs(data.skew())
                uniformity = 1 - min(skewness / 5, 1)
                uniformity_scores.append(uniformity)
        
        if uniformity_scores:
            metrics['uniformity'] = np.mean(uniformity_scores)
        else:
            metrics['uniformity'] = 1.0
        
        # Duplicates (% of non-duplicate rows)
        duplicates = self.data.duplicated().mean()
        metrics['uniqueness'] = 1 - duplicates
        
        # Overall score (weighted average)
        weights = {
            'completeness': 0.35,
            'validity': 0.25,
            'consistency': 0.15,
            'uniformity': 0.15,
            'uniqueness': 0.1
        }
        
        overall_score = sum(metrics[metric] * weight for metric, weight in weights.items())
        
        # Add grade
        if overall_score >= 0.9:
            grade = 'A'
        elif overall_score >= 0.8:
            grade = 'B'
        elif overall_score >= 0.7:
            grade = 'C'
        elif overall_score >= 0.6:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'metrics': metrics
        }

    ## 2. Interactive Data Filtering
    def filter_data(self, filters):
        """
        Apply filters to create a subset of the data
        
        Args:
            filters: Dict of column name: filter condition
        
        Returns:
            EDA object with filtered data
        """
        filtered_data = self.data.copy()
        
        for column, condition in filters.items():
            if column not in self.data.columns:
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
        
        # Create new EDA instance with filtered data
        from copy import deepcopy
        filtered_eda = deepcopy(self)
        filtered_eda.data = filtered_data
        filtered_eda.rows, filtered_eda.cols = filtered_data.shape
        
        # Re-run basic analysis on filtered data
        filtered_eda._identify_column_types()
        
        return filtered_eda

    ## 3. Comparison Analysis
    def compare_datasets(self, other_eda, name1="Dataset 1", name2="Dataset 2"):
        """
        Compare two datasets and identify key differences
        
        Args:
            other_eda: Another EDA object to compare to
            name1: Name for this dataset
            name2: Name for the other dataset
        
        Returns:
            Dict with comparison results
        """
        comparison = {}
        
        # Basic size comparison
        comparison['size'] = {
            name1: {'rows': self.rows, 'columns': self.cols},
            name2: {'rows': other_eda.rows, 'columns': other_eda.cols},
            'difference': {
                'rows': self.rows - other_eda.rows,
                'columns': self.cols - other_eda.cols
            }
        }
        
        # Column comparison
        common_cols = set(self.data.columns).intersection(set(other_eda.data.columns))
        only_in_1 = set(self.data.columns) - set(other_eda.data.columns)
        only_in_2 = set(other_eda.data.columns) - set(self.data.columns)
        
        comparison['columns'] = {
            'common': list(common_cols),
            f'only_in_{name1}': list(only_in_1),
            f'only_in_{name2}': list(only_in_2)
        }
        
        # Value distribution comparison for common columns
        distribution_diff = {}
        
        for col in common_cols:
            if col in self.numeric_cols and col in other_eda.numeric_cols:
                # Compare numeric columns
                stats1 = self.data[col].describe()
                stats2 = other_eda.data[col].describe()
                
                diff = {
                    'mean_diff': stats1['mean'] - stats2['mean'],
                    'mean_pct_diff': (stats1['mean'] - stats2['mean']) / stats2['mean'] * 100 if stats2['mean'] != 0 else 0,
                    'std_diff': stats1['std'] - stats2['std'],
                    'min_diff': stats1['min'] - stats2['min'],
                    'max_diff': stats1['max'] - stats2['max']
                }
                
                distribution_diff[col] = diff
            
            elif col in self.categorical_cols and col in other_eda.categorical_cols:
                # Compare value distributions for categorical columns
                counts1 = self.data[col].value_counts(normalize=True)
                counts2 = other_eda.data[col].value_counts(normalize=True)
                
                # Align the two series for comparison
                aligned_counts = pd.DataFrame({'dataset1': counts1, 'dataset2': counts2}).fillna(0)
                
                # Calculate differences
                aligned_counts['abs_diff'] = (aligned_counts['dataset1'] - aligned_counts['dataset2']).abs()
                aligned_counts['pct_diff'] = aligned_counts['abs_diff'] * 100
                
                distribution_diff[col] = {
                    'max_diff_category': aligned_counts['abs_diff'].idxmax(),
                    'max_diff_pct': aligned_counts['pct_diff'].max(),
                    'total_variation_distance': aligned_counts['abs_diff'].sum() / 2
                }
        
        comparison['distribution_differences'] = distribution_diff
        
        # Data quality comparison if both have quality scores
        if hasattr(self, 'quality_score') and hasattr(other_eda, 'quality_score'):
            comparison['quality'] = {
                name1: self.quality_score,
                name2: other_eda.quality_score
            }
        
        return comparison