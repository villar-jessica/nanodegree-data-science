"""
Gender Equality in Business Leadership Analysis Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class GenderEqualityAnalyzer:
    """
    Gender Equality in Business Leadership Analyzer - COMPLETE PIPELINE.
    
    Includes all preprocessing steps:
    1. Data loading with World Bank '..' handling
    2. Variable mapping (World Bank codes to readable names)
    3. Pivot table (long to wide format)
    4. High-null column removal (>60% threshold)
    5. Temporal imputation
    6. Feature engineering
    7. Model training and evaluation
    
    Attributes:
        data (pd.DataFrame): Original dataset (long format)
        data_mapped (pd.DataFrame): After variable mapping
        data_wide (pd.DataFrame): Pivoted dataset (wide format)
        data_clean (pd.DataFrame): Preprocessed dataset
        features (pd.DataFrame): Features for modeling
        target (pd.Series): Target variable
        variable_mapping (dict): World Bank code to readable name mapping
        models (dict): Trained models
        results (dict): Evaluation metrics
    """
    
    def __init__(self):
        """Initialize analyzer with default World Bank variable mapping."""
        self.data = None
        self.data_mapped = None
        self.data_wide = None
        self.data_clean = None
        self.features = None
        self.target = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Default World Bank variable mapping (can be customized)
        self.variable_mapping = {
            # TARGET (variável que queremos prever)
            'SL.EMP.SMGT.FE.ZS': 'pct_mulheres_gerencia',
            
            # EDUCAÇÃO
            'SE.TER.GRAD.FE.SI.ZS': 'pct_graduadas_stem',
            'SE.TER.ENRR.FE': 'matricula_superior_feminina',
            'SE.SEC.ENRR.FE': 'matricula_secundaria_feminina',
            
            # MERCADO DE TRABALHO
            'SL.TLF.CACT.FE.ZS': 'participacao_trabalho_feminina',
            'SL.TLF.CACT.FM.ZS': 'razao_fm_participacao_trabalho',
            'SL.UEM.TOTL.FE.ZS': 'desemprego_feminino',
            'SL.UEM.TOTL.MA.ZS': 'desemprego_masculino',
            'SL.UEM.ADVN.FE.ZS': 'desemprego_edu_avancada_feminino',
            'SL.UEM.ADVN.MA.ZS': 'desemprego_edu_avancada_masculino',
            'SL.EMP.WORK.FE.ZS': 'empregadas_assalariadas',
            'SL.EMP.WORK.MA.ZS': 'empregados_assalariados',
            
            # LIDERANÇA/EMPRESAS
            'IC.FRM.FEMM.ZS': 'empresas_gerente_feminina',
            'IC.FRM.FEMO.ZS': 'empresas_proprietaria_feminina',
            'SG.GEN.PARL.ZS': 'mulheres_parlamento',
            
            # CONTEXTO TECNOLÓGICO E ECONÔMICO
            'IT.NET.USER.ZS': 'uso_internet',
            'NY.GDP.PCAP.CD': 'pib_per_capita',
            'SI.POV.GINI': 'indice_gini',
            'HD.HCI.OVRL': 'indice_capital_humano'
        }
        
    def load_world_bank_data(self, filepath, series_code_col='Series Code'):
        """
        Load World Bank data in original wide format.
        
        World Bank data typically has:
        - Metadata columns: Country Name, Country Code, Series Code, Series Name
        - Year columns: "2015 [YR2015]", "2016 [YR2016]", etc.
        
        Args:
            filepath (str): Path to Excel file
            series_code_col (str): Column with indicator codes (default: 'Series Code')
            
        Returns:
            pd.DataFrame: Loaded data in original wide format
        """
        try:
            self.data = pd.read_excel(filepath)
            print(f"Data loaded successfully!")
            print(f"  Dimensions: {self.data.shape[0]:,} rows x {self.data.shape[1]} columns")
            
            # Identify year columns (format: "2015 [YR2015]")
            year_cols = [col for col in self.data.columns if 'YR' in col]
            if year_cols:
                first_year = year_cols[0].split('[')[0].strip()
                last_year = year_cols[-1].split('[')[0].strip()
                print(f"  Data period: {first_year} - {last_year}")
                print(f"  Years available: {len(year_cols)}")
            
            return self.data
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found!")
            raise
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
            
    def transform_to_long(self, id_vars=None):
        """
        Transform World Bank wide format to long format.
        
        Converts from:
        - One row per country-indicator
        - Years as columns ("2015 [YR2015]", "2016 [YR2016]", ...)
        
        To:
        - Multiple rows per country (one per year)
        - Year as a column
        - Value as a column
        
        Args:
            id_vars (list): Columns to keep as identifiers 
                           (default: ['Country Name', 'Country Code', 'Series Code'])
            
        Returns:
            pd.DataFrame: Data in long format
        """
        if self.data is None:
            raise ValueError("Data not loaded! Run load_world_bank_data() first.")
        
        print("Transforming from wide to long format...")
        
        # Identify year columns
        year_cols = [col for col in self.data.columns if 'YR' in col]
        if not year_cols:
            raise ValueError("No year columns found! Expected format: '2015 [YR2015]'")
        
        # Default id_vars
        if id_vars is None:
            id_vars = ['Country Name', 'Country Code', 'Series Code']
            # Only include columns that exist
            id_vars = [col for col in id_vars if col in self.data.columns]
        
        print(f"   ID variables: {id_vars}")
        print(f"   Year columns: {len(year_cols)}")
        
        # Melt to long format
        self.data_long = self.data.melt(
            id_vars=id_vars,
            value_vars=year_cols,
            var_name='year_col',
            value_name='value'
        )
        
        # Extract numeric year from "2015 [YR2015]" → 2015
        self.data_long['year'] = self.data_long['year_col'].str.extract(r'(\d{4})').astype(int)
        self.data_long = self.data_long.drop('year_col', axis=1)
        
        # Convert '..' to NaN and make numeric
        self.data_long['value'] = self.data_long['value'].replace('..', np.nan)
        self.data_long['value'] = pd.to_numeric(self.data_long['value'], errors='coerce')
        
        # Remove null values
        before = len(self.data_long)
        self.data_long = self.data_long.dropna(subset=['value'])
        removed = before - len(self.data_long)
        
        print(f"   Transformation complete!")
        print(f"   Long format: {len(self.data_long):,} rows")
        print(f"   Removed {removed:,} rows with null values")
        
        return self.data_long
            
    def map_variables(self, series_code_col='Series Code', custom_mapping=None, keep_unmapped=False):
        """
        Map World Bank indicator codes to readable variable names
        
        Args:
            series_code_col (str): Column containing World Bank codes (default: 'Series Code')
            custom_mapping (dict): Custom mapping dict (default: uses built-in mapping)
            keep_unmapped (bool): Keep variables not in mapping (default: False)
            
        Returns:
            pd.DataFrame: Data with 'variable_name' column added
        """
        if self.data_long is None:
            raise ValueError("Data not loaded! Run load_world_bank_data() first.")
        
        # Use custom mapping if provided, otherwise use default
        mapping = custom_mapping if custom_mapping is not None else self.variable_mapping
        
        print("Mapping World Bank variable codes to readable names...")
        print(f"   Total mapping rules: {len(mapping)}")
        
        # Create variable_name column
        self.data_long['variable_name'] = self.data_long[series_code_col].map(mapping)
        
        # Count mapped vs unmapped
        mapped_count = self.data_long['variable_name'].notna().sum()
        unmapped_count = self.data_long['variable_name'].isna().sum()
        
        print(f"   Mapped: {mapped_count:,} rows ({mapped_count/len(self.data)*100:.1f}%)")
        print(f"   Unmapped: {unmapped_count:,} rows ({unmapped_count/len(self.data)*100:.1f}%)")
        
        # Filter to keep only mapped variables (unless keep_unmapped=True)
        if not keep_unmapped:
            before = len(self.data_long)
            self.data_mapped = self.data_long[self.data_long['variable_name'].notna()].copy()
            removed = before - len(self.data_mapped)
            print(f"   Removed {removed:,} unmapped rows")
            print(f"   Final: {len(self.data_mapped):,} rows with {self.data_mapped['variable_name'].nunique()} unique variables")
        else:
            self.data_mapped = self.data.copy()
            print(f"   Kept all rows (including unmapped)")
        
        return self.data_mapped
    
    def pivot_data(self, index_cols, columns_col='variable_name', values_col='value'):
        """
        Pivot data from long to wide format.
        
        Transforms data where each indicator is in a separate row to format
        where each indicator is a column. This is necessary for modeling.
        
        RATIONALE:
        - Machine learning models need wide format (one row per observation)
        - Each indicator becomes a feature (column)
        - Facilitates feature engineering and analysis
        
        Args:
            index_cols (list): Columns to use as index (e.g., ['Country Name', 'year'])
            columns_col (str): Column containing variable names (default: 'variable_name')
            values_col (str): Column containing values (default: 'value')
            
        Returns:
            pd.DataFrame: Pivoted data in wide format
        """
        if self.data_mapped is None:
            raise ValueError("Data not loaded! Run load_world_bank_data() first.")
        
        print("Pivoting data from long to wide format...")
        print(f"   Index columns: {index_cols}")
        print(f"   Columns from: {columns_col}")
        print(f"   Values from: {values_col}")
        
        # Pivot table
        self.data_wide = self.data_mapped.pivot_table(
            index=index_cols,
            columns=columns_col,
            values=values_col,
            aggfunc='first'  # In case of duplicates, take first value
        ).reset_index()
        
        # Remove column name from index
        self.data_wide.columns.name = None
        
        print(f"Pivoting complete!")
        print(f"Wide format: {self.data_wide.shape[0]} rows x {self.data_wide.shape[1]} columns")
        print(f"Created {self.data_wide.shape[1] - len(index_cols)} indicator columns")
        
        return self.data_wide
    
    def remove_high_null_columns(self, threshold=0.6, verbose=True):
        """
        Remove columns with more than threshold % of null values.
        
        RATIONALE:
        - Columns with >60% nulls have insufficient data for reliable analysis
        - Imputation on such sparse columns introduces too much uncertainty
        - Keeps only columns with enough observations for meaningful patterns
        
        THRESHOLD JUSTIFICATION:
        - 60% nulls = only 40% real data
        - Statistical power decreases significantly below this threshold
        
        Args:
            threshold (float): Maximum proportion of nulls allowed (default: 0.6)
            verbose (bool): Print removed columns (default: True)
            
        Returns:
            pd.DataFrame: Dataset with high-null columns removed
        """
        if self.data_wide is None:
            raise ValueError("Data not pivoted! Run pivot_data() first.")
        
        print(f"Removing columns with >{threshold*100:.0f}% null values...")
        
        # Calculate null percentage per column
        null_pct = self.data_wide.isnull().sum() / len(self.data_wide)
        
        # Identify columns to remove
        cols_to_remove = null_pct[null_pct > threshold].index.tolist()
        
        # Identify columns to keep
        cols_to_keep = null_pct[null_pct <= threshold].index.tolist()
        
        if verbose and len(cols_to_remove) > 0:
            print(f"\n   Columns being removed ({len(cols_to_remove)}):")
            for col in cols_to_remove[:10]:  # Show first 10
                pct = null_pct[col] * 100
                print(f"      - {col[:70]:<70} ({pct:.1f}% null)")
            if len(cols_to_remove) > 10:
                print(f"      ... and {len(cols_to_remove) - 10} more columns")
        
        # Remove columns
        before_cols = self.data_wide.shape[1]
        self.data_wide = self.data_wide[cols_to_keep]
        after_cols = self.data_wide.shape[1]
        
        print(f"\n   Removed {before_cols - after_cols} columns")
        print(f"   Remaining: {after_cols} columns")
        print(f"   Data shape: {self.data_wide.shape[0]} rows x {self.data_wide.shape[1]} columns")
        
        return self.data_wide
    
    def preprocess_data(self, fillna_method='both'):
        """
        Preprocess data with temporal imputation.
        
        This method should be called AFTER pivot_data() and remove_high_null_columns().
        
        IMPUTATION RATIONALE:
        - Forward/backward fill BY COUNTRY preserves temporal trends
        - Appropriate for economic indicators that change gradually
        - Avoids bias from global averages
        
        IMPLICATIONS:
        - Imputed values may slightly underestimate variability
        - Enables complete dataset analysis without massive data loss
        
        Args:
            fillna_method (str): 'ffill', 'bfill', or 'both' (default)
            
        Returns:
            pd.DataFrame: Clean data
            
        Note:
            Imputation is done by country to maintain country-specific trends.
        """
        if self.data_wide is None:
            raise ValueError("Data not pivoted! Run pivot_data() first.")
        
        print("Preprocessing...")
        print(f"   Missing values before: {self.data_wide.isnull().sum().sum()}")
        
        self.data_clean = self.data_wide.copy()
        
        # Temporal imputation by country (if Country Name exists)
        if 'Country Name' in self.data_clean.columns:
            print(f"   Temporal imputation by country...")
            if fillna_method in ['ffill', 'both']:
                self.data_clean = self.data_clean.groupby('Country Name', group_keys=False).apply(
                    lambda group: group.fillna(method='ffill')
                )
            if fillna_method in ['bfill', 'both']:
                self.data_clean = self.data_clean.groupby('Country Name', group_keys=False).apply(
                    lambda group: group.fillna(method='bfill')
                )
        
        # Remove remaining NaNs
        before = len(self.data_clean)
        self.data_clean = self.data_clean.dropna()
        after = len(self.data_clean)
        
        print(f"   Missing values after imputation: {self.data_clean.isnull().sum().sum()}")
        print(f"   Rows removed (remaining NaNs): {before - after}")
        print(f"   Preprocessing complete!")
        print(f"   Final dataset: {self.data_clean.shape[0]} rows x {self.data_clean.shape[1]} columns")
        
        return self.data_clean
    
    def create_features(self, target_col=None, exclude_cols=None):
        """
        Create features excluding categorical identifiers.
        
        CATEGORICAL EXCLUSION - RATIONALE:
        We exclude Country Name, Country Code, Year to:
        - Avoid overfitting to specific countries
        - Focus on GENERALIZABLE economic/social factors
        - Enable model application to new contexts
        
        Args:
            target_col (str): Target column name (if None, tries to auto-detect)
            exclude_cols (list): Columns to exclude (default: ['Country Name', 'Country Code', 'year', 'Year'])
            
        Returns:
            tuple: (features, target)
        """
        if self.data_clean is None:
            raise ValueError("Data not preprocessed! Run preprocess_data() first.")
        
        if exclude_cols is None:
            exclude_cols = ['Country Name', 'Country Code', 'year', 'Year']
        
        # Auto-detect target column if not provided
        if target_col is None:
            possible_targets = [
                'pct_mulheres_gerencia',
                'empresas_gerente_feminina'
            ]
            for col in possible_targets:
                if col in self.data_clean.columns:
                    target_col = col
                    break
            
            if target_col is None:
                raise ValueError(f"Target column not found! Specify target_col parameter.")
        
        print("Creating features...")
        print(f"   Excluded: {exclude_cols}")
        print(f"   Target: {target_col}")
        
        if target_col not in self.data_clean.columns:
            raise ValueError(f"Target column '{target_col}' not found in data!")
        
        feature_cols = [c for c in self.data_clean.columns 
                       if c != target_col and c not in exclude_cols]
        
        self.features = self.data_clean[feature_cols]
        self.target = self.data_clean[target_col]
        self.feature_names = feature_cols
        
        print(f"   Features created!")
        print(f"   {len(feature_cols)} features, {len(self.target)} samples")
        
        return self.features, self.target
    
    def train_models(self, test_size=0.2, random_state=42):
        """
        Train multiple machine learning models.
        
        Models implemented:
        - Linear Regression
        - Random Forest
        - Gradient Boosting
        
        Args:
            test_size (float): Test set proportion (default: 0.2)
            random_state (int): Random seed (default: 42)
            
        Returns:
            dict: Trained models
        """
        if self.features is None or self.target is None:
            raise ValueError("Features not created! Run create_features() first.")
        
        print("Training models...")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Scaling
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_names,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_names,
            index=X_test.index
        )
        
        # Store splits
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=random_state
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=random_state
            )
        }
        
        # Train and evaluate
        for name, model in models.items():
            print(f"\n   Training {name}...")
            
            model.fit(X_train_scaled, y_train)
            
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='r2'
            )
            
            self.models[name] = model
            self.results[name] = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"      R² Test: {self.results[name]['test_r2']:.4f}")
            print(f"      MAE Test: {self.results[name]['test_mae']:.4f}")
            print(f"      CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print("\nAll models trained!")
        return self.models
    
    def plot_correlation_matrix(self, top_n=15, figsize=(14, 10), cmap='coolwarm'):
        """Plot correlation matrix of features."""
        if self.data_clean is None:
            raise ValueError("Data not available!")
        
        # Find target column
        target_cols = [c for c in self.data_clean.columns if 'female' in c.lower() and 'ownership' in c.lower()]
        if not target_cols:
            raise ValueError("Target column not found!")
        target_col = target_cols[0]
        
        corr = self.data_clean.corr()[target_col].sort_values(ascending=False)
        top_features = corr.head(top_n).index.tolist()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            self.data_clean[top_features].corr(), 
            annot=True, fmt='.2f', cmap=cmap,
            center=0, square=True, linewidths=1, 
            cbar_kws={"shrink": 0.8}, ax=ax
        )
        ax.set_title(
            f'Correlation Matrix - Top {top_n} Features', 
            fontsize=16,pad=20
        )
        plt.tight_layout()
        return fig
    
    def plot_scatter_regression(self, x_feature, y_feature=None, figsize=(10, 6)):
        """Plot scatter plot with regression line."""
        if self.data_clean is None:
            raise ValueError("Data not available!")
        
        if y_feature is None:
            target_cols = [c for c in self.data_clean.columns if 'female' in c.lower() and 'ownership' in c.lower()]
            if target_cols:
                y_feature = target_cols[0]
            else:
                raise ValueError("Target column not found!")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(
            self.data_clean[x_feature], 
            self.data_clean[y_feature], 
            alpha=0.5, edgecolors='k', linewidth=0.5
        )
        
        z = np.polyfit(self.data_clean[x_feature], self.data_clean[y_feature], 1)
        p = np.poly1d(z)
        ax.plot(
            self.data_clean[x_feature].sort_values(), 
            p(self.data_clean[x_feature].sort_values()), 
            "r--", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}'
        )
        
        corr = self.data_clean[[x_feature, y_feature]].corr().iloc[0, 1]
        
        ax.set_xlabel(x_feature, fontsize=12)
        ax.set_ylabel(y_feature, fontsize=12)
        ax.set_title(
            f'{x_feature} vs {y_feature}\nCorrelation: {corr:.3f}', 
            fontsize=14, pad=15
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, model_name='Random Forest', top_n=15, figsize=(12, 8)):
        """Plot feature importance from model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found!")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' doesn't have feature_importances_!")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        
        bars = ax.barh(range(len(importance_df)), importance_df['Importance'], color=colors)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(
            f'Top {top_n} Most Important Features - {model_name}', 
            fontsize=14, pad=15
        )
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        for idx, (bar, value) in enumerate(zip(bars, importance_df['Importance'])):
            ax.text(value, idx, f' {value:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        return importance_df
    
    def plot_model_comparison(self, figsize=(15, 5)):
        """Compare performance of all trained models."""
        if not self.results:
            raise ValueError("No models trained! Run train_models() first.")
        
        comparison = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # R² Score
        ax = axes[0]
        x = range(len(comparison))
        width = 0.35
        ax.bar([i-width/2 for i in x], comparison['train_r2'], width, label='Train', alpha=0.8)
        ax.bar([i+width/2 for i in x], comparison['test_r2'], width, label='Test', alpha=0.8)
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score - Train vs Test')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison.index, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # MAE
        ax = axes[1]
        ax.bar([i-width/2 for i in x], comparison['train_mae'], width, label='Train', alpha=0.8)
        ax.bar([i+width/2 for i in x], comparison['test_mae'], width, label='Test', alpha=0.8)
        ax.set_ylabel('MAE')
        ax.set_title('Mean Absolute Error')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison.index, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # RMSE
        ax = axes[2]
        ax.bar([i-width/2 for i in x], comparison['train_rmse'], width, label='Train', alpha=0.8)
        ax.bar([i+width/2 for i in x], comparison['test_rmse'], width, label='Test', alpha=0.8)
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Squared Error')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison.index, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        return comparison
    
    def get_summary_statistics(self):
        """Return descriptive statistics of dataset."""
        if self.data_clean is None:
            raise ValueError("Data not available!")
        
        return self.data_clean.describe()
 

def plot_correlation_analysis(X, y, features, feature_category, top_n=3, figsize=(18, 5), color='steelblue'):
    """
    Plot correlation analysis with scatter plots and regression lines.
    
    This function creates a standardized visualization showing the relationship
    between selected features and the target variable, including correlation
    coefficients and regression lines.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix containing the independent variables
    y : pd.Series
        Target variable (dependent variable)
    features : list
        List of feature names to analyze (must be columns in X)
    feature_category : str
        Category name for the title (e.g., 'Educational Factors', 'Economic Indicators')
    top_n : int, default=3
        Number of top correlated features to display
    figsize : tuple, default=(18, 5)
        Figure size as (width, height)
    color : str, default='steelblue'
        Color for scatter points
        
    Returns:
    --------
    correlation_df : pd.DataFrame
        DataFrame with features and their correlation values, sorted by absolute correlation
    """
    
    # Calculate correlations
    correlation_df = pd.DataFrame({
        'Feature': features,
        'Correlation': [X[feat].corr(y) for feat in features]
    })
    
    # Sort by absolute correlation
    correlation_df['abs_corr'] = correlation_df['Correlation'].abs()
    correlation_df = correlation_df.sort_values('abs_corr', ascending=False)
    
    # Select top N features
    top_features = correlation_df.nlargest(min(top_n, len(correlation_df)), 'abs_corr')['Feature'].values
    n_features = len(top_features)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    if n_features == 1:
        axes = [axes]
    
    # Plot each feature
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(X[feature], y, alpha=0.6, edgecolors='k', linewidth=0.5, s=50, color=color)
        
        # Regression line
        mask = X[feature].notna() & y.notna()
        if mask.sum() > 1:
            z = np.polyfit(X[feature][mask], y[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(X[feature][mask].min(), X[feature][mask].max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2.5, 
                   label=f'y={z[0]:.3f}x+{z[1]:.1f}')
        
        # Get correlation
        corr = X[feature].corr(y)
        
        # Styling
        ax.set_xlabel(feature, fontsize=11, fontweight='bold')
        ax.set_ylabel(y.name if hasattr(y, 'name') else 'Target', fontsize=11)
        ax.set_title(f'Correlation: {corr:.3f}', fontsize=12, pad=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Main title
    plt.tight_layout()
    plt.show()
    
    # Print correlation summary
    print(f"{feature_category.upper()} - CORRELATION SUMMARY")
    print(correlation_df[['Feature', 'Correlation']].to_string(index=False))
    
    return correlation_df.drop('abs_corr', axis=1)
