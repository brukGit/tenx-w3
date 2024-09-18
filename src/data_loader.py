import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.impute import SimpleImputer

class DataLoader:
    def __init__(self, file_path: str, imputation_strategy='mean'):
        self.file_path = file_path
        self.data = None
        self.imputation_strategy = imputation_strategy

      
    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path, sep='|')
            return self.handle_missing_data(self.data)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def handle_missing_data(self, df):
        # Drop fully empty columns
        df = df.dropna(axis=1, how='all')

        # Numeric columns: Impute with mean or median
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if self.imputation_strategy == 'mean':
            numeric_imputer = SimpleImputer(strategy='mean')
        elif self.imputation_strategy == 'median':
            numeric_imputer = SimpleImputer(strategy='median')

        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

        # Categorical columns: Impute with 'Unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

        return df

    def _preprocess_data(self):
        """
        Perform initial preprocessing on the loaded data.
        """
        # Convert date column to datetime
        self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'])

        # Handle missing values
        numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns

        # For numeric columns, fill missing values with median
        for col in numeric_columns:
            self.data[col] = self.data[col].fillna(self.data[col].median())

        # For categorical columns, fill missing values with 'Unknown'
        for col in categorical_columns:
            self.data[col] = self.data[col].fillna('Unknown')

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for numerical columns.
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.data.describe().to_dict()

    def check_missing_values(self) -> Dict[str, int]:
        """
        Check for missing values in each column.
        
        Returns:
            Dict[str, int]: Missing value counts for each column
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.data.isnull().sum().to_dict()