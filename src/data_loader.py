# src/data_loader.py
import pandas as pd
from typing import Dict, Any

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the data from the CSV file.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        self.data = pd.read_csv(self.file_path, sep='|')
        return self.data

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