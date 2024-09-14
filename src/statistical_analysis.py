
# src/statistical_analysis.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple

class StatisticalAnalysis:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def calculate_confidence_interval(self, column: str, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate the confidence interval for a given column.
        
        Args:
            column (str): Name of the column
            confidence (float): Confidence level (default: 0.95)
        
        Returns:
            Tuple[float, float]: Lower and upper bounds of the confidence interval
        """
        data = self.data[column].dropna()
        return stats.t.interval(confidence, len(data)-1, loc=np.mean(data), scale=stats.sem(data))

    def perform_ttest(self, column1: str, column2: str) -> Tuple[float, float]:
        """
        Perform a t-test between two columns.
        
        Args:
            column1 (str): Name of the first column
            column2 (str): Name of the second column
        
        Returns:
            Tuple[float, float]: T-statistic and p-value
        """
        return stats.ttest_ind(self.data[column1].dropna(), self.data[column2].dropna())

    def calculate_correlation(self, column1: str, column2: str) -> float:
        """
        Calculate the Pearson correlation coefficient between two columns.
        
        Args:
            column1 (str): Name of the first column
            column2 (str): Name of the second column
        
        Returns:
            float: Correlation coefficient
        """
        return self.data[[column1, column2]].corr().iloc[0, 1]