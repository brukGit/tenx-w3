# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

class EDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_histograms(self, columns: List[str], output_dir: str):
        """
        Plot histograms for specified numerical columns.
        
        Args:
            columns (List[str]): List of column names to plot
            output_dir (str): Directory to save the plots
        """
        for col in columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(f'{output_dir}/{col}_histogram.png')
            plt.close()

    def plot_correlation_matrix(self, output_dir: str):
        """
        Plot correlation matrix for numerical columns.
        
        Args:
            output_dir (str): Directory to save the plot
        """
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig(f'{output_dir}/correlation_matrix.png')
        plt.close()

    def plot_boxplots(self, columns: List[str], output_dir: str):
        """
        Plot boxplots for specified numerical columns to detect outliers.
        
        Args:
            columns (List[str]): List of column names to plot
            output_dir (str): Directory to save the plots
        """
        for col in columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data[col])
            plt.title(f'Boxplot of {col}')
            plt.savefig(f'{output_dir}/{col}_boxplot.png')
            plt.close()