# scripts/run_analysis.py
import os
import sys
import pandas as pd

# Add the src directory to the Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_dir)

from data_loader import DataLoader
from eda import EDA
from statistical_analysis import StatisticalAnalysis
from advanced_visualizations import AdvancedVisualizations

def main():
    # Load data
    data_loader = DataLoader('../resources/Data/machineLearning.txt')
    data = data_loader.load_data()

    # Perform EDA
    eda = EDA(data)
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    eda.plot_histograms(['TotalPremium', 'TotalClaims'], output_dir)
    eda.plot_correlation_matrix(output_dir)
    eda.plot_boxplots(['TotalPremium', 'TotalClaims'], output_dir)

    # Perform statistical analysis
    stats_analysis = StatisticalAnalysis(data)
    
    ci_premium = stats_analysis.calculate_confidence_interval('TotalPremium')
    print(f"95% Confidence Interval for TotalPremium: {ci_premium}")

    ttest_result = stats_analysis.perform_ttest('TotalPremium', 'TotalClaims')
    print(f"T-test result between TotalPremium and TotalClaims: {ttest_result}")

    correlation = stats_analysis.calculate_correlation('TotalPremium', 'TotalClaims')
    print(f"Correlation between TotalPremium and TotalClaims: {correlation}")

    # Generate advanced plots
    gap = AdvancedVisualizations(data)
    gap.generate_advanced_plots('../resources/Data/machineLearning.txt', 'output', '../notebooks/sa_shapefile.shp')

if __name__ == "__main__":
    main()