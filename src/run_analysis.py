
# scripts/run_analysis.py
import os
import pandas as pd
from src.data_loader import DataLoader
from src.eda import EDA
from src.statistical_analysis import StatisticalAnalysis

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

if __name__ == "__main__":
    main()