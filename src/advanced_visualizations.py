# src/advanced_visualizations.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

class AdvancedVisualizations:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_correlation_heatmap(self, output_path: str):
        """
        Create a heatmap of the correlation matrix for numerical variables.
        
        Args:
            output_path (str): Path to save the output image
        """
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.data[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Heatmap of Numerical Variables')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_premium_by_vehicle_and_cover(self, output_path: str):
        """
        Create a grouped bar chart showing average TotalPremium by VehicleType and CoverType.
        
        Args:
            output_path (str): Path to save the output image
        """
        grouped_data = self.data.groupby(['VehicleType', 'CoverType'])['TotalPremium'].mean().unstack()

        ax = grouped_data.plot(kind='bar', figsize=(15, 8), width=0.8)
        plt.title('Average Total Premium by Vehicle Type and Cover Type')
        plt.xlabel('Vehicle Type')
        plt.ylabel('Average Total Premium')
        plt.legend(title='Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', label_type='center')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_claims_by_province(self, output_path: str, shapefile_path: str):
        """
        Create a geographical heatmap of South Africa showing average TotalClaims by Province.
        
        Args:
            output_path (str): Path to save the output image
            shapefile_path (str): Path to the South Africa shapefile
        """
        # Calculate average claims by province
        claims_by_province = self.data.groupby('Province')['TotalClaims'].mean().reset_index()

        # Load South Africa shapefile
        sa_map = gpd.read_file(shapefile_path)

        # Merge shapefile with claims data
        sa_map = sa_map.merge(claims_by_province, how='left', left_on='NAME_1', right_on='Province')

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        sa_map.plot(column='TotalClaims', ax=ax, legend=True, legend_kwds={'label': 'Average Total Claims'},
                    cmap='YlOrRd', missing_kwds={'color': 'lightgrey'})

        plt.title('Average Total Claims by Province in South Africa')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def generate_advanced_plots(data_path: str, output_dir: str, shapefile_path: str):
    """
    Generate all advanced plots.
    
    Args:
        data_path (str): Path to the CSV data file
        output_dir (str): Directory to save output images
        shapefile_path (str): Path to the South Africa shapefile
    """
    data = pd.read_csv(data_path)
    viz = AdvancedVisualizations(data)

    viz.plot_correlation_heatmap(f"{output_dir}/correlation_heatmap.png")
    viz.plot_premium_by_vehicle_and_cover(f"{output_dir}/premium_by_vehicle_and_cover.png")
    viz.plot_claims_by_province(f"{output_dir}/claims_by_province.png", shapefile_path)

if __name__ == "__main__":
    generate_advanced_plots("../resources/Data/machineLearning.txt", "../scripts/output", "../resources/Data/sa_shapefile/sa_shapefile.shp")