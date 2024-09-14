# tests/test_eda.py
import unittest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from src.eda import EDA

class TestEDA(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame({
            'TotalPremium': [100, 200, 300, 400, 500],
            'TotalClaims': [50, 100, 150, 200, 250],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male']
        })
        self.eda = EDA(self.test_data)

    @patch('matplotlib.pyplot.savefig')
    def test_plot_histograms(self, mock_savefig):
        self.eda.plot_histograms(['TotalPremium', 'TotalClaims'], 'test_output')
        self.assertEqual(mock_savefig.call_count, 2)
        mock_savefig.assert_any_call('test_output/TotalPremium_histogram.png')
        mock_savefig.assert_any_call('test_output/TotalClaims_histogram.png')

    @patch('matplotlib.pyplot.savefig')
    def test_plot_correlation_matrix(self, mock_savefig):
        self.eda.plot_correlation_matrix('test_output')
        mock_savefig.assert_called_once_with('test_output/correlation_matrix.png')

    @patch('matplotlib.pyplot.savefig')
    def test_plot_boxplots(self, mock_savefig):
        self.eda.plot_boxplots(['TotalPremium', 'TotalClaims'], 'test_output')
        self.assertEqual(mock_savefig.call_count, 2)
        mock_savefig.assert_any_call('test_output/TotalPremium_boxplot.png')
        mock_savefig.assert_any_call('test_output/TotalClaims_boxplot.png')

    def test_eda_with_empty_dataframe(self):
        empty_eda = EDA(pd.DataFrame())
        with self.assertRaises(ValueError):
            empty_eda.plot_histograms(['Column1'], 'test_output')
