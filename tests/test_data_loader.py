# tests/test_data_loader.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
from io import StringIO
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader('../resources/Data/machineLearning.txt')
        self.test_data = pd.DataFrame({
            'TotalPremium': [100, 200, 300, np.nan],
            'TotalClaims': [50, 100, 150, 200],
            'Gender': ['Male', 'Female', 'Male', 'Female']
        })
        self.data_loader.data = self.test_data

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = self.test_data
        result = self.data_loader.load_data()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)
        mock_read_csv.assert_called_once_with('../resources/Data/machineLearning.txt')

    def test_get_summary_statistics(self):
        result = self.data_loader.get_summary_statistics()
        self.assertIsInstance(result, dict)
        self.assertIn('TotalPremium', result)
        self.assertIn('TotalClaims', result)
        self.assertEqual(result['TotalPremium']['count'], 3)
        self.assertEqual(result['TotalClaims']['count'], 4)

    def test_check_missing_values(self):
        result = self.data_loader.check_missing_values()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['TotalPremium'], 1)
        self.assertEqual(result['TotalClaims'], 0)
        self.assertEqual(result['Gender'], 0)

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            DataLoader('non_existent_file.txt').load_data()