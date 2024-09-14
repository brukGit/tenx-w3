# tests/test_statistical_analysis.py
import unittest
import pandas as pd
import numpy as np
from scipy import stats
from src.statistical_analysis import StatisticalAnalysis

class TestStatisticalAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame({
            'TotalPremium': [100, 200, 300, 400, 500],
            'TotalClaims': [50, 100, 150, 200, 250],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male']
        })
        self.stats_analysis = StatisticalAnalysis(self.test_data)

    def test_calculate_confidence_interval(self):
        result = self.stats_analysis.calculate_confidence_interval('TotalPremium')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertLess(result[0], result[1])
        
        # Test with different confidence level
        result_90 = self.stats_analysis.calculate_confidence_interval('TotalPremium', confidence=0.90)
        self.assertNotEqual(result, result_90)

    def test_perform_ttest(self):
        result = self.stats_analysis.perform_ttest('TotalPremium', 'TotalClaims')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)  # t-statistic
        self.assertIsInstance(result[1], float)  # p-value

        # Test with same column (should return nan for t-statistic and 1 for p-value)
        result_same = self.stats_analysis.perform_ttest('TotalPremium', 'TotalPremium')
        self.assertTrue(np.isnan(result_same[0]))
        self.assertEqual(result_same[1], 1.0)

    def test_calculate_correlation(self):
        result = self.stats_analysis.calculate_correlation('TotalPremium', 'TotalClaims')
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, -1)
        self.assertLessEqual(result, 1)

        # Test perfect correlation (same column)
        result_perfect = self.stats_analysis.calculate_correlation('TotalPremium', 'TotalPremium')
        self.assertAlmostEqual(result_perfect, 1.0)

    def test_statistical_analysis_with_invalid_column(self):
        with self.assertRaises(KeyError):
            self.stats_analysis.calculate_confidence_interval('NonExistentColumn')

if __name__ == '__main__':
    unittest.main()