import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, List

class HypothesisTesting:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def calculate_risk_ratio(self, group: pd.Series) -> float:
        """
        Calculate the risk ratio for a group.

        Args:
            group (pd.Series): Group data

        Returns:
            float: Risk ratio
        """
        return group['TotalClaims'].sum() / group['TotalPremium'].sum()

    def test_risk_difference(self, feature: str) -> Tuple[float, float]:
        """
        Perform chi-square test for risk differences across categories of a feature.

        Args:
            feature (str): Name of the feature to test

        Returns:
            Tuple[float, float]: Chi-square statistic and p-value
        """
        contingency_table = pd.crosstab(self.data[feature], self.data['TotalClaims'] > 0)
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        return chi2, p_value

    def test_margin_difference(self, feature: str) -> Tuple[float, float]:
        """
        Perform ANOVA test for margin differences across categories of a feature.

        Args:
            feature (str): Name of the feature to test

        Returns:
            Tuple[float, float]: F-statistic and p-value
        """
        groups = [group['TotalPremium'] - group['TotalClaims'] for _, group in self.data.groupby(feature)]
        f_statistic, p_value = stats.f_oneway(*groups)
        return f_statistic, p_value

    def test_gender_risk_difference(self) -> Tuple[float, float]:
        """
        Perform t-test for risk differences between Women and Men.

        Returns:
            Tuple[float, float]: T-statistic and p-value
        """
        women_risk = self.calculate_risk_ratio(self.data[self.data['Gender'] == 'F'])
        men_risk = self.calculate_risk_ratio(self.data[self.data['Gender'] == 'M'])
        t_statistic, p_value = stats.ttest_ind(women_risk, men_risk)
        return t_statistic, p_value

    def run_all_tests(self) -> List[dict]:
        """
        Run all hypothesis tests and return results.

        Returns:
            List[dict]: List of test results
        """
        results = []

        # Test 1: Risk differences across provinces
        chi2, p_value = self.test_risk_difference('Province')
        results.append({
            'test': 'Risk differences across provinces',
            'statistic': chi2,
            'p_value': p_value,
            'reject_null': p_value < 0.05
        })

        # Test 2: Risk differences between zip codes
        chi2, p_value = self.test_risk_difference('PostalCode')
        results.append({
            'test': 'Risk differences between zip codes',
            'statistic': chi2,
            'p_value': p_value,
            'reject_null': p_value < 0.05
        })

        # Test 3: Margin differences between zip codes
        f_statistic, p_value = self.test_margin_difference('PostalCode')
        results.append({
            'test': 'Margin differences between zip codes',
            'statistic': f_statistic,
            'p_value': p_value,
            'reject_null': p_value < 0.05
        })

        # Test 4: Risk differences between Women and Men
        t_statistic, p_value = self.test_gender_risk_difference()
        results.append({
            'test': 'Risk differences between Women and Men',
            'statistic': t_statistic,
            'p_value': p_value,
            'reject_null': p_value < 0.05
        })

        return results