"""
Unit tests for utils module.
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import rolling_z, ewma, vwap, clamp


class TestUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.test_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.02, -0.01, 0.03])
        
    def test_rolling_z(self):
        """Test rolling z-score calculation."""
        # Test with window size 3
        z_scores = rolling_z(self.test_series, window=3)
        
        # Check that z-scores are calculated correctly
        self.assertEqual(len(z_scores), len(self.test_series))
        
        # For window=3, first two values should be NaN or 0 (depending on min_periods)
        # The third value should be a proper z-score
        self.assertFalse(np.isnan(z_scores.iloc[2]))
        
        # Test with different window size
        z_scores_5 = rolling_z(self.test_series, window=5)
        self.assertEqual(len(z_scores_5), len(self.test_series))
        
        # Test with all zeros (should handle division by zero)
        zero_series = pd.Series([0, 0, 0, 0, 0])
        z_zeros = rolling_z(zero_series, window=3)
        self.assertTrue(all(z_zeros == 0))
        
    def test_ewma(self):
        """Test exponential weighted moving average."""
        ewma_result = ewma(self.test_series, span=3)
        
        # Check that result has same length
        self.assertEqual(len(ewma_result), len(self.test_series))
        
        # Check that it's not all NaN
        self.assertFalse(ewma_result.isna().all())
        
        # Test with span=1 (should return original series)
        ewma_1 = ewma(self.test_series, span=1)
        # Convert to same dtype for comparison
        test_series_float = self.test_series.astype(float)
        pd.testing.assert_series_equal(ewma_1, test_series_float)
        
    def test_vwap(self):
        """Test volume weighted average price."""
        prices = pd.Series([100, 101, 102, 103, 104])
        volumes = pd.Series([1000, 2000, 1500, 3000, 1000])
        
        vwap_result = vwap(prices, volumes)
        
        # Check length
        self.assertEqual(len(vwap_result), len(prices))
        
        # Check that first value is the first price
        self.assertEqual(vwap_result.iloc[0], prices.iloc[0])
        
        # Check that last value is the overall VWAP
        expected_vwap = (prices * volumes).sum() / volumes.sum()
        self.assertAlmostEqual(vwap_result.iloc[-1], expected_vwap, places=6)
        
    def test_clamp(self):
        """Test value clamping function."""
        # Test normal clamping
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-1, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)
        
        # Test edge cases
        self.assertEqual(clamp(0, 0, 10), 0)
        self.assertEqual(clamp(10, 0, 10), 10)
        
        # Test with float values
        self.assertAlmostEqual(clamp(5.5, 0.0, 10.0), 5.5, places=6)
        self.assertAlmostEqual(clamp(-0.1, 0.0, 10.0), 0.0, places=6)
        self.assertAlmostEqual(clamp(10.1, 0.0, 10.0), 10.0, places=6)


if __name__ == '__main__':
    unittest.main()
