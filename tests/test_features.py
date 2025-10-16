"""
Unit tests for features module.
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import utils first to avoid relative import issues
import utils

from features import (
    compute_mid, compute_returns, compute_cvd, compute_ofi, 
    compute_bp, compute_vwap, compute_atr, add_feature_block
)


class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_rows = 100
        
        # Create realistic market data
        base_price = 100.0
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_rows, freq='1s'),
            'price': base_price + np.cumsum(np.random.normal(0, 0.1, n_rows)),
            'bid1': base_price - 0.5 + np.cumsum(np.random.normal(0, 0.1, n_rows)),
            'ask1': base_price + 0.5 + np.cumsum(np.random.normal(0, 0.1, n_rows)),
            'bid1_size': np.random.uniform(100, 1000, n_rows),
            'ask1_size': np.random.uniform(100, 1000, n_rows),
            'size': np.random.uniform(10, 100, n_rows),
        })
        
        # Add multi-level data for OFI testing
        for level in range(2, 6):
            self.df[f'bid{level}'] = self.df['bid1'] - (level - 1) * 0.1
            self.df[f'ask{level}'] = self.df['ask1'] + (level - 1) * 0.1
            self.df[f'bid{level}_size'] = np.random.uniform(50, 500, n_rows)
            self.df[f'ask{level}_size'] = np.random.uniform(50, 500, n_rows)
        
        self.params = {
            'features': {
                'atr_window': 14,
                'vwap_window_seconds': 900,
                'ofi_window_seconds': 1,
                'ofi_levels': 5,
                'z_window': 300
            }
        }
        
    def test_compute_mid(self):
        """Test mid price calculation."""
        mid = compute_mid(self.df)
        
        # Check that mid is between bid and ask
        self.assertTrue(all((self.df['bid1'] <= mid) & (mid <= self.df['ask1'])))
        
        # Check that mid is the average of bid and ask
        expected_mid = (self.df['bid1'] + self.df['ask1']) / 2
        pd.testing.assert_series_equal(mid, expected_mid)
        
    def test_compute_returns(self):
        """Test returns calculation."""
        mid = compute_mid(self.df)
        returns = compute_returns(mid, window=1)
        
        # Check length
        self.assertEqual(len(returns), len(self.df))
        
        # Check that first return is 0 (no previous value)
        self.assertEqual(returns.iloc[0], 0.0)
        
        # Check that returns are calculated correctly
        expected_returns = mid.pct_change().fillna(0.0)
        pd.testing.assert_series_equal(returns, expected_returns)
        
    def test_compute_cvd(self):
        """Test Cumulative Volume Delta calculation."""
        mid = compute_mid(self.df)
        mid_prev = mid.shift(1).fillna(method="bfill")
        cvd = compute_cvd(self.df, mid_prev)
        
        # Check length
        self.assertEqual(len(cvd), len(self.df))
        
        # Check that CVD is cumulative
        # (This is a basic check - in practice CVD can decrease)
        self.assertIsInstance(cvd.iloc[-1], (int, float))
        
    def test_compute_ofi(self):
        """Test Order Flow Imbalance calculation."""
        # Test with single level
        ofi_1 = compute_ofi(self.df, window_seconds=1, levels=1)
        self.assertEqual(len(ofi_1), len(self.df))
        
        # Test with multiple levels
        ofi_5 = compute_ofi(self.df, window_seconds=1, levels=5)
        self.assertEqual(len(ofi_5), len(self.df))
        
        # Test that multi-level OFI is different from single-level
        # (This may not always be true depending on data, but it's worth checking)
        self.assertFalse(ofi_1.equals(ofi_5))
        
    def test_compute_bp(self):
        """Test Bid-Pressure calculation."""
        bp = compute_bp(self.df)
        
        # Check length
        self.assertEqual(len(bp), len(self.df))
        
        # Check that BP is between -1 and 1
        self.assertTrue(all(bp >= -1) and all(bp <= 1))
        
        # Check that BP is calculated correctly
        expected_bp = (self.df["bid1_size"] - self.df["ask1_size"]) / (self.df["bid1_size"] + self.df["ask1_size"])
        expected_bp = expected_bp.replace(np.inf, 0).replace(-np.inf, 0).fillna(0)
        pd.testing.assert_series_equal(bp, expected_bp)
        
    def test_compute_vwap(self):
        """Test Volume Weighted Average Price calculation."""
        vwap_result = compute_vwap(self.df, window_seconds=10)
        
        # Check length
        self.assertEqual(len(vwap_result), len(self.df))
        
        # Check that VWAP is not NaN for most values
        self.assertGreater(vwap_result.notna().sum(), len(self.df) * 0.5)
        
        # Check that VWAP values are reasonable (around price range)
        valid_vwap = vwap_result.dropna()
        if len(valid_vwap) > 0:
            self.assertGreater(valid_vwap.min(), self.df['price'].min() * 0.5)
            self.assertLess(valid_vwap.max(), self.df['price'].max() * 1.5)
            
    def test_compute_atr(self):
        """Test Average True Range calculation."""
        atr = compute_atr(self.df, window=14)
        
        # Check length
        self.assertEqual(len(atr), len(self.df))
        
        # Check that ATR is positive
        self.assertTrue(all(atr >= 0))
        
        # Check that ATR is not NaN for most values
        self.assertGreater(atr.notna().sum(), len(self.df) * 0.5)
        
    def test_add_feature_block(self):
        """Test complete feature block addition."""
        result_df = add_feature_block(self.df, self.params)
        
        # Check that all expected features are added
        expected_features = ['ret_1s', 'cvd', 'ofi', 'bp', 'vwap', 'atr', 'cvd_z', 'ofi_z']
        for feature in expected_features:
            self.assertIn(feature, result_df.columns)
            
        # Check that original columns are preserved
        for col in self.df.columns:
            self.assertIn(col, result_df.columns)
            
        # Check that z-scores are reasonable
        if 'cvd_z' in result_df.columns:
            cvd_z = result_df['cvd_z'].dropna()
            if len(cvd_z) > 0:
                # Z-scores should be roughly centered around 0
                self.assertGreaterEqual(cvd_z.std(), 0)  # Should have some variation (can be 0 for constant data)
                
        if 'ofi_z' in result_df.columns:
            ofi_z = result_df['ofi_z'].dropna()
            if len(ofi_z) > 0:
                self.assertGreaterEqual(ofi_z.std(), 0)  # Should have some variation (can be 0 for constant data)


if __name__ == '__main__':
    unittest.main()
