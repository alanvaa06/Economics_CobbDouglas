"""
Unit tests for core economic modeling functions.
"""

import pytest
import numpy as np
from src.economics_models.core import (
    SolowModel,
    FinancialAnalyzer,
    MonteCarloSimulator,
    SolowParameters,
    FinancialParameters,
    MonteCarloParameters,
    calculate_pe_ratio_sensitivity
)


class TestSolowParameters:
    """Test SolowParameters dataclass."""
    
    def test_default_values(self):
        """Test default parameter values."""
        params = SolowParameters()
        assert params.s == 0.20
        assert params.n == 0.005
        assert params.g == 0.02
        assert params.delta == 0.05
        assert params.alpha == 0.35
        assert params.A == 1
        assert params.L == 100
        assert params.K == 10000


class TestSolowModel:
    """Test SolowModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = SolowParameters()
        self.model = SolowModel(self.params)
    
    def test_init(self):
        """Test model initialization."""
        self.setUp()
        assert self.model.params == self.params
    
    def test_production_function(self):
        """Test Cobb-Douglas production function."""
        self.setUp()
        K, A, L = 1000, 1, 100
        Y = self.model.production_function(K, A, L)
        expected = (K ** 0.35) * (A * L) ** 0.65
        assert np.isclose(Y, expected)
    
    def test_next_period_capital(self):
        """Test capital accumulation equation."""
        self.setUp()
        k, y = 1000, 500
        next_k = self.model.next_period_capital(k, y)
        
        investment = self.params.s * y
        depreciation = self.params.delta * k
        expected = (1 + self.params.n + self.params.g) * (k + investment - depreciation)
        assert np.isclose(next_k, expected)
    
    def test_simulation_length(self):
        """Test simulation returns correct array lengths."""
        self.setUp()
        T = 100
        K_t, Y_t, A_t, L_t, growth_rates = self.model.simulate(T)
        
        assert len(K_t) == T
        assert len(Y_t) == T
        assert len(A_t) == T
        assert len(L_t) == T
        assert len(growth_rates) == T - 1
    
    def test_get_final_growth_rate(self):
        """Test final growth rate calculation."""
        self.setUp()
        growth_rate = self.model.get_final_growth_rate(T=50)
        assert isinstance(growth_rate, (int, float))
        assert not np.isnan(growth_rate)


class TestFinancialAnalyzer:
    """Test FinancialAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = FinancialParameters()
        self.analyzer = FinancialAnalyzer(self.params)
    
    def test_init(self):
        """Test analyzer initialization."""
        self.setUp()
        assert self.analyzer.params == self.params
    
    def test_calculate_required_return(self):
        """Test CAPM required return calculation."""
        self.setUp()
        risk_free_rate = 0.03
        required_return = self.analyzer.calculate_required_return(risk_free_rate)
        expected = risk_free_rate + self.params.beta * self.params.equity_risk_premium
        assert np.isclose(required_return, expected)
    
    def test_calculate_justified_pe_ratio_valid(self):
        """Test P/E ratio calculation with valid inputs."""
        self.setUp()
        real_growth_rate = 0.02
        pe_ratio, metrics = self.analyzer.calculate_justified_pe_ratio(real_growth_rate)
        
        assert isinstance(pe_ratio, (int, float))
        assert not np.isnan(pe_ratio)
        assert pe_ratio > 0
        assert 'real_growth_rate' in metrics
        assert 'nominal_growth_rate' in metrics
        assert 'risk_free_rate' in metrics
        assert 'required_return' in metrics
        assert 'earnings_growth_rate' in metrics
    
    def test_calculate_justified_pe_ratio_invalid(self):
        """Test P/E ratio calculation with invalid inputs (growth >= required return)."""
        self.setUp()
        # Use very high growth rate to trigger invalid condition
        real_growth_rate = 0.10
        pe_ratio, metrics = self.analyzer.calculate_justified_pe_ratio(real_growth_rate)
        
        assert np.isnan(pe_ratio)
        assert 'earnings_growth_rate' in metrics
        assert 'required_return' in metrics


class TestMonteCarloSimulator:
    """Test MonteCarloSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.solow_params = SolowParameters()
        self.financial_params = FinancialParameters()
        self.mc_params = MonteCarloParameters(num_simulations=10, num_months=2)
        self.simulator = MonteCarloSimulator(
            self.solow_params, 
            self.financial_params, 
            self.mc_params
        )
    
    def test_init(self):
        """Test simulator initialization."""
        self.setUp()
        assert self.simulator.solow_params == self.solow_params
        assert self.simulator.financial_params == self.financial_params
        assert self.simulator.mc_params == self.mc_params
    
    def test_run_simulation_shape(self):
        """Test simulation returns correct array shapes."""
        self.setUp()
        pe_ratios, earnings_growth_rates = self.simulator.run_simulation()
        
        expected_shape = (self.mc_params.num_simulations, self.mc_params.num_months)
        assert pe_ratios.shape == expected_shape
        assert earnings_growth_rates.shape == expected_shape
    
    @pytest.mark.slow
    def test_run_simulation_values(self):
        """Test simulation returns reasonable values."""
        self.setUp()
        pe_ratios, earnings_growth_rates = self.simulator.run_simulation()
        
        # Filter out NaN values for testing
        valid_pe_ratios = pe_ratios[~np.isnan(pe_ratios)]
        valid_earnings_rates = earnings_growth_rates[~np.isnan(earnings_growth_rates)]
        
        if len(valid_pe_ratios) > 0:
            assert np.all(valid_pe_ratios > 0)
            assert np.all(valid_pe_ratios < 1000)  # Reasonable upper bound
        
        if len(valid_earnings_rates) > 0:
            assert np.all(valid_earnings_rates > -1)  # Reasonable lower bound
            assert np.all(valid_earnings_rates < 1)   # Reasonable upper bound


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_calculate_pe_ratio_sensitivity(self):
        """Test P/E ratio sensitivity analysis."""
        solow_params = SolowParameters()
        financial_params = FinancialParameters()
        g_range = np.array([0.01, 0.02, 0.03])
        
        pe_ratios = calculate_pe_ratio_sensitivity(g_range, solow_params, financial_params)
        
        assert len(pe_ratios) == len(g_range)
        assert all(isinstance(pe, (int, float)) or np.isnan(pe) for pe in pe_ratios)


if __name__ == "__main__":
    pytest.main([__file__])
