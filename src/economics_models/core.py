"""
Core economic modeling functions for Solow growth model and financial analysis.

This module contains the fundamental economic modeling functions extracted from 
the Jupyter notebook, organized following best practices for maintainable code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SolowParameters:
    """Parameters for the Solow growth model."""
    s: float = 0.20          # Savings rate
    n: float = 0.005         # Population growth rate
    g: float = 0.02          # Technological progress rate
    delta: float = 0.05      # Depreciation rate
    alpha: float = 0.35      # Capital output elasticity
    A: float = 1             # Initial technology level
    L: float = 100           # Initial labor force
    K: float = 10000         # Initial capital stock


@dataclass
class FinancialParameters:
    """Parameters for financial analysis and P/E ratio calculations."""
    beta: float = 1.0                    # Beta (systematic risk)
    equity_risk_premium: float = 0.02    # Equity risk premium
    expected_inflation: float = 0.02     # Expected inflation rate
    term_premium: float = 0.01           # Term premium
    retention_rate: float = 0.35         # Earnings retention rate
    tfp: float = 1.1                     # Total factor productivity


@dataclass
class MonteCarloParameters:
    """Parameters for Monte Carlo simulation."""
    num_simulations: int = 1000
    num_months: int = 12
    g_mean: float = 0.02
    g_std: float = 0.005
    inflation_mean: float = 0.02
    inflation_std: float = 0.01
    term_premium_mean: float = 0.005
    term_premium_std: float = 0.002


class SolowModel:
    """Solow growth model implementation."""
    
    def __init__(self, parameters: SolowParameters):
        """Initialize the Solow model with given parameters."""
        self.params = parameters
    
    def next_period_capital(self, k: float, y: float) -> float:
        """
        Calculate next period capital stock.
        
        Args:
            k: Current capital stock
            y: Current output
            
        Returns:
            Next period capital stock
        """
        investment = self.params.s * y
        depreciation = self.params.delta * k
        next_k = (1 + self.params.n + self.params.g) * (k + investment - depreciation)
        return next_k
    
    def production_function(self, K: float, A: float, L: float) -> float:
        """
        Cobb-Douglas production function.
        
        Args:
            K: Capital stock
            A: Technology level
            L: Labor force
            
        Returns:
            Output (Y)
        """
        return (K ** self.params.alpha) * (A * L) ** (1 - self.params.alpha)
    
    def simulate(self, T: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Solow model simulation.
        
        Args:
            T: Number of time periods
            
        Returns:
            Tuple of (K_t, Y_t, A_t, L_t, growth_rates)
        """
        # Initialize arrays
        K_t = np.zeros(T)
        Y_t = np.zeros(T)
        A_t = np.zeros(T)
        L_t = np.zeros(T)
        growth_rates = np.zeros(T - 1)
        
        # Set initial values
        K_t[0] = self.params.K
        A_t[0] = self.params.A
        L_t[0] = self.params.L
        Y_t[0] = self.production_function(K_t[0], A_t[0], L_t[0])
        
        # Run simulation
        for t in range(1, T):
            K_t[t] = self.next_period_capital(K_t[t-1], Y_t[t-1])
            L_t[t] = L_t[t-1] * (1 + self.params.n)
            A_t[t] = A_t[t-1] * (1 + self.params.g)
            Y_t[t] = self.production_function(K_t[t], A_t[t], L_t[t])
            growth_rates[t-1] = (Y_t[t] - Y_t[t-1]) / Y_t[t-1] * 100
        
        return K_t, Y_t, A_t, L_t, growth_rates
    
    def get_final_growth_rate(self, T: int = 100) -> float:
        """
        Get the final growth rate from simulation.
        
        Args:
            T: Number of time periods
            
        Returns:
            Final growth rate as percentage
        """
        _, _, _, _, growth_rates = self.simulate(T)
        return growth_rates[-1]


class FinancialAnalyzer:
    """Financial analysis functions for valuation models."""
    
    def __init__(self, financial_params: FinancialParameters):
        """Initialize with financial parameters."""
        self.params = financial_params
    
    def calculate_required_return(self, risk_free_rate: float) -> float:
        """
        Calculate required return using CAPM.
        
        Args:
            risk_free_rate: Risk-free rate
            
        Returns:
            Required return
        """
        return risk_free_rate + self.params.beta * self.params.equity_risk_premium
    
    def calculate_justified_pe_ratio(self, 
                                   real_growth_rate: float,
                                   earnings_growth_factor: Optional[float] = None) -> Tuple[float, dict]:
        """
        Calculate justified forward P/E ratio.
        
        Args:
            real_growth_rate: Real economic growth rate
            earnings_growth_factor: Factor to adjust earnings growth (default: tfp)
            
        Returns:
            Tuple of (justified_pe_ratio, metrics_dict)
        """
        if earnings_growth_factor is None:
            earnings_growth_factor = self.params.tfp
            
        nominal_growth_rate = real_growth_rate + self.params.expected_inflation
        risk_free_rate = real_growth_rate + self.params.expected_inflation + self.params.term_premium
        required_return = self.calculate_required_return(risk_free_rate)
        earnings_growth_rate = nominal_growth_rate * earnings_growth_factor
        
        metrics = {
            'real_growth_rate': real_growth_rate,
            'nominal_growth_rate': nominal_growth_rate,
            'risk_free_rate': risk_free_rate,
            'required_return': required_return,
            'earnings_growth_rate': earnings_growth_rate
        }
        
        if earnings_growth_rate >= required_return:
            return np.nan, metrics
        
        justified_pe_ratio = (1 - self.params.retention_rate) / (required_return - earnings_growth_rate)
        return justified_pe_ratio, metrics


class MonteCarloSimulator:
    """Monte Carlo simulation for financial risk analysis."""
    
    def __init__(self, 
                 solow_params: SolowParameters, 
                 financial_params: FinancialParameters,
                 mc_params: MonteCarloParameters):
        """Initialize with all parameter sets."""
        self.solow_params = solow_params
        self.financial_params = financial_params
        self.mc_params = mc_params
        self.solow_model = SolowModel(solow_params)
        self.financial_analyzer = FinancialAnalyzer(financial_params)
    
    def run_simulation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Monte Carlo simulation for P/E ratios and earnings growth rates.
        
        Returns:
            Tuple of (pe_ratios, earnings_growth_rates) arrays
        """
        pe_ratios = np.zeros((self.mc_params.num_simulations, self.mc_params.num_months))
        earnings_growth_rates = np.zeros((self.mc_params.num_simulations, self.mc_params.num_months))
        
        for i in range(self.mc_params.num_simulations):
            for month in range(self.mc_params.num_months):
                # Sample random variables
                g = np.random.normal(self.mc_params.g_mean, self.mc_params.g_std)
                inflation = np.random.normal(self.mc_params.inflation_mean, self.mc_params.inflation_std)
                term_premium = np.random.normal(self.mc_params.term_premium_mean, self.mc_params.term_premium_std)
                
                # Update parameters
                temp_solow_params = SolowParameters(
                    s=self.solow_params.s,
                    n=self.solow_params.n,
                    g=g,
                    delta=self.solow_params.delta,
                    alpha=self.solow_params.alpha,
                    A=self.solow_params.A,
                    L=self.solow_params.L,
                    K=self.solow_params.K
                )
                
                temp_financial_params = FinancialParameters(
                    beta=self.financial_params.beta,
                    equity_risk_premium=self.financial_params.equity_risk_premium,
                    expected_inflation=inflation,
                    term_premium=term_premium,
                    retention_rate=self.financial_params.retention_rate,
                    tfp=self.financial_params.tfp
                )
                
                # Calculate metrics
                temp_solow_model = SolowModel(temp_solow_params)
                temp_financial_analyzer = FinancialAnalyzer(temp_financial_params)
                
                real_growth_rate = temp_solow_model.get_final_growth_rate() / 100
                pe_ratio, metrics = temp_financial_analyzer.calculate_justified_pe_ratio(real_growth_rate)
                
                pe_ratios[i, month] = pe_ratio
                earnings_growth_rates[i, month] = metrics['earnings_growth_rate']
        
        return pe_ratios, earnings_growth_rates


def calculate_pe_ratio_sensitivity(g_range: np.ndarray, 
                                 solow_params: SolowParameters,
                                 financial_params: FinancialParameters) -> List[float]:
    """
    Calculate P/E ratio sensitivity to technological growth rate.
    
    Args:
        g_range: Array of technological growth rates to test
        solow_params: Solow model parameters
        financial_params: Financial parameters
        
    Returns:
        List of P/E ratios for each growth rate
    """
    pe_ratios = []
    financial_analyzer = FinancialAnalyzer(financial_params)
    
    for g in g_range:
        temp_params = SolowParameters(
            s=solow_params.s,
            n=solow_params.n,
            g=g,
            delta=solow_params.delta,
            alpha=solow_params.alpha,
            A=solow_params.A,
            L=solow_params.L,
            K=solow_params.K
        )
        
        solow_model = SolowModel(temp_params)
        real_growth_rate = solow_model.get_final_growth_rate() / 100
        pe_ratio, _ = financial_analyzer.calculate_justified_pe_ratio(real_growth_rate)
        pe_ratios.append(pe_ratio)
    
    return pe_ratios
