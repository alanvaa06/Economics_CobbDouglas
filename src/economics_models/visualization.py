"""
Visualization functions for economic models and financial analysis.

This module contains plotting and visualization functions extracted from 
the Jupyter notebook, organized for reusable data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from .core import SolowModel, FinancialAnalyzer, MonteCarloSimulator
from .core import SolowParameters, FinancialParameters, MonteCarloParameters


class EconomicVisualizer:
    """Visualization class for economic models."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """Initialize visualizer with default figure size."""
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_solow_simulation(self, 
                            solow_model: SolowModel, 
                            T: int = 1000,
                            show_all_variables: bool = False) -> None:
        """
        Plot Solow model simulation results.
        
        Args:
            solow_model: Configured Solow model instance
            T: Number of time periods
            show_all_variables: Whether to show all variables or just growth rate
        """
        K_t, Y_t, A_t, L_t, growth_rates = solow_model.simulate(T)
        
        if show_all_variables:
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            plt.plot(np.arange(T), K_t, label='Capital Stock', color='blue')
            plt.plot(np.arange(T), Y_t, label='Output', color='red')
            plt.title('Capital and Output over Time')
            plt.xlabel('Time')
            plt.ylabel('Levels')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 2)
            plt.plot(np.arange(T), A_t, label='Technology Level', color='green')
            plt.plot(np.arange(T), L_t, label='Labor Force', color='orange')
            plt.title('Technology and Labor Force over Time')
            plt.xlabel('Time')
            plt.ylabel('Levels')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 3)
            plt.plot(np.arange(1, T), growth_rates, label='Output Growth Rate', color='purple')
            plt.title('Output Growth Rate over Time')
            plt.xlabel('Time')
            plt.ylabel('Growth Rate (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.figure(figsize=self.figsize)
            plt.plot(np.arange(1, T), growth_rates, label='Output Growth Rate', color='green', linewidth=2)
            plt.title('Output Growth Rate over Time')
            plt.xlabel('Time')
            plt.ylabel('Growth Rate (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_pe_sensitivity(self, 
                          g_range: np.ndarray,
                          solow_params: SolowParameters,
                          financial_params: FinancialParameters) -> None:
        """
        Plot P/E ratio sensitivity to technological growth rate.
        
        Args:
            g_range: Array of technological growth rates
            solow_params: Solow model parameters
            financial_params: Financial parameters
        """
        from .core import calculate_pe_ratio_sensitivity
        
        pe_ratios = calculate_pe_ratio_sensitivity(g_range, solow_params, financial_params)
        
        plt.figure(figsize=self.figsize)
        plt.plot(g_range * 100, pe_ratios, marker='o', linestyle='-', color='blue', linewidth=2, markersize=4)
        
        # Add annotations for every 4th point (if not NaN)
        for i, pe_ratio in enumerate(pe_ratios):
            if not np.isnan(pe_ratio) and i % 4 == 0:
                plt.text(g_range[i] * 100, pe_ratio, f'{pe_ratio:.2f}', 
                        fontsize=8, fontweight='bold', ha='right', va='top')
        
        plt.title('Justified Forward P/E Ratio vs. Technological Growth Rate')
        plt.xlabel('Technological Growth Rate (%)')
        plt.ylabel('Justified Forward P/E Ratio')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_potential_gdp_sensitivity(self, 
                                     solow_params: SolowParameters,
                                     g_min: float = 0.01,
                                     g_max: float = 0.03,
                                     step: float = 0.001) -> None:
        """
        Plot potential GDP growth rate vs technological progress rate.
        
        Args:
            solow_params: Solow model parameters
            g_min: Minimum technological growth rate
            g_max: Maximum technological growth rate
            step: Step size for growth rate range
        """
        g_range = np.arange(g_min, g_max, step)
        potential_gdp_rates = []
        
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
            potential_gdp_rate = solow_model.get_final_growth_rate()
            potential_gdp_rates.append(potential_gdp_rate)
        
        plt.figure(figsize=self.figsize)
        plt.plot(g_range * 100, potential_gdp_rates, label='Potential GDP Growth Rate', 
                color='green', linewidth=2)
        plt.xlabel('Technological Progress Rate % (g)')
        plt.ylabel('Potential GDP Growth Rate (%)')
        plt.title('Potential GDP Growth Rate vs. Technological Progress Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_monte_carlo_results(self, 
                               pe_ratios: np.ndarray, 
                               earnings_growth_rates: np.ndarray,
                               bins: int = 30) -> None:
        """
        Plot Monte Carlo simulation results.
        
        Args:
            pe_ratios: Array of simulated P/E ratios
            earnings_growth_rates: Array of simulated earnings growth rates
            bins: Number of bins for histograms
        """
        # Filter out NaN values
        final_month_pe_ratios = pe_ratios[:, -1][~np.isnan(pe_ratios[:, -1])]
        final_month_earnings_growth_rates = (earnings_growth_rates[:, -1][
            ~np.isnan(earnings_growth_rates[:, -1])] * 100)
        
        # Calculate statistics
        pe_mean, pe_std = np.mean(final_month_pe_ratios), np.std(final_month_pe_ratios)
        earnings_mean, earnings_std = (np.mean(final_month_earnings_growth_rates), 
                                     np.std(final_month_earnings_growth_rates))
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 4))
        
        # Plot P/E ratios
        axs[0].hist(final_month_pe_ratios, bins=bins, alpha=0.75, color='blue', edgecolor='black')
        axs[0].axvline(pe_mean, color='k', linestyle='dashed', linewidth=2, label=f'Mean: {pe_mean:.2f}')
        axs[0].axvline(pe_mean + 2 * pe_std, color='r', linestyle='dashed', linewidth=1, 
                      label=f'±2σ: {pe_mean + 2 * pe_std:.2f}')
        axs[0].axvline(pe_mean - 2 * pe_std, color='r', linestyle='dashed', linewidth=1,
                      label=f'±2σ: {pe_mean - 2 * pe_std:.2f}')
        axs[0].set_title('Distribution of Simulated P/E Ratios')
        axs[0].set_xlabel('P/E Ratio')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Plot Earnings Growth Rates
        axs[1].hist(final_month_earnings_growth_rates, bins=bins, alpha=0.75, color='green', edgecolor='black')
        axs[1].axvline(earnings_mean, color='k', linestyle='dashed', linewidth=2, 
                      label=f'Mean: {earnings_mean:.2f}%')
        axs[1].axvline(earnings_mean + 2 * earnings_std, color='r', linestyle='dashed', linewidth=1,
                      label=f'±2σ: {earnings_mean + 2 * earnings_std:.2f}%')
        axs[1].axvline(earnings_mean - 2 * earnings_std, color='r', linestyle='dashed', linewidth=1,
                      label=f'±2σ: {earnings_mean - 2 * earnings_std:.2f}%')
        axs[1].set_title('Distribution of Simulated Earnings Growth Rates')
        axs[1].set_xlabel('Earnings Growth Rate (%)')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nP/E Ratio Statistics:")
        print(f"Mean: {pe_mean:.3f}")
        print(f"Standard Deviation: {pe_std:.3f}")
        print(f"95% Confidence Interval: [{pe_mean - 2*pe_std:.3f}, {pe_mean + 2*pe_std:.3f}]")
        
        print(f"\nEarnings Growth Rate Statistics:")
        print(f"Mean: {earnings_mean:.3f}%")
        print(f"Standard Deviation: {earnings_std:.3f}%")
        print(f"95% Confidence Interval: [{earnings_mean - 2*earnings_std:.3f}%, {earnings_mean + 2*earnings_std:.3f}%]")


def create_economic_dashboard(solow_params: SolowParameters,
                            financial_params: FinancialParameters,
                            mc_params: MonteCarloParameters) -> None:
    """
    Create a comprehensive dashboard showing all key economic visualizations.
    
    Args:
        solow_params: Solow model parameters
        financial_params: Financial parameters
        mc_params: Monte Carlo parameters
    """
    visualizer = EconomicVisualizer(figsize=(12, 8))
    
    # 1. Solow Model Simulation
    print("=== Solow Growth Model Simulation ===")
    solow_model = SolowModel(solow_params)
    visualizer.plot_solow_simulation(solow_model, T=1000)
    
    # 2. Potential GDP Sensitivity
    print("\n=== Potential GDP Growth Rate Analysis ===")
    visualizer.plot_potential_gdp_sensitivity(solow_params)
    
    # 3. P/E Ratio Sensitivity
    print("\n=== P/E Ratio Sensitivity Analysis ===")
    g_range = np.arange(0.015, 0.031, 0.0005)
    visualizer.plot_pe_sensitivity(g_range, solow_params, financial_params)
    
    # 4. Monte Carlo Simulation
    print("\n=== Monte Carlo Risk Analysis ===")
    print("Running simulation...")
    mc_simulator = MonteCarloSimulator(solow_params, financial_params, mc_params)
    pe_ratios, earnings_growth_rates = mc_simulator.run_simulation()
    visualizer.plot_monte_carlo_results(pe_ratios, earnings_growth_rates)
