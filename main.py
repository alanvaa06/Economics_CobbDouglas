"""
Main execution script for Economics Models.

This script demonstrates the usage of the economics modeling functions
extracted from the Jupyter notebook, organized in a clean, executable format.
"""

import numpy as np
from src.economics_models import (
    SolowModel,
    FinancialAnalyzer,
    MonteCarloSimulator,
    SolowParameters,
    FinancialParameters,
    MonteCarloParameters,
    EconomicVisualizer,
    create_economic_dashboard
)
#%%

def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")


def run_basic_solow_analysis():
    """Run basic Solow model analysis."""
    print_section_header("BASIC SOLOW MODEL ANALYSIS")
    
    # Create default parameters
    solow_params = SolowParameters(
        s=0.20,      # 20% savings rate
        n=0.005,     # 0.5% population growth
        g=0.02,      # 2% technological progress
        delta=0.05,  # 5% depreciation
        alpha=0.35,  # 35% capital elasticity
        A=1,         # Initial technology
        L=100,       # Initial labor force
        K=10000      # Initial capital stock
    )
    
    # Create and run model
    solow_model = SolowModel(solow_params)
    
    print(f"Model Parameters:")
    print(f"  Savings Rate (s): {solow_params.s:.1%}")
    print(f"  Population Growth (n): {solow_params.n:.1%}")
    print(f"  Technology Growth (g): {solow_params.g:.1%}")
    print(f"  Depreciation Rate (δ): {solow_params.delta:.1%}")
    print(f"  Capital Elasticity (α): {solow_params.alpha:.1%}")
    
    # Calculate final growth rate
    final_growth_rate = solow_model.get_final_growth_rate()
    print(f"\nSteady-State Growth Rate: {final_growth_rate:.3f}%")
    
    # Run visualization
    visualizer = EconomicVisualizer()
    visualizer.plot_solow_simulation(solow_model, T=500)
    
    return solow_params, solow_model


def run_financial_analysis(solow_params: SolowParameters):
    """Run financial analysis with P/E ratio calculations."""
    print_section_header("FINANCIAL ANALYSIS - P/E RATIO CALCULATIONS")
    
    # Create financial parameters
    financial_params = FinancialParameters(
        beta=1.0,                    # Market beta
        equity_risk_premium=0.05,    # 5% equity risk premium
        expected_inflation=0.02,     # 2% expected inflation
        term_premium=0.01,           # 1% term premium
        retention_rate=0.35,         # 35% retention rate
        tfp=1.1                      # 10% earnings growth factor
    )
    
    print(f"Financial Parameters:")
    print(f"  Beta: {financial_params.beta:.2f}")
    print(f"  Equity Risk Premium: {financial_params.equity_risk_premium:.1%}")
    print(f"  Expected Inflation: {financial_params.expected_inflation:.1%}")
    print(f"  Term Premium: {financial_params.term_premium:.1%}")
    print(f"  Retention Rate: {financial_params.retention_rate:.1%}")
    print(f"  Earnings Growth Factor: {financial_params.tfp:.2f}")
    
    # Create financial analyzer
    financial_analyzer = FinancialAnalyzer(financial_params)
    
    # Calculate metrics using Solow model output
    solow_model = SolowModel(solow_params)
    real_growth_rate = solow_model.get_final_growth_rate() / 100
    
    pe_ratio, metrics = financial_analyzer.calculate_justified_pe_ratio(real_growth_rate)
    
    print(f"\nCalculated Metrics:")
    print(f"  Real Growth Rate: {metrics['real_growth_rate']:.3%}")
    print(f"  Nominal Growth Rate: {metrics['nominal_growth_rate']:.3%}")
    print(f"  Risk-Free Rate: {metrics['risk_free_rate']:.3%}")
    print(f"  Required Return: {metrics['required_return']:.3%}")
    print(f"  Earnings Growth Rate: {metrics['earnings_growth_rate']:.3%}")
    
    if not np.isnan(pe_ratio):
        print(f"\nJustified Forward P/E Ratio: {pe_ratio:.2f}")
    else:
        print(f"\nJustified Forward P/E Ratio: Cannot be calculated (earnings growth >= required return)")
    
    # Plot sensitivity analysis
    visualizer = EconomicVisualizer()
    g_range = np.arange(0.015, 0.031, 0.0005)
    visualizer.plot_pe_sensitivity(g_range, solow_params, financial_params)
    
    return financial_params


def run_monte_carlo_analysis(solow_params: SolowParameters, financial_params: FinancialParameters):
    """Run Monte Carlo risk analysis."""
    print_section_header("MONTE CARLO RISK ANALYSIS")
    
    # Create Monte Carlo parameters
    mc_params = MonteCarloParameters(
        num_simulations=1000,
        num_months=12,
        g_mean=0.02,
        g_std=0.005,
        inflation_mean=0.02,
        inflation_std=0.01,
        term_premium_mean=0.005,
        term_premium_std=0.002
    )
    
    print(f"Monte Carlo Parameters:")
    print(f"  Number of Simulations: {mc_params.num_simulations:,}")
    print(f"  Time Horizon: {mc_params.num_months} months")
    print(f"  Technology Growth: μ={mc_params.g_mean:.1%}, σ={mc_params.g_std:.1%}")
    print(f"  Inflation: μ={mc_params.inflation_mean:.1%}, σ={mc_params.inflation_std:.1%}")
    print(f"  Term Premium: μ={mc_params.term_premium_mean:.1%}, σ={mc_params.term_premium_std:.1%}")
    
    print(f"\nRunning {mc_params.num_simulations:,} simulations...")
    
    # Run simulation
    mc_simulator = MonteCarloSimulator(solow_params, financial_params, mc_params)
    pe_ratios, earnings_growth_rates = mc_simulator.run_simulation()
    
    print("Simulation completed! Generating visualizations...")
    
    # Visualize results
    visualizer = EconomicVisualizer()
    visualizer.plot_monte_carlo_results(pe_ratios, earnings_growth_rates)
    
    return mc_params


def run_scenario_analysis():
    """Run different economic scenarios."""
    print_section_header("SCENARIO ANALYSIS")
    
    scenarios = {
        "Conservative": SolowParameters(s=0.15, n=0.003, g=0.015, delta=0.06, alpha=0.30),
        "Base Case": SolowParameters(s=0.20, n=0.005, g=0.020, delta=0.05, alpha=0.35),
        "Optimistic": SolowParameters(s=0.25, n=0.007, g=0.025, delta=0.04, alpha=0.40),
    }
    
    financial_params = FinancialParameters(
        beta=1.0,
        equity_risk_premium=0.05,
        expected_inflation=0.02,
        term_premium=0.01,
        retention_rate=0.35,
        tfp=1.1
    )
    
    financial_analyzer = FinancialAnalyzer(financial_params)
    
    print(f"{'Scenario':<12} {'Growth Rate':<12} {'P/E Ratio':<10}")
    print("-" * 40)
    
    for scenario_name, params in scenarios.items():
        solow_model = SolowModel(params)
        growth_rate = solow_model.get_final_growth_rate()
        pe_ratio, _ = financial_analyzer.calculate_justified_pe_ratio(growth_rate / 100)
        
        if not np.isnan(pe_ratio):
            print(f"{scenario_name:<12} {growth_rate:>8.3f}%    {pe_ratio:>6.2f}")
        else:
            print(f"{scenario_name:<12} {growth_rate:>8.3f}%    {'N/A':>6}")


def create_comprehensive_dashboard():
    """Create a comprehensive economic dashboard."""
    print_section_header("COMPREHENSIVE ECONOMIC DASHBOARD")
    
    # Default parameters for dashboard
    solow_params = SolowParameters()
    financial_params = FinancialParameters()
    mc_params = MonteCarloParameters(num_simulations=500)  # Reduced for faster demo
    
    print("Generating comprehensive economic dashboard...")
    print("This may take a few moments...")
    
    create_economic_dashboard(solow_params, financial_params, mc_params)
    
    print("Dashboard generation completed!")

#%%
# 1. Basic Solow Analysis
solow_params, solow_model = run_basic_solow_analysis()

#%%
# 2. Financial Analysis
financial_params = run_financial_analysis(solow_params)

#%%
# 3. Monte Carlo Analysis
mc_params = run_monte_carlo_analysis(solow_params, financial_params)

#%%
# 4. Scenario Analysis
run_scenario_analysis()

#%%
# 5. Comprehensive Dashboard
create_comprehensive_dashboard()
