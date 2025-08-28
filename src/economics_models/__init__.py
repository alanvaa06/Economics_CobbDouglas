"""
Economics Models Package

A comprehensive package for economic modeling and financial analysis,
implementing the Solow growth model and related financial calculations.
"""

from .core import (
    SolowModel,
    FinancialAnalyzer,
    MonteCarloSimulator,
    SolowParameters,
    FinancialParameters,
    MonteCarloParameters,
    calculate_pe_ratio_sensitivity
)

from .visualization import (
    EconomicVisualizer,
    create_economic_dashboard
)

__version__ = "1.0.0"
__author__ = "Economics Model Development Team"

__all__ = [
    # Core classes
    "SolowModel",
    "FinancialAnalyzer", 
    "MonteCarloSimulator",
    
    # Parameter classes
    "SolowParameters",
    "FinancialParameters", 
    "MonteCarloParameters",
    
    # Visualization classes
    "EconomicVisualizer",
    
    # Functions
    "calculate_pe_ratio_sensitivity",
    "create_economic_dashboard",
]
