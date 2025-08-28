# API Reference

## Core Classes

### SolowModel

Implementation of the Solow growth model with Cobb-Douglas production function.

```python
from economics_models import SolowModel, SolowParameters

# Create parameters
params = SolowParameters(
    s=0.20,      # Savings rate
    n=0.005,     # Population growth
    g=0.02,      # Technology growth
    delta=0.05,  # Depreciation rate
    alpha=0.35   # Capital elasticity
)

# Create and run model
model = SolowModel(params)
growth_rate = model.get_final_growth_rate()
```

#### Methods

- `production_function(K, A, L)`: Cobb-Douglas production function
- `next_period_capital(k, y)`: Capital accumulation equation
- `simulate(T)`: Run full simulation for T periods
- `get_final_growth_rate(T)`: Get steady-state growth rate

### FinancialAnalyzer

Financial analysis and P/E ratio calculations using CAPM.

```python
from economics_models import FinancialAnalyzer, FinancialParameters

params = FinancialParameters(
    beta=1.0,
    equity_risk_premium=0.05,
    expected_inflation=0.02
)

analyzer = FinancialAnalyzer(params)
pe_ratio, metrics = analyzer.calculate_justified_pe_ratio(real_growth_rate)
```

#### Methods

- `calculate_required_return(risk_free_rate)`: CAPM required return
- `calculate_justified_pe_ratio(real_growth_rate)`: Forward P/E ratio calculation

### MonteCarloSimulator

Monte Carlo simulation for risk analysis.

```python
from economics_models import MonteCarloSimulator, MonteCarloParameters

mc_params = MonteCarloParameters(
    num_simulations=1000,
    num_months=12
)

simulator = MonteCarloSimulator(solow_params, financial_params, mc_params)
pe_ratios, earnings_rates = simulator.run_simulation()
```

## Parameter Classes

### SolowParameters

Dataclass containing Solow model parameters:

- `s`: Savings rate (default: 0.20)
- `n`: Population growth rate (default: 0.005)
- `g`: Technology growth rate (default: 0.02)
- `delta`: Depreciation rate (default: 0.05)
- `alpha`: Capital elasticity (default: 0.35)
- `A`: Initial technology level (default: 1.0)
- `L`: Initial labor force (default: 100)
- `K`: Initial capital stock (default: 10000)

### FinancialParameters

Dataclass containing financial analysis parameters:

- `beta`: Market beta (default: 1.0)
- `equity_risk_premium`: Market risk premium (default: 0.02)
- `expected_inflation`: Expected inflation (default: 0.02)
- `term_premium`: Term premium (default: 0.01)
- `retention_rate`: Earnings retention rate (default: 0.35)
- `tfp`: Total factor productivity (default: 1.1)

### MonteCarloParameters

Dataclass containing Monte Carlo simulation parameters:

- `num_simulations`: Number of simulations (default: 1000)
- `num_months`: Time horizon (default: 12)
- `g_mean`: Mean technology growth (default: 0.02)
- `g_std`: Technology growth std dev (default: 0.005)
- `inflation_mean`: Mean inflation (default: 0.02)
- `inflation_std`: Inflation std dev (default: 0.01)
- `term_premium_mean`: Mean term premium (default: 0.005)
- `term_premium_std`: Term premium std dev (default: 0.002)

## Visualization Classes

### EconomicVisualizer

Comprehensive visualization for economic models.

```python
from economics_models import EconomicVisualizer

visualizer = EconomicVisualizer()
visualizer.plot_solow_simulation(solow_model)
visualizer.plot_pe_sensitivity(g_range, solow_params, financial_params)
visualizer.plot_monte_carlo_results(pe_ratios, earnings_rates)
```

#### Methods

- `plot_solow_simulation(model, T, show_all_variables)`: Plot simulation results
- `plot_pe_sensitivity(g_range, solow_params, financial_params)`: P/E sensitivity analysis
- `plot_potential_gdp_sensitivity(solow_params)`: GDP growth sensitivity
- `plot_monte_carlo_results(pe_ratios, earnings_rates)`: Monte Carlo results

## Utility Functions

### calculate_pe_ratio_sensitivity

Calculate P/E ratio sensitivity to technology growth rate.

```python
from economics_models import calculate_pe_ratio_sensitivity
import numpy as np

g_range = np.arange(0.01, 0.03, 0.001)
pe_ratios = calculate_pe_ratio_sensitivity(g_range, solow_params, financial_params)
```

### create_economic_dashboard

Generate comprehensive economic analysis dashboard.

```python
from economics_models import create_economic_dashboard

create_economic_dashboard(solow_params, financial_params, mc_params)
```

## Example Usage

### Basic Analysis

```python
from economics_models import *

# Set up parameters
solow_params = SolowParameters(s=0.20, n=0.005, g=0.02)
financial_params = FinancialParameters(beta=1.0, equity_risk_premium=0.05)

# Run Solow model
model = SolowModel(solow_params)
growth_rate = model.get_final_growth_rate()

# Financial analysis
analyzer = FinancialAnalyzer(financial_params)
pe_ratio, metrics = analyzer.calculate_justified_pe_ratio(growth_rate / 100)

print(f"Growth Rate: {growth_rate:.3f}%")
print(f"P/E Ratio: {pe_ratio:.2f}")
```

### Monte Carlo Analysis

```python
# Set up Monte Carlo parameters
mc_params = MonteCarloParameters(num_simulations=1000)

# Run simulation
simulator = MonteCarloSimulator(solow_params, financial_params, mc_params)
pe_ratios, earnings_rates = simulator.run_simulation()

# Visualize results
visualizer = EconomicVisualizer()
visualizer.plot_monte_carlo_results(pe_ratios, earnings_rates)
```
