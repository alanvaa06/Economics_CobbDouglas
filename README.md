# Economics Cobb-Douglas Model

A comprehensive Python package for economic modeling and financial analysis, implementing the Solow growth model with Cobb-Douglas production function. This project includes advanced features such as Monte Carlo simulation, sensitivity analysis, and interactive visualizations for economic research and financial valuation.

## 🎯 Features

- **Solow Growth Model**: Complete implementation with configurable parameters
- **Financial Analysis**: P/E ratio calculations using CAPM and economic growth models
- **Monte Carlo Simulation**: Risk analysis with stochastic modeling
- **Comprehensive Visualizations**: Interactive plots and economic dashboards
- **Sensitivity Analysis**: Parameter sensitivity testing and scenario analysis
- **Professional Architecture**: Clean, modular code following best practices
- **Extensive Testing**: Unit tests with pytest framework
- **Type Hints**: Full type annotation for better code quality
- **Documentation**: Comprehensive API reference and examples

## 📦 Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/economics-models/economics-cobb-douglas.git
cd economics-cobb-douglas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 🚀 Quick Start

### Basic Usage

```python
from economics_models import SolowModel, SolowParameters, EconomicVisualizer

# Create model parameters
params = SolowParameters(
    s=0.20,      # 20% savings rate
    n=0.005,     # 0.5% population growth
    g=0.02,      # 2% technology growth
    delta=0.05,  # 5% depreciation
    alpha=0.35   # 35% capital elasticity
)

# Run Solow model
model = SolowModel(params)
growth_rate = model.get_final_growth_rate()
print(f"Steady-state growth rate: {growth_rate:.3f}%")

# Create visualizations
visualizer = EconomicVisualizer()
visualizer.plot_solow_simulation(model, T=500)
```

### Financial Analysis

```python
from economics_models import FinancialAnalyzer, FinancialParameters

# Set up financial parameters
financial_params = FinancialParameters(
    beta=1.0,                    # Market beta
    equity_risk_premium=0.05,    # 5% equity risk premium
    expected_inflation=0.02,     # 2% expected inflation
    term_premium=0.01,           # 1% term premium
    retention_rate=0.35          # 35% earnings retention
)

# Calculate justified P/E ratio
analyzer = FinancialAnalyzer(financial_params)
pe_ratio, metrics = analyzer.calculate_justified_pe_ratio(growth_rate / 100)

print(f"Justified P/E Ratio: {pe_ratio:.2f}")
print(f"Required Return: {metrics['required_return']:.3%}")
```

### Monte Carlo Simulation

```python
from economics_models import MonteCarloSimulator, MonteCarloParameters

# Set up Monte Carlo parameters
mc_params = MonteCarloParameters(
    num_simulations=1000,
    num_months=12,
    g_mean=0.02,
    g_std=0.005
)

# Run simulation
simulator = MonteCarloSimulator(params, financial_params, mc_params)
pe_ratios, earnings_rates = simulator.run_simulation()

# Visualize results
visualizer.plot_monte_carlo_results(pe_ratios, earnings_rates)
```

### Complete Analysis Dashboard

```python
from economics_models import create_economic_dashboard

# Generate comprehensive dashboard
create_economic_dashboard(params, financial_params, mc_params)
```

## 📊 Run the Complete Analysis

Execute the main analysis script to see all features in action:

```bash
python main.py
```

This will run:
1. Basic Solow model analysis
2. Financial P/E ratio calculations
3. Monte Carlo risk simulation
4. Scenario analysis
5. Comprehensive economic dashboard

## 🏗️ Project Structure

```
economics-cobb-douglas/
├── src/
│   └── economics_models/
│       ├── __init__.py          # Package initialization
│       ├── core.py              # Core economic models
│       └── visualization.py     # Visualization functions
├── tests/
│   ├── __init__.py
│   └── test_core.py            # Unit tests
├── config/
│   └── default_parameters.yaml # Default configuration
├── docs/
│   └── api_reference.md        # API documentation
├── data/                       # Data directory
├── main.py                     # Main execution script
├── pyproject.toml             # Project configuration
├── README.md                  # This file
└── EconomicGrowth.ipynb       # Original Jupyter notebook
```

## 🔧 Configuration

The package uses YAML configuration files for default parameters. See `config/default_parameters.yaml` for all configurable options.

### Key Parameters

- **Solow Model**: Savings rate, population growth, technology progress, depreciation
- **Financial Model**: Beta, risk premiums, inflation expectations
- **Monte Carlo**: Simulation count, time horizon, parameter distributions
- **Visualization**: Plot styling, color schemes, figure sizes

## 📈 Economic Models

### Solow Growth Model

The Solow-Swan model describes long-run economic growth based on:
- **Production Function**: Y = K^α × (A×L)^(1-α)
- **Capital Accumulation**: K_{t+1} = (1+n+g) × (K_t + sY_t - δK_t)
- **Technology Growth**: A_{t+1} = A_t × (1+g)
- **Population Growth**: L_{t+1} = L_t × (1+n)

### Financial Valuation

P/E ratio calculation using:
- **CAPM**: Required Return = Risk-free Rate + β × Market Risk Premium
- **Gordon Growth Model**: P/E = (1-b) / (r - g_e)
- **Economic Integration**: Links macro growth to financial valuation

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/economics_models

# Run specific test file
pytest tests/test_core.py
```

## 📚 Documentation

- [API Reference](docs/api_reference.md) - Complete API documentation
- [Configuration Guide](config/default_parameters.yaml) - Parameter configuration
- [Examples](main.py) - Usage examples and demonstrations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests for new features
- Update documentation for API changes
- Use conventional commit messages

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the Solow-Swan economic growth model
- Implements modern financial valuation techniques
- Built with Python scientific computing stack
- Follows data science best practices

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the [API documentation](docs/api_reference.md)
- Review the example code in `main.py`

---

**Note**: This package is designed for educational and research purposes. Financial models should be validated independently before use in investment decisions.

