# Data Directory

This directory contains data files and results from economic model analyses.

## Structure

- `results/` - Output from model simulations and analyses
- `input/` - Input data files (CSV, JSON, etc.)
- `config/` - Alternative configuration files
- `exports/` - Exported results in various formats

## Data Sources

- Model simulation outputs
- Economic parameter sets
- Analysis results and statistics
- Visualization data

## File Formats

The package supports multiple data formats:
- CSV for tabular data
- JSON for structured data
- YAML for configuration
- Excel files for comprehensive reports

## Usage

Data files are automatically generated when running analyses:

```python
# Results are automatically saved to data/results/
python main.py
```

Configure output location in `config/default_parameters.yaml`:

```yaml
export:
  output_directory: "data/results"
  file_formats: ["csv", "json", "xlsx"]
```
