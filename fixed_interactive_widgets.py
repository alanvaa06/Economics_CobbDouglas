"""
Fixed Interactive Widgets for Solow Model
This corrects the issues in Cell 2 of your notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import interact, FloatSlider, Layout

# First, make sure you have the required functions
def next_period_k(k, s, y, delta, n, g):
    investment = s * y
    depreciation = delta * k
    next_k = (1 + n + g) * (k + investment - depreciation)
    return next_k

def solow_model_simulation_with_growth_rate(s=0.20, n=0.005, g=0.02, delta=0.05, alpha=0.35, A=1, L=100, K=10000, T=1000):
    # Arrays to store simulation results and growth rates
    K_t = np.zeros(T)
    Y_t = np.zeros(T)
    A_t = np.zeros(T)
    L_t = np.zeros(T)
    growth_rates = np.zeros(T-1)

    # Initial values
    K_t[0] = K
    A_t[0] = A
    L_t[0] = L
    Y_t[0] = (K_t[0]**alpha) * (A_t[0]*L_t[0])**(1-alpha)

    # Run the simulation and calculate growth rates
    for t in range(1, T):
        K_t[t] = next_period_k(K_t[t-1], s, Y_t[t-1], delta, n, g)
        L_t[t] = L_t[t-1] * (1 + n)
        A_t[t] = A_t[t-1] * (1 + g)
        Y_t[t] = (K_t[t]**alpha) * (A_t[t]*L_t[t])**(1-alpha)
        growth_rates[t-1] = (Y_t[t] - Y_t[t-1]) / Y_t[t-1] * 100

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, T), growth_rates, label='Output Growth Rate', color='green', linewidth=2)
    plt.title('Output Growth Rate over Time')
    plt.xlabel('Time')
    plt.ylabel('Growth Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# CORRECTED WIDGET CODE
# Create widgets with proper styling
s_widget = widgets.FloatSlider(
    value=0.05, min=0.0, max=1.0, step=0.01, 
    description='Savings Rate:', 
    style={'description_width': 'initial'}, 
    layout=Layout(width='50%')
)

n_widget = widgets.FloatSlider(
    value=0.01, min=0.0, max=0.1, step=0.001, 
    description='Population Growth:', 
    readout_format='.3f', 
    style={'description_width': 'initial'}, 
    layout=Layout(width='50%')
)

g_widget = widgets.FloatSlider(
    value=0.02, min=0.0, max=0.1, step=0.001, 
    description='Tech Progress:', 
    readout_format='.3f', 
    style={'description_width': 'initial'}, 
    layout=Layout(width='50%')
)

delta_widget = widgets.FloatSlider(
    value=0.05, min=0.0, max=0.2, step=0.01, 
    description='Depreciation Rate:', 
    style={'description_width': 'initial'}, 
    layout=Layout(width='50%')
)

alpha_widget = widgets.FloatSlider(
    value=0.35, min=0.0, max=1.0, step=0.01, 
    description='Capital Elasticity:', 
    style={'description_width': 'initial'}, 
    layout=Layout(width='50%')
)

# Create output widget to capture the plots
output_widget = widgets.Output()

# CORRECTED Update function
def update_model(change):
    """Update function that properly captures output"""
    with output_widget:
        clear_output(wait=True)
        # Call the simulation with current widget values
        solow_model_simulation_with_growth_rate(
            s=s_widget.value, 
            n=n_widget.value, 
            g=g_widget.value,
            delta=delta_widget.value, 
            alpha=alpha_widget.value
        )

# Attach observers to all widgets
s_widget.observe(update_model, names='value')
n_widget.observe(update_model, names='value')
g_widget.observe(update_model, names='value')
delta_widget.observe(update_model, names='value')
alpha_widget.observe(update_model, names='value')

# Create container with all widgets
container = widgets.VBox([
    s_widget, 
    n_widget, 
    g_widget, 
    delta_widget, 
    alpha_widget
])

# Display the widgets and output
display(container)
display(output_widget)

# Initial plot with default values
update_model(None)

print("✅ Interactive widgets are now properly configured!")
print("Move the sliders above to see the plot update in real-time.")
