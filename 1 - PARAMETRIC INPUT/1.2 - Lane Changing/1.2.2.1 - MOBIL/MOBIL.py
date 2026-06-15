import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NGM_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _NGM_ROOT not in sys.path:
    sys.path.insert(0, _NGM_ROOT)
from ngm_paths import DIS_SURROUNDING_CSV, dataset_path


def fitness_function(p, ac_change, an_change, ao_change):
    utility_change = ac_change + p * (an_change + ao_change)
    return 1 / (1 + abs(delta_a_th - utility_change))  # Fitness value, higher is better

def select_parents(population, fitnesses):
    # Simple selection: choose the best two individuals
    indexes = np.argsort(fitnesses)[-2:]
    return [population[indexes[0]], population[indexes[1]]]

def crossover(parent1, parent2):
    # Simple crossover: average the parents' values
    return (parent1 + parent2) / 2

def mutate(child):
    mutation_chance = 0.1
    if random.random() < mutation_chance:
        child += random.uniform(-0.1, 0.1)  # Small mutation
    return child

def genetic_algorithm_for_event(ac_change, an_change, ao_change, population_size=20, generations=100):
    population = [random.random() for _ in range(population_size)]  # Initial population of politeness factors
    for _ in range(generations):
        fitnesses = [fitness_function(p, ac_change, an_change, ao_change) for p in population]
        parent1, parent2 = select_parents(population, fitnesses)
        population = [mutate(crossover(parent1, parent2)) for _ in range(population_size)]
    best_index = np.argmax(fitnesses)
    return population[best_index]

def calibrate_per_event_with_ga(df):
    p_optimals = []
    for event_id in df['lane_change_event'].unique():
        event_data = df[df['lane_change_event'] == event_id].iloc[0]
        ac_change = calculate_avg_speed_change(df, event_id, event_data['object_id'])
        an_change = calculate_avg_speed_change(df, event_id, event_data['behind_id_after'])
        ao_change = calculate_avg_speed_change(df, event_id, event_data['behind_id_before'])
        p_optimal = genetic_algorithm_for_event(ac_change, an_change, ao_change)
        p_optimals.append({
            'event_id': event_id,
            'p_optimal': p_optimal,
            'ac_change': ac_change,
            'an_change': an_change,
            'ao_change': ao_change
        })
    return p_optimals

def calculate_avg_speed_change(df, event_id, car_id):
    specific_data = df[(df['lane_change_event'] == event_id) & (df['ID'] == car_id)]
    if specific_data.empty:
        return 0  # Return 0 if no data available for the car
    lane_change_time = specific_data['time'].mean()
    avg_speed_before = specific_data[specific_data['time'] <= lane_change_time]['speed-kf'].mean()
    avg_speed_after = specific_data[specific_data['time'] > lane_change_time]['speed-kf'].mean()
    return avg_speed_after - avg_speed_before

def plot_combined_results(p_optimals_per_event):
    combined_df = pd.DataFrame(p_optimals_per_event)
    
    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Event ID')
    ax1.set_ylabel('Optimal Politeness Factor (p)', color=color)
    ax1.bar(combined_df['event_id'], combined_df['p_optimal'], color=color, label='Politeness Factor (p)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Speed Change', color=color)
    ax2.plot(combined_df['event_id'], combined_df['ac_change'], color='red', marker='o', label='c Speed Change')
    ax2.plot(combined_df['event_id'], combined_df['an_change'], color='green', marker='x', label='n Speed Change')
    ax2.plot(combined_df['event_id'], combined_df['ao_change'], color='purple', marker='^', label='o Speed Change')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    plt.title('Combined Visualization of Lane Change Events')
    plt.show()
    
def visualize_parameter_distributions(all_params):
    # Assuming all_params is a list of lists, with each inner list containing a set of parameters
    param_names = ['T', 'a', 'b', 'v0', 'delta', 'so']
    num_params = len(param_names)
    
    # Convert list of lists into a 2D numpy array for easier column-wise access
    all_params_array = np.array(all_params)
    
    # Create histograms for each parameter
    fig, axs = plt.subplots(1, num_params, figsize=(20, 4))
    for i in range(num_params):
        axs[i].hist(all_params_array[:, i], bins=20, color='skyblue', edgecolor='black')
        axs[i].set_title(param_names[i])
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
def plot_variable_distributions(data, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []
    
    # Determine the variables to plot by excluding specified keys
    variables = [key for key in data[0].keys() if key not in exclude_keys]
    data_values = {var: [d[var] for d in data] for var in variables}
    
    # Calculate the number of rows and columns for plotting
    n_vars = len(variables)
    n_cols = 2
    n_rows = (n_vars + 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows))
    axs = axs.flatten()  # Flatten the axis array for easy iteration
    
    for ax, (var, values) in zip(axs, data_values.items()):
        ax.hist(values, bins=10, edgecolor='black')
        ax.set_title(f'{var} Distribution')
    
    # Hide any unused axes
    for ax in axs[len(variables):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    
# Use the function and plot results
delta_a_th = 0
file_path = dataset_path(DIS_SURROUNDING_CSV)
surrounding_info_with_event_id = pd.read_csv(file_path)
p_optimals_per_event = calibrate_per_event_with_ga(surrounding_info_with_event_id)
plot_combined_results(p_optimals_per_event)

plot_variable_distributions(p_optimals_per_event, exclude_keys=['event_id'])