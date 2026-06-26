import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import os
import sys

_NGM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _NGM_ROOT not in sys.path:
    sys.path.insert(0, _NGM_ROOT)
from ngm_paths import calibration_dataset_paths, load_tgsim_csv

population_size, num_generations, mutation_rate = 40, 80, 0.1 # Calibration parameters
T, a, b, v0, delta, so = None, None, None, None, None, None
most_leading_leader_id = None


def generate_vehicle_groups(datasets):
    """
    Generate vehicle group lists dynamically from datasets.
    Ensures each (ID, run_index) pair is unique.
    Returns a dictionary with keys as group names (e.g., 'I395_A', 'I9094_S') and values as lists of [ID, run-index].
    """
    group_lists = {}
    
    for dataset_key, dataset_path in datasets.items():
        # Read the dataset
        df = load_tgsim_csv(dataset_path)
        df = df.sort_values(by='time')
        df['time'] = df['time'].round(1)
        
        group_A, group_S, group_L = set(), set(), set()  # Use sets to ensure uniqueness
        
        if dataset_key == "I395":
            # Criteria for I395
            for _, row in df.iterrows():
                if row['type-most-common'] == 2 or row['type-most-common'] == 3:
                    group_L.add((row['ID'], row['run-index']))
                elif row['type-most-common'] == 1:
                    group_S.add((row['ID'], row['run-index']))
                elif row['type-most-common'] == 4:
                    group_A.add((row['ID'], row['run-index']))
        else:
            # Determine which column indicates AV/ACC status
            av_column = None
            if "ACC" in df.columns:
                av_column = "ACC"
            elif "AV" in df.columns:
                av_column = "AV"
            
            # Criteria for other datasets
            for _, row in df.iterrows():
                if av_column and str(row[av_column]).strip().lower() == "yes":
                    group_A.add((row['ID'], row['run-index']))
                elif row['type-most-common'] == "small-vehicle":
                    group_S.add((row['ID'], row['run-index']))
                elif row['type-most-common'] == "large-vehicle":
                    group_L.add((row['ID'], row['run-index']))
        
        # Convert sets back to lists
        group_lists[f"{dataset_key}_A"] = [list(item) for item in group_A]
        group_lists[f"{dataset_key}_S"] = [list(item) for item in group_S]
        group_lists[f"{dataset_key}_L"] = [list(item) for item in group_L]
    
    return group_lists

# Example usage
datasets = calibration_dataset_paths()

vehicle_groups = generate_vehicle_groups(datasets)

# Access generated lists
I395_A = vehicle_groups["I395_A"]
I395_S = vehicle_groups["I395_S"]
I395_L = vehicle_groups["I395_L"]

I9094_A = vehicle_groups["I9094_A"]
I9094_S = vehicle_groups["I9094_S"]
I9094_L = vehicle_groups["I9094_L"]

I294l1_A = vehicle_groups["I294l1_A"]
I294l1_S = vehicle_groups["I294l1_S"]
I294l1_L = vehicle_groups["I294l1_L"]

I294l2_A = vehicle_groups["I294l2_A"]
I294l2_S = vehicle_groups["I294l2_S"]
I294l2_L = vehicle_groups["I294l2_L"]



def find_leader_data(df, follower_id, run_index):
    global most_leading_leader_id
    
    follower_data = df[(df['ID'] == follower_id) & (df['run-index'] == run_index)]
    leader_data_dict = {}  # Dictionary to store leader data by leader ID
    
    for index, row in follower_data.iterrows():
        time = row['time']
        follower_x = row[pos]  # Assuming 'pos' is the column for follower's position
        follower_lane = row['lane-kf']  # Assuming the lane information is in 'lane-kf' column
        run_index = row['run-index']  # Assuming 'run-index' is the column for run index

        # Find the nearest leader with a greater position
        leader_data = df[(df['ID'] != follower_id) & (df['time'] == time) & (df['lane-kf'] == follower_lane) & (df[pos] > follower_x) & (df['run-index'] == run_index)]
        
        if not leader_data.empty:
            # Find the nearest leader
            nearest_leader_row = leader_data.loc[leader_data[pos].sub(follower_x).abs().idxmin()]
            
            leader_id = nearest_leader_row['ID']
            leader_x_val = nearest_leader_row[pos]
            leader_speed_val = nearest_leader_row['speed-kf']

            if leader_id not in leader_data_dict:
                leader_data_dict[leader_id] = {'time': [], 'x_val': [], 'speed_val': []}

            leader_data_dict[leader_id]['time'].append(time)
            leader_data_dict[leader_id]['x_val'].append(leader_x_val)
            leader_data_dict[leader_id]['speed_val'].append(leader_speed_val)

    # Choose the most leading leader by selecting the leader with the maximum number of occurrences in the leader_data_dict
    if leader_data_dict:
        most_leading_leader_id = max(leader_data_dict, key=lambda x: len(leader_data_dict[x]['time']))
        leader_data = leader_data_dict[most_leading_leader_id]
        leader_df = pd.DataFrame({'ID': most_leading_leader_id,
                                   'time': leader_data['time'],
                                   pos: leader_data['x_val'],
                                   'speed-kf': leader_data['speed_val'],
                                   'run-index': run_index})
    else:
        leader_df = pd.DataFrame(columns=['ID', 'time', pos, 'speed-kf', 'run-index'])
    
    return leader_df


def extract_subject_and_leader_data(df, follower_id, run_index):
    sdf = df[(df['ID'] == follower_id) & (df['run-index'] == run_index)].round(2)
    ldf = find_leader_data(df, follower_id, run_index).round(2)
    
    # Find the intersection of time frames between leader and subject
    mutual_times = np.intersect1d(ldf['time'], sdf['time'])
    
    # Find the longest continuous segment of mutual time
    max_continuous_mutual_times = []
    continuous_mutual_times = []
    prev_time = None
    for time in mutual_times:
        if prev_time is None or time - prev_time < 0.2:  # Assuming the time step is 0.1
            continuous_mutual_times.append(time)
        else:
            if len(continuous_mutual_times) > len(max_continuous_mutual_times):
                max_continuous_mutual_times = continuous_mutual_times
            continuous_mutual_times = [time]
        prev_time = time
    
    if len(continuous_mutual_times) > len(max_continuous_mutual_times):
        max_continuous_mutual_times = continuous_mutual_times
    
    # Calculate the duration of car-following
    if max_continuous_mutual_times:
        duration = max_continuous_mutual_times[-1] - max_continuous_mutual_times[0]
    else:
        duration = 0  # No mutual time found

    # Filter leader and subject data to include only the longest continuous mutual time
    ldf = ldf[ldf['time'].isin(max_continuous_mutual_times)]
    sdf = sdf[sdf['time'].isin(max_continuous_mutual_times)]
    
    if (isinstance(sdf, list) and not sdf) or (isinstance(sdf, pd.DataFrame) and sdf.empty):
        print(f"No subject data found for Follower ID {follower_id} and Run Index {run_index}.")
        return pd.DataFrame(), pd.DataFrame(), 0, 0  # Return empty DataFrames, duration 0, and start_time 0
    
    else:
        # CAPTURE START TIME BEFORE NORMALIZING
        start_time = sdf['time'].iloc[0]
        ldf['time'], sdf['time'] = ldf['time'] - start_time, sdf['time'] - start_time
        return sdf, ldf, duration, start_time # RETURN START_TIME



def idm_acceleration(v, v_leader, s):
    max_v = 40  # Maximum allowable velocity (adjust as needed)
    max_s = 1000  # Maximum allowable gap (adjust as needed)
    
    s_star = so + max (0,(v * T + (v * (v - v_leader)) / (2 * np.sqrt(a * b))))
    #print (s_star)
    acceleration = a * (1 - (v / min(v0, max_v)) ** delta  - (s_star / min(s, max_s))**2)
    if np.isnan(acceleration):
        acceleration = 0
    #print (acceleration)
    return acceleration

def simulate_car_following(params):
    global T, a, b, v0, so, delta
    T, a, b, v0, so, delta = params  # Include delta from the parameters
    
    num_steps = round(total_time / time_step)
    time = np.linspace(0, total_time, num_steps)
    
    position = np.zeros(num_steps)
    speed = np.zeros(num_steps)
    acl = np.zeros(num_steps)
    
    position[0] = sdf.iloc[0][pos]
    speed[0] = sdf.iloc[0]['speed-kf']
    acl[0] = 0

    for i in range(1, num_steps):
        dt = time_step
        desired_position = position[i - 1] + speed[i - 1] * dt
        
        leader_v = leader_speed[i - 1]
        
        # Calculate acceleration using IDM formula
        acceleration = idm_acceleration(speed[i - 1], leader_v, leader_position[i - 1] - position[i - 1])
        
        acl[i] = acceleration
        speed[i] = speed[i - 1] + acceleration * dt
        position[i] = position[i - 1] + speed[i - 1] * dt + 0.5 * acceleration * (dt ** 2)
        
    return position, speed, acl

def fitness(params):
    # Define weights for position and speed errors
    weight_position = 1.0  # Full weight for position
    weight_speed = 0.5     # Half weight for speed

    # Simulate car following
    sim_position, sim_speed, acl = simulate_car_following(params)
    diff_position = np.array(sim_position) - np.array(target_position)
    diff_speed = np.array(sim_speed) - np.array(target_speed)
    
    # Calculate errors with weights
    mse_position = np.mean(diff_position ** 2) * weight_position
    mse_speed = np.mean(diff_speed ** 2) * weight_speed
    mse = mse_position + mse_speed
    
    rmse_position = np.sqrt(mse_position)
    rmse_speed = np.sqrt(mse_speed)
    rmse = np.sqrt(mse)
    
    mae_position = np.mean(np.abs(diff_position)) * weight_position
    mae_speed = np.mean(np.abs(diff_speed)) * weight_speed
    mae = mae_position + mae_speed
    
    mape_position = np.mean(np.abs(diff_position / np.array(target_position))) * 100 * weight_position
    mape_speed = np.mean(np.abs(diff_speed / np.array(target_speed))) * 100 * weight_speed
    mape = (mape_position + mape_speed) / 2
    
    nrmse_position = rmse_position / (np.max(target_position) - np.min(target_position))
    nrmse_speed = rmse_speed / (np.max(target_speed) - np.min(target_speed))
    nrmse = (nrmse_position * weight_position + nrmse_speed * weight_speed) / (weight_position + weight_speed)
    
    sse_position = np.sum(diff_position ** 2) * weight_position
    sse_speed = np.sum(diff_speed ** 2) * weight_speed
    sse = sse_position + sse_speed
    
    ss_res_position = np.sum(diff_position ** 2) * weight_position
    ss_tot_position = np.sum((np.array(target_position) - np.mean(target_position)) ** 2)
    r2_position = 1 - (ss_res_position / ss_tot_position)

    ss_res_speed = np.sum(diff_speed ** 2) * weight_speed
    ss_tot_speed = np.sum((np.array(target_speed) - np.mean(target_speed)) ** 2)
    r2_speed = 1 - (ss_res_speed / ss_tot_speed)
    
    r2 = (r2_position * weight_position + r2_speed * weight_speed) / (weight_position + weight_speed)
    
    total_diff = np.sum(np.abs(diff_position)) * weight_position + np.sum(np.abs(diff_speed)) * weight_speed
    
    # Fitness is the inverse of total error to maximize fitness
    fitness_value = 1.0 / (total_diff + 1e-5)
    
    # Store all error metrics in a dictionary
    error_metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'NRMSE': nrmse,
        'SSE': sse,
        'R-squared': r2,
        'Total Difference': total_diff
    }
    
    return fitness_value, error_metrics  # Return fitness and all error metrics


def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(child):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] += random.uniform(-0.1, 0.1)
    return child

# Modify genetic_algorithm function
def genetic_algorithm():
    # Define parameter ranges including delta
    t_range = (0.5, 2.5)   # Time headway in seconds
    a_range = (0.3, 3)     # Minimum gap in meters
    b_range = (0.5, 3)     # Comfortable deceleration in m/s^2
    v0_range = (5, 35)     # Desired velocity in m/s
    so_range = (1, 5)      # Acceleration relaxation time in seconds
    delta_range = (3.8, 4.2)  # Exponent in IDM formula

    # Initialize population with random parameter values
    population = [[random.uniform(*range_) for range_ in (t_range, a_range, b_range, v0_range, so_range, delta_range)]
                  for _ in range(population_size)]
    
    best_error = float('inf')  # Initialize best error
    best_individual = None  # Initialize best individual
    best_metrics = None  # Initialize best error metrics
    
    for generation in range(num_generations):
        # Evaluate fitness and errors
        fitness_and_errors = [fitness(individual) for individual in population]
        
        # Sorting population based on fitness (first element of tuple), in descending order
        population_sorted = sorted(zip(population, fitness_and_errors), key=lambda x: x[1][0], reverse=True)
        
        # Update population with sorted individuals
        population = [ind for ind, _ in population_sorted]
        
        # Update best individual and best error if a better one is found
        current_best_error = population_sorted[0][1][1]['Total Difference']  # Error is the second element of the fitness_and_errors tuple
        if current_best_error < best_error:
            best_error = current_best_error
            best_individual = population_sorted[0][0]
            best_metrics = population_sorted[0][1][1]  # Best error metrics
        
        # Parent selection (top half of the sorted population)
        parents = population[:len(population) // 2]
        
        # Generate children via crossover and mutation
        children = []
        while len(children) < (population_size - len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            children.extend([mutate(child1), mutate(child2)])
        
        # The new population consists of the parents and the children
        population = parents + children[:population_size - len(parents)]
    
    # Return the best individual, best error, and best error metrics after all generations
    return best_individual, best_error, best_metrics


def plot_simulation(timex, leader_position, target_position, sim_position, leader_speed, target_speed, sim_speed, follower_id, most_leading_leader_id, run_index, save_dir):
    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    plt.plot(timex, leader_position, label='Leader')
    plt.plot(timex, target_position, label='Target')
    plt.plot(timex, sim_position, label='Simulated Follower')
    plt.xlabel('time (sec)')
    plt.ylabel('Position (m)')
    plt.title(f'Position vs time, FID: {follower_id}, LID: {int(most_leading_leader_id)}, run: {run_index}')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(timex, leader_speed, label='Leader')
    plt.plot(timex, target_speed, label='Target')
    plt.plot(timex, sim_speed, label='Simulated Follower')
    plt.xlabel('time (sec)')
    plt.ylabel('Speed (m/s)')
    plt.title(f'Speed vs time, FID: {follower_id}, LID: {int(most_leading_leader_id)}, run: {run_index}')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(save_dir, f'{outname}_FID_{follower_id}_LID_{int(most_leading_leader_id)}_run_{run_index}.png')
    plt.savefig(plot_filename)
    #plt.show()
    plt.close()

def visualize_parameter_distributions(all_params):
    # Assuming all_params is a list of lists, with each inner list containing a set of parameters
    param_names = ['T', 'a', 'b', 'v0', 'so', 'delta']

    num_params = len(param_names)
    
    # Convert list of lists into a 2D numpy array for easier column-wise access
    all_params_array = np.array(all_params)

    print(f"Shape of all_params_array: {all_params_array.shape}")
    print(f"Number of param_names: {len(param_names)}")
    
    # Create histograms for each parameter
    fig, axs = plt.subplots(1, num_params, figsize=(20, 4))
    for i in range(num_params):
        axs[i].hist(all_params_array[:, i], bins=20, color='skyblue', edgecolor='black')
        axs[i].set_title(param_names[i])
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

    # Optionally, create box plots for each parameter
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_params_array, labels=param_names, patch_artist=True)
    plt.title('Distribution of IDM Parameters')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



    #### FUNCTIONS

    def find_leader_data(df, follower_id, run_index):
    global most_leading_leader_id
    
    follower_data = df[(df['ID'] == follower_id) & (df['run-index'] == run_index)]
    leader_data_dict = {}  # Dictionary to store leader data by leader ID
    
    for index, row in follower_data.iterrows():
        time = row['time']
        follower_x = row[pos]  # Assuming 'pos' is the column for follower's position
        follower_lane = row['lane-kf']  # Assuming the lane information is in 'lane-kf' column
        run_index = row['run-index']  # Assuming 'run-index' is the column for run index

        # Find the nearest leader with a greater position
        leader_data = df[(df['ID'] != follower_id) & (df['time'] == time) & (df['lane-kf'] == follower_lane) & (df[pos] > follower_x) & (df['run-index'] == run_index)]
        
        if not leader_data.empty:
            # Find the nearest leader
            nearest_leader_row = leader_data.loc[leader_data[pos].sub(follower_x).abs().idxmin()]
            
            leader_id = nearest_leader_row['ID']
            leader_x_val = nearest_leader_row[pos]
            leader_speed_val = nearest_leader_row['speed-kf']

            if leader_id not in leader_data_dict:
                leader_data_dict[leader_id] = {'time': [], 'x_val': [], 'speed_val': []}

            leader_data_dict[leader_id]['time'].append(time)
            leader_data_dict[leader_id]['x_val'].append(leader_x_val)
            leader_data_dict[leader_id]['speed_val'].append(leader_speed_val)

    # Choose the most leading leader by selecting the leader with the maximum number of occurrences in the leader_data_dict
    if leader_data_dict:
        most_leading_leader_id = max(leader_data_dict, key=lambda x: len(leader_data_dict[x]['time']))
        leader_data = leader_data_dict[most_leading_leader_id]
        leader_df = pd.DataFrame({'ID': most_leading_leader_id,
                                   'time': leader_data['time'],
                                   pos: leader_data['x_val'],
                                   'speed-kf': leader_data['speed_val'],
                                   'run-index': run_index})
    else:
        leader_df = pd.DataFrame(columns=['ID', 'time', pos, 'speed-kf', 'run-index'])
    
    return leader_df


def extract_subject_and_leader_data(df, follower_id, run_index):
    sdf = df[(df['ID'] == follower_id) & (df['run-index'] == run_index)].round(2)
    ldf = find_leader_data(df, follower_id, run_index).round(2)
    
    # Find the intersection of time frames between leader and subject
    mutual_times = np.intersect1d(ldf['time'], sdf['time'])
    
    # Find the longest continuous segment of mutual time
    max_continuous_mutual_times = []
    continuous_mutual_times = []
    prev_time = None
    for time in mutual_times:
        if prev_time is None or time - prev_time < 0.2:  # Assuming the time step is 0.1
            continuous_mutual_times.append(time)
        else:
            if len(continuous_mutual_times) > len(max_continuous_mutual_times):
                max_continuous_mutual_times = continuous_mutual_times
            continuous_mutual_times = [time]
        prev_time = time
    
    if len(continuous_mutual_times) > len(max_continuous_mutual_times):
        max_continuous_mutual_times = continuous_mutual_times
    
    # Calculate the duration of car-following
    if max_continuous_mutual_times:
        duration = max_continuous_mutual_times[-1] - max_continuous_mutual_times[0]
    else:
        duration = 0  # No mutual time found

    # Filter leader and subject data to include only the longest continuous mutual time
    ldf = ldf[ldf['time'].isin(max_continuous_mutual_times)]
    sdf = sdf[sdf['time'].isin(max_continuous_mutual_times)]
    
    if (isinstance(sdf, list) and not sdf) or (isinstance(sdf, pd.DataFrame) and sdf.empty):
        print(f"No subject data found for Follower ID {follower_id} and Run Index {run_index}.")
        return pd.DataFrame(), pd.DataFrame(), 0  # Return empty DataFrames and duration 0
    
    else:
        start_time = sdf['time'].iloc[0]
        ldf['time'], sdf['time'] = ldf['time'] - start_time, sdf['time'] - start_time
        return sdf, ldf, duration



def idm_acceleration(v, v_leader, s):
    max_v = 40  # Maximum allowable velocity (adjust as needed)
    max_s = 1000  # Maximum allowable gap (adjust as needed)
    
    s_star = so + max (0,(v * T + (v * (v - v_leader)) / (2 * np.sqrt(a * b))))
    #print (s_star)
    acceleration = a * (1 - (v / min(v0, max_v)) ** delta  - (s_star / min(s, max_s))**2)
    if np.isnan(acceleration):
        acceleration = 0
    #print (acceleration)
    return acceleration

def simulate_car_following(params):
    global T, a, b, v0, so, delta
    T, a, b, v0, so, delta = params  # Include delta from the parameters
    
    num_steps = round(total_time / time_step)
    time = np.linspace(0, total_time, num_steps)
    
    position = np.zeros(num_steps)
    speed = np.zeros(num_steps)
    acl = np.zeros(num_steps)
    
    position[0] = sdf.iloc[0][pos]
    speed[0] = sdf.iloc[0]['speed-kf']
    acl[0] = 0

    for i in range(1, num_steps):
        dt = time_step
        desired_position = position[i - 1] + speed[i - 1] * dt
        
        leader_v = leader_speed[i - 1]
        
        # Calculate acceleration using IDM formula
        acceleration = idm_acceleration(speed[i - 1], leader_v, leader_position[i - 1] - position[i - 1])
        
        acl[i] = acceleration
        speed[i] = speed[i - 1] + acceleration * dt
        position[i] = position[i - 1] + speed[i - 1] * dt + 0.5 * acceleration * (dt ** 2)
        
    return position, speed, acl

def fitness(params):
    # Define weights for position and speed errors
    weight_position = 1.0  # Full weight for position
    weight_speed = 0.5     # Half weight for speed

    # Simulate car following
    sim_position, sim_speed, acl = simulate_car_following(params)
    diff_position = np.array(sim_position) - np.array(target_position)
    diff_speed = np.array(sim_speed) - np.array(target_speed)
    
    # Calculate errors with weights
    mse_position = np.mean(diff_position ** 2) * weight_position
    mse_speed = np.mean(diff_speed ** 2) * weight_speed
    mse = mse_position + mse_speed
    
    rmse_position = np.sqrt(mse_position)
    rmse_speed = np.sqrt(mse_speed)
    rmse = np.sqrt(mse)
    
    mae_position = np.mean(np.abs(diff_position)) * weight_position
    mae_speed = np.mean(np.abs(diff_speed)) * weight_speed
    mae = mae_position + mae_speed
    
    mape_position = np.mean(np.abs(diff_position / np.array(target_position))) * 100 * weight_position
    mape_speed = np.mean(np.abs(diff_speed / np.array(target_speed))) * 100 * weight_speed
    mape = (mape_position + mape_speed) / 2
    
    nrmse_position = rmse_position / (np.max(target_position) - np.min(target_position))
    nrmse_speed = rmse_speed / (np.max(target_speed) - np.min(target_speed))
    nrmse = (nrmse_position * weight_position + nrmse_speed * weight_speed) / (weight_position + weight_speed)
    
    sse_position = np.sum(diff_position ** 2) * weight_position
    sse_speed = np.sum(diff_speed ** 2) * weight_speed
    sse = sse_position + sse_speed
    
    ss_res_position = np.sum(diff_position ** 2) * weight_position
    ss_tot_position = np.sum((np.array(target_position) - np.mean(target_position)) ** 2)
    r2_position = 1 - (ss_res_position / ss_tot_position)

    ss_res_speed = np.sum(diff_speed ** 2) * weight_speed
    ss_tot_speed = np.sum((np.array(target_speed) - np.mean(target_speed)) ** 2)
    r2_speed = 1 - (ss_res_speed / ss_tot_speed)
    
    r2 = (r2_position * weight_position + r2_speed * weight_speed) / (weight_position + weight_speed)
    
    total_diff = np.sum(np.abs(diff_position)) * weight_position + np.sum(np.abs(diff_speed)) * weight_speed
    
    # Fitness is the inverse of total error to maximize fitness
    fitness_value = 1.0 / (total_diff + 1e-5)
    
    # Store all error metrics in a dictionary
    error_metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'NRMSE': nrmse,
        'SSE': sse,
        'R-squared': r2,
        'Total Difference': total_diff
    }
    
    return fitness_value, error_metrics  # Return fitness and all error metrics

  
    


def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(child):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] += random.uniform(-0.1, 0.1)
    return child

# Modify genetic_algorithm function
def genetic_algorithm():
    # Define parameter ranges including delta
    t_range = (0.5, 2.5)   # Time headway in seconds
    a_range = (0.3, 3)     # Minimum gap in meters
    b_range = (0.5, 3)     # Comfortable deceleration in m/s^2
    v0_range = (5, 35)     # Desired velocity in m/s
    so_range = (1, 5)      # Acceleration relaxation time in seconds
    delta_range = (3.8, 4.2)  # Exponent in IDM formula

    # Initialize population with random parameter values
    population = [[random.uniform(*range_) for range_ in (t_range, a_range, b_range, v0_range, so_range, delta_range)]
                  for _ in range(population_size)]
    
    best_error = float('inf')  # Initialize best error
    best_individual = None  # Initialize best individual
    best_metrics = None  # Initialize best error metrics
    
    for generation in range(num_generations):
        # Evaluate fitness and errors
        fitness_and_errors = [fitness(individual) for individual in population]
        
        # Sorting population based on fitness (first element of tuple), in descending order
        population_sorted = sorted(zip(population, fitness_and_errors), key=lambda x: x[1][0], reverse=True)
        
        # Update population with sorted individuals
        population = [ind for ind, _ in population_sorted]
        
        # Update best individual and best error if a better one is found
        current_best_error = population_sorted[0][1][1]['Total Difference']  # Error is the second element of the fitness_and_errors tuple
        if current_best_error < best_error:
            best_error = current_best_error
            best_individual = population_sorted[0][0]
            best_metrics = population_sorted[0][1][1]  # Best error metrics
        
        # Parent selection (top half of the sorted population)
        parents = population[:len(population) // 2]
        
        # Generate children via crossover and mutation
        children = []
        while len(children) < (population_size - len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            children.extend([mutate(child1), mutate(child2)])
        
        # The new population consists of the parents and the children
        population = parents + children[:population_size - len(parents)]
    
    # Return the best individual, best error, and best error metrics after all generations
    return best_individual, best_error, best_metrics


def plot_simulation(timex, leader_position, target_position, sim_position, leader_speed, target_speed, sim_speed, follower_id, most_leading_leader_id, run_index, save_dir):
    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    plt.plot(timex, leader_position, label='Leader')
    plt.plot(timex, target_position, label='Target')
    plt.plot(timex, sim_position, label='Simulated Follower')
    plt.xlabel('time (sec)')
    plt.ylabel('Position (m)')
    plt.title(f'Position vs time, FID: {follower_id}, LID: {int(most_leading_leader_id)}, run: {run_index}')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(timex, leader_speed, label='Leader')
    plt.plot(timex, target_speed, label='Target')
    plt.plot(timex, sim_speed, label='Simulated Follower')
    plt.xlabel('time (sec)')
    plt.ylabel('Speed (m/s)')
    plt.title(f'Speed vs time, FID: {follower_id}, LID: {int(most_leading_leader_id)}, run: {run_index}')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(save_dir, f'{outname}_FID_{follower_id}_LID_{int(most_leading_leader_id)}_run_{run_index}.png')
    plt.savefig(plot_filename)
    #plt.show()
    plt.close()

def visualize_parameter_distributions(all_params):
    # Assuming all_params is a list of lists, with each inner list containing a set of parameters
    param_names = ['T', 'a', 'b', 'v0', 'so', 'delta']

    num_params = len(param_names)
    
    # Convert list of lists into a 2D numpy array for easier column-wise access
    all_params_array = np.array(all_params)

    print(f"Shape of all_params_array: {all_params_array.shape}")
    print(f"Number of param_names: {len(param_names)}")
    
    # Create histograms for each parameter
    fig, axs = plt.subplots(1, num_params, figsize=(20, 4))
    for i in range(num_params):
        axs[i].hist(all_params_array[:, i], bins=20, color='skyblue', edgecolor='black')
        axs[i].set_title(param_names[i])
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

    # Optionally, create box plots for each parameter
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_params_array, labels=param_names, patch_artist=True)
    plt.title('Distribution of IDM Parameters')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


### CALIBRATION PROCESS


import os
import pandas as pd
import numpy as np


##################################################################################################
datasets = {
    "df395": calibration_dataset_paths()["I395"],
    "df9094": calibration_dataset_paths()["I9094"],
    "df294l1": calibration_dataset_paths()["I294l1"],
    "df294l2": calibration_dataset_paths()["I294l2"],
}

groups = {
    "df395": ["I395_A", "I395_S", "I395_L"],
    "df9094": ["I9094_L", "I9094_S", "I9094_A"],
    "df294l1": ["I294l1_L", "I294l1_S", "I294l1_A"],
    "df294l2": ["I294l2_L", "I294l2_S", "I294l2_A"],
}

# Save directory for plots and data
save_dir = os.path.join(os.path.dirname(__file__), "Results", "idm_batch_plots")
#####################################################################################################

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# Iterate through each dataset and group
for df_key, df_path in datasets.items():
    df = load_tgsim_csv(df_path)
    df = df.sort_values(by='time')
    df['time'] = df['time'].round(1)
    pos = "yloc-kf" if df_key == "df395" else "xloc-kf"
    
    for group in groups[df_key]:
        # Define the output CSV file name for this group's parameters
        outname = f"IDM_Params_{group}.csv"
        output_csv_path = os.path.join(save_dir, outname)
        
        if os.path.exists(output_csv_path):
            print(f"Parameter file {output_csv_path} already exists. Skipping group {group}...")
            continue
        
        # Define the current group
        AVs = eval(group)
        
        # Lists to store results for the group
        params_list = []
        all_simulations_list = [] # <<< NEW: List to store simulation DataFrames

        for data in AVs:
            follower_id, run_index = data
            
            # Unpack the new start_time variable
            sdf, ldf, duration, start_time = extract_subject_and_leader_data(df, follower_id, run_index)
            print(f"Processing Follower ID: {follower_id}, Run Index: {run_index}")
        
            if sdf.empty:
                print(f"No data found for Follower ID {follower_id}, Run Index {run_index}. Skipping...")
                continue

            total_time = len(ldf) * 0.1
            time_step, num_steps = 0.1, round(total_time / 0.1)
            timex = np.linspace(0, total_time, num_steps)
            leader_position, leader_speed = ldf[pos].tolist(), ldf['speed-kf'].tolist()
            target_position, target_speed = sdf[pos].tolist(), sdf['speed-kf'].tolist()
            
            best_params, best_error, best_metrics = genetic_algorithm()
        
            if best_params is None or best_error is None or not best_metrics:
                print(f"Skipping Follower ID {follower_id}, Run Index {run_index} due to missing optimization results.")
                continue
        
            # Store the optimized parameters and error metrics
            params_list.append([follower_id, run_index, duration] + best_params + [best_error] + list(best_metrics.values()))
            
            # --- Simulation and Plotting (Existing) ---
            sim_position, sim_speed, acl = simulate_car_following(best_params)
            plot_simulation(
                timex, leader_position, target_position, sim_position,
                leader_speed, target_speed, sim_speed, follower_id,
                most_leading_leader_id, run_index, save_dir
            )

            # --- NEW: Create and store the simulation DataFrame ---
            sim_data = {
                'ID': follower_id,
                'run-index': run_index,
                'time': np.round(timex + start_time, 1), # Add start_time back for original timestamp
                pos: sim_position,
                'speed-kf': sim_speed,
                'sim_acceleration': acl
            }

            # To keep the format similar to the input, copy other relevant columns from the original data
            if len(sdf) == len(timex):
                 for col in ['lane-kf', 'type-most-common', 'ACC', 'AV']:
                     if col in sdf.columns:
                         sim_data[col] = sdf[col].values

            sim_df = pd.DataFrame(sim_data)
            all_simulations_list.append(sim_df)
            # --- End of New Section ---

        # After processing all vehicles in the group, save the collected data
        
        # Save the parameters to CSV (Existing logic)
        if params_list:
            metrics_names = list(best_metrics.keys())
            columns = ['Follower_ID', 'Run_Index', 'Duration', 'T', 'a', 'b', 'v0', 'so', 'delta', 'Error'] + metrics_names
            params_df = pd.DataFrame(params_list, columns=columns)
            params_df.to_csv(output_csv_path, index=False)
            print(f"Saved parameters for {group} to {output_csv_path}")

        # --- NEW: Save the concatenated simulation data to a separate CSV ---
        if all_simulations_list:
            sim_outname = f"IDM_Simulated_{group}.csv"
            sim_output_csv_path = os.path.join(save_dir, sim_outname)
            
            final_simulations_df = pd.concat(all_simulations_list, ignore_index=True)
            final_simulations_df.to_csv(sim_output_csv_path, index=False)
            print(f"Saved simulated trajectories for {group} to {sim_output_csv_path}")
        # --- End of New Section ---