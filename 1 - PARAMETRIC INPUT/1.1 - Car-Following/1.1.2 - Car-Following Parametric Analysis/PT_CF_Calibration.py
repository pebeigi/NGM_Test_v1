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


population_size, num_generations, mutation_rate = 40, 80, 0.1  #simulation parameters
accl_max, v_desired, Tcorr, RT = 3.0, 36.0, 20.0, 0.6 #suggested values from the paper and v_desired=36 is the v_desired from the data
most_leading_leader_id = None


def generate_vehicle_groups(datasets):
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
    leader_data_dict = {}
    
    for index, row in follower_data.iterrows():
        time = row['time']
        follower_x = row[pos]
        follower_lane = row['lane-kf']
        run_index = row['run-index']

        #find the leader
        leader_data = df[(df['ID'] != follower_id) & (df['time'] == time) & (df['lane-kf'] == follower_lane) & (df[pos] > follower_x) & (df['run-index'] == run_index)]
        
        if not leader_data.empty:
            nearest_leader_row = leader_data.loc[leader_data[pos].sub(follower_x).abs().idxmin()]
            
            leader_id = nearest_leader_row['ID']
            leader_x_val = nearest_leader_row[pos]
            leader_speed_val = nearest_leader_row['speed-kf']

            if leader_id not in leader_data_dict:
                leader_data_dict[leader_id] = {'time': [], 'x_val': [], 'speed_val': []}

            leader_data_dict[leader_id]['time'].append(time)
            leader_data_dict[leader_id]['x_val'].append(leader_x_val)
            leader_data_dict[leader_id]['speed_val'].append(leader_speed_val)

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
    
    #find the intersection of time frames between leader and subject
    mutual_times = np.intersect1d(ldf['time'], sdf['time'])
    
    #find the longest continuous segment of mutual time
    max_continuous_mutual_times = []
    continuous_mutual_times = []
    prev_time = None
    for time in mutual_times:
        if prev_time is None or time - prev_time < 0.2:  #the time step is 0.1
            continuous_mutual_times.append(time)
        else:
            if len(continuous_mutual_times) > len(max_continuous_mutual_times):
                max_continuous_mutual_times = continuous_mutual_times
            continuous_mutual_times = [time]
        prev_time = time
    
    if len(continuous_mutual_times) > len(max_continuous_mutual_times):
        max_continuous_mutual_times = continuous_mutual_times
    
    #filter leader and subject data to include only the longest continuous mutual time
    ldf = ldf[ldf['time'].isin(max_continuous_mutual_times)]
    sdf = sdf[sdf['time'].isin(max_continuous_mutual_times)]
    
    if (isinstance(sdf, list) and not sdf) or (isinstance(sdf, pd.DataFrame) and sdf.empty):
        print(f"No subject data found for Follower ID {follower_id} and Run Index {run_index}.")
        empty_df = pd.DataFrame()
        # Return a default start_time of 0 when no data is found
        return empty_df, empty_df, 0
    
    else:
        # Capture the start_time before normalizing
        start_time = sdf['time'].iloc[0]
        ldf['time'], sdf['time'] = ldf['time'] - start_time, sdf['time'] - start_time
        # Return the captured start_time
        return sdf, ldf, start_time
    

def acceleration_calculator(i, t, vehicle, accl_max, v_desired, Gamma1, Gamma2, Wm, Wc, Tmax, Alpha, Beta, Tcorr, RT, prng):
    So_D = 3 #default value by Talebpour
    if (vehicle['gap'] - So_D) > 0.1:
        Seff = vehicle['gap'] - So_D
    else:
        Seff = 0.1 #default value by Talebpour

    if vehicle['deltav'] > (Seff / Tmax): #correct
        Tau = Seff / vehicle['deltav']
    else:
        Tau = Tmax

    if vehicle['deltav'] == 0:
        vehicle['deltav'] = 0.0000001 #default value by Talebpour
    if Alpha == 0:
        Alpha = 0.0000001 #default value by Talebpour
    
    Zprime = Tau / (2.0 * Alpha * vehicle['speed']) # correct
    Zdoubleprime = 0.0

    #if Wc * Zprime >= 1:
    if Wc * Zprime > 0:
        a0 = 1
        #Zstar = (-1 * math.sqrt(2.0 * math.log(a0 * Wc * Zprime))) / (math.sqrt(2.0 * math.pi)) #default by Talebpour
        Zstar = -np.sqrt(2 * np.log(a0 * Wc * Zprime / np.sqrt(2 * np.pi))) #sharika
    else:
        Zstar = 0.0
    Astar = (2.0 / Tau) * ((Seff / Tau) - vehicle['deltav'] + (Alpha * vehicle['speed'] * Zstar)) #correct
    
    for NewtonCounter in range(3):
        X = Astar 
        if X >= 0:
            if X == 0:
                X = 0.0000001 #default value by Talebpour
            Uptprime = Gamma1 * math.pow(X, Gamma1 - 1)
            Uptdoubleprime = Gamma1 * (Gamma1 - 1) * math.pow(X, Gamma1 - 2)
        else:
            Uptprime = Wm * Gamma2 * pow(-X, Gamma2 - 1)
            Uptdoubleprime = -Wm * Gamma2 * (Gamma2 - 1) * pow(-X, Gamma2 - 2)

        Z = (vehicle['deltav'] + (0.5 * X * Tau) - (Seff / Tau)) / (Alpha * vehicle['deltav']) # density of a Gaussian # correct
        #fn = norm.cdf(Z)
        fn = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-Z ** 2.0 / 2.0) #sharika PDF
        F = Uptprime - Wc * fn * Zprime # correct
        
        Fprime = Uptdoubleprime - Wc * fn * (Z * math.pow(Zprime, 2.0) + Zdoubleprime)
        if Fprime == 0:
            Fprime = 0.000000000001 #default value by Talebpour

        Astar = Astar - (F / Fprime)

    X = Astar
    if X >= 0:
        Uptprime = Gamma1 * (X ** max(Gamma1 - 1, 0))  # Ensures no negative base issue
        Uptdoubleprime = Gamma1 * (Gamma1 - 1) * (X ** max(Gamma1 - 2, 0))
    else:
        Uptprime = Wm * Gamma2 * ((-X) ** max(Gamma2 - 1, 0))
        Uptdoubleprime = -Wm * Gamma2 * (Gamma2 - 1) * ((-X) ** max(Gamma2 - 2, 0))

    """ original code with the negative issue
    if X >= 0:
        Uptprime = Gamma1 * math.pow(X, Gamma1 - 1)
        Uptdoubleprime = Gamma1 * (Gamma1 - 1) * math.pow(X, Gamma1 - 2)
    else:
        Uptprime = Wm * Gamma2 * math.pow(-X, Gamma2 - 1)
        Uptdoubleprime = -Wm * Gamma2 * (Gamma2 - 1) * math.pow(-X, Gamma2 - 2)
    """
    
    Z = (vehicle['deltav'] + (0.5 * Astar * Tau) - (Seff / Tau)) / (Alpha * vehicle['deltav'])
    # fn = norm.cdf(Z) #default value by Talebpour
    fn = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-Z ** 2.0 / 2.0) #sharika PDF
    
    F = Uptprime - Wc * fn * Zprime
    Fprime = Uptdoubleprime - Wc * fn * (Z * math.pow(Zprime, 2.0) + Zdoubleprime)
    if Fprime == 0:
        Fprime = 0.000000000001

    Var = -1.0 / (Beta * Fprime)
    
    Random_Wiener = np.random.rand()
    
    Yt = math.exp(-1 * 0.1 / Tau) + math.sqrt(24.0 * 0.1 / Tau) * Random_Wiener #default value by Talebpour
    
    accl_cf = Astar + Var * Yt
    accl_ff = accl_max * (1 - (vehicle['speed'] / v_desired))
    accl_ = np.minimum(accl_cf, accl_ff)

    if accl_ > 3: #default value by Talebpour
        accl_ = 3
    elif accl_ < -8: #default value by Talebpour
        accl_ = -8
    return accl_, fn, Wc * fn
    
###### 2nd Version for Nachuan #####
def acceleration_calculator(i, t, vehicle, accl_max, v_desired, Gamma1, Gamma2,
                        Wm, Wc, Tmax, Alpha, Beta, Tcorr, RT, prng,
                        prev_accel):

    So_D = 3
    Seff = max(vehicle['gap'] - So_D, 0.1)

    deltav = vehicle['deltav'] if abs(vehicle['deltav']) > 1e-6 else 1e-6
    speed  = max(vehicle['speed'], 0.1)
    Alpha  = max(Alpha, 1e-6)

    Tau = Seff/deltav if deltav > (Seff/Tmax) else Tmax

    Zprime = Tau / (2 * Alpha * speed)
    Zdoubleprime = 0.0

    if Wc * Zprime >= 1:
        Zstar = -np.sqrt(2*np.log(Wc * Zprime))
    else:
        Zstar = 0.0

    Astar = (2/Tau)*((Seff/Tau) - deltav + Alpha*speed*Zstar)

    # Limit Astar before Newton to avoid blow-up
    Astar = np.clip(Astar, -5, 2)

    # No Yt random noise
    accl_cf = Astar

    accl_ff = accl_max * (1 - speed/v_desired)

    accl_raw = min(accl_cf, accl_ff)

    # Final smoothing
    accl = 0.3 * prev_accel + 0.7 * accl_raw

    return np.clip(accl, -8, 3)

def simulate_car_following(params):
    global Tmax, Alpha, Beta, Wc, Gamma1, Gamma2, Wm
    Tmax, Alpha, Beta, Wc, Gamma1, Gamma2, Wm = params
    
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
        
        acceleration, _, _ = acceleration_calculator(i, time[i], {'gap': leader_position[i-1] - position[i-1], 'deltav': leader_speed[i-1] - speed[i-1], 'speed': speed[i-1], 'vehID': follower_id}, accl_max, v_desired, Gamma1, Gamma2, Wm, Wc, Tmax, Alpha, Beta, Tcorr, RT, np.random.default_rng())

        acl[i] = acceleration
        speed[i] = speed[i - 1] + acceleration * dt
        position[i] = position[i - 1] + speed[i-1] * dt + 0.5 * acceleration * (dt**2)
        
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

def genetic_algorithm():
    #define parameter ranges for PT model
    Tmax_range = (2, 8.0)
    Alpha_range = (0, 0.6)
    Beta_range = (2, 8)
    Wc_range = (60000, 130000)
    Gamma1_range = (0.3, 2.0)
    Gamma2_range = (0.3, 2.0)
    Wm_range = (2, 8.0)

    #population with random parameter values
    population = [[random.uniform(*range_) for range_ in (Tmax_range, Alpha_range, Beta_range, Wc_range, Gamma1_range, Gamma2_range, Wm_range)]
                  for _ in range(population_size)]
    
    best_error = float('inf')
    best_individual = None
    best_metrics = None
    
    for generation in range(num_generations):
        #evaluate fitness and errors
        fitness_and_errors = [fitness(individual) for individual in population]
        population_sorted = sorted(zip(population, fitness_and_errors), key=lambda x: x[1][0], reverse=True)
        population = [ind for ind, _ in population_sorted]
        
        #Update best individual and best error if a better one is found
        current_best_error = population_sorted[0][1][1]['Total Difference']  # Error is the second element of the fitness_and_errors tuple
        if current_best_error < best_error:
            best_error = current_best_error
            best_individual = population_sorted[0][0]
            best_metrics = population_sorted[0][1][1]  # Best error metrics
        
        #Parent selection (top half of the sorted population)
        parents = population[:len(population) // 2]
        
        children = []
        while len(children) < (population_size - len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            children.extend([mutate(child1), mutate(child2)])
        population = parents + children[:population_size - len(parents)]
    
    #return the best individual, best error, and best error metrics after all generations
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
    plt.close()

def visualize_parameter_distributions(all_params):
    param_names = ['Tmax', 'Alpha', 'Beta', 'Wc', 'Gamma1', 'Gamma2', 'Wm']
    num_params = len(param_names)
    
    #convert list of lists into a 2D numpy array for easier column-wise access
    all_params_array = np.array(all_params)
    
    #histograms for each parameter
    fig, axs = plt.subplots(1, num_params, figsize=(20, 4))
    for i in range(num_params):
        axs[i].hist(all_params_array[:, i], bins=20, color='skyblue', edgecolor='black')
        axs[i].set_title(param_names[i])
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

    #create box plots for each parameter
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_params_array, labels=param_names, patch_artist=True)
    plt.title('Distribution of PT Model Parameters')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



import os
import pandas as pd
import numpy as np

###############################################################################################################
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
    "df294l2": ["I294l2_L", "I294l2_S", "I294l2_A"]}

# Save directory for plots
save_dir = os.path.join(os.path.dirname(__file__), "Results", "pt_batch_plots")
################################################################################################################


# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# Iterate through each dataset and group
for df_key, df_path in datasets.items():
    df = load_tgsim_csv(df_path)
    df = df.sort_values(by='time')
    df['time'] = df['time'].round(1)
    #pos = "xloc-kf"
    pos = "yloc-kf" if df_key == "df395" else "xloc-kf"
    
    for group in groups[df_key]:
        # Define the output CSV file name for this group's parameters
        outname = f"PT_Params_{group}.csv"
        output_csv_path = os.path.join(save_dir, outname)
        
        # Check if the CSV file already exists
        if os.path.exists(output_csv_path):
            print(f"Parameter file {output_csv_path} already exists. Skipping group {group}...")
            continue
        
        # Define the current group and initialize lists for results
        AVs = eval(group)
        params_list = []
        # NEW: List to store simulation DataFrames for the group
        all_simulations_list = [] 

        for data in AVs:
            follower_id, run_index = data
            # Unpack the new start_time variable from the modified function
            sdf, ldf, start_time = extract_subject_and_leader_data(df, follower_id, run_index)
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
            
            # Store optimized parameters
            params_list.append([follower_id, run_index] + best_params + [best_error] + list(best_metrics.values()))
            
            # Simulate and plot with best parameters (existing code)
            sim_position, sim_speed, acl = simulate_car_following(best_params)
            plot_simulation(
                timex, leader_position, target_position, sim_position,
                leader_speed, target_speed, sim_speed, follower_id,
                most_leading_leader_id, run_index, save_dir)
            
            # ✨ NEW: Create and store a DataFrame for the current simulation
            sim_data = {
                'ID': follower_id,
                'run-index': run_index,
                'time': np.round(timex + start_time, 1), # Restore the original timestamp
                pos: sim_position,
                'speed-kf': sim_speed,
                'sim_acceleration': acl
            }

            # Copy other relevant columns from the original data to match the format
            if len(sdf) == len(timex):
                 for col in ['lane-kf', 'type-most-common', 'ACC', 'AV']:
                     if col in sdf.columns:
                         sim_data[col] = sdf[col].values

            sim_df = pd.DataFrame(sim_data)
            all_simulations_list.append(sim_df)
            # --- End of New Section ---
        
        # After processing all vehicles in the group, save the collected data
        
        # Save parameters to CSV (existing logic, moved for clarity)
        if params_list:
            # Visualize parameter distributions
            visualize_parameter_distributions(all_params)
            
            metrics_names = list(best_metrics.keys())
            columns = ['Follower_ID', 'Run_Index', 'Tmax', 'Alpha', 'Beta', 'Wc', 'Gamma1', 'Gamma2', 'Wm', 'Error'] + metrics_names
            params_df = pd.DataFrame(params_list, columns=columns)
            params_df.to_csv(output_csv_path, index=False)
            print(f"Saved parameters for {group} to {output_csv_path}")
            
        # NEW: Save the concatenated simulation data to a separate CSV file
        if all_simulations_list:
            sim_outname = f"PT_Simulated_{group}.csv"
            sim_output_csv_path = os.path.join(save_dir, sim_outname)
            
            final_simulations_df = pd.concat(all_simulations_list, ignore_index=True)
            final_simulations_df.to_csv(sim_output_csv_path, index=False)
            print(f"Saved simulated trajectories for {group} to {sim_output_csv_path}")
        # --- End of New Section ---
