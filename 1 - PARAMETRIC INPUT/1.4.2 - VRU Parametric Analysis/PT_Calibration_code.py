# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 18:41:49 2025

@author: TBP

Demonstration: Prospect Theory (PT) model for pedestrians & bicycles,
with expanded collision-weight parameters:

  - Wc_pp    (ped–ped collision weight)
  - Wc_pb    (ped–bike collision weight, from pedestrian's perspective)
  - Wc_pbar  (ped–barrier collision weight)
  - Wc_bp    (bike–ped collision weight, from bicyclist's perspective)
  - Wc_bb    (bike–bike collision weight)
  - Wc_bbar  (bike–barrier collision weight)

Plus PT parameters for each mode:

  - eta_ped, xi_ped, tau_ped, v_pref_ped
    (Pedestrian amplitude, nonlinearity, reaction time, and desired speed)
  - eta_bike, xi_bike, tau_bike, v_pref_bike
    (Bicycle amplitude, nonlinearity, reaction time, and desired speed)

We calibrate each ID with a genetic algorithm (GA). Each ID gets its own best-fit parameter set.
All lanes are filtered so that we only keep LANES_TO_KEEP = [1, 39, 40, 41].
"""

import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os
import sys

_NGM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _NGM_ROOT not in sys.path:
    sys.path.insert(0, _NGM_ROOT)
from ngm_paths import FOGGY_BOTTOM_CSV, dataset_path

##############################################################################
# 1) GLOBAL SETTINGS
##############################################################################

CSV_PATH = dataset_path(FOGGY_BOTTOM_CSV)
# Since x and y are in meters, we don't need PIX_TO_M anymore.

PEDESTRIAN_CODE = 0
BICYCLE_CODE    = 1

# We only keep these lanes (if you still need lane filtering)
LANES_TO_KEEP = [1, 39, 40, 41]

NEIGHBOR_RADIUS = 5.0  # meters
SAMPLE_STEP     = 1

MIN_PRESENCE    = 50   # e.g. min # of timesteps

POP_SIZE        = 20
GENERATIONS     = 40
MUTATION_RATE   = 0.2
ELITISM_COUNT   = 4

r2_threshold = 0.95

OUTPUT_DIR = "./pt_collision_weights_expanded"
os.makedirs(OUTPUT_DIR, exist_ok=True)

##############################################################################
# 4) PARAM VECTOR: 14 parameters (6 collision + 8 PT)
##############################################################################

def random_pt_params():
    return [
        random.uniform(170, 210),    # Wc_pp
        random.uniform(180, 320),    # Wc_pb
        random.uniform(25, 120),     # Wc_pbar
        random.uniform(1500, 2250),  # Wc_bp
        random.uniform(1550, 2200),  # Wc_bb
        random.uniform(900, 1650),   # Wc_bbar
        random.uniform(1, 5),        # eta_ped
        random.uniform(1, 5),        # xi_ped
        random.uniform(0.3, 1.5),    # tau_ped
        1.358,                       # v_pref_ped (fixed)
        random.uniform(2, 6),        # eta_bike
        random.uniform(2, 6),        # xi_bike
        random.uniform(0.3, 1.0),    # tau_bike
        3.073                        # v_pref_bike (fixed)
    ]

def clamp_pt_params(p):
    p[0] = min(max(p[0], 10),  2000)  
    p[1] = min(max(p[1], 10),  3000)  
    p[2] = min(max(p[2], 10),  2000)  
    p[3] = min(max(p[3], 10),  3000)  
    p[4] = min(max(p[4], 10),  5000)  
    p[5] = min(max(p[5], 10),  3000)  

    p[6] = min(max(p[6], 0.1),  10)   
    p[7] = min(max(p[7], 0.5),  10)   
    p[8] = min(max(p[8], 0.1),  3.0)  
    p[9] = 1.358                     

    p[10]= min(max(p[10],0.1),  10)   
    p[11]= min(max(p[11],0.5),  10)   
    p[12]= min(max(p[12],0.1),  3.0)  
    p[13]= 3.073                     

    return p

def mutate_pt_params(p, rate=0.2):
    for i in range(len(p)):
        if random.random() < rate:
            if i in [0,1,2,3,4,5]:     
                p[i] += random.uniform(-50, 50)
            elif i in [6,7,8,10,11,12]:
                p[i] += random.uniform(-0.5, 0.5)
    return clamp_pt_params(p)

def crossover_pt_params(p1, p2):
    idx = random.randint(1, len(p1) - 1)
    c1 = p1[:idx] + p2[idx:]
    c2 = p2[:idx] + p1[idx:]
    return clamp_pt_params(c1), clamp_pt_params(c2)

##############################################################################
# 5) PT UTILITY: picking best (v,theta)
##############################################################################

def choose_pt_velocity_direction(px, py, vx, vy, agent_type, pvec,
                                neighbors, lane_id, dt=0.1):
    # Use collision weights (even though boundaries are ignored)
    Wc_pp, Wc_pb, Wc_pbar, Wc_bp, Wc_bb, Wc_bbar = pvec[:6]

    if agent_type == PEDESTRIAN_CODE:
        eta, xi, tau, v_pref = pvec[6:10]
        half_w = 0.2
    else:
        eta, xi, tau, v_pref = pvec[10:14]
        half_w = 0.4

    curr_speed = math.hypot(vx, vy)
    curr_theta = 0.0 if curr_speed < 1e-9 else math.atan2(vy, vx)

    speed_candidates = np.linspace(0, 1.2 * v_pref, 7)
    angle_candidates = np.linspace(curr_theta - math.radians(60),
                                   curr_theta + math.radians(60), 9)

    best_utility = -1e9
    best_v = curr_speed
    best_th = curr_theta

    def subjective_value(v_, th_):
        align = max(0.0, math.cos(th_ - curr_theta))
        sp_ratio = v_ / max(1e-9, v_pref)
        base = eta * align
        if base <= 0:
            return 0.0
        exponent = (sp_ratio ** xi)
        return base ** exponent

    # In this simplified version, we ignore boundaries, so no use of distance
    def collision_prob(px2, py2):
        for (nx_, ny_, vx_n, vy_n, ttype) in neighbors:
            other_hw = 0.2 if ttype == PEDESTRIAN_CODE else 0.4
            if math.hypot(px2 - nx_, py2 - ny_) < half_w + other_hw:
                return 1.0
        return 0.0

    def collision_weight(px2, py2):
        min_dist = float('inf')
        min_type = None
        for (nx_, ny_, vx_n, vy_n, ttype) in neighbors:
            dist_ij = math.hypot(px2 - nx_, py2 - ny_)
            if dist_ij < min_dist:
                min_dist = dist_ij
                min_type = ttype
        if min_type is None:
            return 0.0
        else:
            if agent_type == PEDESTRIAN_CODE:
                return Wc_pp if min_type == PEDESTRIAN_CODE else Wc_pb
            else:
                return Wc_bb if min_type == BICYCLE_CODE else Wc_bp

    for sp in speed_candidates:
        for ag in angle_candidates:
            px2 = px + sp * math.cos(ag) * dt
            py2 = py + sp * math.sin(ag) * dt
            val = subjective_value(sp, ag)
            p_col = collision_prob(px2, py2)
            # Only penalize if collision probability is high
            w_c = collision_weight(px2, py2) if p_col > 0.5 else 0.0
            utility = (1.0 - p_col) * val - p_col * w_c
            if utility > best_utility:
                best_utility = utility
                best_v = sp
                best_th = ag

    vx_fin = best_v * math.cos(best_th)
    vy_fin = best_v * math.sin(best_th)
    ax = (vx_fin - vx) / dt
    ay = (vy_fin - vy) / dt
    return ax, ay

def pt_acceleration(px, py, vx, vy, agent_type, pvec,
                    neighbors, lane_id, dt=0.1):
    return choose_pt_velocity_direction(px, py, vx, vy, agent_type, pvec,
                                        neighbors, lane_id, dt)

##############################################################################
# 6) SINGLE-AGENT SIMULATION (MODIFIED TO BE DYNAMIC)
##############################################################################

def simulate_pt_single_agent(px, py, vx, vy, agent_type, pvec,
                             neighbors, lane_id,
                             total_dt=1.0, substeps=5):
    """
    Forward-Euler style integration for the focal agent *and* neighbors.
    Now each substep updates everyone so that the environment is dynamic.
    Returns the focal agent's final (x, y, vx, vy) after total_dt.
    """
    h = total_dt / substeps

    # 1) Combine the focal agent (index 0) + neighbors into a single list
    agents = []
    # The focal agent is first:
    agents.append({
        'x': px,
        'y': py,
        'vx': vx,
        'vy': vy,
        'lane': lane_id,
        'type': agent_type
    })
    # Then each neighbor is appended:
    for nb in neighbors:
        agents.append({
            'x':  nb[0],
            'y':  nb[1],
            'vx': nb[2],
            'vy': nb[3],
            'lane': lane_id,
            'type': nb[4]
        })

    # 2) Substeps: in each step, compute accelerations for all agents, then update.
    for _ in range(substeps):
        accel_list = []
        # Compute acceleration for each agent i
        for i, ag in enumerate(agents):
            # Build a local neighbor list for agent i
            local_nb = []
            for j, other_ag in enumerate(agents):
                if j == i:
                    continue
                dx = ag['x'] - other_ag['x']
                dy = ag['y'] - other_ag['y']
                dist = math.hypot(dx, dy)
                if dist <= NEIGHBOR_RADIUS:
                    local_nb.append((other_ag['x'], other_ag['y'],
                                     other_ag['vx'], other_ag['vy'],
                                     other_ag['type']))
            ax, ay = pt_acceleration(ag['x'], ag['y'], ag['vx'], ag['vy'],
                                     ag['type'], pvec, local_nb, ag['lane'],
                                     dt=h)
            accel_list.append((ax, ay))

        # Now update everyone based on the accelerations
        for i, ag in enumerate(agents):
            ax, ay = accel_list[i]
            ag['vx'] += ax * h
            ag['vy'] += ay * h
            ag['x']  += ag['vx'] * h
            ag['y']  += ag['vy'] * h

    # Return the final state of the focal agent (index 0)
    return agents[0]['x'], agents[0]['y'], agents[0]['vx'], agents[0]['vy']

##############################################################################
# 7) BUILD FRAME DATA
##############################################################################

def build_frame_data(df, neighbor_radius=NEIGHBOR_RADIUS, sample_step=SAMPLE_STEP):
    frame_data = {}
    grouped = df.groupby('time')
    times_sorted = sorted(grouped.groups.keys())
    sampled = times_sorted[::sample_step]

    for t in sampled:
        group_t = grouped.get_group(t)
        user_list = []
        for _, row in group_t.iterrows():
            user_list.append({
                'id': row['id'],
                'type': row['type_most_common'],
                'x': row['xloc_kf'],
                'y': row['yloc_kf'],
                'vx': row['speed_kf_x'],
                'vy': row['speed_kf_y'],
                'lane': (int(row['lane_kf']) if not pd.isnull(row['lane_kf']) else -1),
                'neighbors': []
            })
        for i in range(len(user_list)):
            px_i, py_i = user_list[i]['x'], user_list[i]['y']
            nbs = []
            for j in range(len(user_list)):
                if i == j:
                    continue
                px_j, py_j = user_list[j]['x'], user_list[j]['y']
                if math.hypot(px_i - px_j, py_i - py_j) <= neighbor_radius:
                    vx_j = user_list[j]['vx']
                    vy_j = user_list[j]['vy']
                    ttype = user_list[j]['type']
                    nbs.append((px_j, py_j, vx_j, vy_j, ttype))
            user_list[i]['neighbors'] = nbs
        frame_data[t] = user_list
    return frame_data

##############################################################################
# 8) METRIC: Compute error metrics
##############################################################################

def compute_pt_metrics(frame_data, pvec):
    times_list = sorted(frame_data.keys())
    if len(times_list) < 2:
        return {
            'MSE': 1e9, 'RMSE': 1e9, 'MAE': 1e9,
            'MAPE': 1e9, 'NRMSE': 1e9, 'SSE': 1e9,
            'R-squared': -999, 'Error': 1e9
        }

    x_errs, y_errs = [], []
    x_obs_vals, y_obs_vals = [], []
    time_map = {t: {u['id']: u for u in frame_data[t]} for t in times_list}

    for i in range(len(times_list) - 1):
        t1, t2 = times_list[i], times_list[i+1]
        dt = t2 - t1
        if dt <= 0:
            continue
        id_map1 = time_map[t1]
        id_map2 = time_map[t2]
        common_ids = set(id_map1.keys()) & set(id_map2.keys())
        for aid in common_ids:
            u1, u2 = id_map1[aid], id_map2[aid]
            px1, py1 = u1['x'], u1['y']
            vx1, vy1 = u1['vx'], u1['vy']
            a_type = u1['type']
            lane1 = u1['lane']
            nb1 = u1['neighbors']

            # Observed next-step
            px2_obs, py2_obs = u2['x'], u2['y']

            # Predict next-step
            pxp, pyp, _, _ = simulate_pt_single_agent(
                px1, py1, vx1, vy1, a_type, pvec,
                nb1, lane1, total_dt=dt, substeps=5
            )
            x_errs.append(pxp - px2_obs)
            y_errs.append(pyp - py2_obs)
            x_obs_vals.append(px2_obs)
            y_obs_vals.append(py2_obs)

    x_err_arr, y_err_arr = np.array(x_errs), np.array(y_errs)
    SSE = np.sum(x_err_arr**2 + y_err_arr**2)
    n = len(x_err_arr)
    MSE = SSE / n
    RMSE = math.sqrt(MSE)
    MAE = np.mean(np.abs(np.concatenate([x_err_arr, y_err_arr])))
    
    # Simple MAPE calculation
    x_obs_arr, y_obs_arr = np.array(x_obs_vals), np.array(y_obs_vals)
    pos_obs = np.hypot(x_obs_arr, y_obs_arr)
    px_pred = x_obs_arr + x_err_arr
    py_pred = y_obs_arr + y_err_arr
    pred_r = np.hypot(px_pred, py_pred)
    mask = pos_obs > 1e-9
    MAPE = np.mean(np.abs((pred_r[mask] - pos_obs[mask]) / pos_obs[mask])) * 100 if np.any(mask) else 1e9

    dx_range = x_obs_arr.max() - x_obs_arr.min() if x_obs_arr.size > 0 else 1e-9
    dy_range = y_obs_arr.max() - y_obs_arr.min() if y_obs_arr.size > 0 else 1e-9
    NRMSE = RMSE / (((dx_range + dy_range) * 0.5) if (dx_range + dy_range) > 0 else 1e-9)
    xm, ym = x_obs_arr.mean(), y_obs_arr.mean()
    SStot = np.sum((x_obs_arr - xm)**2) + np.sum((y_obs_arr - ym)**2)
    r2 = 1.0 - (SSE / SStot) if SStot > 1e-9 else -999

    total_diff = 1.0 / (1.0 + MSE)
    return {
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE,
        'MAPE': MAPE,
        'NRMSE': NRMSE,
        'SSE': SSE,
        'R-squared': r2,
        'Error': total_diff
    }

##############################################################################
# 9) GENETIC ALGORITHM
##############################################################################

def evaluate_population(frame_data, population):
    return [compute_pt_metrics(frame_data, pvec)['Error'] for pvec in population]

def run_ga_for_id(frame_data_sub, pop_size=POP_SIZE, generations=GENERATIONS, r2_threshold=r2_threshold):
    population = [clamp_pt_params(random_pt_params()) for _ in range(pop_size)]
    best_vec, best_score = None, -1e9
    for gen in range(generations):
        scores = evaluate_population(frame_data_sub, population)
        pop_sorted = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        # Update best solution so far:
        if pop_sorted[0][1] > best_score:
            best_score = pop_sorted[0][1]
            best_vec = pop_sorted[0][0]
        curr_metrics = compute_pt_metrics(frame_data_sub, best_vec)
        if curr_metrics['R-squared'] >= r2_threshold:
            print(f"Early stopping at gen {gen+1} because R-squared ({curr_metrics['R-squared']:.3f}) reached threshold {r2_threshold}.")
            best_score = pop_sorted[0][1]
            best_vec = pop_sorted[0][0]
            break
        new_pop = [pop_sorted[i][0] for i in range(ELITISM_COUNT)]
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(pop_sorted)[0], random.choice(pop_sorted)[0]
            c1, c2 = crossover_pt_params(p1, p2)
            new_pop.append(mutate_pt_params(c1, MUTATION_RATE))
            if len(new_pop) < pop_size:
                new_pop.append(mutate_pt_params(c2, MUTATION_RATE))
        population = new_pop[:pop_size]
        if (gen + 1) % 10 == 0:
            print(f"GA: gen={gen+1}, best fitness={best_score:.5f}")
    final_metrics = compute_pt_metrics(frame_data_sub, best_vec)
    return best_vec, best_score, final_metrics

##############################################################################
# 10) PREDICTION AND MAIN FUNCTION
##############################################################################

def get_predictions_for_subdata(frame_data_sub, params, focus_id):
    times_list = sorted(frame_data_sub.keys())
    if len(times_list) < 2:
        return pd.DataFrame([])
    
    time_dict = {t: {u['id']: u for u in frame_data_sub[t]} for t in times_list}
    rows = []
    for i in range(len(times_list) - 1):
        t1, t2 = times_list[i], times_list[i+1]
        dt = t2 - t1
        if dt <= 0:
            continue
        id_map1, id_map2 = time_dict[t1], time_dict[t2]
        if focus_id not in id_map1 or focus_id not in id_map2:
            continue
        u1, u2 = id_map1[focus_id], id_map2[focus_id]
        px1, py1 = u1['x'], u1['y']
        vx1, vy1 = u1['vx'], u1['vy']
        lane1 = u1['lane']
        nb1 = u1['neighbors']
        utype = u1['type']
        x2_obs, y2_obs = u2['x'], u2['y']
        ax_pred, ay_pred = pt_acceleration(px1, py1, vx1, vy1, utype, params, nb1, lane1, dt=dt/SAMPLE_STEP)
        pxp, pyp, vxp, vyp = simulate_pt_single_agent(px1, py1, vx1, vy1, utype, params, nb1, lane1,
                                                      total_dt=dt, substeps=5)
        rows.append({
            'id': focus_id, 'time': t2,
            'x_obs': x2_obs, 'y_obs': y2_obs,
            'vx_obs': u2['vx'], 'vy_obs': u2['vy'],
            'x_pred': pxp, 'y_pred': pyp,
            'vx_pred': vxp, 'vy_pred': vyp,
            'ax_pred': ax_pred, 'ay_pred': ay_pred,
            'type_most_common': utype
        })
    return pd.DataFrame(rows)

def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df['type_most_common'].isin([PEDESTRIAN_CODE, BICYCLE_CODE])].copy()
    df = df[df['lane_kf'].isin(LANES_TO_KEEP)].copy()
    df.sort_values(by=['time', 'id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} rows for ped/bike data with lanes {LANES_TO_KEEP}.")
    
    plt.ion()
    
    param_names = [
        "Wc_pp", "Wc_pb", "Wc_pbar", "Wc_bp",
        "Wc_bb", "Wc_bbar",
        "eta_ped", "xi_ped", "tau_ped", "v_pref_ped",
        "eta_bike", "xi_bike", "tau_bike", "v_pref_bike"
    ]
    
    unique_ids = df['id'].unique()
    all_results, all_predictions = [], []
    
    for agent_id in unique_ids:
        df_id = df[df['id'] == agent_id]
        times_id = df_id['time'].unique()
        if len(times_id) < MIN_PRESENCE:
            print(f"ID={agent_id}: skip (presence < {MIN_PRESENCE}).")
            continue
        df_sub = df[df['time'].isin(times_id)].copy()
        frame_data_sub = build_frame_data(df_sub, neighbor_radius=NEIGHBOR_RADIUS)
        if len(frame_data_sub) < 2:
            continue

        best_vec, best_fit, best_mets = run_ga_for_id(frame_data_sub, r2_threshold=r2_threshold)

        rowdict = {
            'id': agent_id,
            'best_fitness': best_fit,
            'MSE': best_mets['MSE'],
            'RMSE': best_mets['RMSE'],
            'MAE': best_mets['MAE'],
            'MAPE': best_mets['MAPE'],
            'NRMSE': best_mets['NRMSE'],
            'SSE': best_mets['SSE'],
            'R-squared': best_mets['R-squared'],
            'Error': best_mets['Error']
        }
        for i, param_value in enumerate(best_vec):
            rowdict[param_names[i]] = param_value
        all_results.append(rowdict)
        print(f"ID={agent_id} => SSE={best_mets['SSE']:.3f}, MSE={best_mets['MSE']:.3f}, RMSE={best_mets['RMSE']:.3f}, R2={best_mets['R-squared']:.3f}")
        
        # Generate predictions and plot intermediate results
        df_pred_sub = get_predictions_for_subdata(frame_data_sub, best_vec, agent_id)
        all_predictions.append(df_pred_sub)
        
        if not df_pred_sub.empty:
            plt.figure()
            plt.plot(df_pred_sub['time'], df_pred_sub['x_obs'], marker='o', label='Observed x')
            plt.plot(df_pred_sub['time'], df_pred_sub['x_pred'], marker='x', label='Predicted x')
            plt.xlabel("Time")
            plt.ylabel("X position")
            plt.title(f"Agent {agent_id} - X Position - PT MODEL")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.pause(1)
            plt.close()
            
            plt.figure()
            plt.plot(df_pred_sub['time'], df_pred_sub['y_obs'], marker='o', label='Observed y')
            plt.plot(df_pred_sub['time'], df_pred_sub['y_pred'], marker='x', label='Predicted y')
            plt.xlabel("Time")
            plt.ylabel("Y position")
            plt.title(f"Agent {agent_id} - Y Position - PT MODEL")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.pause(1)
            plt.close()
    
    df_res = pd.DataFrame(all_results)
    out_csv = os.path.join(OUTPUT_DIR, "pt_expanded_collision_calib.csv")
    df_res.to_csv(out_csv, index=False)
    print("Saved results to:", out_csv)
    
    df_pred_all = pd.concat(all_predictions, ignore_index=True)
    out_pred_csv = os.path.join(OUTPUT_DIR, "pt_expanded_collision_predictions.csv")
    df_pred_all.to_csv(out_pred_csv, index=False)
    print("Saved predictions to:", out_pred_csv)
    
    # Plot distributions for parameters and errors
    plot_columns = param_names + ['best_fitness', 'MSE', 'RMSE', 'MAE', 'MAPE', 'NRMSE', 'SSE', 'R-squared', 'Error']
    for col in plot_columns:
        plt.figure()
        plt.hist(df_res[col].dropna(), bins=20, edgecolor='black')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()
        plt.pause(1)
        plt.close()
    
    plt.ioff()

if __name__=="__main__":
    main()
