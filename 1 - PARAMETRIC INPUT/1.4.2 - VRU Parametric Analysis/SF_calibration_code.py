# -*- coding: utf-8 -*-
"""
Calibrates a mixed pedestrian+bicycle Social Force model.

Pedestrians (Indices 0..5, 6 parameters):
  1) vAlpha0_ped   (fixed)
  2) tauAlpha_ped  (fixed)
  3) A_pp          (ped–ped interaction intensity)
  4) B_pp          (ped–ped interaction range)
  5) A_wall_ped    (wall interaction intensity)
  6) B_wall_ped    (wall interaction range)

Bicycles (Indices 6..18, 13 parameters from Wang et al.):
  7)  m_gamma       (fixed)
  8)  tau_gamma     (fixed)
  9)  v_gamma0      (fixed)
  10) a_gamma_max   (fixed)
  11) b_gamma       (fixed)
  12) eta_gamma     (fixed)
  13) mu_gamma      (fixed)
  14) eps_m         (variable)
  15) A_w           (variable)
  16) B_w           (variable)
  17) A_s           (variable)
  18) B_s           (variable)
  19) T_i           (fixed)

Additional features:
  - Distribution plots for each metric are saved.
  - For each ID, after calibration, predictions are obtained and then compared graphically with the observed positions.
 
@author: TBP
@date: 2025-02-01
"""

import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.animation as animation

_NGM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _NGM_ROOT not in sys.path:
    sys.path.insert(0, _NGM_ROOT)
from ngm_paths import FOGGY_BOTTOM_CSV, dataset_path

##############################################################################
# 1) GLOBAL SETTINGS & PATHS
##############################################################################

CSV_PATH = dataset_path(FOGGY_BOTTOM_CSV)
# PIX_TO_M is not needed because data is already in meters.
PEDESTRIAN_CODE = 0  # code used in dataset for pedestrians
BICYCLE_CODE    = 1  # code used in dataset for bicycles

lane = [1, 39, 40, 41]  # lanes to keep

NEIGHBOR_RADIUS = 5.0
SAMPLE_STEP     = 1

r2_threshold = 0.95

POP_SIZE        = 20
GENERATIONS     = 40
MUTATION_RATE   = 0.2
FITNESS_METRIC  = "TotalDiff"
ELITISM_COUNT   = 4
USE_TOURNAMENT_SELECTION = False
TOURNAMENT_SIZE = 3

SAVE_PLOTS      = True
PLOT_DIR        = "./sfm_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

##############################################################################
# 2) BOUNDARY POLYGONS
##############################################################################
POLYGON_METERS = {}

##############################################################################
# 3) GEOMETRY / NEIGHBOR UTILITIES
##############################################################################
# Not used since polygons are empty (no wall forces).

##############################################################################
# 4) PEDESTRIAN (Helbing–Molnár)
##############################################################################

def elliptical_b_value(r_ab, vbeta, e_beta, delta_t=0.1):
    R1 = math.hypot(r_ab[0], r_ab[1])
    rx = r_ab[0] - vbeta * delta_t * e_beta[0]
    ry = r_ab[1] - vbeta * delta_t * e_beta[1]
    R2 = math.hypot(rx, ry)
    denom = vbeta * delta_t
    big_expr = (R1 + R2)**2 - denom**2
    if big_expr < 0:
        return 0.0
    return 0.5 * math.sqrt(big_expr)

def elliptical_ped_repulsion(px, py, nx, ny, vx_b, vy_b, A_pp, B_pp, delta_t=0.1):
    r_ab = (px - nx, py - ny)
    speed_b = math.hypot(vx_b, vy_b)
    if speed_b < 1e-9:
        e_b = (0.0, 0.0)
    else:
        e_b = (vx_b/speed_b, vy_b/speed_b)
    eps = 1e-5
    def potential(rx, ry):
        b_val = elliptical_b_value((rx, ry), speed_b, e_b, delta_t)
        return A_pp * math.exp(- b_val / (B_pp + 1e-9))
    pot0 = potential(r_ab[0], r_ab[1])
    pot_px = potential(r_ab[0] + eps, r_ab[1])
    dV_dx = (pot_px - pot0) / eps
    pot_py = potential(r_ab[0], r_ab[1] + eps)
    dV_dy = (pot_py - pot0) / eps
    fx = - dV_dx
    fy = - dV_dy
    return fx, fy

def compute_ped_accel(px, py, vx, vy, neighbors, polygons, lane_id, ped_params):
    vAlpha0_ped, tauAlpha_ped, A_pp, B_pp, A_wall_ped, B_wall_ped = ped_params
    m_ped = 80
    speed_obs = math.hypot(vx, vy)
    if speed_obs > 1e-3:
        ex, ey = vx / speed_obs, vy / speed_obs
    else:
        ex, ey = 1.0, 0.0
    ax_drive = (vAlpha0_ped * ex - vx) / tauAlpha_ped * m_ped
    ay_drive = (vAlpha0_ped * ey - vy) / tauAlpha_ped * m_ped

    ax_pp, ay_pp = 0.0, 0.0
    for nb in neighbors:
        nx_, ny_, vx_b, vy_b, nb_type = nb
        if nb_type == PEDESTRIAN_CODE:  # ped–ped interaction
            fx, fy = elliptical_ped_repulsion(px, py, nx_, ny_, vx_b, vy_b, A_pp, B_pp)
            ax_pp += fx
            ay_pp += fy

    ax_wall, ay_wall = 0.0, 0.0
    if lane_id in polygons:
        # (No wall repulsion if polygons is empty)
        pass

    ax = ax_drive + ax_pp + ax_wall
    ay = ay_drive + ay_pp + ay_wall

    a_val = math.hypot(ax / m_ped, ay / m_ped)
    if a_val < 0.3 * speed_obs / 0.1:
        return ax / m_ped, ay / m_ped
    else:
        sc = 0.3 * speed_obs / (0.1 * a_val)
        return (ax / m_ped * sc, ay / m_ped * sc)

##############################################################################
# 5) BICYCLE (Wang et al.)
##############################################################################

def v_gamma_safe(g_gamma, v_gamma, v_delta, b_gamma, b_delta, eta_gamma):
    inside = b_gamma**2 * eta_gamma**2 + b_gamma * (2 * g_gamma - eta_gamma * v_gamma + (v_delta**2) / b_delta)
    if inside < 0:
        return 0.0
    return -b_gamma * eta_gamma + math.sqrt(inside)

def line_circle_intersect(bx, by, v_relx, v_rely, ax, ay, radius):
    v_len = math.hypot(v_relx, v_rely)
    if v_len < 1e-9:
        return False
    cx = bx - ax
    cy = by - ay
    dotVV = v_len * v_len
    dotVC = cx * v_relx + cy * v_rely
    t_star = -dotVC / dotVV
    if t_star < 0:
        t_star = 0.0
    lx = cx + t_star * v_relx
    ly = cy + t_star * v_rely
    dist_sq = lx * lx + ly * ly
    return (dist_sq <= radius * radius)

def get_lane_orientation_and_extents(polygon):
    if len(polygon) < 2:
        return 'horizontal', 0.0, 0.0
    x1, y1 = polygon[0]
    x2, y2 = polygon[1]
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) > abs(dy):
        orientation = 'horizontal'
        all_ys = [pt[1] for pt in polygon]
        return orientation, min(all_ys), max(all_ys)
    else:
        orientation = 'vertical'
        all_xs = [pt[0] for pt in polygon]
        return orientation, min(all_xs), max(all_xs)

def compute_bike_accel(px, py, vx, vy, neighbors, polygons, lane_id, bike_params):
    (m_gamma, tau_gamma, v_gamma0, a_gamma_max, b_gamma,
     eta_gamma, mu_gamma, eps_m,
     A_w, B_w,
     A_s, B_s,
     T_i) = bike_params

    speed_obs = math.hypot(vx, vy)
    if speed_obs < 1e-9:
        ex, ey = 1.0, 0.0
    else:
        ex, ey = vx / speed_obs, vy / speed_obs

    # Driving force
    fx_drive = (m_gamma / tau_gamma) * (v_gamma0 * ex - vx)
    fy_drive = (m_gamma / tau_gamma) * (v_gamma0 * ey - vy)
    fx_sum, fy_sum = fx_drive, fy_drive

    # Car-following along the direction (ex,ey)
    front_bikes = []
    for nb in neighbors:
        nx_, ny_, vx_n, vy_n, nb_type = nb
        if nb_type != BICYCLE_CODE:
            continue
        dx, dy = nx_ - px, ny_ - py
        dotp = dx * ex + dy * ey
        if dotp > 0:
            dist = math.hypot(dx, dy)
            front_bikes.append((dist, nx_, ny_, vx_n, vy_n))
    front_bikes.sort(key=lambda x: x[0])
    if front_bikes:
        dist, nx_f, ny_f, vx_f, vy_f = front_bikes[0]
        v_f = math.hypot(vx_f, vy_f)
        g_gamma = max(dist - 1.5, 0.1)
        v_safe = v_gamma_safe(g_gamma, speed_obs, v_f, b_gamma, b_gamma, eta_gamma)
        dist_f = math.hypot(nx_f - px, ny_f - py)
        if dist_f > 1e-9:
            n_att_x = (nx_f - px) / dist_f
            n_att_y = (ny_f - py) / dist_f
            f_att_mag = (v_safe - speed_obs) / tau_gamma
            if f_att_mag > a_gamma_max:
                f_att_mag = a_gamma_max
            fx_sum += f_att_mag * n_att_x
            fy_sum += f_att_mag * n_att_y

    # Boundary repulsion: (skipped if polygons is empty)
    fx_wall, fy_wall = 0.0, 0.0
    if lane_id in polygons:
        pass
    fx_sum += fx_wall
    fy_sum += fy_wall

    # Bike–Bike repulsion
    dt = 0.1
    for nb in neighbors:
        nx_, ny_, vx_n, vy_n = nb[:4]
        rx = px - nx_
        ry = py - ny_
        dist_r = math.hypot(rx, ry)
        if dist_r < 1e-9:
            continue
        dvx = vx - vx_n
        dvy = vy - vy_n
        dist_dv = math.hypot(dvx, dvy)
        R1 = dist_r
        rx2 = rx - dvx * dt
        ry2 = ry - dvy * dt
        R2 = math.hypot(rx2, ry2)
        inside_ = (R1 + R2)**2 - (dist_dv * dt)**2
        if inside_ < 0:
            B_gd = 0.0
        else:
            B_gd = 0.5 * math.sqrt(inside_)
        f_bike_mag = A_s * math.exp(- B_gd / (B_s + 1e-9))
        n_hat_x = rx / dist_r
        n_hat_y = ry / dist_r
        fx_sum += f_bike_mag * n_hat_x
        fy_sum += f_bike_mag * n_hat_y

    # Bike–Ped collision avoidance
    ped_radius = 0.3
    for nb in neighbors:
        nx_, ny_, vx_n, vy_n, nb_type = nb
        if nb_type == PEDESTRIAN_CODE:
            v_relx = vx - vx_n
            v_rely = vy - vy_n
            if line_circle_intersect(px, py, v_relx, v_rely, nx_, ny_, ped_radius):
                rel_speed = math.hypot(v_relx, v_rely)
                if rel_speed > 1e-9:
                    dxp = px - nx_
                    dyp = py - ny_
                    dist_p = math.hypot(dxp, dyp)
                    if dist_p < 1e-9:
                        continue
                    nxp = dxp / dist_p
                    nyp = dyp / dist_p
                    f_ga_mag = rel_speed / T_i
                    fx_sum += f_ga_mag * nxp
                    fy_sum += f_ga_mag * nyp

    ax = fx_sum / m_gamma
    ay = fy_sum / m_gamma
    a_val = math.hypot(ax, ay)
    if a_val > a_gamma_max:
        sc = a_gamma_max / a_val
        ax *= sc
        ay *= sc

    return ax, ay

##############################################################################
# 6) MAIN ACCELERATION WRAPPER
##############################################################################

def compute_sfm_acceleration(px, py, vx, vy, neighbors, polygons, lane_id, params, user_type):
    ped_params = params[0:6]
    bike_params = params[6:19]
    if user_type == PEDESTRIAN_CODE:
        return compute_ped_accel(px, py, vx, vy, neighbors, polygons, lane_id, ped_params)
    elif user_type == BICYCLE_CODE:
        return compute_bike_accel(px, py, vx, vy, neighbors, polygons, lane_id, bike_params)
    else:
        return (0.0, 0.0)

##############################################################################
# 7) SIMULATION: Forward-Euler Integration (MODIFIED TO BE DYNAMIC)
##############################################################################

def simulate_sfm_substeps(px, py, vx, vy, neighbors, lane_id, polygons, params, user_type,
                          total_dt=1.0, substeps=10):
    """
    Previously, this function updated only the focal agent with static neighbors.
    Now it updates the focal agent *and* all neighbors in each substep, making the
    simulation dynamic. It still returns only the focal agent's final state.
    """
    h = total_dt / substeps

    # Combine the focal agent (index 0) + neighbors into a single list:
    agents = []
    agents.append({
        'x': px, 'y': py, 'vx': vx, 'vy': vy,
        'lane': lane_id, 'type': user_type
    })
    for nb in neighbors:
        agents.append({
            'x':  nb[0],
            'y':  nb[1],
            'vx': nb[2],
            'vy': nb[3],
            'lane': lane_id,
            'type': nb[4]
        })

    for _ in range(substeps):
        accel_list = []
        # Compute acceleration for each agent
        for i, ag in enumerate(agents):
            # Build local neighbor list for agent i (within NEIGHBOR_RADIUS)
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
            ax, ay = compute_sfm_acceleration(ag['x'], ag['y'], ag['vx'], ag['vy'],
                                              local_nb, polygons, ag['lane'],
                                              params, ag['type'])
            accel_list.append((ax, ay))

        # Now update each agent
        for i, ag in enumerate(agents):
            ax, ay = accel_list[i]
            ag['vx'] += ax * h
            ag['vy'] += ay * h
            ag['x']  += ag['vx'] * h
            ag['y']  += ag['vy'] * h

    # Return the focal agent's final position and velocity
    return agents[0]['x'], agents[0]['y'], agents[0]['vx'], agents[0]['vy']

##############################################################################
# 7b) BUILD FRAME DATA
##############################################################################

def build_frame_data(df, neighbor_radius=5.0, sample_step=1):
    frame_data = {}
    grouped = df.groupby('time')
    times_sorted = sorted(grouped.groups.keys())
    sampled_times = times_sorted[::sample_step]

    for t in sampled_times:
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
                'ax_obs': row['acceleration_kf_x'],
                'ay_obs': row['acceleration_kf_y'],
                'lane': int(row['lane_kf']) if not pd.isnull(row['lane_kf']) else -1,
                'neighbors': []
            })
        for i in range(len(user_list)):
            px, py = user_list[i]['x'], user_list[i]['y']
            nb_list = []
            for j in range(len(user_list)):
                if i == j:
                    continue
                nx_ = user_list[j]['x']
                ny_ = user_list[j]['y']
                vx_n = user_list[j]['vx']
                vy_n = user_list[j]['vy']
                nb_type = user_list[j]['type']
                if math.hypot(px - nx_, py - ny_) <= neighbor_radius:
                    nb_list.append((nx_, ny_, vx_n, vy_n, nb_type))
            user_list[i]['neighbors'] = nb_list
        frame_data[t] = user_list
    return frame_data

##############################################################################
# 8) METRICS: Compute error metrics
##############################################################################

def compute_sfm_metrics_velchange_with_r2(frame_data, polygons, params):
    x_errs = []
    y_errs = []
    x_obs_vals = []
    y_obs_vals = []
    time_dict = {}
    times_list = sorted(frame_data.keys())
    for t in times_list:
        id_map = {u['id']: u for u in frame_data[t]}
        time_dict[t] = id_map

    for i in range(len(times_list)-1):
        t1 = times_list[i]
        t2 = times_list[i+1]
        dt = t2 - t1
        if dt <= 0:
            continue
        id_map1 = time_dict[t1]
        id_map2 = time_dict[t2]
        common_ids = set(id_map1.keys()) & set(id_map2.keys())
        for agent_id in common_ids:
            u1 = id_map1[agent_id]
            u2 = id_map2[agent_id]
            px1, py1 = u1['x'], u1['y']
            vx1, vy1 = u1['vx'], u1['vy']
            lane1 = u1['lane']
            nb1 = u1['neighbors']
            x2_obs, y2_obs = u2['x'], u2['y']
            px_end, py_end, _, _ = simulate_sfm_substeps(px1, py1, vx1, vy1,
                                                         nb1, lane1, polygons, params, u1['type'],
                                                         total_dt=dt, substeps=10)
            x_errs.append(px_end - x2_obs)
            y_errs.append(py_end - y2_obs)
            x_obs_vals.append(x2_obs)
            y_obs_vals.append(y2_obs)

    if len(x_errs) == 0:
        return {
            'Error': 1e9, 'MSE': 1e9, 'RMSE': 1e9, 'MAE': 1e9,
            'MAPE': 1e9, 'NRMSE': 1e9, 'SSE': 1e9, 'R-squared': -999.0,
            'TotalDiff': 1e9
        }

    x_err_arr = np.array(x_errs)
    y_err_arr = np.array(y_errs)
    x_obs_arr = np.array(x_obs_vals)
    y_obs_arr = np.array(y_obs_vals)

    SSE = np.sum(x_err_arr**2 + y_err_arr**2)
    n = len(x_err_arr)
    MSE = SSE / n
    RMSE = math.sqrt(MSE)
    MAE = np.mean(np.abs(np.concatenate([x_err_arr, y_err_arr])))
    total_diff = 1.0 / (1.0 + SSE)

    pos_obs = np.hypot(x_obs_arr, y_obs_arr)
    pos_pred = np.hypot(x_obs_arr + x_err_arr, y_obs_arr + y_err_arr)
    if np.any(pos_obs > 1e-9):
        MAPE = np.mean(np.abs((pos_pred - pos_obs) / pos_obs)) * 100.0
    else:
        MAPE = 1e9

    dx = np.ptp(x_obs_arr)
    dy = np.ptp(y_obs_arr)
    NRMSE = RMSE / ((dx + dy) * 0.5) if (dx + dy) > 0 else 1e9

    xm = np.mean(x_obs_arr)
    ym = np.mean(y_obs_arr)
    SStot = np.sum((x_obs_arr - xm)**2) + np.sum((y_obs_arr - ym)**2)
    r2 = 1 - SSE / SStot if SStot > 1e-9 else -999.0

    return {
        'Error': total_diff,
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE,
        'MAPE': MAPE,
        'NRMSE': NRMSE,
        'SSE': SSE,
        'R-squared': r2,
        'TotalDiff': total_diff
    }

##############################################################################
# 9) GA for Social Force Model Calibration
##############################################################################

def crossover(p1, p2):
    idx = random.randint(1, len(p1) - 1)
    c1 = p1[:idx] + p2[idx:]
    c2 = p2[:idx] + p1[idx:]
    return c1, c2

def random_params():
    # Pedestrian parameters
    vAlpha0_ped = 1.358  # fixed
    tauAlpha_ped = 0.5   # fixed
    A_pp = random.uniform(500, 3000)
    B_pp = random.uniform(0.05, 2.0)
    A_wall = random.uniform(500, 3000)
    B_wall = random.uniform(0.05, 2.0)
    # Bicycle parameters
    m_gamma = 200
    tau_gamma = 0.5
    v_gamma0 = 3.073
    a_gamma_max = 4
    b_gamma = 4
    eta_gamma = 0.3
    mu_gamma = 1.2
    eps_m = random.uniform(0.5, 5.0)
    A_w = random.uniform(500, 3000)
    B_w = random.uniform(0.05, 2.0)
    A_s = random.uniform(500, 3000)
    B_s = random.uniform(0.05, 2.0)
    T_i = 0.97

    return [
        vAlpha0_ped, tauAlpha_ped, A_pp, B_pp, A_wall, B_wall,
        m_gamma, tau_gamma, v_gamma0, a_gamma_max, b_gamma, eta_gamma, mu_gamma, eps_m,
        A_w, B_w, A_s, B_s, T_i
    ]

def clamp_params(p):
    # Pedestrian parameters
    p[0] = 1.358
    p[1] = 0.5
    p[2] = min(max(p[2], 500), 3000)
    p[3] = min(max(p[3], 0.05), 2.0)
    p[4] = min(max(p[4], 500), 3000)
    p[5] = min(max(p[5], 0.05), 2.0)
    # Bicycle parameters
    p[6] = 200
    p[7] = 0.5
    p[8] = 3.073
    p[9] = 4
    p[10] = 4
    p[11] = 0.3
    p[12] = 1.2
    p[13] = min(max(p[13], 0.5), 5.0)
    p[14] = min(max(p[14], 500), 3000)
    p[15] = min(max(p[15], 0.05), 2.0)
    p[16] = min(max(p[16], 500), 3000)
    p[17] = min(max(p[17], 0.05), 2.0)
    p[18] = 0.97
    return p

def mutate(p, rate=0.2):
    for i in [2, 3, 4, 5, 13, 14, 15, 16, 17]:
        if random.random() < rate:
            if i in [3, 5, 15, 17]:
                p[i] += random.uniform(-0.2, 0.2)
            else:
                p[i] += random.uniform(-100, 100)
    return clamp_params(p)

def tournament_selection(pop, fitness_vals, k=3):
    chosen = random.sample(list(zip(pop, fitness_vals)), k)
    best = max(chosen, key=lambda x: x[1])[0]
    return best

def run_ga_for_subdata(frame_data_sub, polygons, pop_size=20, generations=40,
                       early_stop_threshold=1e-3, patience=10, r2_threshold=r2_threshold):
    population = [clamp_params(random_params()) for _ in range(pop_size)]
    best_params = None
    best_metrics = None
    best_err = float('inf')
    no_improve_count = 0
    prev_best_err = float('inf')

    for gen in range(generations):
        all_metrics = [compute_sfm_metrics_velchange_with_r2(frame_data_sub, polygons, ind)
                       for ind in population]
        errors = [m['Error'] for m in all_metrics]

        for i, e in enumerate(errors):
            if e < best_err:
                best_err = e
                best_params = population[i][:]
                best_metrics = all_metrics[i]

        # Early stop based on R-squared threshold
        if best_metrics is not None and best_metrics['R-squared'] >= r2_threshold:
            print(f"Early stopping at gen {gen+1}: R-squared ({best_metrics['R-squared']:.3f}) >= threshold {r2_threshold}")
            break

        # Early stop based on lack of improvement
        if abs(prev_best_err - best_err) < early_stop_threshold:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at gen {gen+1}: no significant improvement (best error={best_err:.6f})")
                break
        else:
            no_improve_count = 0
        prev_best_err = best_err

        # Selection and reproduction
        fitness_vals = [1.0 / (e + 1e-9) for e in errors]
        pop_fit = sorted(zip(population, fitness_vals), key=lambda x: x[1], reverse=True)
        sorted_pop = [p for p, f in pop_fit]
        new_pop = sorted_pop[:ELITISM_COUNT]

        if not USE_TOURNAMENT_SELECTION:
            half = pop_size // 2
            parents = sorted_pop[:half]
            while len(new_pop) < pop_size:
                p1, p2 = random.sample(parents, 2)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1, MUTATION_RATE)
                c2 = mutate(c2, MUTATION_RATE)
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)
        else:
            while len(new_pop) < pop_size:
                p1 = tournament_selection(population, fitness_vals, k=TOURNAMENT_SIZE)
                p2 = tournament_selection(population, fitness_vals, k=TOURNAMENT_SIZE)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1, MUTATION_RATE)
                c2 = mutate(c2, MUTATION_RATE)
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)

        population = new_pop[:pop_size]
        print(f"Gen {gen+1}/{generations}, best Error={best_err:.6f}, R2= {best_metrics['R-squared']:.3f}")

    return best_params, best_metrics

##############################################################################
# 10) GET PREDICTIONS FOR A SINGLE ID
##############################################################################

def get_predictions_for_subdata(frame_data_sub, polygons, params, focus_id):
    times_list = sorted(frame_data_sub.keys())
    if len(times_list) < 2:
        return pd.DataFrame([])
    time_dict = {}
    for t in times_list:
        id_map = {u['id']: u for u in frame_data_sub[t]}
        time_dict[t] = id_map
    rows = []
    for i in range(len(times_list) - 1):
        t1 = times_list[i]
        t2 = times_list[i+1]
        dt = t2 - t1
        if dt <= 0:
            continue
        id_map1 = time_dict[t1]
        id_map2 = time_dict[t2]
        if focus_id not in id_map1 or focus_id not in id_map2:
            continue
        u1 = id_map1[focus_id]
        px1, py1 = u1['x'], u1['y']
        vx1, vy1 = u1['vx'], u1['vy']
        lane1 = u1['lane']
        nb1 = u1['neighbors']
        utype = u1['type']

        u2 = id_map2[focus_id]
        x2_obs, y2_obs = u2['x'], u2['y']
        vx2_obs, vy2_obs = u2['vx'], u2['vy']

        ax_pred, ay_pred = compute_sfm_acceleration(px1, py1, vx1, vy1,
                                                    nb1, polygons, lane1,
                                                    params, utype)
        px_end, py_end, vx_end, vy_end = simulate_sfm_substeps(
            px1, py1, vx1, vy1, nb1, lane1, polygons, params, utype,
            total_dt=dt, substeps=10
        )
        rows.append({
            'id': focus_id,
            'time': t2,
            'x_obs': x2_obs,
            'y_obs': y2_obs,
            'vx_obs': vx2_obs,
            'vy_obs': vy2_obs,
            'x_pred': px_end,
            'y_pred': py_end,
            'vx_pred': vx_end,
            'vy_pred': vy_end,
            'ax_pred': ax_pred,
            'ay_pred': ay_pred
        })
    return pd.DataFrame(rows)

##############################################################################
# 11) MAIN
##############################################################################

def main():
    # Load data
    df = pd.read_csv(CSV_PATH)
    df = df[df['time'] < 15 * 60]
    df = df[df['type_most_common'].isin([PEDESTRIAN_CODE, BICYCLE_CODE])].copy()
    df = df[df['lane_kf'].isin(lane)].copy()
    df.sort_values(by=['time', 'id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Records for ped/bike: {len(df)}")

    # (Optional) Estimate free speeds if desired
    # Determine presence threshold (15% quantile)
    presence_counts = df.groupby('id')['time'].nunique()
    presence_threshold = presence_counts.quantile(0.15)
    print(f"Minimum presence threshold = {presence_threshold:.2f}")

    unique_ids = df['id'].unique()
    results_rows = []
    all_predictions = []

    # Enable interactive plotting
    plt.ion()

    for agent_id in unique_ids:
        df_id = df[df['id'] == agent_id]
        times_for_id = df_id['time'].unique()
        if len(times_for_id) < presence_threshold:
            print(f"ID={agent_id} has < {presence_threshold:.1f} time steps => skipping calibration.")
            continue

        df_sub = df[df['time'].isin(times_for_id)].copy()

        # Determine lane mode and type mode
        lane_mode = df_id['lane_kf'].dropna()
        lane_val = lane_mode.mode()[0] if not lane_mode.empty else -1
        type_mode = df_id['type_most_common'].dropna()
        type_val = type_mode.mode()[0] if not type_mode.empty else -1

        frame_data_sub = build_frame_data(df_sub, neighbor_radius=NEIGHBOR_RADIUS, sample_step=SAMPLE_STEP)
        if len(frame_data_sub) < 2:
            print(f"ID={agent_id}: not enough distinct frames => skip.")
            continue

        # Run GA with empty polygons
        best_params, best_metrics = run_ga_for_subdata(frame_data_sub, {}, pop_size=POP_SIZE, generations=GENERATIONS)
        rowdict = {
            'id': agent_id,
            'Error': best_metrics['Error'],
            'MSE': best_metrics['MSE'],
            'RMSE': best_metrics['RMSE'],
            'MAE': best_metrics['MAE'],
            'MAPE': best_metrics['MAPE'],
            'NRMSE': best_metrics['NRMSE'],
            'SSE': best_metrics['SSE'],
            'R-squared': best_metrics['R-squared'],
            'Total Difference': best_metrics['TotalDiff'],
            'lane_kf': lane_val,
            'type_most_common': type_val,
            # Calibrated Pedestrian Parameters:
            'vAlpha0_ped': best_params[0],
            'tauAlpha_ped': best_params[1],
            'A_pp': best_params[2],
            'B_pp': best_params[3],
            'A_wall': best_params[4],
            'B_wall': best_params[5],
            # Calibrated Bicycle Parameters:
            'm_gamma': best_params[6],
            'tau_gamma': best_params[7],
            'v_gamma0': best_params[8],
            'a_gamma_max': best_params[9],
            'b_gamma': best_params[10],
            'eta_gamma': best_params[11],
            'mu_gamma': best_params[12],
            'eps_m': best_params[13],
            'A_w': best_params[14],
            'B_w': best_params[15],
            'A_s': best_params[16],
            'B_s': best_params[17],
            'T_i': best_params[18]
        }
        results_rows.append(rowdict)
        print(f"ID={agent_id} => Error={best_metrics['Error']:.4f}, R2={best_metrics['R-squared']:.3f}")

        df_pred_sub = get_predictions_for_subdata(frame_data_sub, {}, best_params, agent_id)
        all_predictions.append(df_pred_sub)

        # Plots for this ID
        if not df_pred_sub.empty:
            # X position comparison
            plt.figure()
            plt.plot(df_pred_sub['time'], df_pred_sub['x_obs'], 'ro-', label='Observed x')
            plt.plot(df_pred_sub['time'], df_pred_sub['x_pred'], 'bx--', label='Simulated x')
            plt.xlabel("Time")
            plt.ylabel("X Position")
            plt.title(f"ID={agent_id} - X Position Comparison - SF MODEL")
            plt.legend()
            plt.tight_layout()
            plt.draw()
            plt.pause(1)
            plt.close()

            # Y position comparison
            plt.figure()
            plt.plot(df_pred_sub['time'], df_pred_sub['y_obs'], 'ro-', label='Observed y')
            plt.plot(df_pred_sub['time'], df_pred_sub['y_pred'], 'bx--', label='Simulated y')
            plt.xlabel("Time")
            plt.ylabel("Y Position")
            plt.title(f"ID={agent_id} - Y Position Comparison - SF MODEL")
            plt.legend()
            plt.tight_layout()
            plt.draw()
            plt.pause(1)
            plt.close()

    # Summarize results
    df_results = pd.DataFrame(results_rows, columns=[
        'id','Error','MSE','RMSE','MAE','MAPE','NRMSE','SSE','R-squared','Total Difference',
        'lane_kf','type_most_common',
        'vAlpha0_ped','tauAlpha_ped','A_pp','B_pp','A_wall','B_wall',
        'm_gamma','tau_gamma','v_gamma0','a_gamma_max','b_gamma','eta_gamma','mu_gamma',
        'eps_m','A_w','B_w','A_s','B_s','T_i'
    ])
    out_csv = r"./calibration_per_id.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"Saved per-ID calibration metrics to {out_csv}")

    df_pred = pd.concat(all_predictions, ignore_index=True)
    out_pred_csv = r"./calibration_predictions.csv"
    df_pred.to_csv(out_pred_csv, index=False)
    print(f"Saved per-ID predictions to {out_pred_csv}")

    # Distribution plots
    metrics_cols = ['Error','MSE','RMSE','MAE','MAPE','NRMSE','SSE','R-squared','Total Difference']
    for col in metrics_cols:
        if df_results.empty:
            print("No results to plot for distribution of", col)
            continue
        plt.figure()
        df_results[col].hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(os.path.join(PLOT_DIR, f"dist_{col}.png"))
        plt.close()

    plt.ioff()
    print("Done.")

if __name__ == "__main__":
    main()
