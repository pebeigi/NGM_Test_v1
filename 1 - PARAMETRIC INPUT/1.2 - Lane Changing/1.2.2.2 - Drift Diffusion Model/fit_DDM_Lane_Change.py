import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.stats import f
from mpi4py import MPI
import copy

from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
import pickle

import random
from scipy.stats import norm
import time
from numba import jit
from geneticalgorithm import geneticalgorithm as ga
import psutil

from multiprocessing import Pool


# === MPI Initialization ===
# paralel computing needed because it is very computationally expensive
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  # should be equal to the number of draws

# Get total system memory (in bytes)
total_memory = psutil.virtual_memory().total/ (1024 ** 3)
print(total_memory/52)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NGM_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _NGM_ROOT not in sys.path:
    sys.path.insert(0, _NGM_ROOT)
from ngm_paths import ORGANIZED_DATA_PKL, dataset_path

############ Load data #######################################
with open(dataset_path(ORGANIZED_DATA_PKL), "rb") as f:
    organized_data = pickle.load(f)

# the data is in the form of dictionaries: first get the type of vehicle, they get the filename (which is the vehicle id)
# for each vehicle id, there are a few continuous trajectories to be selected from 

veh_type=0 # small_vehs, 1 for large vehicles 
if veh_type==0:
  sample_interval=3 # 1 sample out if every few
else:
  sample_interval=1 # big vehicles
selected_fnames=[]
k=0
for key in organized_data[veh_type]:
  k =k+1
  if k%sample_interval==0:
    selected_fnames.append(key)

print("Number of Vehicles:", len(selected_fnames))
#now get the number of LCs for each class
left_lc_count=0
right_lc_count=0
for fname in selected_fnames:
  #print(fname, len(organized_data[veh_type][fname]))
  for i in range(len(organized_data[veh_type][fname])):
    traj=(organized_data[veh_type][fname][i])
    if traj[1]==traj[0]+1:
      right_lc_count=right_lc_count+1
    elif traj[1]==traj[0]-1:
      left_lc_count=left_lc_count+1
print("right lc count:", right_lc_count)
print("left lc count:", left_lc_count)


######################### Select spec and reorder data  ################################################
cpu_per_task=0.1
total_cpu=5
num_per_group=700

data_lens=[]
seg_lens=[]
for fname in selected_fnames:
    data_lens.append(0)
    for traj_data in organized_data[veh_type][fname]:
        data_lens[-1]=data_lens[-1]+(len(traj_data[-3]))
        seg_lens.append(len(traj_data[-3]))
data_lens=np.array(data_lens)
selected_fnames=np.array(selected_fnames)

sorted_idxes=np.argsort(data_lens)
sorted_lens=data_lens[sorted_idxes]
sorted_fnames=selected_fnames[sorted_idxes]

# group them
num_groups = len(data_lens)//num_per_group  + 1 *(len(data_lens)%num_per_group!=0)

group_data=[]
group_fnames=[]
for g in range(num_groups):
    sub_fnames = sorted_fnames[g*num_per_group : (g+1)*num_per_group]
    group_fnames.append(sub_fnames)
    sub_data={}
    for sub_fname in sub_fnames:
        sub_data[sub_fname]=organized_data[veh_type][sub_fname]
    group_data.append(sub_data)


###############################  Read Model #############################################################
delta_t=0.1 # time step size
# t is for i and tau is for j


##### below are the functions needed for ddm yto calculate the probability
@jit(nopython=True, cache=True)
def m(i=None, j=None, rates=None):
  if i==j:
    return 0
  rel_rates=rates[j:i]
  return sum(rel_rates)*delta_t

@jit(nopython=True, cache=True)
def f(x, i, y, j, m_vals_t_tau, sigma ):
  t=i*delta_t
  tau=j*delta_t
  f = (1/np.sqrt(2*np.pi*(sigma**2)*(t-tau)))*np.exp(-(x-y-m_vals_t_tau[i][j])**2/ (2*(sigma**2) * (t-tau)))
  
  return f

@jit(nopython=True, cache=True)
def sai(a, i, y, j, m_vals_t_tau, sigma, rates):
  t=i*delta_t
  tau=j*delta_t
  sai=f(a, i, y, j, m_vals_t_tau, sigma)/ 2 *(0-rates[i]-  (a-y-m_vals_t_tau[i][j])/(t-tau)   )
  #print("sai", sai)
  return sai

@jit(nopython=True, cache=True)
def p_max(A_plus, i, x0, i0, m_vals_t_tau, sigma, rates, p_maxes, processed):
  t=i*delta_t
  t0 = i0*delta_t

  if processed[i]==1:
    return p_maxes[i], p_maxes # the length of p_maxes gradually increases

  p_max=-2*sai(A_plus, i, x0, i0, m_vals_t_tau, sigma, rates)
  #print("p_max", p_max, i, i0)
  for i_k in range(len(m_vals_t_tau)):
    t_k=i_k*delta_t


    if i_k>=i:
      break
    p_max=p_max+2*delta_t* p_maxes[i_k] * sai(A_plus, i, A_plus, i_k, m_vals_t_tau, sigma, rates)
  p_max=max(p_max, 0)
  p_maxes[i]=(delta_t*p_max)
  #print("P_maxes", p_maxes)

  return delta_t*p_max, p_maxes

@jit(nopython=True, cache=True)
def construct_m(rates):
  m_vals_t_tau=np.zeros((len(rates), len(rates)))
  for i in range(len(rates)):
    t=i*delta_t
    for j in range(len(rates)):
      tau=j*delta_t
      if j<=i:
        m_vals_t_tau[i][j] = m(i, j, rates )
  return m_vals_t_tau

@jit(nopython=True, cache=True)
def get_p_and_f_max(A_plus, i, A0, i0, m_vals_t_tau, sigma, rates):
  p_maxes=np.zeros(len(rates))
  f_maxes=np.zeros(len(rates))
  processed=np.zeros(len(rates))
  f_max=0
  for i in range(len(rates)):
    t=i*delta_t
    if i!=0:
      p_plus, p_maxes=p_max(A_plus, i, A0, i0, m_vals_t_tau, sigma, rates, p_maxes, processed)
    else:
      p_plus=0
    processed[i]=1
    f_max = f_max + p_plus
    f_maxes[i]=f_max
  return p_maxes, f_maxes

def drift_one_dir(veh_segment, random_params, dir=1):# the drift of one direction
  # 1 means right and -1 means left
  beta_L, beta_R, beta_G, beta_V, beta_MLC, G0, sigma, alpha = random_params

  # belows is the data for the vehicle trajectory segments, organized from the file that is read
  old_lane, new_lane, lc_dir, is_MLC_left, is_MLC_right, spds, lead_spds, lead_hws, left_lead_spds, left_lead_hws, left_lead_gaps, right_lead_spds, right_lead_hws, right_lead_gaps, left_follow_spds, left_follow_hws, left_follow_gaps, right_follow_spds, right_follow_hws, right_follow_gaps, left_gap_increase, right_gap_increase = veh_segment 

  # initial evidence
  A0 = 10 - alpha*lead_hws[0]
  A_plus = 20

  if dir==1:
    beta_dir=beta_R
    adj_follow_gaps=right_follow_gaps
    adj_lead_spds=right_lead_spds
    is_MLC = is_MLC_right
  else:
    beta_dir=beta_L
    adj_follow_gaps=left_follow_gaps
    adj_lead_spds=left_lead_spds
    is_MLC = is_MLC_left

  rates=beta_dir + beta_G*np.arctan(adj_follow_gaps-G0) + beta_V*np.arctan(adj_lead_spds-lead_spds) + is_MLC*beta_MLC # at an interval of every 0.1
  
  m_vals_t_tau=construct_m(rates)

  p_maxes, f_maxes = get_p_and_f_max(A_plus, i, A0, 0, m_vals_t_tau, sigma, rates)

  return p_maxes, f_maxes


# simulate the individual vehicles

def simulate_individual(fname, random_params, veh_segments):
  # iterate over all segments for the vehicle of the same ID
  segment_costs=[] # costs of each segment

  for i in range(len(veh_segments)):
    veh_segment=veh_segments[i]
    old_lane=veh_segment[0]
    new_lane=veh_segment[1]
    lc_dir=veh_segment[2]

    if old_lane == 10: # the left most lane, drift only possible to the right
      # no left lane
      p_max_right, f_max_right = drift_one_dir(veh_segment, random_params, dir=1)
      p_max_left = np.zeros(len(p_max_right))
      f_max_left = np.zeros(len(f_max_right))

    elif old_lane == 15: # the right most lane, drift only possible to the left
      # no right lane
      p_max_left, f_max_left = drift_one_dir(veh_segment, random_params, dir=-1)
      p_max_right = np.zeros(len(p_max_left))
      f_max_right = np.zeros(len(f_max_left))

    else: # middle lanes, drift 
      # both lanes exist
      p_max_right, f_max_right = drift_one_dir(veh_segment, random_params, dir=1)
      p_max_left, f_max_left = drift_one_dir(veh_segment, random_params, dir=-1)

    if lc_dir==0:
      likelihood = (1-f_max_right[-1]) * (1-f_max_left[-1])
    elif lc_dir==1: # lc to the right
      likelihood = p_max_right[-1] * (1-f_max_left[-1])
    elif lc_dir==-1:
      likelihood = (1-f_max_right[-1]) * p_max_left[-1]

    
    segment_costs.append(np.log(max(likelihood, np.exp(-100))))
  
  return sum(segment_costs) 

# the lists for all the log-likelihoods and parameters for the sake of checking 
all_lls=[]
all_params=[]


# the cost function for genetic algorithm
def ddm_cost(params):
    # need to broad the parameters
    params=comm.bcast(params, root=0) # from root 0 broad cast the parameters
    if rank == 0:
        rank0_start = time.time()

    # if the optimization is finished, rank 1 communicates stop signal to all other ranks
    if isinstance(params, str):
        return "stop" # to make other workers exit the loop when needed

    # read the variables and construct the parameter distributions, for avs, the standard deviations should all be zero
    beta_L_mean, beta_R_mean, beta_G_mean, beta_V_mean, beta_MLC_mean, G0_mean, sigma_mean, alpha_mean, std_beta_L, std_beta_R, std_beta_G, std_beta_V, std_beta_MLC, std_G0, std_alpha, cov_beta_G_beta_V, cov_beta_G_alpha, cov_beta_V_alpha = params
    mean_vals= np.array([beta_L_mean, beta_R_mean, beta_G_mean, beta_V_mean, beta_MLC_mean, G0_mean, sigma_mean, alpha_mean])
    std_vals=np.array([std_beta_L, std_beta_R, std_beta_G, std_beta_V, std_beta_MLC, std_G0, 0, std_alpha]) # sigmal is assumed to be deterministic
    cov_matrix=np.diag(std_vals**2)
    
   
    cov_matrix[2,3]=params[15]*params[10]*params[11]
    cov_matrix[3,2]=params[15]*params[10]*params[11]
    cov_matrix[2,7]=params[16]*params[10]*params[14]
    cov_matrix[7,2]=params[16]*params[10]*params[14]
    cov_matrix[3,7]=params[17]*params[11]*params[14]
    cov_matrix[7,3]=params[17]*params[11]*params[14]

    # if not positive semidefinite, this cov matrix is not valid and it is out of bound
    if np.all(np.linalg.eigvals(cov_matrix) >= 0)==False:  # check for positive definitiveness
        added_cost=10000000000
        oob=True
    else:
        added_cost = 0
        oob=False

    # draw parameters and organize data into parallelizable format
    
    params_all_members=[]
    for g in range(num_groups):
        params_all_members.append([])
        group_len=len(group_fnames[g])
        for gl in range(group_len):
            random_params_base = np.random.multivariate_normal(mean_vals, cov_matrix, size=int(num_trials)//2)
            # reflect these variables to get the conjugates
            random_params_conjugate = 2*mean_vals - random_params_base
            random_params=np.concatenate((random_params_base, random_params_conjugate))
            params_all_members[-1].append(random_params)
    all_params_groups=[]
    for trial in range(num_trials):
        all_params_groups.append([])
        for g in range(num_groups):
            all_params_groups[-1].append([])
            group_len=len(group_fnames[g])
            for gl in range(group_len):
                all_params_groups[-1][-1].append(params_all_members[g][gl][trial])

    # Start iterating
    # the number of trials would be splitted via comm
    # the samples would be splitted via ray
    costs_all_trials=[]

    params_group = all_params_groups[rank]

    trial_costs=[]
    
    for group in range(num_groups):
        params_this_group=params_group[group]
        sub_fnames=group_fnames[group]
        sub_data=group_data[group]
        time_group_start=time.time()

        # if out of bound (non positive-semidefinite cov matrix), just assign a large cost everywhere in the loop, so that the computation is not wasted
        
        if oob==True:
            for i_f in range(len(sub_fnames)):
                trial_costs.append(1000000)
            
        elif oob==False: 
            with Pool(8) as p:
                input_args_parallel = []
                for i_f in range(len(sub_fnames)):
                   input_args_parallel.append(tuple([sub_fnames[i_f], params_this_group[i_f], sub_data[sub_fnames[i_f]]]))
                group_costs=p.starmap(simulate_individual, input_args_parallel)
            trial_costs.append(sum(group_costs))

    tot_trial_cost=sum(trial_costs)
    
    costs_all_trials = comm.gather(tot_trial_cost, root=0)
    
        
    # for rank 0, save the progress 
    if rank==0:
        with open("progress/gen"+str(int(len(all_lls)//pop_size))+"ind"+ str(int(len(all_lls)%pop_size)) + ".pkl", "wb") as f:
            if len(all_lls)==0:
                pickle.dump([all_lls, all_params], f)
            else:
                pickle.dump([[all_lls[-1]], [all_params[-1]]], f)

        ll=-sum(costs_all_trials)/len(costs_all_trials) + added_cost
        all_lls.append(ll)
        all_params.append(params)

        print("ll", ll)
        print("Rank0 calculated ll in {0} seconds".format(time.time()-rank0_start))

        return ll
    
    return None


############################################ Perform Optimization ###############################################
# Genetic Algo set up
pop_size=50 # for each generation of GA, the number of individuals
num_trials=size # number of simulations (draws) for each vehicle. Size is defined earlier (It should be just 1 for AV)
if rank == 0:
    
    algorithm_param = {'max_num_iteration': 1000, # 1000 iterations
    'population_size':pop_size, # 50 samples
    'mutation_probability':0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type':'uniform',
    'max_iteration_without_improv':200}
    
     # simulate 100 trials, or any even number
    
    
    beta_L_range=[-5, 5] # left side constant
    beta_R_range=[-5, 5] # right side constant
    beta_G_range=[0, 5] # gap coefficient
    beta_V_range=[0, 5] # speed coefficient
    beta_MLC_range=[0, 5] # the constant of MLC
    G0_range=[5, 50] # the gap above which there is an incentive to perform a LC
    sigma_range=[0.001, 10] # noise range
    alpha_range = [-10, 10] # the coefficients of the headway


    # standard deviations
    # these should all be zero for AVs
    
    std_beta_L_range=[0, 5 ] # left side constant
    std_beta_R_range=[0, 5] # right side constant
    std_beta_G_range=[0, 10] # gap coefficient
    std_beta_V_range=[0, 5] # speed coefficient
    std_beta_MLC_range=[0, 5] # the constant of MLC
    std_G0_range=[0, 10] # the gap above which there is an incentive to perform a LC
    std_alpha_range = [0, 10] # the coefficients of the headway
    
    # consider three covariances for now # this is the covariance coefficient
    cov_beta_G_beta_V_range =[-1, 1]
    cov_beta_G_alpha_range = [-1, 1]
    cov_beta_V_alpha_range = [-1, 1]
    
    
    var_names = ["beta_L_mean", "beta_R_mean", "beta_G_mean", "beta_V_mean", "beta_MLC_mean", "G0_mean", "sigma", "alpha_mean", "std_beta_L", "std_beta_R", "std_beta_G", "std_beta_V", "std_beta_MLC", "std_G0", "std_alpha", "cov_beta_G_beta_V", "cov_beta_G_alpha", "cov_beta_V_alpha"]
    var_ranges = np.array([beta_L_range, beta_R_range, beta_G_range, beta_V_range, beta_MLC_range, G0_range, sigma_range, alpha_range, std_beta_L_range, std_beta_R_range, std_beta_G_range, std_beta_V_range, std_beta_MLC_range, std_G0_range, std_alpha_range, cov_beta_G_beta_V_range, cov_beta_G_alpha_range, cov_beta_V_alpha_range])
    
    ddm_model=ga(function=ddm_cost, dimension=len(var_ranges), variable_type='real',variable_boundaries=var_ranges, function_timeout=20000, algorithm_parameters=algorithm_param, progress_bar=False)
    
    
    ddm_model.run()

    best_params = ddm_model.best_variable
    best_cost = ddm_model.best_function

    with open("progress/final_result.pkl", "wb") as f:
            pickle.dump([best_params, best_cost], f)

    ddm_cost("stop")


else:
    # for other ranks, just keep assisiting rank 0 if needed. Stop them if rank 0 stops 
    while True:

        worker_result = ddm_cost(None)
        if worker_result=="stop":
            print(f"Rank {rank} exiting")
            break


