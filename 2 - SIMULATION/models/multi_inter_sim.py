import os
import sys
import traci
import traci.constants as tc
import sumolib
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from sumolib import checkBinary
import math
from scipy.stats import norm
import ray
import pickle
import io
from numba import njit
import subprocess
import random
from . import ns
from . import coop_models as coop
from . import signal_control
from .paths import (
    MODEL_PARAMS_DIR,
    MODELS_DIR,
    TEMPLATES_DIR,
    load_mobil_results_csv,
    mobil_params_from_csv_row,
)

HOLD_SEC = 1
# SUMO TraCI: traci.vehicle.changeLane(..., duration_s); not LC_NO_RETURN (see freeway_sim).
TRACI_CHANGE_LANE_DURATION_S = 3.0
# TraCI handoff: use SUMO default CF + LC when within this distance of the next intersection.
# EW (EB/WB): next_intersection_distance is (approx.) meters to the intersection along x; use d2i < HANDOFF
#   so vehicles slightly past the reference (negative d2i) still hand off — avoids MOBIL/DDM at the stop line.
# NS (NB/SB): distance uses y; sign differs — use abs(d2i) < HANDOFF (do not use raw d2i < HANDOFF on NS).
HANDOFF_DIST_M = 80.0

#Fix Progress Bar
#def run_sim_multi_inter(user_input_data):
def run_sim_multi_inter(user_input_data, progress_cb=None, is_running_check=None):
    ################################################## First Geometry Construction ###########################################################
    
    num_NS_lanes = user_input_data["Geometry"]['Num_Lanes_NS']
    num_EW_lanes = user_input_data["Geometry"]['Num_Lanes_EW']
    west_central_len = user_input_data["Geometry"]["West_Central_Length"]
    central_east_len = user_input_data["Geometry"]["Central_East_Length"]

    # these are immutable
    num_inters = 3
    
    main_node_names = ["West_Most"] # the nodes from west-most to east-most
    for i in range(num_inters):
        main_node_names.append("Intersection"+str(i+1))
    main_node_names.append("East_Most")
    
    
    # file names 
    # template file
    # Pedram: Replace the template file location
    multi_inters_template_file = str(
        TEMPLATES_DIR / f"multi_intersections_EW{num_EW_lanes}_NS{num_NS_lanes}_template.net.xml"
    )
     # function to write the new arterial file 
    new_file_name = str(MODELS_DIR / "multi_intersections.net.xml")  # the adjusted  network file
    net_file_name = new_file_name # occasionally this variable is used to
    ###### net file names in simulation
    route_file_name = str(MODELS_DIR / "all_trips.trips.xml")  # trip file will be saved here

    config_file = str(MODELS_DIR / "multi_intersections.sumocfg")  # config file, will set the  route_file_name and new_file_name in the simulation loop

    END_APPROACH = 300  # meters (was 100)
    MINOR_ROAD_LEN = 300  # meters north/south from intersection (adjust as needed)
    distance_between_nodes = [END_APPROACH, west_central_len, central_east_len, END_APPROACH]

    # the x coordination of intersections define
    inter1_x = END_APPROACH
    inter2_x = END_APPROACH + west_central_len
    inter3_x = END_APPROACH + west_central_len + central_east_len
    inter_thresh = 50
    
    # the new location of nodes
    main_node_xs=[0]
    for d in distance_between_nodes:
        main_node_xs.append(main_node_xs[-1]+d)
    
    # now convert the main_nodes into dict:
    new_node_xs={} # the x coordinates of the main road
    for i in range(len(main_node_names)):
        new_node_xs[main_node_names[i]] = main_node_xs[i]


    new_node_ys = {}
    # Main road junctions y = 0
    for name in main_node_names:
        new_node_ys[name] = 0.0

    # Cross road ends: set y based on desired minor road length
    for i in range(num_inters):
        new_node_ys[f"Intersection{i + 1}"] = 0.0
        new_node_ys[f"North_Most{i + 1}"] = MINOR_ROAD_LEN
        new_node_ys[f"South_Most{i + 1}"] = -MINOR_ROAD_LEN

    # now change the northmost and southmost point of crossing roads
    for i in range(num_inters):
        new_node_xs["North_Most"+str(i+1)] = new_node_xs["Intersection"+str(i+1)]
        new_node_xs["South_Most"+str(i+1)] = new_node_xs["Intersection"+str(i+1)]

    
    def construct_geometry(new_file_name):
        tree = ET.parse(multi_inters_template_file)
        root = tree.getroot()
        
        for junction in root.findall("junction"): # reset the junction location
            junction.set("shape", "")
            junction_id = junction.get("id")
            if junction_id in new_node_xs:
                junction.set("x", str(np.round(new_node_xs[junction_id], 2)))
            if junction_id in new_node_ys:
                junction.set("y", str(np.round(new_node_ys[junction_id], 2)))
                
        
        for edge in root.findall("edge"): # the new length need to be recalculated automatically
            if "length" in edge.attrib:
                del edge.attrib["length"]
            if "shape" in edge.attrib:
                del edge.attrib["shape"]
        
        tree.write(new_file_name)
        # Run netconvert to automatically update the attributes
        try:
            subprocess.run([
                "netconvert",
                "--sumo-net-file", new_file_name,
                "-o", new_file_name
            ], check=True)
            print("Netconvert finished successfully. Output saved at:", new_file_name)
        except subprocess.CalledProcessError as e:
            print("Netconvert failed with error:", e)
    
    construct_geometry(new_file_name) # construct the new geometry
    
    # get conections between lanes single_inter_template_file
    def parse_connection_vias(net_file):
        tree = ET.parse(net_file)
        root = tree.getroot()
    
        connections = {}
    
        for conn in root.iter("connection"):
            from_edge = conn.get("from")
            to_edge   = conn.get("to")
            from_lane = conn.get("fromLane")
            to_lane   = conn.get("toLane")
            via       = conn.get("via")  # <-- this is what you want
    
            # Some connections don't have 'via' (straight through), skip or store None
            if not from_edge+"_"+str(from_lane) in connections:
                connections[from_edge+"_"+str(from_lane)]={}
            connections[from_edge+"_"+str(from_lane)][to_edge+"_"+str(to_lane)] = via
    
        return connections

    connections_between_edges = parse_connection_vias(new_file_name)
    
    # define lanes (define for the max possible number of lanes here)
    # In SUMO the lanes are by segments, therefore for finding neighbors, we need to connect them to form real lanes
    
    EB_lane0_names = ["EB01_0", connections_between_edges["EB01_0"]["EB12_0"], "EB12_0", connections_between_edges["EB12_0"]["EB23_0"], "EB23_0", connections_between_edges["EB23_0"]["EB34_0"], "EB34_0"]
    WB_lane0_names = ["WB34_0", connections_between_edges["WB34_0"]["WB23_0"], "WB23_0", connections_between_edges["WB23_0"]["WB12_0"], "WB12_0", connections_between_edges["WB12_0"]["WB01_0"], "WB01_0"]
    NB1_lane0_names = ["NB1_South_0", connections_between_edges["NB1_South_0"]["NB1_North_0"], "NB1_North_0"]
    NB2_lane0_names = ["NB2_South_0", connections_between_edges["NB2_South_0"]["NB2_North_0"], "NB2_North_0"]
    NB3_lane0_names = ["NB3_South_0", connections_between_edges["NB3_South_0"]["NB3_North_0"], "NB3_North_0"]
    SB1_lane0_names = ["SB1_North_0", connections_between_edges["SB1_North_0"]["SB1_South_0"], "SB1_South_0"]
    SB2_lane0_names = ["SB2_North_0", connections_between_edges["SB2_North_0"]["SB2_South_0"], "SB2_South_0"]
    SB3_lane0_names = ["SB3_North_0", connections_between_edges["SB3_North_0"]["SB3_South_0"], "SB3_South_0"]


    ''' # old
    WB_lane0_names = ["WB34_0",  ":Intersection3_7_0" ,  "WB23_0", ":Intersection2_7_0" , "WB12_0", ":Intersection1_7_0"  , "WB01_0"]
    
    NB1_lane0_names = ["NB1_South_0", ":Intersection1_13_0", "NB1_North_0" ]
    NB2_lane0_names = ["NB2_South_0", ":Intersection2_13_0", "NB2_North_0" ]
    NB3_lane0_names = ["NB3_South_0", ":Intersection3_13_0", "NB3_North_0" ]
    
    SB1_lane0_names = ["SB1_North_0", ":Intersection1_1_0", "SB1_South_0"]
    SB2_lane0_names = ["SB2_North_0", ":Intersection2_1_0", "SB2_South_0"]
    SB3_lane0_names = ["SB3_North_0", ":Intersection3_1_0", "SB3_South_0"]
    '''

    def construct_lane_list(template, num_lanes, all_lane_names):
        for n in range(num_lanes):
            all_lane_names.append([])
            for seg in template:
                if seg is None:
                    continue
                all_lane_names[-1].append(seg[:-1] + str(n))
        return all_lane_names
    
    all_lane_names = []
    all_lane_names = construct_lane_list(EB_lane0_names, 5, all_lane_names )
    all_lane_names = construct_lane_list(WB_lane0_names, 5, all_lane_names )
    
    all_lane_names = construct_lane_list(NB1_lane0_names, 5, all_lane_names )
    all_lane_names = construct_lane_list(NB2_lane0_names, 5, all_lane_names )
    all_lane_names = construct_lane_list(NB3_lane0_names, 5, all_lane_names )
    
    all_lane_names = construct_lane_list(SB1_lane0_names, 5, all_lane_names )
    all_lane_names = construct_lane_list(SB2_lane0_names, 5, all_lane_names )
    all_lane_names = construct_lane_list(SB3_lane0_names, 5, all_lane_names )
    
    
    # for each lane number define the next one:
    next_lanes={}
    last_lanes={}
    for lane_names in all_lane_names:
        for i in range(len(lane_names)):
            if i!=len(lane_names)-1 and i!=0:
                next_lanes[lane_names[i]]= lane_names[i+1]
                last_lanes[lane_names[i]] = lane_names[i-1]
            if i==0:
                next_lanes[lane_names[i]]= lane_names[i+1]
                last_lanes[lane_names[i]] = None
            if i==len(lane_names)-1:
                next_lanes[lane_names[i]]= None
                last_lanes[lane_names[i]] = lane_names[i-1]


    
    def get_lane_lengths(net_file_name):
        with open(net_file_name, 'r') as f:
            xml_content = f.read()
        
        lane_seg_lengths = {}
        
        try:
            root = ET.parse(io.StringIO(xml_content)).getroot()
        except ET.ParseError as e:
            print(f"Error parsing XML content: {e}")
            return {}
    
        # Iterate through all <edge> elements
        for edge in root.findall('edge'):
            # Iterate through all <lane> elements within each edge
            for lane in edge.findall('lane'):
                lane_id = lane.get('id')
                length_str = lane.get('length')
                
                if lane_id and length_str:
                    try:
                        lane_seg_lengths[lane_id] = float(length_str)
                    except ValueError:
                        print(f"Warning: Could not convert length '{length_str}' for lane '{lane_id}' to a float.")
    
        return lane_seg_lengths
    
    lane_lengths = get_lane_lengths(net_file_name)
    #print("all_lanes", all_lane_names)
    # to get the global location we convert to the global coordinate for longitudinal location
    lane_segment_to_lane_list ={} 
    for lanes in all_lane_names:
        for lane in lanes:
            lane_segment_to_lane_list[lane]=np.array(lanes)
    # positions along lane
    pos_adjust_by_lanes = {}
    # the start of weaving area is 0, as a reference
    for lanes in all_lane_names:
        adjustment = 0
        #print(lanes)
        for lane in lanes:
            pos_adjust_by_lanes[lane] = adjustment
            adjustment = adjustment +  lane_lengths[lane[:-1]+"0"] # lane 0 always exists and they are of the same length
            
                    
   
    lane_segment_to_idx={}
    for lane_idx in range(len(all_lane_names)):
        lanes = all_lane_names[lane_idx]
        for lane in lanes:
            lane_segment_to_idx[lane] = lane_idx
  
    
    # define vehicle inputs and select driving model
    in_names = ["EB01","WB34", "NB1_South", "NB2_South", "NB3_South", "SB1_North", "SB2_North", "SB3_North"]
    out_names = ["WB01", "EB34", "SB1_South", "SB2_South", "SB3_South", "NB1_North", "NB2_North", "NB3_North"]
    user_edge_names = ["West", "East", "South West", "South", "South East", "North West", "North", "North East"] # map to user input of users
    origins_ = []
    destinations_ = []
    in_flows=[]
    
    # need to think about how to let users decide on inputs
    for i in range(len(in_names)):
        in_name = in_names[i]
        for j in range(len(out_names)):
            if j==i:
                continue
            out_name = out_names[j]
            # now to format in and out flows
            in_user_edge_name = user_edge_names[i]
            out_user_edge_name = user_edge_names[j]
            in_flows.append(user_input_data['Vehicle_Flows'][in_user_edge_name][out_user_edge_name]) # based on user input             
    
            origins_.append(in_name)
            destinations_.append(out_name)
            #print(in_name, out_name)

    # Signal control: Webster from volumes or manual override from user; arterial offsets
    signal_plans = signal_control.get_multi_signal_plans(
        user_input_data, in_flows, origins_, destinations_, user_edge_names,
        in_names, num_EW_lanes, num_NS_lanes
    )
    sig = user_input_data.get("Signal_Control") or {}
    offset_1_2 = int(sig.get("offset_1_2", 0))
    offset_2_3 = int(sig.get("offset_2_3", 0))
    signal_control.apply_multi_signal_to_net(new_file_name, signal_plans, offset_1_2=offset_1_2, offset_2_3=offset_2_3)
    
    ########################### Simulation Hyperparameters ##############################################################################
    
    sim_visualization = user_input_data["Sim_Visualization"] # whether user wants visualization
    sim_time =  user_input_data["Sim_Time"] # 1 hr of simulation
    
    min_t = 0
    t_step = user_input_data["Sim_StepSize"]

    # --- Collision / rear-end safety tuning (ported from freeway_sim.py) ---
    # Keep models unchanged; only constrain parameters for realism/safety.
    #
    # Optional guards (OFF by default to avoid changing the model logic):
    # - Enable_Safety_SpeedCap: post-process speed to guarantee a minimum gap (changes control logic)
    # - Set_SpeedMode_Safety: forces SUMO safety checks even under setSpeed (changes behavior)
    ENABLE_SAFETY_SPEED_CAP = bool(user_input_data.get("Enable_Safety_SpeedCap", False))
    PT_STOCHASTIC_ENABLE = bool(user_input_data.get("PT_Stochastic_Enable", False))
    PT_STOCHASTIC_SIGMA = float(user_input_data.get("PT_Stochastic_Sigma", 0.0))
    SET_SPEEDMODE_SAFETY = bool(user_input_data.get("Set_SpeedMode_Safety", False))
    SAFETY_EXTRA_GAP = float(user_input_data.get("Safety_Extra_Gap", 0.5))
    # Clamp extremely aggressive IDM samples (used by our custom longitudinal model)
    IDM_MIN_T = float(user_input_data.get("IDM_Min_T", 0.8))      # s
    IDM_MIN_B = float(user_input_data.get("IDM_Min_b", 2.0))      # m/s^2
    IDM_MIN_S0 = float(user_input_data.get("IDM_Min_s0", 2.0))    # m
    # (No SUMO vType tau/decel/emergencyDecel guarding here — we keep the mechanism consistent
    # across freeway/multi/single and handle safety via clamps + optional runtime guards.)
    
    ####################### Vehicle Input Flow Characteritics ###############################################################################
    
    ## Vehicle Flow Variables

    HV_rate = user_input_data["Vehicle_Flows"].get("HV_rate", 0.0)
    AV_rate = user_input_data["Vehicle_Flows"].get("AV_rate", 0.0)
    CAV_rate = user_input_data["Vehicle_Flows"].get("CAV_rate", 0.0)
    CAHV_rate = user_input_data["Vehicle_Flows"].get("CAHV_rate", 0.0)
    SV_rate = user_input_data["Vehicle_Flows"].get(
        "SV_rate",
        max(0.0, 1.0 - HV_rate - AV_rate - CAV_rate - CAHV_rate),
    )

    # Vehicle classes:
    #   0: SV, 1: AV, 2: HV, 3: CAV (AV size), 4: CAHV (HV size)
    veh_lens = {0: 4.5, 1: 4.5, 2: 12.0, 3: 4.5, 4: 12.0}
    
    ############################################ Driving Model Selection #################################################
    LC_model_name = user_input_data["LC_Model"]
    CF_model_name = user_input_data["CF_Model"]
    CF_default = user_input_data["CF_Default_Params"] # True or false
    LC_default = user_input_data["LC_Default_Params"] #  True or false

    min_hw = 1.5 # this is when generating the vehicles only just to make sure they do not overlap
    min_gap =2 # min gap for DDM lc Model
    
    # loading default model
    model_folder = str(MODEL_PARAMS_DIR) + os.sep
    IDM_param_data = {
        0: pd.read_csv(model_folder + "merged_IDM_S.csv"),
        1: pd.read_csv(model_folder + "merged_IDM_A.csv"),
        2: pd.read_csv(model_folder + "merged_IDM_L.csv"),
        4: pd.read_csv(model_folder + "merged_IDM_CL.csv"),
    }
    PT_param_data = {
        0: pd.read_csv(model_folder + "merged_PT_S.csv"),
        1: pd.read_csv(model_folder + "merged_PT_A.csv"),
        2: pd.read_csv(model_folder + "merged_PT_L.csv"),
        4: pd.read_csv(model_folder + "merged_PT_CL.csv"),
    }
    
    MOBIL_data = load_mobil_results_csv()  # not class-based
    veh_class_names = {0: "Small Vehicle", 1: "Automated Vehicle", 2: "Heavy Vehicle"}
     
    # functions to sample the model params
    def sample_IDM(veh_class=None): # 0 for sv, 1 for A, 2 for LV
        base_class = 1 if veh_class == 3 else 2 if veh_class == 4 else veh_class
        if CF_default == False and CF_model_name == "IDM": # user customized and this model is chosen
            
            mean_T = user_input_data["CF_Parameters"]["T"][veh_class_names[base_class]]["Mean"]
            std_T = user_input_data["CF_Parameters"]["T"][veh_class_names[base_class]]["Std"]
            
            mean_a = user_input_data["CF_Parameters"]["a"][veh_class_names[base_class]]["Mean"]
            std_a = user_input_data["CF_Parameters"]["a"][veh_class_names[base_class]]["Std"]
            
            mean_b = user_input_data["CF_Parameters"]["b"][veh_class_names[base_class]]["Mean"]
            std_b = user_input_data["CF_Parameters"]["b"][veh_class_names[base_class]]["Std"]

            mean_v0 = user_input_data["CF_Parameters"]["v_0"][veh_class_names[base_class]]["Mean"]
            std_v0 = user_input_data["CF_Parameters"]["v_0"][veh_class_names[base_class]]["Std"]

            mean_s0 = user_input_data["CF_Parameters"]["s_0"][veh_class_names[base_class]]["Mean"]
            std_s0 = user_input_data["CF_Parameters"]["s_0"][veh_class_names[base_class]]["Std"]

            random_T = max(0.001, random.gauss(mean_T, std_T))
            random_a = max(0.001, random.gauss(mean_a, std_a))
            random_b = max(0.001, random.gauss(mean_b, std_b))
            random_v0 = max(0.001, random.gauss(mean_v0, std_v0))
            random_s0 = max(0.001, random.gauss(mean_s0, std_s0))

            driving_params = np.array([ random_T, random_a, random_b, random_v0, random_s0, 4], dtype=np.float64)
            return driving_params
            
        
        else: #sample default, if this model is not chosen the model will not be used
            _idm_csv_key = 4 if veh_class == 4 else (1 if veh_class == 3 else veh_class)
            samples = IDM_param_data[_idm_csv_key]
            samples = samples[samples["T"]>0]
            picked_row = samples.sample(n=1).iloc[0]
            driving_params = np.array(picked_row[["T", "a", "b", "v0", "so", "delta"]].values, dtype = np.float64 )
            # Apply 50% of the sample speeds
            driving_params[3] = max(0.001, 0.5 * driving_params[3])
            return driving_params
    
    def sample_PT(veh_class=None):
        # "T_max", "α", "β", "W_c", "Gamma1", "Gamma2", "W"
        base_class = 1 if veh_class == 3 else 2 if veh_class == 4 else veh_class
        if CF_default == False and CF_model_name == "PT":
            mean_T_max = user_input_data["CF_Parameters"]["T_max"][veh_class_names[base_class]]["Mean"]
            std_T_max= user_input_data["CF_Parameters"]["T_max"][veh_class_names[base_class]]["Std"]
            
            mean_alpha = user_input_data["CF_Parameters"]["α"][veh_class_names[base_class]]["Mean"]
            std_alpha = user_input_data["CF_Parameters"]["α"][veh_class_names[base_class]]["Std"]
            
            mean_beta = user_input_data["CF_Parameters"]["β"][veh_class_names[base_class]]["Mean"]
            std_beta = user_input_data["CF_Parameters"]["β"][veh_class_names[base_class]]["Std"]

            mean_Wc = user_input_data["CF_Parameters"]["W_c"][veh_class_names[base_class]]["Mean"]
            std_Wc = user_input_data["CF_Parameters"]["W_c"][veh_class_names[base_class]]["Std"]

            mean_Gamma1 = user_input_data["CF_Parameters"]["Gamma1"][veh_class_names[base_class]]["Mean"]
            std_Gamma1 = user_input_data["CF_Parameters"]["Gamma1"][veh_class_names[base_class]]["Std"]

            mean_Gamma2 = user_input_data["CF_Parameters"]["Gamma2"][veh_class_names[base_class]]["Mean"]
            std_Gamma2 = user_input_data["CF_Parameters"]["Gamma2"][veh_class_names[base_class]]["Std"]

            mean_Wm = user_input_data["CF_Parameters"]["W"][veh_class_names[base_class]]["Mean"]
            std_Wm = user_input_data["CF_Parameters"]["W"][veh_class_names[base_class]]["Std"]

            random_T_max = max(0.001, random.gauss(mean_T_max, std_T_max))
            random_alpha = max(0.001, random.gauss(mean_alpha, std_alpha))
            random_beta = max(0.001, random.gauss(mean_beta, std_beta))
            random_Wc = max(0.001, random.gauss(mean_Wc, std_Wc))
            random_Gamma1 = max(0.001, random.gauss(mean_Gamma1, std_Gamma1))
            random_Gamma2 = max(0.001, random.gauss(mean_Gamma2, std_Gamma2))
            random_Wm = max(0.001, random.gauss(mean_Wm, std_Wm))
            
            driving_params = np.array([ random_T_max, random_alpha, random_beta, random_Wc, random_Gamma1, random_Gamma2, random_Wm ], dtype=np.float64)
            return driving_params 
        
        else:
            _pt_csv_key = 4 if veh_class == 4 else (1 if veh_class == 3 else veh_class)
            samples = PT_param_data[_pt_csv_key]

            picked_row = samples.sample(n=1).iloc[0]
            driving_params = np.array(picked_row[['Tmax', 'Alpha', 'Beta', 'Wc', 'Gamma1','Gamma2', 'Wm']].values, dtype=np.float64)
            return driving_params
    
    def sample_MOBIL(): # sample from all data where both MLC and DLC exists
        if LC_default == False and LC_model_name == "MOBIL":
            
            mean_p_disc = user_input_data["LC_Parameters"]['Disc: p_opt'][veh_class_names[veh_class]]["Mean"]
            std_p_disc= user_input_data["LC_Parameters"]['Disc: p_opt'][veh_class_names[veh_class]]["Std"]
            mean_ath_disc = user_input_data["LC_Parameters"]['Disc: a_th'][veh_class_names[veh_class]]["Mean"]
            std_ath_disc= user_input_data["LC_Parameters"]['Disc: a_th'][veh_class_names[veh_class]]["Std"]

            mean_b_disc = user_input_data["LC_Parameters"]['Disc: b_safe'][veh_class_names[veh_class]]["Mean"]
            std_b_disc = user_input_data["LC_Parameters"]['Disc: b_safe'][veh_class_names[veh_class]]["Std"]
            mean_b_mand = user_input_data["LC_Parameters"]['Mand: b_safe'][veh_class_names[veh_class]]["Mean"]
            std_b_mand = user_input_data["LC_Parameters"]['Mand: b_safe'][veh_class_names[veh_class]]["Std"]

            random_p_disc = max(0.001, random.gauss(mean_p_disc, std_p_disc))
            random_ath_disc = random.gauss(mean_ath_disc, std_ath_disc)
            random_b_disc = max(0.1, random.gauss(mean_b_disc, std_b_disc))
            random_b_mand = max(0.1, random.gauss(mean_b_mand, std_b_mand))

            return np.array([random_p_disc, random_ath_disc, random_b_disc, random_b_mand], dtype=np.float64)

        else:
            if MOBIL_data.empty:
                return mobil_params_from_csv_row({})
            picked_row = MOBIL_data.sample(n=1).iloc[0]
            return mobil_params_from_csv_row(picked_row)
    
    def sample_DDM(): # 
        if LC_default == False and LC_model_name == "DDM":
            mean_alpha_h = user_input_data["LC_Parameters"]["α_h"][veh_class_names[veh_class]]["Mean"]
            std_alpha_h = user_input_data["LC_Parameters"]["α_h"][veh_class_names[veh_class]]["Std"]
            
            mean_beta0_left = user_input_data["LC_Parameters"]["β_0_left"][veh_class_names[veh_class]]["Mean"]
            std_beta0_left = user_input_data["LC_Parameters"]["β_0_left"][veh_class_names[veh_class]]["Std"]

            mean_beta0_right = user_input_data["LC_Parameters"]["β_0_right"][veh_class_names[veh_class]]["Mean"]
            std_beta0_right = user_input_data["LC_Parameters"]["β_0_right"][veh_class_names[veh_class]]["Std"]

            mean_beta_G = user_input_data["LC_Parameters"]["β_G"][veh_class_names[veh_class]]["Mean"]
            std_beta_G = user_input_data["LC_Parameters"]["β_G"][veh_class_names[veh_class]]["Std"]

            mean_G_0 = user_input_data["LC_Parameters"]["G_0"][veh_class_names[veh_class]]["Mean"]
            std_G_0 = user_input_data["LC_Parameters"]["G_0"][veh_class_names[veh_class]]["Std"]
            
            mean_beta_V = user_input_data["LC_Parameters"]["β_V"][veh_class_names[veh_class]]["Mean"]
            std_beta_V = user_input_data["LC_Parameters"]["β_V"][veh_class_names[veh_class]]["Std"]
            
            mean_beta_MLC = user_input_data["LC_Parameters"]["β_MLC"][veh_class_names[veh_class]]["Mean"]
            std_beta_MLC = user_input_data["LC_Parameters"]["β_MLC"][veh_class_names[veh_class]]["Std"]
            
            mean_sigma = user_input_data["LC_Parameters"]["σ"][veh_class_names[veh_class]]["Mean"]
            std_sigma = user_input_data["LC_Parameters"]["σ"][veh_class_names[veh_class]]["Std"]


            random_alpha_h = random.gauss(mean_alpha_h, std_alpha_h)
            random_beta0_left = random.gauss(mean_beta0_left, std_beta0_left)
            random_beta0_right =  random.gauss(mean_beta0_right, std_beta0_right)
            random_beta_G =  random.gauss(mean_beta_G, std_beta_G)
            random_G_0 = random.gauss(mean_G_0, std_G_0)
            random_beta_V = max(0.001, random.gauss(mean_beta_V, std_beta_V))
            random_beta_MLC =  random.gauss(mean_beta_MLC, std_beta_MLC)
            random_sigma = max(0.001, random.gauss(mean_sigma, std_sigma))

            driving_params = np.array([random_alpha_h , random_beta0_left, random_beta0_right, random_beta_G, random_G_0 , random_beta_V, random_beta_MLC , random_sigma], dtype=np.float64)

        else:
            driving_params = np.array([0.08, -3.5, -4.2, 0.2737, 8.69, 0.6808, 87.0, 8.458], dtype=np.float64)
            
        return driving_params
        
    ####################################### Vehicle Generation ###############################################################
    # function to generate vehicles
    
    def generate_agents(output_file="all_trips.trips.xml"):
        all_veh_generation_times={}
        origins=[]
        destinations=[]
        gen_times=[]
        agent_type_by_id={}
        veh_origins_by_id={}
        veh_destinations_by_id={}
        IDM_params_by_id = {}
        PT_params_by_id = {}
        MOBIL_params_by_id ={}
        DDM_params_by_id = {}
        tech_type_by_id = {}
        # now do vehicles first
        for in_idx in range(len(origins_)):
            in_name =  origins_[in_idx]
            out_name = destinations_[in_idx]
            demand = in_flows[in_idx]
            if demand== 0 :
                generation_times=np.array([])
            else:
                generation_times=np.cumsum(np.maximum(np.random.exponential(3600/demand, 100000), min_hw))
                generation_times=np.round(generation_times[(generation_times>=min_t) & (generation_times<=sim_time)],1)
            
            if not in_name in all_veh_generation_times:
                all_veh_generation_times[in_name] = {}
            
            all_veh_generation_times[in_name][out_name] = generation_times
    
    
            for gen_t in generation_times:
                origins.append(in_name)
                destinations.append(out_name)
                gen_times.append(gen_t)
    
        
        origins=np.array(origins)
        destinations=np.array(destinations)
        gen_times=np.array(gen_times) 
        
        sort_indices=np.argsort(gen_times)
        gen_times=gen_times[sort_indices]
        origins=origins[sort_indices]
        destinations=destinations[sort_indices]
    
        # update vehicle id
        agent_ids=np.array([idx+1 for idx in range(len(gen_times))])
        # now write the trip file
        root = ET.Element("routes")

        # Match freeway_sim: SUMO picks insertion lane from `from` that fits the route to `to`.
        _depart_lane = str(user_input_data.get("Depart_Lane", "best")).strip()
        if _depart_lane not in ("best", "random", "free", "allowed", "first"):
            _depart_lane = "best"

        for i in range(len(agent_ids)):

            veh_class = np.random.choice([0, 1, 2, 3, 4], p=[SV_rate, AV_rate, HV_rate, CAV_rate, CAHV_rate])
            agent_type_by_id[str(agent_ids[i])] = veh_class
            tech_type_by_id[str(agent_ids[i])] = (
                "SV" if veh_class == 0 else
                "AV" if veh_class == 1 else
                "HV" if veh_class == 2 else
                "CAV" if veh_class == 3 else
                "CAHV")
              
            veh_origins_by_id[str(agent_ids[i])] = str(origins[i])
            veh_destinations_by_id[str(agent_ids[i])] = str(destinations[i])
            if True: # regardless of the model selected, we save the parameters
                    idm_params = sample_IDM(veh_class)
                    PT_params = sample_PT(veh_class)
                    MOBIL_params = sample_MOBIL()
                    DDM_params = sample_DDM()
                
                    # Clamp aggressive IDM samples (prevents rear-end crashes / unrealistic following)
                    T, a, b, v0, s0, delta = idm_params
                    T = max(float(T), float(IDM_MIN_T))
                    b = max(float(b), float(IDM_MIN_B))
                    s0 = max(float(s0), float(IDM_MIN_S0))
                    idm_params = np.array([T, float(a), b, float(v0), s0, float(delta)], dtype=np.float64)

                    IDM_params_by_id[str(agent_ids[i])] = idm_params
                    PT_params_by_id[str(agent_ids[i])] = PT_params
                    MOBIL_params_by_id[str(agent_ids[i])] = MOBIL_params
                    DDM_params_by_id[str(agent_ids[i])] = DDM_params
                
                    veh_len = veh_lens[veh_class]
                    if veh_len<10:
                        guishape="passenger"
                    else:
                        guishape="truck"

                    veh_type = ET.SubElement(root, "vType", {
                        "id": str(agent_ids[i]),
                        "accel": str(a),
                        "decel": str(b),
                        "length": str(veh_len),
                        "minGap": str(s0),
                        "tau": str(T),
                        "maxSpeed": str(v0),
                        "carFollowModel": "IDM",
                        "guiShape": guishape,
                        "vClass": "truck" if veh_len > 10 else "passenger",
                    })
                    
    
                    # Now create the vehicle/trip element referring to this type
                    trip = ET.SubElement(root, "trip", {
                        "id": str(agent_ids[i]),
                        "type": str(agent_ids[i]) ,
                        "depart": str(gen_times[i]),
                        "from": str(origins[i]),
                        "to": str(destinations[i]),
                        "departLane": _depart_lane,
                        "departSpeed": "max" # need to change this
                    })
            
    
        # Save XML
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        return agent_type_by_id, tech_type_by_id, veh_origins_by_id, veh_destinations_by_id, IDM_params_by_id, PT_params_by_id, MOBIL_params_by_id, DDM_params_by_id
            
    ######## generate agents with paramters
    
    agent_type_by_id, tech_type_by_id, veh_origins_by_id, veh_destinations_by_id, IDM_params_by_id, PT_params_by_id, MOBIL_params_by_id, DDM_params_by_id = generate_agents(output_file=route_file_name)
    
    #print(IDM_params_by_id)
    ################################################### Leader/ Follower Finding Function and Driving Models #######################################
    
    def find_leader(global_pos, lane_idx_, lane_seg, lanes,  global_poses, vs, lengths):
        # lanes is the lanes of all vehicles
        # lane is the 
        if lane_idx_<0:
            return 0, 5, global_pos+500, 50

        #if lane_seg[-2:]=="-1":
           # print(lane_idx_, lane_seg)
        
        if lane_segment_to_lane_list.get(lane_seg) is None:
            return 0, 5, global_pos+500, 50
        # Same SUMO lane only (exclude parallel lanes on the same segment; fixes PT/CF leader).
        lead_inds = np.where((global_poses>global_pos+0.001) & (lanes == lane_seg) )[0]# leader indicator
        
        if len(lead_inds)==0:
            return 0, 5, global_pos+500, 50
        else:
            lead_xs = global_poses[lead_inds]
            lead_vs = vs[lead_inds]
            lead_lens = lengths[lead_inds]
            #find the closest
            nearest_idx = np.argmin(lead_xs)
            return 1, lead_lens[nearest_idx], lead_xs[nearest_idx], lead_vs[nearest_idx]
    
    def find_follower(global_pos, lane_idx_, lane_seg, lanes, global_poses, vs, lengths):
         
        if lane_idx_<0:
            return 0, 5, global_pos+500, 50

        #if lane_seg[-2:]=="-1":
            #print(lane_idx_, lane_seg)
        
        if lane_segment_to_lane_list.get(lane_seg) is None:
            return 0, 5, global_pos-500, 0.1
        follow_inds = np.where((global_poses<global_pos-0.001) & (lanes == lane_seg) )[0]# follower indicator
        
        if len(follow_inds)==0:
            return 0, 5, global_pos-500, 0.1
        else:
            follow_xs = global_poses[follow_inds]
            follow_vs = vs[follow_inds]
            follow_lens = lengths[follow_inds]
            #find the closest
            nearest_idx = np.argmax(follow_xs)
            return 1, follow_lens[nearest_idx], follow_xs[nearest_idx], follow_vs[nearest_idx]
    
    ## LC models
    # define LC models
    
    @njit
    def DDM_LC(DDM_params, direction, is_MLC, adj_gap, adj_lead_v, lead_v ):
        alpha_h, beta_0_left, beta_0_right, beta_G, G_0, beta_V, beta_MLC, sigma =  DDM_params 
        if direction == 1: # left
            beta_0 = beta_0_left
        else:
            beta_0 = beta_0_right
    
        mu = beta_0 + beta_G*np.arctan(adj_gap-G_0) + beta_V*np.arctan(adj_lead_v - lead_v) + beta_MLC*is_MLC
        return mu # drift rate
    
    ## CF models (IDM reference + PT on deviations; same structure as freeway_sim)
    @njit
    def IDM(IDM_params, v, v_leader, x, x_leader, length, length_leader):
        T, a, b, v0, s0, delta = IDM_params
        s_star = s0 + v * T + v * (v - v_leader) / (2 * np.sqrt(a * b))
        gap = x_leader - x - 0.5 * (length + length_leader)
        gap = max(gap, 0.1)
        acc = a * (1 - (v / v0) ** delta - (s_star / gap) ** 2)
        return acc

    @njit
    def PT_relative_IDM(PT_params, IDM_params, v, v_leader, x, x_leader, length, length_leader):
        tau_max, alpha_v, beta_PT, wc, gamma1, gamma2, wm = PT_params
        a_ref = IDM(IDM_params, v, v_leader, x, x_leader, length, length_leader)
        delta_vals = np.linspace(-3.0, 3.0, 25)
        a0 = 1.0
        U_acc_plus = (
            (0.5 * wm + 0.5 * (1 - wm) * (np.tanh(delta_vals / a0) + 1))
            * (delta_vals / a0)
            * (1 + (delta_vals / a0) ** 2) ** (0.5 * gamma1 - 0.5)
        )
        U_acc_minus = (
            (0.5 * wm + 0.5 * (1 - wm) * (np.tanh(delta_vals / a0) + 1))
            * (delta_vals / a0)
            * (1 + (delta_vals / a0) ** 2) ** (0.5 * gamma2 - 0.5)
        )
        U_acc = U_acc_plus * (delta_vals >= 0) + U_acc_minus * (delta_vals < 0)
        delta_v = v - v_leader
        sn = x_leader - x - 0.5 * (length + length_leader)
        den = alpha_v * max(v_leader, 0.1)
        an_vals = a_ref + delta_vals
        col_vars = (delta_v + 0.5 * an_vals * tau_max - sn / tau_max) / den
        p_cld = np.zeros(len(delta_vals))
        for i_c in range(len(col_vars)):
            p_cld[i_c] = 0.5 * (1 + math.erf(col_vars[i_c] / math.sqrt(2)))
        U_tot = U_acc - p_cld * wc
        U_tot = np.minimum(U_tot, 700 / beta_PT)
        choice_weights = np.exp(beta_PT * U_tot)
        ps = (choice_weights + 0.0000001) / sum(choice_weights + 0.0000001)
        acc = sum(ps * an_vals)
        return acc

    @njit
    def _cf_accel(use_pt, pt_or_idm_params, idm_params, v, v_leader, x, x_leader, length, length_leader):
        if use_pt == 1:
            return PT_relative_IDM(pt_or_idm_params, idm_params, v, v_leader, x, x_leader, length, length_leader)
        return IDM(pt_or_idm_params, v, v_leader, x, x_leader, length, length_leader)

    @njit
    def MOBIL_LC(Mobil_params, CF_params, use_pt, idm_params, left_lane_exists, right_lane_exists, MLC_left, MLC_right, self_info,
                 leader_info, follower_info, left_leader_info, left_follower_info, right_leader_info,
                 right_follower_info):
        if left_lane_exists == 0 and right_lane_exists == 0:
            return 0
        politeness, a_thresh, b_safe_disc, b_safe_mand = Mobil_params
        x_self, v_self, length_self = self_info
        x_leader, v_leader, length_leader = leader_info
        x_follower, v_follower, length_follower = follower_info
        x_left_leader, v_left_leader, length_left_leader = left_leader_info
        x_right_leader, v_right_leader, length_right_leader = right_leader_info
        x_left_follower, v_left_follower, length_left_follower = left_follower_info
        x_right_follower, v_right_follower, length_right_follower = right_follower_info

        no_lc_acc_self = _cf_accel(use_pt, CF_params, idm_params, v_self, v_leader, x_self, x_leader, length_self, length_leader)
        no_lc_acc_follower = _cf_accel(use_pt, CF_params, idm_params, v_follower, v_self, x_follower, x_self, length_follower, length_self)

        lc_left_acc_self = -100.0
        lc_left_acc_follower = -100.0
        lc_left_acc_left_follower = -100.0

        if left_lane_exists:
            lc_left_acc_self = _cf_accel(use_pt, CF_params, idm_params, v_self, v_left_leader, x_self, x_left_leader, length_self,
                                        length_left_leader)
            lc_left_acc_follower = _cf_accel(use_pt, CF_params, idm_params, v_follower, v_leader, x_follower, x_leader, length_follower,
                                            length_leader)
            lc_left_acc_left_follower = _cf_accel(use_pt, CF_params, idm_params, v_left_follower, v_self, x_left_follower, x_self,
                                                 length_left_follower, length_self)

        lc_right_acc_self = -100.0
        lc_right_acc_follower = -100.0
        lc_right_acc_right_follower = -100.0

        if right_lane_exists:
            lc_right_acc_self = _cf_accel(use_pt, CF_params, idm_params, v_self, v_right_leader, x_self, x_right_leader, length_self,
                                         length_right_leader)
            lc_right_acc_follower = lc_left_acc_follower
            lc_right_acc_right_follower = _cf_accel(use_pt, CF_params, idm_params, v_right_follower, v_self, x_right_follower, x_self,
                                                   length_right_follower, length_self)

        b_thresh_left = b_safe_mand if MLC_left == 1 else b_safe_disc
        left_safe = left_lane_exists and (lc_left_acc_left_follower > -b_thresh_left) and (
                        lc_left_acc_self > -b_thresh_left)

        b_thresh_right = b_safe_mand if MLC_right == 1 else b_safe_disc
        right_safe = right_lane_exists and (lc_right_acc_right_follower > -b_thresh_right) and (
                        lc_right_acc_self > -b_thresh_right)

        if MLC_left == 1:
            return 1 if left_safe else 0
        if MLC_right == 1:
            return -1 if right_safe else 0

        no_lc_acc_left_follower = _cf_accel(use_pt, CF_params, idm_params, v_left_follower, v_left_leader, x_left_follower, x_left_leader,
                                           length_left_follower, length_left_leader)
        no_lc_acc_right_follower = _cf_accel(use_pt, CF_params, idm_params, v_right_follower, v_right_leader, x_right_follower,
                                            x_right_leader, length_right_follower, length_right_leader)

        left_incentive = -999.0
        right_incentive = -999.0

        if left_lane_exists:
            left_incentive = (lc_left_acc_self - no_lc_acc_self) + politeness * (
                        lc_left_acc_follower - no_lc_acc_follower + lc_left_acc_left_follower - no_lc_acc_left_follower)

        if right_lane_exists:
            right_incentive = (lc_right_acc_self - no_lc_acc_self) + politeness * (
                        lc_right_acc_follower - no_lc_acc_follower + lc_right_acc_right_follower - no_lc_acc_right_follower)

        if left_safe and right_safe:
            if left_incentive > right_incentive and left_incentive > a_thresh:
                return 1
            elif right_incentive > left_incentive and right_incentive > a_thresh:
                return -1
        elif left_safe and left_incentive > a_thresh:
            return 1
        elif right_safe and right_incentive > a_thresh:
            return -1

        return 0

    acc_max=5
    acc_min=-10
    acc_step_size = 0.1

    ################################################# Run simulation Function ################################################
    
    # Parse XML
    #print(lane_lengths)
    #print("P_adjust", pos_adjust_by_lanes)
    tree = ET.parse(config_file)
    root = tree.getroot()
    veh_is_initialized={}
    all_evidences_left = {}
    all_evidences_right = {}
    for vid in veh_origins_by_id:
        veh_is_initialized[vid] = 0
        all_evidences_left[vid] = []
        all_evidences_right[vid] = []
        
    # Change net-file
    for inp in root.findall("input"):
        net = inp.find("net-file")
        net.set("value", net_file_name)
        route = inp.find("route-files")
        route.set("value", route_file_name)
    # Change time values
    time_elem = root.find("time")
    if time_elem is not None:
        end = time_elem.find("end")
        end.set("value", str(sim_time))  # 2 hours
        step = time_elem.find("step-length")
        step.set("value", str(t_step))
    # Save back to file (overwrite)
    tree.write(config_file)
    
    collect_data = bool(user_input_data.get("Sim_DataCollection", True))
    collect_every_steps = 1
    if collect_data:
        sample_freq = float(user_input_data.get("Sim_DataFreq", t_step))
        collect_every_steps = max(1, int(round(sample_freq / t_step))) if t_step > 0 else 1
    all_data = {"time":[], "id":[], "type":[], "x":[], "y":[], "v":[], "theta":[], "road":[], "length":[] , "lane":[], "lane_pos":[] } if collect_data else None
    
    if sim_visualization==True: # whether user chooses visualization or not 
        sumo_binary = checkBinary('sumo-gui')
    else:
        sumo_binary = checkBinary('sumo')

    traci.start([
        sumo_binary,
        "-c", config_file,
        "--lanechange.duration", "2",
    ])

    # ---- Communication config (already in your file) ----
    comm_cfg = user_input_data.get("Comm_Params", {"Range": 30.0, "Lookahead": 5, "Latency": 0, "Loss": 0.0})
    if isinstance(comm_cfg, dict):
        comm_cfg = {
            "Range": comm_cfg.get("Range", 30.0),
            "Lookahead": comm_cfg.get("Lookahead", 5),
            "Latency": comm_cfg.get("Latency", 0),
            "Loss": comm_cfg.get("Loss", 0.0),
        }

    # ✅ IMPORTANT: bus call must end here (closed parentheses)
    bus = ns.CommunicationBus(
        range_m=float(comm_cfg["Range"]),
        m_max=int(comm_cfg["Lookahead"]),
        latency_steps=int(comm_cfg["Latency"]),
        loss_rate=float(comm_cfg["Loss"]),
        connected_tech=("CAV", "CAHV"),
    )

    # ---- Colors (your freeway-style map) ----
    COLOR_MAP = {
        # Requested palette:
        # - CAV & CAHV: blue
        # - AV: green
        # - SV & HV: gray
        "SV": (160, 160, 160, 255),
        "HV": (160, 160, 160, 255),
        "AV": (0, 180, 0, 255),
        "CAV": (0, 102, 204, 255),
        "CAHV": (0, 102, 204, 255),
    }
    color_set = {}

    t = 0
    # need to initialize evidence for ddm
    
    if LC_model_name == "DDM":
        A_left_by_id={}
        A_right_by_id={}
    
    last_speed = {}
    last_acc = {}
    speed_mode_set = {}
    #Fix Progress Bar
    last_progress = -1
    step_count = 0
    progress_every = max(1, int(round(1.0 / t_step)))  # update about once per simulated second
    lc_every_steps = max(1, int(user_input_data.get("LC_Every_Steps", 10)))
    # Effective dt between LC decisions (used inside DDM SDE since evidence updates every N steps now)
    lc_dt = lc_every_steps * t_step

    def next_intersection_distance(road, x, y):
        """Distance (m) to the next intersection along the current approach."""
        if road == "EB01":
            return inter1_x - x
        if road == "EB12":
            return inter2_x - x
        if road == "EB23":
            return inter3_x - x
        if road == "WB34":
            return x - inter3_x
        if road == "WB23":
            return x - inter2_x
        if road == "WB12":
            return x - inter1_x
        if road.startswith("NB"):
            return 0.0 - y
        if road.startswith("SB"):
            return y - 0.0
        return 1e9

    def get_next_route_edge(vid):
        """Next normal edge on the vehicle route after the current road (not final trip destination)."""
        try:
            r = traci.vehicle.getRoute(vid)
            ce = traci.vehicle.getRoadID(vid)
        except traci.TraCIException:
            return None
        if str(ce).startswith(":"):
            return None
        try:
            idx = list(r).index(ce)
        except ValueError:
            return None
        if idx + 1 < len(r):
            return str(r[idx + 1])
        return None

    def required_lane_for_next_move(road, next_edge):
        """
        Dedicated lane index for the upcoming turn at the next intersection, or None if any lane is valid
        (e.g. straight-through on EB/WB continuation).
        """
        if next_edge is None:
            return None
        ne = str(next_edge)
        if road in ("EB01", "EB12", "EB23"):
            k = road[-1]
            if ne == f"NB{k}_North":
                return num_EW_lanes - 1
            if ne == f"SB{k}_South":
                return 0
            return None
        if road in ("WB12", "WB23", "WB34"):
            k = road[-2]
            if ne == f"NB{k}_North":
                return 0
            if ne == f"SB{k}_South":
                return num_EW_lanes - 1
            return None
        if road.startswith("NB"):
            if ne.startswith("WB"):
                return num_NS_lanes - 1
            if ne.startswith("EB"):
                return 0
            return None
        if road.startswith("SB"):
            if ne.startswith("EB"):
                return num_NS_lanes - 1
            if ne.startswith("WB"):
                return 0
            return None
        return None

    veh_subscribed_ids = set()
    veh_sub_vars = (
        tc.VAR_POSITION,
        tc.VAR_SPEED,
        tc.VAR_ANGLE,
        tc.VAR_ROAD_ID,
        tc.VAR_LANE_ID,
        tc.VAR_LANEPOSITION,
        tc.VAR_LENGTH,
        tc.VAR_TYPE,
        tc.VAR_LANE_INDEX,
    )

    while t < sim_time:
        traci.simulationStep()
        # (optional) remove this line, it slows things down:
        # print("time:", t, traci.simulation.getTime())

        t += t_step
        t = np.round(t, 1)

        # --- progress update ---
        step_count += 1
        do_lc_step = (step_count % lc_every_steps == 0)
        if progress_cb is not None and (step_count % progress_every == 0 or t >= sim_time):
            pct = int(min(99, max(0, (t / sim_time) * 100)))
            if pct != last_progress:
                progress_cb(pct)
                last_progress = pct
        
        # save first just for computation purpose
        all_ids = traci.vehicle.getIDList()
        all_id_set = set(all_ids)
        new_ids = all_id_set.difference(veh_subscribed_ids)
        for _vid in new_ids:
            try:
                traci.vehicle.subscribe(_vid, veh_sub_vars)
                veh_subscribed_ids.add(_vid)
            except Exception:
                pass
        veh_subscribed_ids.intersection_update(all_id_set)
        
        ids = np.zeros(len(all_ids)).astype(str)
        types = np.zeros(len(all_ids))
        xs= np.zeros(len(all_ids))
        ys= np.zeros(len(all_ids))
        vs= np.zeros(len(all_ids))
        thetas= np.zeros(len(all_ids))
        roads= np.zeros(len(all_ids)).astype(str)
        lanes= np.zeros(len(all_ids)).astype(str)
        lengths= np.zeros(len(all_ids))
        lane_poses=np.zeros(len(all_ids))
        global_poses=np.zeros(len(all_ids)) # map the pos within lane to a global frame
        lane_indexes = np.zeros(len(all_ids))  # from the lane name to lane_id
    
        
        for i_id in range(len(all_ids)):
            vid = all_ids[i_id]
            sub = traci.vehicle.getSubscriptionResults(vid) or {}
            # ---- Color once when vehicle is first seen ----
            if vid not in color_set:
                tech_lbl = tech_type_by_id.get(str(vid), "SV")
                traci.vehicle.setColor(vid, COLOR_MAP.get(tech_lbl, (255, 255, 255, 255)))
                color_set[vid] = True
            # Optional: enforce SUMO safety checks even under setSpeed (OFF by default).
            if SET_SPEEDMODE_SAFETY and vid not in speed_mode_set:
                try:
                    traci.vehicle.setSpeedMode(vid, 31)
                except Exception:
                    pass
                speed_mode_set[vid] = True
                
            if veh_origins_by_id[vid] == 0:
                #traci.vehicle.setLaneChangeMode(vid, 256)
                #traci.vehicle.setSpeedMode(vid, 32)
                veh_origins_by_id[vid] = 1
            veh_origins_by_id[vid]=1
            x, y = sub.get(tc.VAR_POSITION, traci.vehicle.getPosition(vid))
            speed = float(sub.get(tc.VAR_SPEED, traci.vehicle.getSpeed(vid)))
            prev_v = last_speed.get(vid, speed)
            a_est = (speed - prev_v) / (t_step if t_step > 0 else 1.0)
            last_speed[vid] = speed
            last_acc[vid] = a_est
            edge = sub.get(tc.VAR_ROAD_ID, traci.vehicle.getRoadID(vid))
            lane = sub.get(tc.VAR_LANE_ID, traci.vehicle.getLaneID(vid))
            lane_pos = float(sub.get(tc.VAR_LANEPOSITION, traci.vehicle.getLanePosition(vid)))
            vtype = sub.get(tc.VAR_TYPE, traci.vehicle.getTypeID(vid))
            length = float(sub.get(tc.VAR_LENGTH, traci.vehicle.getLength(vid)))
            theta = float(sub.get(tc.VAR_ANGLE, traci.vehicle.getAngle(vid)))
            
            if not lane in pos_adjust_by_lanes: # may be slow
                global_pos =  lane_pos
            else:
                global_pos = lane_pos + pos_adjust_by_lanes[lane] 
            # for those within the intersection turning return itself (treating turning lanes as its own kind)
            #lane_idx = lane_segment_to_idx[lane]
            lane_idx = int(sub.get(tc.VAR_LANE_INDEX, int(lane[-1])))
            
            
            #print(vid, x, y, speed, edge, lane, lane_pos, vtype, length)
            #vtype = traci.vehicle.getTypeID(vid)  # or map to 0/1 if needed
        
            # Then append to lists
            ids[i_id]=(vid)
            types[i_id]=(int(agent_type_by_id[str(vid)]))
            xs[i_id]=(x)
            ys[i_id]=(y)
            vs[i_id]=(speed)
            thetas[i_id]=(theta)
            roads[i_id]=(edge)
            lanes[i_id]=(lane)
            lengths[i_id]=(length)
            lane_poses[i_id]=(lane_pos)
            global_poses[i_id] = global_pos
            lane_indexes[i_id] = lane_idx
        speed_by_vid = {str(all_ids[i]): float(vs[i]) for i in range(len(all_ids))}
        lane_index_by_vid = {str(all_ids[i]): int(lane_indexes[i]) for i in range(len(all_ids))}

        # -------------------- ADD THIS BLOCK HERE (between the two loops) --------------------
        states_conn = {}
        for i, vid in enumerate(all_ids):
            tech = tech_type_by_id.get(str(vid), "SV")
            if tech not in ("CAV", "CAHV"):
                continue
            states_conn[str(vid)] = ns.VehicleState(
                vid=str(vid),
                lane=str(lanes[i]),
                global_pos=float(global_poses[i]),
                v=float(vs[i]),
                a=float(last_acc.get(str(vid), 0.0)),
                length=float(lengths[i]),
                tech=str(tech),
            )

        bus.step(states_conn)
        # -----------------------------------------------------------------------------------

        for i_id in range(len(all_ids)):
            # IDM
            # now do ddm
            vid = all_ids[i_id]
            dest = veh_destinations_by_id[vid]
            x = xs[i_id]
            y = ys[i_id]
            v = vs[i_id]
            lane_idx = lane_indexes[i_id]
            global_pos = global_poses[i_id]
            lane =  str(lanes[i_id])
            length = lengths[i_id]
            road = roads[i_id]
            old_lane_idx = lane_index_by_vid.get(str(vid))
            if old_lane_idx is None:
                old_lane_idx = traci.vehicle.getLaneIndex(vid)

            d2i = next_intersection_distance(road, x, y)

            # Internal junction connectors: SUMO handles trajectories
            if str(lane).startswith(":"):
                continue

            # Near the next intersection: full SUMO for CF + LC (no IDM/PT/MOBIL/DDM overrides).
            if road.startswith("NB") or road.startswith("SB"):
                _sumo_near_inter = abs(d2i) < HANDOFF_DIST_M
            else:
                # EW: d2i < 0 means nose past the intersection reference — still SUMO (queued / past stop line)
                _sumo_near_inter = d2i < HANDOFF_DIST_M
            if _sumo_near_inter:
                continue

            JUNCTION_CORE = 8.0  # meters (small). Tune 3-10.
            # If vehicle is extremely close to the intersection center, let SUMO clear it
            near_core = (
                (abs(x - inter1_x) < JUNCTION_CORE)
                or (abs(x - inter2_x) < JUNCTION_CORE)
                or (abs(x - inter3_x) < JUNCTION_CORE)
            ) and (abs(y) < JUNCTION_CORE)
            if near_core:
                continue

            SIGNAL_ZONE = 20.0  # meters; tune 50–120 depending on approach length

            tls_list = traci.vehicle.getNextTLS(vid)  # [(tlsID, linkIndex, dist, state), ...]
            if tls_list:
                tls_id, tls_link, tls_dist, tls_state = tls_list[0]

                # If close to the signal, let SUMO handle speed / stopping / starting
                if tls_dist < SIGNAL_ZONE:
                    traci.vehicle.setSpeed(vid, -1)  # IMPORTANT: give back speed control to SUMO
                    continue

            # --- Next movement from route (not final destination) → correct turn lane before the intersection ---
            next_e = get_next_route_edge(vid)
            req_lane = required_lane_for_next_move(road, next_e)
            # Outside SUMO handoff: nudge into dedicated left/right turn lane; MOBIL/DDM disabled for that lateral control
            if do_lc_step and req_lane is not None and old_lane_idx != req_lane:
                traci.vehicle.changeLane(vid, int(req_lane), TRACI_CHANGE_LANE_DURATION_S)
            use_mobil_lc = req_lane is None

            # find route attribute
            left_lane_exists = 0
            right_lane_exists = 0
            MLC_left = 0 
            MLC_right = 0

            # MLC for EW (use next route edge so multi-leg routes still get correct mandatory LC at each intersection)
            if road=="EB01" or road=="EB12" or road=="EB23":
                if next_e == "NB"+road[-1]+"_North" and int(lane[-1])<num_EW_lanes-1 :
                    MLC_left = 1
                elif next_e == "SB"+road[-1]+"_South" and int(lane[-1])>0 :
                    MLC_right = 1
            elif road=="WB12" or road=="WB23" or road=="WB34":
                if next_e == "NB"+road[-2]+"_North" and int(lane[-1])>0 :
                    MLC_right= 1
                elif next_e == "SB"+road[-2]+"_South" and int(lane[-1])<num_EW_lanes-1:
                    MLC_left = 1
            # MLC for NS
            elif road[0:2]== "NB" : 
                if next_e is not None and str(next_e).startswith("EB") and int(lane[-1])>0:
                    MLC_right = 1
                elif next_e is not None and str(next_e).startswith("WB") and int(lane[-1])<num_NS_lanes-1:
                    MLC_left = 1
            elif road[0:2]== "SB": 
                if next_e is not None and str(next_e).startswith("EB") and int(lane[-1])<num_NS_lanes-1:
                    MLC_left = 1
                elif next_e is not None and str(next_e).startswith("WB") and int(lane[-1])>0:
                    MLC_right = 1
            

            if int(lane[-1])>=1:
                right_lane_exists = 1
            if "NB" in lane or "SB" in lane:
                if int(lane[-1])<num_NS_lanes-1 and MLC_right == 0:
                    left_lane_exists = 1
            elif "EB" in lane or "WB" in lane:
                if int(lane[-1])<num_EW_lanes-1 and MLC_right == 0:
                    left_lane_exists = 1

            # Bounds for TraCI changeLane — subscription lane index vs. lane-ID suffix can diverge briefly;
            # never request lane index outside [0, n_lanes-1] (SUMO raises on -1).
            try:
                n_lanes_here = max(1, int(traci.edge.getLaneNumber(road)))
            except Exception:
                n_lanes_here = max(
                    1,
                    int(num_NS_lanes) if road.startswith(("NB", "SB")) else int(num_EW_lanes),
                )

            # the name of the right lane seg 
            left_lane = lane[:-1]+str(int(lane[-1])+1)
            right_lane = lane[:-1]+str(int(lane[-1])-1)

            # find_leader(global_pos, lane_idx, lane_seg, lanes,  global_poses, vs, lengths)
    
            leader_exists, leader_len, leader_global_x, leader_v = find_leader(global_pos, lane_idx, lane, lanes, global_poses, vs, lengths)
            follower_exists, follower_len, follower_global_x, follower_v = find_follower(global_pos, lane_idx, lane, lanes, global_poses, vs, lengths)
            
            left_leader_exists, left_leader_len, left_leader_global_x, left_leader_v = find_leader(global_pos, lane_idx+1, left_lane, lanes, global_poses, vs, lengths)
            left_follower_exists, left_follower_len, left_follower_global_x, left_follower_v = find_follower(global_pos, lane_idx+1, left_lane, lanes, global_poses, vs, lengths)
            right_leader_exists, right_leader_len, right_leader_global_x, right_leader_v = find_leader(global_pos, lane_idx-1, right_lane, lanes, global_poses, vs, lengths)
            right_follower_exists, right_follower_len, right_follower_global_x, right_follower_v = find_follower(global_pos, lane_idx-1, right_lane, lanes, global_poses, vs, lengths)

            # ===================== CAV COOPERATIVE CONTROL (ADD THIS) =====================
            tech_label = tech_type_by_id.get(str(vid), "SV")

            # If we're on an internal lane (":"), do not override SUMO control
            if str(lane).startswith(":"):
                tech_label = "SV"

            if tech_label in ("CAV", "CAHV"):
                # Build ego/leader states
                ego_state = states_conn.get(
                    str(vid),
                    ns.VehicleState(
                        vid=str(vid),
                        lane=str(lane),
                        global_pos=float(global_pos),
                        v=float(v),
                        a=float(last_acc.get(str(vid), 0.0)),
                        length=float(length),
                        tech=str(tech_label),
                    ),
                )

                leader_state = ns.VehicleState(
                    vid="leader",
                    lane=ego_state.lane,
                    global_pos=float(leader_global_x),
                    v=float(leader_v),
                    a=0.0,
                    length=float(leader_len),
                    tech="SV",
                )

                # Connected preceding vehicles in the same lane (from bus)
                pcs = bus.get_preceding_connected(ego_state)

                # Use IDM params as base for C-IDM (same as freeway)
                CF_params = IDM_params_by_id[str(vid)]

                cidm_cfg = user_input_data.get("CIDM_Params", {"K_v": 0.2, "K_a": 0.05, "s_ref": 50.0})
                acc = coop.c_idm_accel(
                    CF_params, ego_state, leader_state, pcs, bus,
                    K_v=float(cidm_cfg.get("K_v", 0.2)),
                    K_a=float(cidm_cfg.get("K_a", 0.05)),
                    s_ref=float(cidm_cfg.get("s_ref", 50.0)),
                )

                # --- SAFETY CLAMP (prevents unrealistic braking/negative speeds) ---
                new_v = max(0.0, float(v) + float(acc) * float(t_step))
                # --- Anti rear-end safety cap using TraCI leader ---
                if ENABLE_SAFETY_SPEED_CAP and float(t_step) > 0:
                    try:
                        leader_info = traci.vehicle.getLeader(vid, 200.0)
                    except Exception:
                        leader_info = None
                    if leader_info is not None:
                        leader_id, gap_bb = leader_info
                        try:
                            leader_speed = float(speed_by_vid.get(str(leader_id), traci.vehicle.getSpeed(leader_id)))
                        except Exception:
                            leader_speed = float(leader_v)
                        try:
                            s0_self = float(IDM_params_by_id[str(vid)][4])
                        except Exception:
                            s0_self = 2.0
                        min_gap_bb = max(0.5, s0_self + float(SAFETY_EXTRA_GAP))
                        v_cap = leader_speed + (float(gap_bb) - min_gap_bb) / float(t_step)
                        new_v = float(min(new_v, max(0.0, v_cap)))
                traci.vehicle.setSpeed(vid, new_v)

                # C-MOBIL only when no dedicated turn lane is imposed (use_mobil_lc); turns use TraCI above.
                # Expensive decision gated to LC steps.
                if do_lc_step and use_mobil_lc:
                    cm_cfg = user_input_data.get("CMOBIL_Params", {"kappa": 0.1, "gamma": 1.0})
                    kappa = float(cm_cfg.get("kappa", 0.1))
                    gamma = float(cm_cfg.get("gamma", 1.0))

                    Mobil_params = MOBIL_params_by_id[str(vid)]

                    lane_choice = coop.c_mobil_decision(
                        Mobil_params, CF_params,
                        ego_state, leader_state,
                        ns.VehicleState("f", ego_state.lane, float(follower_global_x), float(follower_v), 0.0,
                                        float(follower_len), "SV"),
                        ns.VehicleState("ll", str(left_lane), float(left_leader_global_x), float(left_leader_v), 0.0,
                                        float(left_leader_len), "SV"),
                        ns.VehicleState("lf", str(left_lane), float(left_follower_global_x), float(left_follower_v),
                                        0.0, float(left_follower_len), "SV"),
                        ns.VehicleState("rl", str(right_lane), float(right_leader_global_x), float(right_leader_v), 0.0,
                                        float(right_leader_len), "SV"),
                        ns.VehicleState("rf", str(right_lane), float(right_follower_global_x), float(right_follower_v),
                                        0.0, float(right_follower_len), "SV"),
                        left_lane_exists, right_lane_exists, MLC_left, MLC_right,
                        bus=bus, kappa=kappa, gamma_lc=gamma,
                        left_lane_str=str(left_lane), right_lane_str=str(right_lane),
                    )

                    if lane_choice != 0:
                        _tgt_lc = int(old_lane_idx) + int(lane_choice)
                        if 0 <= _tgt_lc < n_lanes_here:
                            traci.vehicle.changeLane(vid, _tgt_lc, TRACI_CHANGE_LANE_DURATION_S)

                # IMPORTANT: we handled this vehicle; skip normal IDM/PT + MOBIL/DDM below
                continue
            # ============================================================================

            # first do car following, then do lane changing
            
            if CF_model_name == "IDM":
                CF_params = IDM_params_by_id[str(vid)]
                acc = IDM(CF_params, v, leader_v, global_pos, leader_global_x, length , leader_len)
                
            elif CF_model_name == "PT":
                CF_params = PT_params_by_id[str(vid)]
                acc = PT_relative_IDM(
                    CF_params,
                    IDM_params_by_id[str(vid)],
                    v,
                    leader_v,
                    global_pos,
                    leader_global_x,
                    length,
                    leader_len,
                )
                if PT_STOCHASTIC_ENABLE and PT_STOCHASTIC_SIGMA > 0.0:
                    acc = acc + float(np.random.normal(0.0, PT_STOCHASTIC_SIGMA))

            # --- SAFETY CLAMP (prevents unrealistic braking/negative speeds) ---
            new_v = max(0.0, float(v) + float(acc) * float(t_step))
            # --- Anti rear-end safety cap using TraCI leader (more robust than our global leader mapping) ---
            if ENABLE_SAFETY_SPEED_CAP and float(t_step) > 0:
                try:
                    leader_info = traci.vehicle.getLeader(vid, 200.0)  # (leaderID, gap in m) or None
                except Exception:
                    leader_info = None
                if leader_info is not None:
                    leader_id, gap_bb = leader_info
                    try:
                        leader_speed = float(speed_by_vid.get(str(leader_id), traci.vehicle.getSpeed(leader_id)))
                    except Exception:
                        leader_speed = float(leader_v)
                    try:
                        s0_self = float(IDM_params_by_id[str(vid)][4])
                    except Exception:
                        s0_self = 2.0
                    min_gap_bb = max(0.5, s0_self + float(SAFETY_EXTRA_GAP))
                    v_cap = leader_speed + (float(gap_bb) - min_gap_bb) / float(t_step)
                    new_v = float(min(new_v, max(0.0, v_cap)))
            traci.vehicle.setSpeed(vid, new_v)

            # Non-CAV LC: gate the whole decision (MOBIL_LC / DDM_LC) to LC steps only.
            if do_lc_step and use_mobil_lc and LC_model_name == "MOBIL":
                self_info = np.array([global_pos, v, length])
                leader_info = np.array([leader_global_x, leader_v, leader_len] )
                follower_info = np.array([follower_global_x, follower_v, follower_len] )
                left_leader_info = np.array([left_leader_global_x, left_leader_v, left_leader_len] )
                left_follower_info = np.array([left_follower_global_x, left_follower_v, left_follower_len] )
                right_leader_info = np.array([right_leader_global_x, right_leader_v, right_leader_len] )
                right_follower_info = np.array([right_follower_global_x, right_follower_v, right_follower_len] )

                Mobil_params = MOBIL_params_by_id[str(vid)]
                _idm_m = IDM_params_by_id[str(vid)]
                _use_pt = 1 if CF_model_name == "PT" else 0
                _cf_m = PT_params_by_id[str(vid)] if CF_model_name == "PT" else _idm_m

                lane_choice = MOBIL_LC(Mobil_params, _cf_m, _use_pt, _idm_m, left_lane_exists, right_lane_exists , MLC_left, MLC_right, self_info, leader_info, follower_info, left_leader_info, left_follower_info, right_leader_info, right_follower_info )

                _tgt_m = int(old_lane_idx) + int(lane_choice)
                if _tgt_m >= 0 and _tgt_m < n_lanes_here:
                    traci.vehicle.changeLane(vid, _tgt_m, 1.0)

            elif do_lc_step and use_mobil_lc and LC_model_name == "DDM":
                hw =  min(20, (leader_global_x-global_pos)/(v+0.01) ) # avoid too low initial evidence for now

                DDM_params = DDM_params_by_id[str(vid)]
                alpha_h = DDM_params[0]
                sigma = DDM_params[-1]
                if not vid in A_left_by_id:
                    A_left_by_id[vid] = 10 - alpha_h*hw # initial evidence
                if not vid in A_right_by_id:
                    A_right_by_id[vid] = 10 - alpha_h*hw
                all_evidences_left[vid].append(A_left_by_id[vid] )
                all_evidences_right[vid].append(A_right_by_id[vid] )

                # SDE integrates over lc_dt = lc_every_steps * t_step since evidence now updates every N steps.
                if left_lane_exists:
                    left_follow_gap = global_pos - left_follower_global_x - length
                    left_mu = DDM_LC(DDM_params, 1, MLC_left, left_follow_gap, left_leader_v, leader_v )
                    A_left_by_id[vid]  = A_left_by_id[vid]  + left_mu*lc_dt + np.random.normal()*sigma*np.sqrt(lc_dt)

                if right_lane_exists: # as set above, if it is mlc to one direction, the other direction not considered
                    right_follow_gap = global_pos - right_follower_global_x - length
                    right_mu = DDM_LC(DDM_params, -1, MLC_right, right_follow_gap, right_leader_v, leader_v )
                    A_right_by_id[vid]  = A_right_by_id[vid]  + right_mu*lc_dt + np.random.normal()*sigma*np.sqrt(lc_dt)

                if A_right_by_id[vid]*right_lane_exists > A_left_by_id[vid]*left_lane_exists and A_right_by_id[vid]>20:
                    if right_follow_gap>min_gap and right_leader_global_x - global_pos - right_leader_len >min_gap:
                        _tgt_r = int(old_lane_idx) - 1
                        if _tgt_r >= 0:
                            traci.vehicle.changeLane(vid, _tgt_r, 0)
                            del A_right_by_id[vid]

                elif  A_right_by_id[vid]*right_lane_exists < A_left_by_id[vid]*left_lane_exists and A_left_by_id[vid]>20:
                    if left_follow_gap>min_gap and left_leader_global_x - global_pos - left_leader_len >min_gap:
                        _tgt_l = int(old_lane_idx) + 1
                        if _tgt_l < n_lanes_here:
                            traci.vehicle.changeLane(vid, _tgt_l, 0)
                            del A_left_by_id[vid]
            
        if collect_data and (step_count % collect_every_steps == 0):
            all_data["time"].append(np.ones(len(ids)) * t)  # extend flattens the array into the list
            all_data["id"].append(ids)
            all_data["type"].append(types)
            all_data["x"].append(xs)
            all_data["y"].append(ys)
            all_data["v"].append(vs)
            all_data["theta"].append(thetas)
            all_data["length"].append(lengths)
            all_data["road"].append(roads)
            all_data["lane"].append(lanes)
            all_data["lane_pos"].append(lane_poses)
            
    
    if collect_data:
        for key in all_data:
            all_data[key] = np.concatenate(all_data[key])
        all_data = pd.DataFrame(all_data)
    else:
        all_data = pd.DataFrame(columns=["time", "id", "type", "x", "y", "v", "theta", "road", "length", "lane", "lane_pos"])

    traci.close()
    
    return all_data
    
