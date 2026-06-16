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
from numba import njit, prange
import subprocess
import random
from sklearn.cluster import KMeans
from . import signal_control
from . import ns
from . import coop_models as coop
from .paths import (
    MODEL_PARAMS_DIR,
    MODELS_DIR,
    TEMPLATES_DIR,
    load_mobil_results_csv,
    mobil_params_from_csv_row,
)

# Align with arterial (multi_inter_sim): TraCI changeLane duration and approach handoff distance.
# SUMO TraCI: traci.vehicle.changeLane(..., duration_s); not LC_NO_RETURN (see freeway_sim).
TRACI_CHANGE_LANE_DURATION_S = 3.0
HANDOFF_DIST_M = 80.0


def tail_sumo_error(path="sumo.err.log", n=40):
    try:
        with open(path, "r", errors="ignore") as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return "(no sumo.err.log found)"


if __name__ == "__main__":
    sumo_cmd = [
        "sumo",
        "-c", "your.sumocfg",
        "--log", "sumo.log",
        "--error-log", "sumo.err.log",
        "--message-log", "sumo.msg.log",
        "--verbose",
    ]

    subprocess.Popen(
        sumo_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

#traci.init(8813)  # or traci.start(sumo_cmd) if not using remote-port


def run_sim_single_inter(user_input_data, progress_cb=None, is_running_check=None):

    ################################################## First Geometry Construction ###########################################################
    
    num_NS_lanes = user_input_data["Geometry"]['Num_Lanes_NS']
    num_EW_lanes = user_input_data["Geometry"]['Num_Lanes_EW']
    intersect_control = user_input_data["Geometry"]['Intersection_Control']
    sidewalk_width = user_input_data["Geometry"]['Walkway_Width']# 8m sidewalk the old one

    # these are immutable
    #num_inters = 3
    
    #main_node_names = ["West_Most"] # the nodes from west-most to east-most
    #for i in range(num_inters):
        #main_node_names.append("Intersection"+str(i+1))
    #main_node_names.append("East_Most")
    
    # file names 
    # template file
    # Pedram: Replace the template file location
    if intersect_control  == "Signal":
        single_inter_template_file = str(
            TEMPLATES_DIR / f"single_intersection_EW{num_EW_lanes}_NS{num_NS_lanes}_template.net.xml"
        )
    else:
        single_inter_template_file = str(
            TEMPLATES_DIR / f"single_intersection_EW{num_EW_lanes}_NS{num_NS_lanes}_stop_sign_template.net.xml"
        )
     # function to write the new arterial file 
    new_file_name = str(MODELS_DIR / "single_intersection.net.xml")  # the adjusted  network file
    net_file_name = new_file_name # occasionally this variable is used to
    ###### net file names in simulation
    route_file_name = str(MODELS_DIR / "all_trips.trips.xml")  # trip file will be saved here
    

    config_file = str(MODELS_DIR / "single_intersection.sumocfg")  # config file, will set the  route_file_name and new_file_name in the simulation loop

    # adjust the length of legs based on user input
    segment_length =  user_input_data["Geometry"]["Road_Length"]
    East_L = segment_length
    West_L = segment_length
    North_L = segment_length
    South_L = segment_length
    
    new_junction_xs = {
    "East_in":  East_L, 
    "West_in": -West_L, 
    "North_in": 0, 
    "South_in": 0, 
    "East_out":  East_L, 
    "West_out": -West_L, 
    "North_out": 0, 
    "South_out": 0, 
    "Intersection": 0
    }

    new_junction_ys = {
        "East_in":  0, 
        "West_in":  0, 
        "North_in": North_L, 
        "South_in": -South_L, 
        "East_out":  0, 
        "West_out":  0, 
        "North_out": North_L, 
        "South_out": -South_L, 
        "Intersection": 0
    }
    
    

    def construct_geometry(new_file_name, sidewalk_width, intersection_control):
        """
        sidewalk_width: float (e.g., 1.5)
        intersection_control: str ("signal" or "all_way_stop")
        """
        tree = ET.parse(single_inter_template_file)
        root = tree.getroot()
        
        # Map your UI labels to SUMO junction types
        #control_map = {
        #    "signal": "traffic_light",
        #    "all_way_stop": "allway_stop"
        #}
        #sumo_control_type = control_map.get(intersection_control, "traffic_light")
        
        # 1. Update Junctions (Location and Control Type)
        for junction in root.findall("junction"):
            junction_id = junction.get("id")
            
            # Reset shape so netconvert recalculates it
            junction.set("shape", "")
            
            # Update Control Type (Only for the main intersection, usually named 'Intersection')
            #if junction_id == "Intersection":
                #junction.set("type", sumo_control_type)
            
            # Update Coordinates
            if junction_id in new_junction_xs:
                junction.set("x", str(np.round(new_junction_xs[junction_id], 2)))
                junction.set("y", str(np.round(new_junction_ys[junction_id], 2)))     
        
        # 2. Update Sidewalk Width
        # We look for lanes that allow pedestrians and update their width attribute
        for edge in root.findall("edge"):
            # Reset edge geometry for netconvert
            if "length" in edge.attrib:
                del edge.attrib["length"]
            if "shape" in edge.attrib:
                del edge.attrib["shape"]
                
            for lane in edge.findall("lane"):
                # Check if this lane is a sidewalk (allows pedestrians)
                allow = lane.get("allow", "")
                if "pedestrian" in allow:
                    lane.set("width", str(np.round(sidewalk_width, 2)))
        
        # 3. Save and Run Netconvert
        tree.write(new_file_name)
        
        try:
            # We add --junctions.scurve-stretch to help with geometry smoothing 
            # when widths change significantly
            subprocess.run([
                "netconvert",
                "--sumo-net-file", new_file_name,
                "--offset.disable-normalization", "true",
                "-o", new_file_name
            ], check=True)
            print(f"Netconvert successful. Control: {intersection_control}, Sidewalk: {sidewalk_width}m")
        except subprocess.CalledProcessError as e:
            print("Netconvert failed with error:", e)
    
    construct_geometry(new_file_name, sidewalk_width, intersect_control) # construct the new geometry

    # Signal control: Webster from volumes or manual override from user
    if intersect_control  == "Signal":
        signal_plan = signal_control.get_single_signal_plan(
            user_input_data, num_EW_lanes, num_NS_lanes
        )
        signal_control.apply_single_signal_to_net(new_file_name, signal_plan)

    # edges for peds at intersection
    intersection_edges = [":Intersection_c0", ":Intersection_w0", ":Intersection_c1", ":Intersection_w1", ":Intersection_c2", ":Intersection_w2",":Intersection_c3", ":Intersection_w3"]

    # define lanes (define for the max possible number of lanes here)
    # In SUMO the lanes are by segments, therefore for finding neighbors, we need to connect them to form real lanes
    ############################# For vehicles #########################################

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
    EB_lane1_names  = [ "EB_West_1",  connections_between_edges["EB_West_1"]["EB_East_1"] , "EB_East_1" ] # this belongs to the same lane
    WB_lane1_names = ["WB_East_1",  connections_between_edges["WB_East_1"]["WB_West_1"] , "WB_West_1"]
    NB_lane1_names = ["NB_South_1", connections_between_edges["NB_South_1"]["NB_North_1"]  , "NB_North_1"]
    SB_lane1_names = ["SB_North_1", connections_between_edges["SB_North_1"]["SB_South_1"] , "SB_South_1"]
    
    
    def construct_lane_list(template, num_lanes, all_lane_names ):
        for n in range(num_lanes):
            all_lane_names.append([])
            for seg in template:
                all_lane_names[-1].append(seg[:-1]+str(n))
        
        return all_lane_names
    
    all_lane_names = []
    all_lane_names = construct_lane_list(EB_lane1_names, 5, all_lane_names )
    all_lane_names = construct_lane_list(WB_lane1_names, 5, all_lane_names )
    all_lane_names = construct_lane_list(NB_lane1_names, 5, all_lane_names )
    all_lane_names = construct_lane_list(SB_lane1_names, 5, all_lane_names )
    
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
        for lane in lanes:
            pos_adjust_by_lanes[lane] = adjustment
            adjustment = adjustment +  lane_lengths[lane[:-1]+"0"] # lane 0 always exists and they are of the same length
            
                    
    lane_segment_to_idx={}
    for lane_idx in range(len(all_lane_names)):
        lanes = all_lane_names[lane_idx]
        for lane in lanes:
            lane_segment_to_idx[lane] = lane_idx
    # all edges, needed for lane reformatting
    all_edges = ["EB_West", "WB_East", "SB_North", "NB_South", "WB_West", "EB_East", "NB_North", "SB_South"]
    
    ###################################### Specific Geometry for ATM lanes ###########################
    
    
    # convert string to pologon vertices
    def convert_shape(shape_str):
        # shape_str: string like "6.40,8.40 -6.40,8.40"
        xs = []
        ys = []
    
        coord_pairs = shape_str.strip().split()  # split by space
        for pair in coord_pairs:
            xi, yi = pair.split(',')
            xs.append(float(xi))
            ys.append(float(yi))
    
        # Close the polygon
        xs.append(xs[0])
        ys.append(ys[0])
        return np.array(xs), np.array(ys)
    #
    def get_rectangle(shape_str, lane_width) : # here they are all rectangles
        xs = [] # 
        ys = []
        coord_pairs = shape_str.strip().split()  # split by space
        for pair in coord_pairs:
            xi, yi = pair.split(',')
            xs.append(float(xi))
            ys.append(float(yi))
    
        mag = np.sqrt((xs[1]-xs[0])**2 + (ys[1]-ys[0])**2)
        dx = (xs[1]-xs[0])/mag
        dy = (ys[1]-ys[0])/mag
    
        left_dx = dy*lane_width/2
        left_dy = -dx*lane_width/2
    
        right_dx = -dy*lane_width/2
        right_dy = dx*lane_width/2
    
        x_coords = np.array([xs[0]+left_dx, xs[0]+right_dx, xs[1]+right_dx, xs[1]+left_dx, xs[0]+left_dx])
        y_coords = np.array([ys[0]+left_dy, ys[0]+right_dy, ys[1]+right_dy, ys[1]+left_dy, ys[0]+left_dy])
    
        return x_coords, y_coords


    ### get the shapes of each ped edge and their lengths
    tree = ET.parse(net_file_name)
    root = tree.getroot()
    
    ped_edge_shapes = {}
    bike_edge_shapes = {}
    lens_by_edge={} # the length of each lane
    # Iterate over edges
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        function = edge.get("function")  # may be None
        for lane in edge.findall("lane"):
            shape = lane.get("shape")
            allow = lane.get("allow")  # e.g., "car bus pedestrian"
            lens_by_edge[edge_id] = float(lane.get("length"))
            
            # Include lane if it's a pedestrian edge or allows pedestrians
            if function in ["walkingarea", "crossing"] or (allow and "pedestrian" in allow):
                #print(f"Edge {edge_id}, Lane {lane.get('id')}, Shape: {shape}, Allow: {allow}")
                #
                #plt.plot(shape_xs, shape_ys)
                if edge_id[-2]=="w": # waiting area
                    shape_xs, shape_ys = convert_shape(shape)
                    
                elif edge_id[1]=="B": # it is a road
                    shape_xs, shape_ys = get_rectangle(shape, sidewalk_width)
                    
                else: # crossing
                    shape_xs, shape_ys = get_rectangle(shape, sidewalk_width)
                    
                ped_edge_shapes[edge_id]=[shape_xs, shape_ys]
            
            if (allow and "bicycle" in allow) and edge_id[1]=="B":
                shape_xs, shape_ys = get_rectangle(shape, 2) # width is 2m
                bike_edge_shapes[edge_id]=[shape_xs, shape_ys]
                
    #print(bike_edge_shapes)             
    ### The min and max x and y for the intersection considered by peds
    intersection_x_min = min(ped_edge_shapes[':Intersection_w0'][0])-20
    intersection_x_max = max(ped_edge_shapes[':Intersection_w1'][0])+20
    intersection_y_min = min(ped_edge_shapes[':Intersection_w3'][1])-20
    intersection_y_max = max(ped_edge_shapes[':Intersection_w1'][1])+20

    ## x range of EW crosswalks and y range of NS crosswalks
    EW_cross_x_min = min(ped_edge_shapes[':Intersection_c0'][0])-2
    EW_cross_x_max = max(ped_edge_shapes[':Intersection_c0'][0])+2
    NS_cross_y_min = min(ped_edge_shapes[':Intersection_c3'][1])-2
    NS_cross_y_max = max(ped_edge_shapes[':Intersection_c3'][1])+2

    # Arrival tolerance (meters) for ATM destination checks.
    # If this is too small, agents may never be considered "arrived".
    dest_tol = float(user_input_data.get("ATM_Dest_Tolerance", 1.0))

    ''' intersection layout
    w0  c0  w1
    
    c3      c1
    
    w3  c2  w2
    
    '''
    # now more precisely define the areas
    
    ped_WB_West_xs = ped_edge_shapes["WB_West"][0]
    ped_WB_West_ys = ped_edge_shapes["WB_West"][1]
    ped_WB_East_xs = ped_edge_shapes["WB_East"][0]
    ped_WB_East_ys = ped_edge_shapes["WB_East"][1]
    
    ped_EB_West_xs = ped_edge_shapes["EB_West"][0]
    ped_EB_West_ys = ped_edge_shapes["EB_West"][1]
    ped_EB_East_xs = ped_edge_shapes["EB_East"][0]
    ped_EB_East_ys = ped_edge_shapes["EB_East"][1]
    
    ped_NB_North_xs = ped_edge_shapes["NB_North"][0]
    ped_NB_North_ys = ped_edge_shapes["NB_North"][1]
    ped_NB_South_xs = ped_edge_shapes["NB_South"][0]
    ped_NB_South_ys = ped_edge_shapes["NB_South"][1]
    
    ped_SB_North_xs = ped_edge_shapes["SB_North"][0]
    ped_SB_North_ys = ped_edge_shapes["SB_North"][1]
    ped_SB_South_xs = ped_edge_shapes["SB_South"][0]
    ped_SB_South_ys = ped_edge_shapes["SB_South"][1]

    # bikes
    bike_WB_West_xs = bike_edge_shapes["WB_West"][0]
    bike_WB_West_ys = bike_edge_shapes["WB_West"][1]
    bike_WB_East_xs = bike_edge_shapes["WB_East"][0]
    bike_WB_East_ys = bike_edge_shapes["WB_East"][1]
    
    bike_EB_West_xs = bike_edge_shapes["EB_West"][0]
    bike_EB_West_ys = bike_edge_shapes["EB_West"][1]
    bike_EB_East_xs = bike_edge_shapes["EB_East"][0]
    bike_EB_East_ys = bike_edge_shapes["EB_East"][1]
    
    bike_NB_North_xs = bike_edge_shapes["NB_North"][0]
    bike_NB_North_ys = bike_edge_shapes["NB_North"][1]
    bike_NB_South_xs = bike_edge_shapes["NB_South"][0]
    bike_NB_South_ys = bike_edge_shapes["NB_South"][1]
    
    bike_SB_North_xs = bike_edge_shapes["SB_North"][0]
    bike_SB_North_ys = bike_edge_shapes["SB_North"][1]
    bike_SB_South_xs = bike_edge_shapes["SB_South"][0]
    bike_SB_South_ys = bike_edge_shapes["SB_South"][1]
    
    # peds 
    west_cross_xs = ped_edge_shapes[":Intersection_c3"][0]
    west_cross_ys = ped_edge_shapes[":Intersection_c3"][1]
    north_cross_xs = ped_edge_shapes[":Intersection_c0"][0]
    north_cross_ys = ped_edge_shapes[":Intersection_c0"][1]
    east_cross_xs = ped_edge_shapes[":Intersection_c1"][0]
    east_cross_ys = ped_edge_shapes[":Intersection_c1"][1]
    south_cross_xs = ped_edge_shapes[":Intersection_c2"][0]
    south_cross_ys = ped_edge_shapes[":Intersection_c2"][1]
    
    NW_waiting_xs = np.array([max(ped_WB_West_xs), max(ped_WB_West_xs), max(ped_WB_West_xs)+sidewalk_width, max(ped_SB_North_xs), max(ped_SB_North_xs), min(ped_SB_North_xs), max(ped_WB_West_xs)])
    NW_waiting_ys = np.array([max(ped_WB_West_ys), min(ped_WB_West_ys), min(ped_WB_West_ys), min(ped_SB_North_ys)-sidewalk_width, min(ped_SB_North_ys), min(ped_SB_North_ys), max(ped_WB_West_ys) ])
    ped_edge_shapes[":Intersection_w0"] = [NW_waiting_xs, NW_waiting_ys]

    NE_waiting_xs = np.array([min(ped_WB_East_xs), min(ped_WB_East_xs), min(ped_WB_East_xs)-sidewalk_width, min(ped_NB_North_xs), min(ped_NB_North_xs), max(ped_NB_North_xs), min(ped_WB_East_xs) ])
    NE_waiting_ys = np.array([max(ped_WB_East_ys), min(ped_WB_East_ys), min(ped_WB_East_ys), min(ped_NB_North_ys)-sidewalk_width, min(ped_NB_North_ys),  min(ped_NB_North_ys), max(ped_WB_East_ys) ])
    ped_edge_shapes[":Intersection_w1"] = [NE_waiting_xs, NE_waiting_ys]
    
    SE_waiting_xs = np.array([min(ped_EB_East_xs), min(ped_EB_East_xs), min(ped_EB_East_xs)-sidewalk_width , min(ped_NB_South_xs), min(ped_NB_South_xs), max(ped_NB_South_xs), min(ped_EB_East_xs)])
    SE_waiting_ys = np.array([min(ped_EB_East_ys), max(ped_EB_East_ys), max(ped_EB_East_ys), max(ped_NB_South_ys)+sidewalk_width, max(ped_NB_South_ys), max(ped_NB_South_ys), min(ped_EB_East_ys) ])
    ped_edge_shapes[":Intersection_w2"] = [SE_waiting_xs, SE_waiting_ys]

    SW_waiting_xs = np.array([max(ped_EB_West_xs), max(ped_EB_West_xs), max(ped_EB_West_xs)+sidewalk_width , max(ped_SB_South_xs), max(ped_SB_South_xs), min(ped_SB_South_xs), max(ped_EB_West_xs)])
    SW_waiting_ys = np.array([min(ped_EB_West_ys), max(ped_EB_West_ys), max(ped_EB_West_ys), max(ped_SB_South_ys)+sidewalk_width,  max(ped_SB_South_ys), max(ped_SB_South_ys), min(ped_EB_West_ys)])
    ped_edge_shapes[":Intersection_w3"] = [SW_waiting_xs, SW_waiting_ys]
    
    intersection_xbounds = [NW_waiting_xs, north_cross_xs, NE_waiting_xs, east_cross_xs, SE_waiting_xs, south_cross_xs, SW_waiting_xs, west_cross_xs ]
    intersection_ybounds = [NW_waiting_ys, north_cross_ys, NE_waiting_ys, east_cross_ys, SE_waiting_ys, south_cross_ys, SW_waiting_ys, west_cross_ys ]
    
    # now define a function to combine the areas (given a route)
    
    # sample the boundaries at 0.1s interval to form dense points
    def convert_bound_to_pts(x_bounds, y_bounds):
        
        num_pts = int(max((max(x_bounds)-min(x_bounds))//0.1, (max(y_bounds)-min(y_bounds))//0.1))
    
        x_pts = (np.linspace(x_bounds[0], x_bounds[1], num_pts))
        y_pts = (np.linspace(y_bounds[0], y_bounds[1], num_pts))
    
        return x_pts, y_pts

    def convert_all_bounds_to_pts(x_bounds, y_bounds):
        all_xpts = []
        all_ypts = []

        for i in range(len(x_bounds)-1):
            num_pts = int(max(np.absolute(x_bounds[i] - x_bounds[i+1])//0.1, (np.absolute(y_bounds[i] - y_bounds[i+1]))//0.1))
            
            x_pts = (np.linspace(x_bounds[i], x_bounds[i+1], num_pts))
            y_pts = (np.linspace(y_bounds[i], y_bounds[i+1], num_pts))
            all_xpts.append(x_pts)
            all_ypts.append(y_pts)
        
        x_pts= np.concatenate(all_xpts)
        y_pts= np.concatenate(all_ypts)
         
        return x_pts, y_pts

    # now do spline pts for bikes
    #bike_SB_North_xs = bike_edge_shapes["SB_North"][0]
    #bike_SB_North_ys = bike_edge_shapes["SB_North"][1]
    #bike_SB_South_xs = bike_edge_shapes["SB_South"][0]
    #bike_SB_South_ys = bike_edge_shapes["SB_South"][1]
    # bike bd points for each edge
    bike_xbounds_by_edge = {}
    bike_ybounds_by_edge = {}
    bike_xcenters_by_edge = {}
    bike_ycenters_by_edge = {}

    bike_destination_bound_xs = {}
    bike_destination_bound_ys = {}
    for edge_id in bike_edge_shapes: # this does not restrict to destination edge
        bike_rec_xpts, bike_rec_ypts= convert_all_bounds_to_pts(bike_edge_shapes[edge_id][0], bike_edge_shapes[edge_id][1])
        #bike_bound_xpts, bike_bound_ypts = convert_bound_to_pts(bike_edge_shapes[edge_id][0], bike_edge_shapes[edge_id][1])
        if "SB" in edge_id:
            min_y = np.min(bike_rec_ypts)
            mask = np.isclose(bike_rec_ypts, min_y)
            bike_bound_xpts = bike_rec_xpts[mask]
            bike_bound_ypts = bike_rec_ypts[mask]
        
        # NB: take top boundary (max y)
        elif "NB" in edge_id:
            max_y = np.max(bike_rec_ypts)
            mask = np.isclose(bike_rec_ypts, max_y)
            bike_bound_xpts = bike_rec_xpts[mask]
            bike_bound_ypts = bike_rec_ypts[mask]
        
        # EB: take right boundary (max x)
        elif "EB" in edge_id:
            max_x = np.max(bike_rec_xpts)
            mask = np.isclose(bike_rec_xpts, max_x)
            bike_bound_xpts = bike_rec_xpts[mask]
            bike_bound_ypts = bike_rec_ypts[mask]
        
        # WB: take left boundary (min x)
        elif "WB" in edge_id:
            min_x = np.min(bike_rec_xpts)
            mask = np.isclose(bike_rec_xpts, min_x)
            bike_bound_xpts = bike_rec_xpts[mask]
            bike_bound_ypts = bike_rec_ypts[mask]
        
        
        #print(edge_id)
        #plt.plot(bike_bound_xpts, bike_bound_ypts)
                                                                
        #plt.scatter(bike_rec_xpts, bike_rec_ypts)
        #plt.show()

        bike_destination_bound_xs[edge_id] = bike_bound_xpts
        bike_destination_bound_ys[edge_id] = bike_bound_ypts
                                                                
                                                                
        if ("SB" in edge_id) or ("NB" in edge_id):
            #print( "SB", set(bike_rec_xpts),)
            
            ypt_min = np.min(bike_rec_ypts)
            ypt_max = np.max(bike_rec_ypts)
            valid_indices = np.where((bike_rec_ypts > ypt_min) & (bike_rec_ypts < ypt_max))[0]
            bike_rec_xpts = bike_rec_xpts[valid_indices]
            bike_rec_ypts = bike_rec_ypts[valid_indices]
        
            mean_xbd = np.mean(bike_rec_xpts)
            bound_xs1 = bike_rec_xpts[bike_rec_xpts > mean_xbd]   # right boundary
            bound_xs2 = bike_rec_xpts[bike_rec_xpts < mean_xbd]   # left boundary
            bound_ys1 = bike_rec_ypts[bike_rec_xpts > mean_xbd]
            bound_ys2 = bike_rec_ypts[bike_rec_xpts < mean_xbd]

            order1 = np.argsort(bound_ys1)
            order2 = np.argsort(bound_ys2)
            bound_xs1 = bound_xs1[order1]
            bound_ys1 = bound_ys1[order1]
            bound_xs2 = bound_xs2[order2]
            bound_ys2 = bound_ys2[order2]
        
            center_xs = (bound_xs1 + bound_xs2) / 2
            center_ys = (bound_ys1 + bound_ys2) / 2
        
            bike_xbounds_by_edge[edge_id] = [bound_xs1, bound_xs2]
            bike_ybounds_by_edge[edge_id] = [bound_ys1, bound_ys2]
            bike_xcenters_by_edge[edge_id] = center_xs
            bike_ycenters_by_edge[edge_id] = center_ys
        
        
        # EB/WB logic: filter by x-range, split by mean y (top vs bottom)
        elif ("EB" in edge_id) or ("WB" in edge_id):
            #print( "EWB", set(bike_rec_ypts))
           
            
            
            xpt_min = np.min(bike_rec_xpts)
            xpt_max = np.max(bike_rec_xpts)
            valid_indices = np.where((bike_rec_xpts > xpt_min) & (bike_rec_xpts < xpt_max))[0]
            bike_rec_xpts = bike_rec_xpts[valid_indices]
            bike_rec_ypts = bike_rec_ypts[valid_indices]
        
            mean_ybd = np.mean(bike_rec_ypts)
            
            bound_ys1 = bike_rec_ypts[bike_rec_ypts > mean_ybd]   # upper boundary
            bound_ys2 = bike_rec_ypts[bike_rec_ypts < mean_ybd]   # lower boundary
            bound_xs1 = bike_rec_xpts[bike_rec_ypts > mean_ybd]
            bound_xs2 = bike_rec_xpts[bike_rec_ypts < mean_ybd]

            order1 = np.argsort(bound_xs1)
            order2 = np.argsort(bound_xs2)
            bound_xs1 = bound_xs1[order1]
            bound_ys1 = bound_ys1[order1]
            bound_xs2 = bound_xs2[order2]
            bound_ys2 = bound_ys2[order2]

            center_xs = (bound_xs1 + bound_xs2) / 2
            center_ys = (bound_ys1 + bound_ys2) / 2
        
            bike_xbounds_by_edge[edge_id] = [bound_xs1, bound_xs2]
            bike_ybounds_by_edge[edge_id] = [bound_ys1, bound_ys2]
            bike_xcenters_by_edge[edge_id] = center_xs
            bike_ycenters_by_edge[edge_id] = center_ys
    
        #plt.plot(center_xs, center_ys)
        #plt.plot(bound_xs1, bound_ys1)
        #plt.plot(bound_xs2, bound_ys2)
        #plt.show()
        #print(center_xs, center_ys)
        #print(bound_xs1, bound_ys1)
        #print(bound_xs2, bound_ys2)
    # given multiple boundaries with point representation, return the joint boundary in the form of corners or dense pts
    def find_joint_boundary(x_lists, y_lists):
        all_boundaries = []
        for i in range(len(x_lists)):
            for ii in range(1, len(x_lists[i])):
                # x1, x2, y1, y2
                
                if x_lists[i][ii-1]==x_lists[i][ii]:
                    all_boundaries.append([x_lists[i][ii-1], x_lists[i][ii], min(y_lists[i][ii-1],y_lists[i][ii]), max(y_lists[i][ii-1],y_lists[i][ii]) ])
                elif y_lists[i][ii-1]==y_lists[i][ii]:
                    all_boundaries.append([min(x_lists[i][ii], x_lists[i][ii-1]), max(x_lists[i][ii], x_lists[i][ii-1]), y_lists[i][ii-1], y_lists[i][ii] ])
                else:
                    all_boundaries.append([x_lists[i][ii-1] , x_lists[i][ii], y_lists[i][ii-1], y_lists[i][ii] ])
        
        # rounding needed to be precise 
        all_boundaries = np.round(np.array(all_boundaries), 1)
    
        # Convert rows to tuples so we can use np.unique with return_counts
        rows_as_tuples = [tuple(row) for row in all_boundaries]
        unique_rows, counts = np.unique(rows_as_tuples, return_counts=True, axis=0)
        
        # Keep only rows that appear exactly once
        boundaries = np.array([row for row, count in zip(unique_rows, counts) if count == 1])
        
        boundary_xs = boundaries[:,0:2]
        boundary_ys = boundaries[:,2:]

        # next: make sparse points within lines so that one can find the nearest point on the boundary
        boundary_x_pts=[]
        boundary_y_pts=[]
        for i in range(len(boundary_xs)):
            bound_xs = boundary_xs[i]
            bound_ys = boundary_ys[i]
    
            x_pts, y_pts = convert_bound_to_pts(bound_xs, bound_ys)
            
    
            boundary_x_pts.append(x_pts)
            boundary_y_pts.append(y_pts)
    
        boundary_x_pts = np.round(np.concatenate(boundary_x_pts),1)
        boundary_y_pts = np.round(np.concatenate(boundary_y_pts),1)
        
        
        return boundary_xs , boundary_ys, boundary_x_pts, boundary_y_pts
        # the later two are scattered points with representations at an interavl of 0.1m

    # construct the boundaries of peds by OD
    # Initialize the main dictionaries
    # x and y pts for each side of the boundary per od
    ped_route_xbounds_by_od = {} 
    ped_route_ybounds_by_od = {}
    # x and y pts for the center between the boundaries per od
    ped_route_xcenters_by_od={} 
    ped_route_ycenters_by_od={}
    
    # --- Eastbound (EB) Routes ---
    # EB_West to EB_East (Forward)
    ped_route_xbounds_by_od.setdefault("EB_West", {})["EB_East"] = [ped_EB_West_xs, SW_waiting_xs, south_cross_xs, SE_waiting_xs, ped_EB_East_xs ]
    ped_route_ybounds_by_od.setdefault("EB_West", {})["EB_East"] = [ped_EB_West_ys, SW_waiting_ys, south_cross_ys, SE_waiting_ys, ped_EB_East_ys ]
    # EB_East to EB_West (Reverse)
    # NOTE: The provided segments are the same as the forward route, which suggests a simple reverse path
    ped_route_xbounds_by_od.setdefault("EB_East", {})["EB_West"] = [ped_EB_East_xs, SE_waiting_xs, south_cross_xs, SW_waiting_xs, ped_EB_West_xs ] # Corrected segment order
    ped_route_ybounds_by_od.setdefault("EB_East", {})["EB_West"] = [ped_EB_East_ys, SE_waiting_ys, south_cross_ys, SW_waiting_ys, ped_EB_West_ys ] # Corrected segment order
    
    # --- Westbound (WB) Routes ---
    # WB_West to WB_East (Forward)
    ped_route_xbounds_by_od.setdefault("WB_West", {})["WB_East"] = [ped_WB_West_xs, NW_waiting_xs, north_cross_xs, NE_waiting_xs, ped_WB_East_xs ]
    ped_route_ybounds_by_od.setdefault("WB_West", {})["WB_East"] = [ped_WB_West_ys, NW_waiting_ys, north_cross_ys, NE_waiting_ys, ped_WB_East_ys ]
    # WB_East to WB_West (Reverse)
    ped_route_xbounds_by_od.setdefault("WB_East", {})["WB_West"] = [ped_WB_East_xs, NE_waiting_xs, north_cross_xs, NW_waiting_xs, ped_WB_West_xs ] # Corrected segment order
    ped_route_ybounds_by_od.setdefault("WB_East", {})["WB_West"] = [ped_WB_East_ys, NE_waiting_ys, north_cross_ys, NW_waiting_ys, ped_WB_West_ys ] # Corrected segment order

    # --- Southbound (SB) Routes ---
    # SB_North to SB_South (Forward)
    ped_route_xbounds_by_od.setdefault("SB_North", {})["SB_South"] = [ped_SB_North_xs, NW_waiting_xs, west_cross_xs, SW_waiting_xs, ped_SB_South_xs ]
    ped_route_ybounds_by_od.setdefault("SB_North", {})["SB_South"] = [ped_SB_North_ys, NW_waiting_ys, west_cross_ys, SW_waiting_ys, ped_SB_South_ys ]
    # SB_South to SB_North (Reverse) - **Corrected typo** in segment order for S->N
    ped_route_xbounds_by_od.setdefault("SB_South", {})["SB_North"] = [ped_SB_South_xs, SW_waiting_xs, west_cross_xs, NW_waiting_xs, ped_SB_North_xs ]
    ped_route_ybounds_by_od.setdefault("SB_South", {})["SB_North"] = [ped_SB_South_ys, SW_waiting_ys, west_cross_ys, NW_waiting_ys, ped_SB_North_ys ]
    
    # --- Northbound (NB) Routes ---
    # NB_North to NB_South (Forward) - **Corrected typo**: N->S should be starting North and ending South
    ped_route_xbounds_by_od.setdefault("NB_North", {})["NB_South"] = [ped_NB_North_xs, NE_waiting_xs, east_cross_xs, SE_waiting_xs, ped_NB_South_xs ]
    ped_route_ybounds_by_od.setdefault("NB_North", {})["NB_South"] = [ped_NB_North_ys, NE_waiting_ys, east_cross_ys, SE_waiting_ys, ped_NB_South_ys ]
    # NB_South to NB_North (Reverse) - **Corrected typo** in segment order for S->N
    ped_route_xbounds_by_od.setdefault("NB_South", {})["NB_North"] = [ped_NB_South_xs, SE_waiting_xs, east_cross_xs, NE_waiting_xs, ped_NB_North_xs ]
    ped_route_ybounds_by_od.setdefault("NB_South", {})["NB_North"] = [ped_NB_South_ys, SE_waiting_ys, east_cross_ys, NE_waiting_ys, ped_NB_North_ys ]


    for origin in ped_route_xbounds_by_od:
        ped_route_xcenters_by_od[origin]={}
        ped_route_ycenters_by_od[origin]={}
        for destination in ped_route_xbounds_by_od[origin]:
            all_x_bounds = ped_route_xbounds_by_od[origin][destination]
            all_y_bounds = ped_route_ybounds_by_od[origin][destination]
            
            # discretize
            bxs, bys, bx_pts, by_pts = find_joint_boundary(all_x_bounds, all_y_bounds)
            # exclude the two sides of the boundaries (between the bound)
            valid_bnd_ind = ((bx_pts>-West_L+0.01) & (bx_pts<East_L-0.01) & (by_pts>-South_L+0.01) & (by_pts<North_L-0.01)  )
            valid_bx_pts = bx_pts[valid_bnd_ind ]
            valid_by_pts = by_pts[valid_bnd_ind ]

            
            
            if ("EB" in origin) or ("EB" in destination) or ("WB" in origin) or ("WB" in destination):
                # 1. Sort all points by X to ensure they are in order
                sort_idx = np.argsort(valid_bx_pts)
                sorted_all_x = valid_bx_pts[sort_idx]
                sorted_all_y = valid_by_pts[sort_idx]
                
                # 2. Find unique X values
                # If your data has floating point noise, use np.round(sorted_all_x, 3)
                unique_xs = np.unique(sorted_all_x)
                
                side1_x, side1_y = [], []
                side2_x, side2_y = [], []
                
                # 3. For each X, find the two Y values (min and max)
                for ux in unique_xs:
                    # Get all Y values that share this exact X
                    y_values_at_x = sorted_all_y[sorted_all_x == ux]
                    
                    if len(y_values_at_x) >= 2:
                        # Assign the lower Y to one side and higher Y to the other
                        side1_x.append(ux)
                        side1_y.append(np.min(y_values_at_x))
                        
                        side2_x.append(ux)
                        side2_y.append(np.max(y_values_at_x))
                
                # 4. Convert back to numpy arrays for your existing logic
                sorted_bx1s = np.array(side1_x)
                sorted_bx2s = np.array(side2_x)
                sorted_by1s = np.array(side1_y)
                sorted_by2s = np.array(side2_y)
                
                # 5. Calculate the centerline
                ped_route_xbounds_by_od[origin][destination] = [sorted_bx1s, sorted_bx2s]
                ped_route_ybounds_by_od[origin][destination] = [sorted_by1s, sorted_by2s]
                
                center_xs = sorted_bx1s 
                center_ys = np.round((sorted_by1s + sorted_by2s) / 2, 1)
                
            else:
                # cluster x for NS bounds
                # 1. Sort all points by Y to ensure they are in a vertical order (South to North)
                sort_idx = np.argsort(valid_by_pts)
                sorted_all_x = valid_bx_pts[sort_idx]
                sorted_all_y = valid_by_pts[sort_idx]
                
                # 2. Find unique Y values 
                # (Add np.round(..., 2) if the coordinates have floating-point noise)
                unique_ys = np.unique(sorted_all_y)
                
                side1_x, side1_y = [], []
                side2_x, side2_y = [], []
                
                # 3. For each Y, find the two X values (Left rail and Right rail)
                for uy in unique_ys:
                    # Get all X values that share this exact Y vertical station
                    x_values_at_y = sorted_all_x[sorted_all_y == uy]
                    
                    if len(x_values_at_y) >= 2:
                        # Assign the lower X (West side) to one side and higher X (East side) to the other
                        side1_y.append(uy)
                        side1_x.append(np.min(x_values_at_y))
                        
                        side2_y.append(uy)
                        side2_x.append(np.max(x_values_at_y))
                
                # 4. Convert back to numpy arrays
                sorted_bx1s = np.array(side1_x)
                sorted_bx2s = np.array(side2_x)
                sorted_by1s = np.array(side1_y)
                sorted_by2s = np.array(side2_y)
                
                # 5. Store in your OD dictionary
                ped_route_xbounds_by_od[origin][destination] = [sorted_bx1s, sorted_bx2s]
                ped_route_ybounds_by_od[origin][destination] = [sorted_by1s, sorted_by2s]
                
                # Calculate the vertical centerline
                # center_ys is the vertical flow, center_xs is the horizontal midpoint
                center_ys = sorted_by1s 
                center_xs = np.round((sorted_bx1s + sorted_bx2s) / 2, 1)
    
            ped_route_xcenters_by_od[origin][destination] = center_xs 
            ped_route_ycenters_by_od[origin][destination] = center_ys 

    # function to find nearest boundary point
    def find_nearest_boundary(x, y, boundary_x_pts, boundary_y_pts):
        # find nearest boundary pt and compute the distnace
        boundary_pts = np.stack([boundary_x_pts, boundary_y_pts], axis=1)  # shape (N,2)
        query_pt = np.array([x, y])
    
        dists = np.linalg.norm(boundary_pts - query_pt, axis=1)  # Euclidean distances
        idx_min = np.argmin(dists)
    
        nearest_bound_x = boundary_x_pts[idx_min]
        nearest_bound_y = boundary_y_pts[idx_min]
        min_dist = dists[idx_min]
        
        return nearest_bound_x, nearest_bound_y, min_dist

    # the destination can also be seen as a line boundary for peds and bikes (to randomly choose a target on it)
    destination_boundary={}
    destination_boundary_pts={}
    
    for edge in all_edges:
        edge_x_bounds = ped_edge_shapes[edge][0]
        edge_y_bounds = ped_edge_shapes[edge][1]
        if edge[-4:]=="West":
            destination_boundary[edge] = [ [min(edge_x_bounds), min(edge_x_bounds)],  [min(edge_y_bounds), max(edge_y_bounds)] ]
        elif edge[-4:]=="East":
            destination_boundary[edge] = [ [max(edge_x_bounds), max(edge_x_bounds)],  [min(edge_y_bounds), max(edge_y_bounds)] ]
        elif edge[-5:]=="North":
            destination_boundary[edge] = [ [min(edge_x_bounds), max(edge_x_bounds)],  [max(edge_y_bounds), max(edge_y_bounds)] ]
        elif edge[-5:]=="South":
            destination_boundary[edge] = [ [min(edge_x_bounds), max(edge_x_bounds)],  [min(edge_y_bounds), min(edge_y_bounds)] ]
            
    for edge in all_edges:
        x_pts, y_pts = convert_bound_to_pts(destination_boundary[edge][0], destination_boundary[edge][1])
        destination_boundary_pts[edge] = [x_pts, y_pts]
    
    ####################################### Input Demands #################################
    # define vehicle input nodes 
    in_names = ["EB_West", "WB_East", "SB_North", "NB_South"]
    out_names = ["WB_West", "EB_East", "NB_North", "SB_South"] # all outputs
    

    # Ped/Bike sidewalk demand.
    # We still assume ATMs do not "turn" (no corner-to-corner walking), but we DO allow
    # two-way usage on each leg: agents can start on any of the 8 sidewalks and their
    # destination is the opposite sidewalk on the same corridor.
    #
    # This matches the ODs that are pre-built in `ped_route_*_by_od` above:
    # EB_West <-> EB_East, WB_East <-> WB_West, NB_South <-> NB_North, SB_North <-> SB_South.
    all_ped_in_edges = [
        "EB_West", "EB_East",
        "WB_East", "WB_West",
        "SB_North", "SB_South",
        "NB_South", "NB_North",
    ]
    all_ped_out_edges = [
        "EB_East", "EB_West",
        "WB_West", "WB_East",
        "SB_South", "SB_North",
        "NB_North", "NB_South",
    ]
    #all_bike_in_edges = ["EB_West", "WB_East", "SB_North", "NB_South"]
    #all_bike_out_edges = ["EB_East", "WB_West", "SB_South", "NB_North"]
    # Bike ODs: two per approach — through + right turn (US-style). Left turns excluded
    # (SUMO net: SB_North→EB_East is dir="l"; EB_West→NB_North / WB_East→SB_South / NB_South→WB_West are lefts and not listed).
    all_bike_in_edges = ["EB_West", "EB_West",
                         "WB_East", "WB_East",
                         "SB_North",
                         "NB_South", "NB_South"]
    all_bike_out_edges = ["EB_East", "SB_South",
                          "WB_West", "NB_North",
                          "SB_South",
                          "NB_North", "EB_East"]

    #all_bike_in_edges = ["EB_West", "WB_East", "SB_North", "NB_South"]
    #all_bike_out_edges = ["EB_East", "WB_West", "SB_South", "NB_North"]
    
    ped_start_rel_locs = {"EB_West":0, "WB_East":0, "SB_North":0, "NB_South":0, "WB_West":1, "EB_East":1, "NB_North":1, "SB_South":1 } 
    # 1 means at end and 0 means at the start (we randomize still)

    LT_names = ["NB_North", "SB_South", "EB_East", "WB_West"]
    RT_names = ["SB_South", "NB_North", "WB_West", "EB_East"]
    Through_names =["EB_East", "WB_West", "SB_South", "NB_North" ]

    _bounds = ("East-Bound", "West-Bound", "South-Bound", "North-Bound")
    _vf = user_input_data["Vehicle_Flows"]
    in_flows = [ _vf[b]["volume"] for b in _bounds ]
    # Default 10% left / 10% right per approach when keys omitted (matches GUI defaults).
    LT_ratios = [ float(_vf[b].get("LT_Ratio", 0.1)) for b in _bounds ]
    RT_ratios = [ float(_vf[b].get("RT_Ratio", 0.1)) for b in _bounds ]
    
                
        
    ########################### Simulation Hyperparameters ##############################################################################
    
    sim_visualization = user_input_data["Sim_Visualization"] # whether user wants visualization
    sim_time =  user_input_data["Sim_Time"] # 1 hr of simulation
    max_t = sim_time
    
    min_t = 0
    t_step = user_input_data["Sim_StepSize"]
    
    ####################### Vehicle Input Flow Characteritics ###############################################################################
    
    ## Vehicle Flow Variables
    
    HV_rate = float(user_input_data["Vehicle_Flows"].get("HV_rate", 0.0))
    AV_rate = float(user_input_data["Vehicle_Flows"].get("AV_rate", 0.0))
    CAV_rate = float(user_input_data["Vehicle_Flows"].get("CAV_rate", 0.0))
    CAHV_rate = float(user_input_data["Vehicle_Flows"].get("CAHV_rate", 0.0))
    SV_rate = float(user_input_data["Vehicle_Flows"].get("SV_rate", max(0.0, 1.0 - HV_rate - AV_rate - CAV_rate - CAHV_rate)))
    # Vehicle classes:
    #   0: SV, 1: AV, 2: HV, 3: CAV (AV size), 4: CAHV (HV size)
    veh_lens = {0: 4.5, 1: 4.5, 2: 12.0, 3: 4.5, 4: 12.0} # m of each class of vehicles

    ######################## ATM Input Flow Characteristics #####################################################
    # NOTE:
    # - The UI may not populate ATM fields when Ped/Bike are disabled.
    # - Avoid KeyErrors and divide-by-zero by deriving enable flags from volumes.
    ped_volume = float(user_input_data.get("Ped_Volume", 0) or 0)
    bike_volume = float(user_input_data.get("Bike_Volume", 0) or 0)
    ped_allowed = bool(user_input_data.get("Ped_Allowed", ped_volume > 0))
    bike_allowed = bool(user_input_data.get("Bike_Allowed", bike_volume > 0))

    # flow per OD pair
    
    ped_od_flow = (ped_volume / len(all_ped_in_edges)) if (ped_allowed and ped_volume > 0 and len(all_ped_in_edges) > 0) else 0.0
    bike_od_flow = (bike_volume / len(all_bike_in_edges)) if (bike_allowed and bike_volume > 0 and len(all_bike_in_edges) > 0) else 0.0

    # size of atms
    ped_radius = 0.3
    bike_radius = 1
    
    ############################################ Model Selection #################################################
    LC_model_name = user_input_data["LC_Model"]
    CF_model_name = user_input_data["CF_Model"]

    # ATM model is only required when any ATM is enabled.
    # Default to "SF" so the simulation can run even if the ATM page was skipped.
    atm_model_name = user_input_data.get("ATM_Model", "SF")
    Ped_model_name = atm_model_name
    Bike_model_name = atm_model_name
    
    CF_default = user_input_data["CF_Default_Params"] # True or false
    LC_default = user_input_data["LC_Default_Params"] #  True or false
    Ped_default = user_input_data.get("ATM_Default_Params", True)
    Bike_default = user_input_data.get("ATM_Default_Params", True)
    # Code-only (no GUI): True = native SUMO dynamics for VRUs via TraCI (setSpeed -1); False = SF/PT TraCI control.
    VRU_USE_SUMO_DEFAULT = bool(user_input_data.get("VRU_USE_SUMO_DEFAULT", True))

    min_hw = 1.5 # this is when generating the vehicles only just to make sure they do not overlap
    min_gap =2 # min gap for DDM lc Model

    # --- Collision / rear-end safety tuning (ported from freeway_sim.py) ---
    # Keep models (IDM/PT/MOBIL/DDM) unchanged; only constrain parameters for realism/safety.
    #
    # Optional guards (OFF by default to avoid changing the model logic):
    # - Enable_Safety_SpeedCap: post-process speed to guarantee a minimum gap (changes control logic)
    # - Set_SpeedMode_Safety: forces SUMO safety checks even under setSpeed (changes behavior)
    ENABLE_SAFETY_SPEED_CAP = bool(user_input_data.get("Enable_Safety_SpeedCap", False))
    PT_STOCHASTIC_ENABLE = bool(user_input_data.get("PT_Stochastic_Enable", False))
    PT_STOCHASTIC_SIGMA = float(user_input_data.get("PT_Stochastic_Sigma", 0.0))
    SET_SPEEDMODE_SAFETY = bool(user_input_data.get("Set_SpeedMode_Safety", False))
    # Extra bumper-to-bumper gap (m) required beyond the vehicle's own minGap (s0) if safety cap is enabled.
    SAFETY_EXTRA_GAP = float(user_input_data.get("Safety_Extra_Gap", 0.5))
    # Clamp extremely aggressive IDM samples (used by our custom longitudinal model)
    IDM_MIN_T = float(user_input_data.get("IDM_Min_T", 0.8))      # s
    IDM_MIN_B = float(user_input_data.get("IDM_Min_b", 2.0))      # m/s^2
    IDM_MIN_S0 = float(user_input_data.get("IDM_Min_s0", 2.0))    # m
    
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
    
    SF_atm_data = pd.read_csv(model_folder+"SF_atm_params.csv")
    PT_atm_data = pd.read_csv(model_folder+"PT_atm_params.csv")
    
    veh_class_names = {0: "Small Vehicle", 1: "Automated Vehicle", 2: "Heavy Vehicle"}
    
    
     
    # functions to sample the model params
    def sample_IDM(veh_class=None): # 0 for sv, 1 for A, 2 for HV
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
            samples = IDM_param_data[_idm_csv_key] # the sample data frame
            samples = samples[samples["T"]>0]
            picked_row = samples.sample(n=1).iloc[0]
            driving_params = np.array(picked_row[["T", "a", "b", "v0", "so", "delta"]].values, dtype = np.float64 ) 
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
            #driving_params = np.array([ mean_T_max, mean_alpha, mean_beta, mean_Wc, mean_Gamma1, mean_Gamma2, mean_Wm ], dtype=np.float64)
            return driving_params 
        
        else:
            _pt_csv_key = 4 if veh_class == 4 else (1 if veh_class == 3 else veh_class)
            samples = PT_param_data[_pt_csv_key] # the sample data frame
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
    
    # sample drift-diffusion model
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

    # sample SF_ped
    # vAlpha0_ped, tauAlpha_ped, A_pp, B_pp, A_wall_ped, B_wall_ped 
    def sample_SF_Ped():
        if Ped_default == False and Ped_model_name == "SF":
        
          
            mean_v_alpha = user_input_data["ATM_Parameters"]['Ped: v_α'][veh_class_names[veh_class]]["Mean"]
            std_v_alpha = user_input_data["ATM_Parameters"]['Ped: v_α'][veh_class_names[veh_class]]["Std"]
            
            mean_tau_alpha = user_input_data["ATM_Parameters"]['Ped: τ_α'][veh_class_names[veh_class]]["Mean"]
            std_tau_alpha = user_input_data["ATM_Parameters"]['Ped: τ_α'][veh_class_names[veh_class]]["Std"]
            
            mean_app = user_input_data["ATM_Parameters"]['Ped: A_pp'][veh_class_names[veh_class]]["Mean"]
            std_app = user_input_data["ATM_Parameters"]['Ped: A_pp'][veh_class_names[veh_class]]["Std"]
            
            mean_bpp = user_input_data["ATM_Parameters"]['Ped: B_pp'][veh_class_names[veh_class]]["Mean"]
            std_bpp = user_input_data["ATM_Parameters"]['Ped: B_pp'][veh_class_names[veh_class]]["Std"]
            
            mean_awall = user_input_data["ATM_Parameters"]['Ped: A_wall'][veh_class_names[veh_class]]["Mean"]
            std_awall = user_input_data["ATM_Parameters"]['Ped: A_wall'][veh_class_names[veh_class]]["Std"]
            
            mean_bwall = user_input_data["ATM_Parameters"]['Ped: B_wall'][veh_class_names[veh_class]]["Mean"]
            std_bwall = user_input_data["ATM_Parameters"]['Ped: B_wall'][veh_class_names[veh_class]]["Std"]
    
            # 2. Sample from Gaussian Distributions
            # Using a small positive minimum (0.001) for parameters that must be positive.
            random_v_alpha = max(0.001, random.gauss(mean_v_alpha, std_v_alpha))
            random_tau_alpha = max(0.001, random.gauss(mean_tau_alpha, std_tau_alpha))
            random_app = max(0.001, random.gauss(mean_app, std_app))
            random_bpp = max(0.001, random.gauss(mean_bpp, std_bpp))
            random_awall = max(0.001, random.gauss(mean_awall, std_awall))
            random_bwall = max(0.001, random.gauss(mean_bwall, std_bwall))

            # 3. Combine into a NumPy array (v_α, τ_α, A_pp, B_pp, A_wall, B_wall)
            driving_params = np.array([
                random_v_alpha, random_tau_alpha, random_app, 
                random_bpp, random_awall, random_bwall 
            ], dtype=np.float64)
            return driving_params

        else:
            # Sample from existing data (assuming SF_Ped_data exists and has matching columns)
            picked_row = SF_atm_data.sample(n=1).iloc[0]
            # NOTE: You must replace the column names below with the actual column names in SF_Ped_data
            ped_params = picked_row[[ 'vAlpha0_ped', 'tauAlpha_ped', 'A_pp', 'B_pp', 'A_wall', 'B_wall']].values
            
            return np.array(ped_params, dtype=np.float64)

    # sample SF_bike
    # m_gamma, tau_gamma, v_gamma0, a_gamma_max, b_gamma, eta_gamma, mu_gamma, eps_m, A_w, B_w, A_s, B_s, T_i
    def sample_SF_Bike():
        # Parameters for Bicycles (SF Model, using Bike: prefix)
        
        if Bike_default == False and Bike_model_name == "SF":
            
            mean_tau_gamma = user_input_data["ATM_Parameters"]['Bike: τ_γ'][veh_class_names[veh_class]]["Mean"]
            std_tau_gamma = user_input_data["ATM_Parameters"]['Bike: τ_γ'][veh_class_names[veh_class]]["Std"]
            
            # v_γ (Desired Speed)
            mean_v_gamma = user_input_data["ATM_Parameters"]['Bike: v_γ'][veh_class_names[veh_class]]["Mean"]
            std_v_gamma = user_input_data["ATM_Parameters"]['Bike: v_γ'][veh_class_names[veh_class]]["Std"]
            
            # a_γ (Maximum Acceleration)
            mean_a_gamma = user_input_data["ATM_Parameters"]['Bike: a_γ'][veh_class_names[veh_class]]["Mean"]
            std_a_gamma = user_input_data["ATM_Parameters"]['Bike: a_γ'][veh_class_names[veh_class]]["Std"]
            
            # b_γ (Comfortable Braking Deceleration)
            mean_b_gamma = user_input_data["ATM_Parameters"]['Bike: b_γ'][veh_class_names[veh_class]]["Mean"]
            std_b_gamma = user_input_data["ATM_Parameters"]['Bike: b_γ'][veh_class_names[veh_class]]["Std"]
            
            # η_γ (Exponent for distance decay)
            mean_eta_gamma = user_input_data["ATM_Parameters"]['Bike: η_γ'][veh_class_names[veh_class]]["Mean"]
            std_eta_gamma = user_input_data["ATM_Parameters"]['Bike: η_γ'][veh_class_names[veh_class]]["Std"]
            
            # ε_m (Minimum time gap to leader)
            mean_epsilon_m = user_input_data["ATM_Parameters"]['Bike: ε_m'][veh_class_names[veh_class]]["Mean"]
            std_epsilon_m = user_input_data["ATM_Parameters"]['Bike: ε_m'][veh_class_names[veh_class]]["Std"]
            
            # A_w (Wall interaction parameter A)
            mean_aw = user_input_data["ATM_Parameters"]['Bike: A_w'][veh_class_names[veh_class]]["Mean"]
            std_aw = user_input_data["ATM_Parameters"]['Bike: A_w'][veh_class_names[veh_class]]["Std"]
            
            # B_w (Wall interaction parameter B)
            mean_bw = user_input_data["ATM_Parameters"]['Bike: B_w'][veh_class_names[veh_class]]["Mean"]
            std_bw = user_input_data["ATM_Parameters"]['Bike: B_w'][veh_class_names[veh_class]]["Std"]
            
            # A_s (Social interaction parameter A)
            mean_as = user_input_data["ATM_Parameters"]['Bike: A_s'][veh_class_names[veh_class]]["Mean"]
            std_as = user_input_data["ATM_Parameters"]['Bike: A_s'][veh_class_names[veh_class]]["Std"]
            
            # B_s (Social interaction parameter B)
            mean_bs = user_input_data["ATM_Parameters"]['Bike: B_s'][veh_class_names[veh_class]]["Mean"]
            std_bs = user_input_data["ATM_Parameters"]['Bike: B_s'][veh_class_names[veh_class]]["Std"]
            
            # τ (General relaxation time / time constant)
            mean_tau = user_input_data["ATM_Parameters"]['Bike: τ'][veh_class_names[veh_class]]["Mean"]
            std_tau = user_input_data["ATM_Parameters"]['Bike: τ'][veh_class_names[veh_class]]["Std"]
            

            random_tau_gamma = max(0.001, random.gauss(mean_tau_gamma, std_tau_gamma))
            random_v_gamma = max(0.001, random.gauss(mean_v_gamma, std_v_gamma))
            random_a_gamma = max(0.001, random.gauss(mean_a_gamma, std_a_gamma))
            random_b_gamma = max(0.001, random.gauss(mean_b_gamma, std_b_gamma))
            random_eta_gamma = max(0.001, random.gauss(mean_eta_gamma, std_eta_gamma))
            random_epsilon_m = max(0.001, random.gauss(mean_epsilon_m, std_epsilon_m))
            random_aw = max(0.001, random.gauss(mean_aw, std_aw))
            random_bw = max(0.001, random.gauss(mean_bw, std_bw))
            random_as = max(0.001, random.gauss(mean_as, std_as))
            random_bs = max(0.001, random.gauss(mean_bs, std_bs))
            random_tau = max(0.001, random.gauss(mean_tau, std_tau))

            driving_params = np.array([200, random_tau_gamma, random_v_gamma, random_a_gamma, random_b_gamma, random_eta_gamma, 1.2, random_epsilon_m, random_aw, random_bw, random_as, random_bs, random_tau], dtype=np.float64)
            return driving_params
    
        else:
            # Sample from existing data (assuming SF_Bike_data exists and has matching columns)
            picked_row = SF_atm_data.sample(n=1).iloc[0]
            
            bike_params = picked_row[['m_gamma', 'tau_gamma', 'v_gamma0', 'a_gamma_max', 'b_gamma',
       'eta_gamma', 'mu_gamma', 'eps_m', 'A_w', 'B_w', 'A_s', 'B_s', 'T_i']].values
            
            return np.array(bike_params, dtype=np.float64)


    # pt ped
    def sample_PT_Ped():
        if Ped_default == False and Ped_model_name == "PT":
          
            mean_Wc_p= user_input_data["ATM_Parameters"]["w_c p-p"][veh_class_names[veh_class]]["Mean"]
            std_Wc_p = user_input_data["ATM_Parameters"]["w_c p-p"][veh_class_names[veh_class]]["Std"]
            
            mean_Wc_b = user_input_data["ATM_Parameters"]["w_c p-b"][veh_class_names[veh_class]]["Mean"]
            std_Wc_b = user_input_data["ATM_Parameters"]["w_c p-b"][veh_class_names[veh_class]]["Std"]
            
            mean_Wc_bar = user_input_data["ATM_Parameters"]["w_c p_bar"][veh_class_names[veh_class]]["Mean"]
            std_Wc_bar = user_input_data["ATM_Parameters"]["w_c p_bar"][veh_class_names[veh_class]]["Std"]

            mean_eta= user_input_data["ATM_Parameters"]["η_ped"][veh_class_names[veh_class]]["Mean"]
            std_eta = user_input_data["ATM_Parameters"]["η_ped"][veh_class_names[veh_class]]["Std"]
            
            mean_xi = user_input_data["ATM_Parameters"]["ξ_ped"][veh_class_names[veh_class]]["Mean"]
            std_xi = user_input_data["ATM_Parameters"]["ξ_ped"][veh_class_names[veh_class]]["Std"]
            
            mean_tau = user_input_data["ATM_Parameters"]["τ_ped"][veh_class_names[veh_class]]["Mean"]
            std_tau = user_input_data["ATM_Parameters"]["τ_ped"][veh_class_names[veh_class]]["Std"]

            mean_vpref = user_input_data["ATM_Parameters"]["v_desired_ped"][veh_class_names[veh_class]]["Mean"]
            std_vpref = user_input_data["ATM_Parameters"]["v_desired_ped"][veh_class_names[veh_class]]["Std"]
            
    
            # Using a small positive minimum (0.001) for parameters that must be positive.
            random_Wc_p = max(0.001, random.gauss(mean_Wc_p, std_Wc_p))
            random_Wc_b = max(0.001, random.gauss(mean_Wc_b, std_Wc_b))
            random_Wc_bar = max(0.001, random.gauss(mean_Wc_bar, std_Wc_bar))
            random_eta = max(0.001, random.gauss(mean_eta, std_eta))
            random_xi = max(0.001, random.gauss(mean_xi, std_xi))
            random_tau = max(0.001, random.gauss(mean_tau, std_tau))
            random_vpref = max(1, random.gauss(mean_vpref, std_vpref))

            driving_params = np.array([random_Wc_p, random_Wc_b, random_Wc_bar,random_eta, random_xi, random_tau,random_vpref], dtype=np.float64)
            return driving_params

        else:
            # Sample from existing data (assuming SF_Ped_data exists and has matching columns)
            picked_row = PT_atm_data.sample(n=1).iloc[0]
            # NOTE: You must replace the column names below with the actual column names in SF_Ped_data
            ped_params = picked_row[[ 'Wc_pp', 'Wc_pb', 'Wc_pbar', 'eta_ped', 'xi_ped', 'tau_ped', 'v_pref_ped']].values
            
            return np.array(ped_params, dtype=np.float64)

    # finally sample PT_bike
    def sample_PT_Bike():
        if Bike_default == False and Bike_model_name == "PT":
          
            mean_Wc_p= user_input_data["ATM_Parameters"]["w_c b-p"][veh_class_names[veh_class]]["Mean"]
            std_Wc_p = user_input_data["ATM_Parameters"]["w_c b-p"][veh_class_names[veh_class]]["Std"]
            
            mean_Wc_b = user_input_data["ATM_Parameters"]["w_c b-b"][veh_class_names[veh_class]]["Mean"]
            std_Wc_b = user_input_data["ATM_Parameters"]["w_c b-b"][veh_class_names[veh_class]]["Std"]
            
            mean_Wc_bar = user_input_data["ATM_Parameters"]["w_c b_bar"][veh_class_names[veh_class]]["Mean"]
            std_Wc_bar = user_input_data["ATM_Parameters"]["w_c b_bar"][veh_class_names[veh_class]]["Std"]

            mean_eta= user_input_data["ATM_Parameters"]["η_bike"][veh_class_names[veh_class]]["Mean"]
            std_eta = user_input_data["ATM_Parameters"]["η_bike"][veh_class_names[veh_class]]["Std"]
            
            mean_xi = user_input_data["ATM_Parameters"]["ξ_bike"][veh_class_names[veh_class]]["Mean"]
            std_xi = user_input_data["ATM_Parameters"]["ξ_bike"][veh_class_names[veh_class]]["Std"]
            
            mean_tau = user_input_data["ATM_Parameters"]["τ_bike"][veh_class_names[veh_class]]["Mean"]
            std_tau = user_input_data["ATM_Parameters"]["τ_bike"][veh_class_names[veh_class]]["Std"]

            mean_vpref = user_input_data["ATM_Parameters"]["v_desired_bike"][veh_class_names[veh_class]]["Mean"]
            std_vpref = user_input_data["ATM_Parameters"]["v_desired_bike"][veh_class_names[veh_class]]["Std"]
            
    
            # Using a small positive minimum (0.001) for parameters that must be positive.
            random_Wc_p = max(0.001, random.gauss(mean_Wc_p, std_Wc_p))
            random_Wc_b = max(0.001, random.gauss(mean_Wc_b, std_Wc_b))
            random_Wc_bar = max(0.001, random.gauss(mean_Wc_bar, std_Wc_bar))
            random_eta = max(0.001, random.gauss(mean_eta, std_eta))
            random_xi = max(0.001, random.gauss(mean_xi, std_xi))
            random_tau = max(0.001, random.gauss(mean_tau, std_tau))
            random_vpref = max(1, random.gauss(mean_vpref, std_vpref))

            driving_params = np.array([random_Wc_p, random_Wc_b, random_Wc_bar,random_eta, random_xi, random_tau,random_vpref], dtype=np.float64)
            return driving_params

        else:
            # Sample from existing data (assuming SF_Ped_data exists and has matching columns)
            picked_row = PT_atm_data.sample(n=1).iloc[0]
            # NOTE: You must replace the column names below with the actual column names in SF_Ped_data
            bike_params = picked_row[[ 'Wc_bp', 'Wc_bb', 'Wc_bbar', 'eta_bike', 'xi_bike', 'tau_bike', 'v_pref_bike']].values
            
            return np.array(bike_params, dtype=np.float64)
        
    ####################################### Vehicle Generation ###############################################################
    # function to generate vehicles
    
    # generate the agents based on user input demand

    # --- CONSTANTS FOR AGENT TYPES ---
    PED_TYPE_ID = "pedestrian_type"
    BIKE_TYPE_ID = "bicycle_type"
    # --------------------------------------------------------------------------
    
    def generate_agents(net_file_name, route_file_name):
        
        # 1. INITIALIZATION
        origins = []
        destinations = []
        gen_times = []
        are_vehs = []  # 1: Car/Truck, 0: Pedestrian, -2: Bike
    
        agent_type_by_id = {}
        tech_type_by_id = {}
        veh_destinations_by_id={}
        veh_origins_by_id = {}
        ped_ods = {}
        bike_ods = {}
        bike_dest_bounds = {}
        # bike_params
        
        ped_SF_params_by_id = {}
        ped_PT_params_by_id = {}
        bike_SF_params_by_id = {}
        bike_PT_params_by_id = {}
        # vehicle params
        IDM_params_by_id = {}
        PT_params_by_id = {}
        MOBIL_params_by_id ={}
        DDM_params_by_id = {}
        

        # 2. TRAFFIC DEMAND GENERATION (CARS/TRUCKS)
        for in_idx, in_name in enumerate(in_names):
            LT_name = LT_names[in_idx]
            RT_name = RT_names[in_idx]
            Through_name = Through_names[in_idx]
            
            LT_demand = LT_ratios[in_idx] * in_flows[in_idx]
            RT_demand = RT_ratios[in_idx] * in_flows[in_idx]
            Through_demand = (1 - LT_ratios[in_idx] - RT_ratios[in_idx]) * in_flows[in_idx]
    
            # Helper function to generate and filter times
            def generate_times(demand):
                if demand <= 0:
                    return np.array([])
                
                # Assuming min_hw is the minimum head-way (time)
                times = np.cumsum(np.maximum(np.random.exponential(3600 / demand, 100000), min_hw))
                return np.round(times[(times >= min_t) & (times <= max_t)], 1)
    
            # Generate times for all vehicle movements
            lt_times = generate_times(LT_demand)
            rt_times = generate_times(RT_demand)
            thru_times = generate_times(Through_demand)
    
            # Collect data for final sorting
            for t in lt_times:
                origins.append(in_name); destinations.append(LT_name); gen_times.append(t); are_vehs.append(1)
                
            for t in rt_times:
                origins.append(in_name); destinations.append(RT_name); gen_times.append(t); are_vehs.append(1)
            for t in thru_times:
                origins.append(in_name); destinations.append(Through_name); gen_times.append(t); are_vehs.append(1)
    
        # 3. PEDESTRIAN DEMAND GENERATION
        if ped_od_flow > 0: 
            
            for e_idx in range(len(all_ped_in_edges)):
                in_edge = all_ped_in_edges[e_idx]
                out_edge = all_ped_out_edges[e_idx]

                ped_gen_times = np.cumsum(np.maximum(np.random.exponential(3600 / ped_od_flow, 100000), 0.001))
                ped_gen_times = np.round(ped_gen_times[(ped_gen_times >= min_t) & (ped_gen_times <= max_t)], 1)
                for gen_t in ped_gen_times:
                    origins.append(in_edge); destinations.append(out_edge); gen_times.append(gen_t); are_vehs.append(0)
    
        # 4. BIKE DEMAND GENERATION (AS PERSONS WITH BIKE VTYPE)
        if bike_od_flow > 0:
            

            for e_idx in range(len(all_bike_in_edges)):
                bike_gen_times = np.cumsum(np.maximum(np.random.exponential(3600 / bike_od_flow, 100000), 0.001))
                bike_gen_times = np.round(bike_gen_times[(bike_gen_times >= min_t) & (bike_gen_times <= max_t)], 1)
                in_edge = all_bike_in_edges[e_idx]
                out_edge = all_bike_out_edges[e_idx]
                for gen_t in bike_gen_times:
                    origins.append(in_edge); destinations.append(out_edge); gen_times.append(gen_t); are_vehs.append(-2)
                    #print(in_edge, out_edge, gen_t)
    
        # 5. SORT AND ASSIGN IDS
        origins = np.array(origins)
        destinations = np.array(destinations)
        gen_times = np.array(gen_times)
        are_vehs = np.array(are_vehs)
        
        sort_indices = np.argsort(gen_times)
        gen_times = gen_times[sort_indices]
        origins = origins[sort_indices]
        destinations = destinations[sort_indices]
        are_vehs = are_vehs[sort_indices]
    
        agent_ids = np.array([idx + 1 for idx in range(len(gen_times))])
    
        # 6. WRITE XML FILE
        root = ET.Element("routes")

        # Match freeway_sim: SUMO picks insertion lane from `from` that fits the route to `to`.
        _depart_lane = str(user_input_data.get("Depart_Lane", "best")).strip()
        if _depart_lane not in ("best", "random", "free", "allowed", "first"):
            _depart_lane = "best"

        # --- Define Generic Types First (For Pedestrians and Bikes) ---
    
        # Default Pedestrian vType (vClass="pedestrian")
        ET.SubElement(root, "vType", {
            "id": PED_TYPE_ID,
            "vClass": "pedestrian", 
            "maxSpeed": "1.39", # Average walking speed (m/s)
            # More realistic body footprint (also makes them look larger in GUI)
            "length": "0.5",
            "width": "0.6",
            "guiShape": "pedestrian"
        })
    
        # Default Bicycle vType (vClass="bicycle") - used for persons if they are modeled as persons that walk/use bike lanes
        # NOTE: It's better to model bikes as proper vehicles (vClass="bicycle") using <trip> for road movement.
        # However, since the original code used <person>, we define a type for that person.
    

        ET.SubElement(root, "vType", {
            "id": BIKE_TYPE_ID,
            "vClass": "bicycle",
            "maxSpeed": "5.55",     # ~20 km/h
            "length": "1.8",
            "width": "0.6",
            "guiShape": "bicycle"
        })
    
        # --- Generate Agent Definitions (Trips and Persons) ---
    
        for i in range(len(agent_ids)):
            agent_id = str(agent_ids[i])
            gen_t = str(gen_times[i])
            origin = str(origins[i])
            destination = str(destinations[i])
    
            if are_vehs[i] == 1:  # CARS/TRUCKS (vClass="passenger" or "truck")
                veh_destinations_by_id[agent_id] = destination
                veh_origins_by_id[agent_id] = origin 
                veh_class_idx = np.random.choice(
                    [0, 1, 2, 3, 4],
                    p=[SV_rate, AV_rate, HV_rate, CAV_rate, CAHV_rate],
                )
                agent_type_by_id[agent_id] = veh_class_idx
                tech_type_by_id[agent_id] = (
                    "SV" if veh_class_idx == 0 else
                    "AV" if veh_class_idx == 1 else
                    "HV" if veh_class_idx == 2 else
                    "CAV" if veh_class_idx == 3 else
                    "CAHV"
                )

                idm_params = sample_IDM(veh_class_idx)
                PT_params = sample_PT(veh_class_idx)
                MOBIL_params = sample_MOBIL()
                DDM_params = sample_DDM()

                # Clamp aggressive IDM samples (prevents rear-end crashes / unrealistic following)
                try:
                    T, a, b, v0, s0, delta = idm_params
                    T = max(float(T), float(IDM_MIN_T))
                    b = max(float(b), float(IDM_MIN_B))
                    s0 = max(float(s0), float(IDM_MIN_S0))
                    idm_params = np.array([T, float(a), b, float(v0), s0, float(delta)], dtype=np.float64)
                except Exception:
                    pass
                
                IDM_params_by_id[str(agent_id)] = idm_params
                PT_params_by_id[str(agent_id)] = PT_params
                MOBIL_params_by_id[str(agent_id)] = MOBIL_params
                DDM_params_by_id[str(agent_id)] = DDM_params
                
                
                # Dynamic vType definition for car-following model (IDM)
                
                T, a, b, v0, s0, delta = idm_params
                veh_len = veh_lens[veh_class_idx]
                guishape = "truck" if veh_len > 10 else "passenger"
                    
                    # Define vType for this specific vehicle/driver combination
                ET.SubElement(root, "vType", {
                        "id": agent_id, # Use agent ID as type ID since params are unique
                        "accel": str(a),
                        "decel": str(b),
                        "length": str(veh_len),
                        "minGap": str(s0),
                        "tau": str(T),
                        "maxSpeed": str(v0),
                        "carFollowModel": "IDM",
                        "guiShape": guishape,
                        "vClass": "truck" if veh_len > 10 else "passenger"
                })
                    
                # Create the trip element
                ET.SubElement(root, "trip", {
                        "id": agent_id,
                        "type": agent_id, 
                        "depart": gen_t,
                        "from": origin,
                        "to": destination,
                        "departLane": _depart_lane,
                        "departSpeed": "max" 
                })
    
            elif are_vehs[i] == 0:  # PEDESTRIANS (using <person>)
                #print(gen_t)
                agent_type_by_id[agent_id] = -1
                
            
                # Depart position on the sidewalk of the origin edge
                depart_pos = np.random.uniform(0, lens_by_edge.get(origin, 100))
                
                # Create the person element, referencing the pedestrian vType
                person = ET.SubElement(root, "person", {
                    "id": agent_id,
                    "type": PED_TYPE_ID, # Assign the generic pedestrian type
                    "depart": gen_t,
                    "departPos": str(depart_pos),
                    "departSpeed": str(np.random.uniform(0.5, 3)), # Initial speed
                })
                
                # Define the walk stage
                ET.SubElement(person, "walk", {
                    "from": origin,
                    "to": destination,
                    "departPosLat": str(np.random.uniform(-2, 2)), # Lateral position on sidewalk/walkingarea
                })
                
                # Store O/D and movement parameters
                destination_boundary_xs, destination_boundary_ys = destination_boundary_pts.get(destination, ([0], [0]))
                random_dest_idx = np.random.randint(0, len(destination_boundary_xs))
                
                ped_ods[agent_id] = [origin, destination, destination_boundary_xs[random_dest_idx], destination_boundary_ys[random_dest_idx]]
               
                ped_SF_params_by_id[agent_id] = sample_SF_Ped()
                ped_PT_params_by_id[agent_id] = sample_PT_Ped()
    
    
            elif are_vehs[i] == -2:  # BICYCLES (using <person> or <trip>)
                # Note: Modeling bikes as <person> with <walk> is unusual unless they are 
                # restricted to pedestrian areas. If they are using dedicated bike lanes, 
                # they should be modeled as <trip> with vClass="bicycle".
                # Sticking to the original <person> structure, but using the bike vType.
                
                agent_type_by_id[agent_id] = -2
            
                # Depart position
                depart_pos = np.random.uniform(0, lens_by_edge.get(origin, 100))
    
                # Create the person element, referencing the bike vType
                ET.SubElement(root, "trip", {
                    "id": agent_id,
                    "type": BIKE_TYPE_ID,     # <-- bicycle vType
                    "depart": gen_t,
                    "from": origin,
                    "to": destination,
                    "departLane": _depart_lane,
                    "departSpeed": "max",
                })
                
                # Store O/D and movement parameters
                
                destination_boundary_xs = bike_destination_bound_xs[destination]
                destination_boundary_ys = bike_destination_bound_ys[destination]
                origin_edge_boundary_xs = bike_destination_bound_xs[origin] 
                origin_edge_boundary_ys = bike_destination_bound_ys[origin] 
                
                random_dest_idx = np.random.randint(0, len(destination_boundary_xs))
                random_dest_idx0 = np.random.randint(0, len(origin_edge_boundary_xs))
            
                bike_ods[agent_id] = [origin, destination, destination_boundary_xs[random_dest_idx], destination_boundary_ys[random_dest_idx]]
                # changed here:destination by edge instead otherwise they may move out of boundary
                # the ending x and y of the origin edge and destination edge
                bike_dest_bounds[agent_id] = {origin:[origin_edge_boundary_xs[random_dest_idx0], origin_edge_boundary_ys[random_dest_idx0]], destination:[destination_boundary_xs[random_dest_idx], destination_boundary_ys[random_dest_idx]]}
                
                bike_SF_params_by_id[agent_id] = sample_SF_Bike()
                bike_PT_params_by_id[agent_id] = sample_PT_Bike()
                
    
        # 7. SAVE XML
        # Use minidom to pretty-print (compatible with older Python versions)
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t")
        with open(route_file_name, "w", encoding="utf-8") as f:
            f.write(xmlstr)
    
        print(f"Successfully generated {len(agent_ids)} agents and saved to {route_file_name}")
        
        return tech_type_by_id, agent_type_by_id, veh_origins_by_id , veh_destinations_by_id, ped_ods, ped_SF_params_by_id, ped_PT_params_by_id, bike_ods, bike_SF_params_by_id, bike_PT_params_by_id, IDM_params_by_id, PT_params_by_id, MOBIL_params_by_id, DDM_params_by_id, bike_dest_bounds
        
    tech_type_by_id, agent_type_by_id, veh_origins_by_id , veh_destinations_by_id,  ped_ods , ped_SF_params_by_id, ped_PT_params_by_id,  bike_ods , bike_SF_params_by_id, bike_PT_params_by_id, IDM_params_by_id, PT_params_by_id, MOBIL_params_by_id, DDM_params_by_id, bike_dest_bounds = generate_agents(net_file_name, route_file_name)

    
    #################################################### Boundary Points generation for bikes and peds #########################################
    ped_bound_pts ={} # two boundaries
    ped_central_line_pts = {} # two boundaries and a central line
    for id in ped_ods:
        #print(id)
        origin = ped_ods[id][0]
        destination = ped_ods[id][1]
        origin_xbounds = ped_edge_shapes[origin][0]
        origin_ybounds = ped_edge_shapes[origin][1]
        destination_xbounds = ped_edge_shapes[destination][0]
        destination_ybounds = ped_edge_shapes[destination][1]
    
        
        boundary1_xs, boundary2_xs = ped_route_xbounds_by_od[origin][destination]
        boundary1_ys, boundary2_ys = ped_route_ybounds_by_od[origin][destination]
    
        center_xs, center_ys = ped_route_xcenters_by_od[origin][destination], ped_route_ycenters_by_od[origin][destination]
        ped_bound_pts[id] = [boundary1_xs, boundary1_ys, boundary2_xs, boundary2_ys]  # first bound's x and y and then second boundary's.
        ped_central_line_pts[id] = [center_xs, center_ys] # just x and y
    
    '''
    bike_bound_pts ={} # two boundaries
    bike_central_line_pts = {}
    for id in bike_ods:
        #print(id)
        origin = bike_ods[id][0]
        destination = bike_ods[id][1]
        #origin_xbounds = bike_edge_shapes[origin][0]
        #origin_ybounds = bike_edge_shapes[origin][1]
        destination_xbounds = bike_destination_bound_xs[destination]
        destination_ybounds = bike_destination_bound_ys[destination]
    
        
        boundary1_xs, boundary2_xs = ped_route_xbounds_by_od[origin][destination]
        boundary1_ys, boundary2_ys = ped_route_ybounds_by_od[origin][destination]
    
        center_xs, center_ys = ped_route_xcenters_by_od[origin][destination], ped_route_ycenters_by_od[origin][destination]
        bike_bound_pts[id] = [boundary1_xs, boundary1_ys, boundary2_xs, boundary2_ys]  # first bound's x and y and then second boundary's.
        bike_central_line_pts[id] = [center_xs, center_ys] # just x and y
        
    '''
    ################################################### Leader/ Follower Finding Function and Driving Models #######################################
    
    def find_leader(global_pos, lane_idx_, lane_seg, lanes,  global_poses, vs, lengths):
        # lanes is the lanes of all vehicles
        # lane is the 
        if lane_idx_<0:
            return 0, 5, global_pos+500, 50

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
    
    
    ######################################################## Vehicle driving models ######################################################
    
    @njit
    def DDM_LC(DDM_params, direction, is_MLC, adj_gap, adj_lead_v, lead_v ):
        alpha_h, beta_0_left, beta_0_right, beta_G, G_0, beta_V, beta_MLC, sigma =  DDM_params 
        if direction == 1: # left
            beta_0 = beta_0_left
        else:
            beta_0 = beta_0_right
    
        mu = beta_0 + beta_G*np.arctan(adj_gap-G_0) + beta_V*np.arctan(adj_lead_v - lead_v) + beta_MLC*is_MLC
        return mu # drift rate
    
    ## CF models (IDM reference + PT on deviations; aligned with freeway_sim)
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
    a0 = 1

    def PT_plot(PT_params, v, v_leader, x, x_leader, length, length_leader): #
        # Legacy debug plot (absolute accel grid; not used by PT_relative_IDM runtime)
        an_vals = np.linspace(acc_min, acc_max, int((acc_max - acc_min) / acc_step_size) + 1)
        #Tmax, Alpha, Beta, Wc, Gamma1,Gamma2, Wm = PT_params
        tau_max, alpha_v, beta_PT, wc, gamma1, gamma2, wm = PT_params

        # utility of acc when there is no collision
        U_acc_plus = (  0.5*wm + 0.5*(1-wm)*( np.tanh(an_vals/a0)+1 )  ) * (an_vals/a0) * ( 1 + (an_vals/a0)**2 )**(0.5*gamma1-0.5)  
        U_acc_minus = (  0.5*wm + 0.5*(1-wm)*( np.tanh(an_vals/a0)+1 )  ) * (an_vals/a0) * ( 1 + (an_vals/a0)**2 )**(0.5*gamma2-0.5)
        U_acc = U_acc_plus*(an_vals>=0)*1 +  U_acc_minus*(an_vals<0)*1
        #U_acc = np.minimum(U_acc, 700)
        # collision prob of each possibility
        delta_v = v -  v_leader 
        sn = x_leader - x - 0.5 * (length + length_leader)
        den = alpha_v * max(v_leader, 0.1)
        col_vars = (delta_v + 1/2*an_vals*tau_max - sn/tau_max) / den
        p_cld = np.zeros(len(an_vals))
        for i_c in range(len(col_vars)):
            p_cld[i_c] = 0.5*(1+math.erf(col_vars[i_c]/math.sqrt(2))) 
    
        U_tot = U_acc  - p_cld*wc 

        scaled_U = beta_PT*U_tot
        scaled_U = scaled_U - (max(scaled_U)-700)
        
        choice_weights=np.exp(scaled_U)
        ps = (choice_weights+0.0000001)/sum(choice_weights+0.0000001)
    
        acc = sum(ps*an_vals)

        print("PT_params", PT_params)
        
        plt.plot(an_vals, ps, label="p_tot")
        plt.plot(an_vals, p_cld, label="p_cld")
        plt.legend()
        plt.show()
    #'''
    ################################################## Ped models ############################################################
    cutoff_radius=10 # say the ped considers other objects within this amount
    # model params for peds
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
        rx=r_ab[0]
        ry=r_ab[1]
        r = math.hypot(rx, ry)
        # mke sure this is not too small:
        min_r = 0.2
        if r<min_r:
            s = min_r / (r + 1e-12)
            rx*= s
            ry*= s
            r_ab = (rx, ry)
            r =0.2
            
        speed_b = np.hypot(vx_b, vy_b)
        if speed_b < 1e-9:
            e_b = (0.0, 0.0)
        else:
            e_b = (vx_b/speed_b, vy_b/speed_b)
        eps = 1e-5 
        def potential(rx, ry):
            b_val = elliptical_b_value((rx, ry), speed_b, e_b, delta_t)
            return A_pp * np.exp(- b_val / (B_pp + 1e-9))
        pot0 = potential(r_ab[0], r_ab[1])
        pot_px = potential(r_ab[0] + eps, r_ab[1])
        dV_dx = (pot_px - pot0) / eps
        pot_py = potential(r_ab[0], r_ab[1] + eps)
        dV_dy = (pot_py - pot0) / eps
        fx = - dV_dx
        fy = - dV_dy
        return fx, fy

    # SF for pedestrian
    def compute_ped_accel_SF(px, py, vx, vy, dest_x, dest_y, neighbors, bound1_xs, bound1_ys, bound2_xs, bound2_ys, center_xs, center_ys, ped_params):
        # now there is two boundaries
        vAlpha0_ped, tauAlpha_ped, A_pp, B_pp, A_wall_ped, B_wall_ped = ped_params
        m_ped = 80
        speed_obs = math.hypot(vx, vy)
        dist_to_dest = np.sqrt((dest_x-px)**2+(dest_y-py)**2)+10**(-9)
        ex = (dest_x-px)/dist_to_dest
        ey = (dest_y-py)/dist_to_dest
        
        ax_drive = (vAlpha0_ped * ex - vx) / tauAlpha_ped #* m_ped   #### here there should not be weight if we are normalizing by weight
        ay_drive = (vAlpha0_ped * ey - vy) / tauAlpha_ped #* m_ped
        ax_pp, ay_pp = 0.0, 0.0
        
        nxs, nys, vxs_b, vys_b, nb_types = neighbors  ############# what are these (nx_ and ny_) neighbor x and neighbor y
        # ignore cars
        rel_nxs = nxs[nb_types<0]
        rel_nys = nys[nb_types<0]
        rel_vxs_b = vxs_b[nb_types<0]
        rel_vys_b = vys_b[nb_types<0]
    
        if len(rel_vys_b)>0:
            for nei_i in range(len(rel_vys_b)):
                fx, fy = elliptical_ped_repulsion(px, py, rel_nxs[nei_i], rel_nys[nei_i], rel_vxs_b[nei_i], rel_vys_b[nei_i], A_pp, B_pp)
                ax_pp += fx
                ay_pp += fy
        
        all_bound_xs = [bound1_xs, bound2_xs] # both boundaries (if )
        all_bound_ys = [bound1_ys, bound2_ys]
        dists_to_bound = []
        # new logic: calculate which bound the ped is closer to, and then get the distance to the center line
        
        for b_i in range(len(all_bound_xs)):
            bound_xs = all_bound_xs[b_i]
            bound_ys = all_bound_ys[b_i]
            dist = float(np.sqrt(np.min((bound_xs-px)**2 + (bound_ys-py)**2))) # np.sqrt(min((bound_xs-px)**2+(bound_ys-py)**2))
            dists_to_bound.append(dist)
    
        d_to_wall = min(dists_to_bound)
        fmag =  A_wall_ped* math.exp(- d_to_wall / max(B_wall_ped, 1e-9))
        
        # calculate the direction centerline
        '''
        dists_to_center = np.sqrt((px-center_xs)**2+(py-center_ys)**2)
        nearest_center_idx = np.argmin(dists_to_center)
        nearest_center_x = center_xs[nearest_center_idx]
        nearest_center_y = center_ys[nearest_center_idx]
        dist_to_center = dists_to_center[nearest_center_idx]
        '''

        # squared distances (no sqrt needed for ranking)
        d2 = (px - center_xs)**2 + (py - center_ys)**2
        
        # indices of 20 nearest center points, which dampens the jump
        k = 5
        nearest_idxs = np.argpartition(d2, k)[:k]
        
        # average their positions
        nearest_center_x = np.mean(center_xs[nearest_idxs])
        nearest_center_y = np.mean(center_ys[nearest_idxs])
        
        # distance to the averaged center
        dx = nearest_center_x - px
        dy = nearest_center_y - py
        dist_to_center = math.hypot(dx, dy) + 1e-6

        dist_to_center_thresh = 0.5
        if dist_to_center>dist_to_center_thresh: 
            ax_wall = ((nearest_center_x-px)/dist_to_center)*fmag
            ay_wall = ((nearest_center_y-py)/dist_to_center)*fmag
        else:
            ax_wall = 0
            ay_wall =0
            
        #print("fmag", fmag, A_wall_ped)
        #print("ped wall", ax_wall, ay_wall)
        #print("ped_pp", ax_pp, ay_pp)
        #print("ped_drive", ax_drive, ay_drive)
    
        ax = ax_drive + 0.1*ax_pp + 0.1* ax_wall ####################### The force from wall always assumed to be zero?
        ay = ay_drive + 0.1*ay_pp + 0.1*ay_wall
        acc_x = ax/m_ped
        acc_y = ay/m_ped

        speed = math.hypot(vx, vy)
        a_val = math.hypot(acc_x, acc_y)

        '''
        if a_val > 1e-9 and speed > 1e-9:
            cap = 0.3 * speed / max(t_step, 1e-9)
            if a_val > cap:
                scale = cap / a_val
                acc_x *= scale; acc_y *= scale
        '''
        return acc_x, acc_y
    

    DT = 0.05 # look ahead time
    def v_gamma_safe(g_gamma, v_gamma, v_delta, b_gamma, b_delta, eta_gamma):
        inside = b_gamma**2 * eta_gamma**2 + b_gamma * (2*g_gamma - eta_gamma*v_gamma + (v_delta**2)/max(b_delta,1e-9))
        if inside < 0: return 0.0
        return -b_gamma * eta_gamma + math.sqrt(inside)   

    # SF for bikes
    def compute_bike_accel_SF(px, py, vx, vy, dest_x, dest_y, neighbors, bound1_xs, bound1_ys, bound2_xs, bound2_ys, center_xs, center_ys, bike_params):
        # now there is two boundaries
        # for neighbor types, -1 is ped and -2 is bike, 
        #print("running_bike_sf")
        m_gamma, tau_gamma, v_gamma0, a_gamma_max, b_gamma, eta_gamma, mu_gamma, eps_m, A_w, B_w, A_s, B_s, T_i = bike_params
        x_neis, y_neis, vx_neis, vy_neis, type_neis = neighbors
        
        speed_obs = np.hypot(vx, vy)
        
        dist_to_dest = np.sqrt((dest_x-px)**2+(dest_y-py)**2)+10**(-9)
        ex = (dest_x-px)/dist_to_dest
        ey = (dest_y-py)/dist_to_dest
        
        fx_drive = (m_gamma/tau_gamma) * (v_gamma0*ex - vx) 
        fy_drive = (m_gamma/tau_gamma) * (v_gamma0*ey - vy) 
        fx_sum, fy_sum = fx_drive, fy_drive
    
        nei_bike_idxes = np.where( (type_neis == -2) )[0]
        nei_ped_idxes = np.where( (type_neis == -1) )[0]
        
        nei_bike_xs = x_neis[nei_bike_idxes]
        nei_bike_ys = y_neis[nei_bike_idxes]
        nei_bike_vxs = vx_neis[nei_bike_idxes]
        nei_bike_vys = vy_neis[nei_bike_idxes]
        
        nei_ped_xs = x_neis[nei_ped_idxes]
        nei_ped_ys = y_neis[nei_ped_idxes]
        nei_ped_vxs = vx_neis[nei_ped_idxes]
        nei_ped_vys = vy_neis[nei_ped_idxes]
        
        # check bikes in the front
        front_bike_idxes = np.where( (type_neis == -2) & (vx*vx_neis + vy*vy_neis>0) )[0]
        front_bike_xs = x_neis[nei_bike_idxes]
        front_bike_ys = y_neis[nei_bike_idxes]
        front_bike_vxs = vx_neis[nei_bike_idxes]
        front_bike_vys = vy_neis[nei_bike_idxes]
        
         
        ##### handle the closest front bike
        if len(front_bike_xs) > 0:  
            closest_bike_idx = np.argmin((front_bike_xs-px)**2 + (front_bike_ys-py)**2 )
            nei_bike_x = front_bike_xs[closest_bike_idx]
            nei_bike_y = front_bike_ys[closest_bike_idx]
            nei_bike_vx = front_bike_vxs[closest_bike_idx]
            nei_bike_vy = front_bike_vys[closest_bike_idx]
            v_f = np.hypot(nei_bike_vx, nei_bike_vy)
            dist_to_bike = np.sqrt((nei_bike_x-px)**2+(nei_bike_y-py)**2)
            g_gamma = max(dist_to_bike, 0.1)
    
            v_safe = v_gamma_safe(g_gamma, speed_obs, v_f, b_gamma, b_gamma, eta_gamma) 
            f_att = (v_safe - speed_obs) /tau_gamma
            f_att = max(min(f_att, a_gamma_max), -a_gamma_max) # should it be min?
            fx_sum = fx_sum + f_att * ex
            fy_sum = fy_sum + f_att * ey
    
        # handle other bikes
        if len(nei_bike_xs)>0:
            rxs, rys = px - nei_bike_xs, py - nei_bike_ys
    
            R1s = np.hypot(rxs, rys)
            dvxs, dvys = vx - nei_bike_vxs, vy - nei_bike_vys
            
            R2s = np.hypot(rxs - dvxs*DT, rys - dvys*DT)
            
            dist_dvs = np.hypot(dvxs, dvys)
            insides = (R1s + R2s)**2 - (dist_dvs * DT)**2
    
            B_gds = (insides >= 0)*0.5*np.sqrt(insides)
            f_mags = A_s * np.exp(- B_gds / max(B_s, 1e-9))
            
            fx_mags = f_mags*(rxs/R1s)
            fy_mags = f_mags*(rys/R1s)
            
            fx_sum = fx_sum+sum(fx_mags)
            fy_sum = fy_sum+sum(fy_mags)
    
        # handle peds
        if len(nei_ped_xs)>0: # should parallelize it in the future
            for pi in range(len(nei_ped_xs)):
                nx, ny, nvx, nvy = nei_ped_xs[pi], nei_ped_ys[pi], nei_ped_vxs[pi], nei_ped_vys[pi]
                v_relx, v_rely = vx - nvx, vy - nvy
                vlen2 = v_relx*v_relx + v_rely*v_rely
                if vlen2 > 1e-12:
                    cx, cy = (nx - px), (ny - py)
                    t_star = max(0.0, - (cx*v_relx + cy*v_rely) / vlen2)
                    lx = cx + t_star*v_relx; ly = cy + t_star*v_rely
                    if lx*lx + ly*ly <= (ped_radius*2)**2: # not sure what is here
                        rel_speed = math.sqrt(vlen2)
                        dist = math.hypot(px-nx, py-ny)
                        if dist > 1e-9:
                            nxp, nyp = (px-nx)/dist, (py-ny)/dist
                            f_ga = rel_speed / max(T_i, 1e-9)
                            fx_sum += f_ga * nxp; fy_sum += f_ga * nyp
    
        # handle boundary
        all_bound_xs = [bound1_xs, bound2_xs] # both boundaries (if )
        all_bound_ys = [bound1_ys, bound2_ys]
        dists_to_bound = []
        
        for b_i in range(len(all_bound_xs)):
            bound_xs = all_bound_xs[b_i]
            bound_ys = all_bound_ys[b_i]
            dist = np.sqrt(min((bound_xs-px)**2+(bound_ys-py)**2))
            dists_to_bound.append(dist)
    
        d_to_wall = min(dists_to_bound)
        fmag =  A_w* math.exp(- d_to_wall / max(B_w, 1e-9))
        # calculate the direction centerline
        dists_to_center = np.sqrt((px-center_xs)**2+(py-center_ys)**2+0.0001)
        nearest_center_idx = np.argmin(dists_to_center)
        nearest_center_x = center_xs[nearest_center_idx]
        nearest_center_y = center_ys[nearest_center_idx]
        dist_to_center = dists_to_center[nearest_center_idx]
        if dist_to_center>0.1:
            
            fx_center = ((nearest_center_x-px)/dist_to_center)*fmag
            fy_center = ((nearest_center_y-py)/dist_to_center)*fmag
        else:
            fx_center = 0
            fy_center = 1
        fx_sum = fx_sum + fx_center
        fy_sum = fy_sum + fy_center
    
        return fx_sum/m_gamma, fy_sum/ m_gamma

    # choose PT for both bikes and peds, same formulation
    @njit()
    def choose_pt_acceleration(px, py, vx, vy, dest_x, dest_y, pvec,
                                    neighbors, bound_xs, bound_ys, dt, tp):
        # Use collision weights (even though boundaries are ignored)
        Wc_p, Wc_b, Wc_bar, eta, xi, tau, v_pref = pvec # the collision with ped bike, bound
        if tp==-1:
            half_w = ped_radius # change here
        else:
            half_w = bike_radius
        x_neis, y_neis, vx_neis, vy_neis, type_neis = neighbors
    
        curr_speed = math.hypot(vx, vy)
        curr_theta = 0.0 if curr_speed < 1e-9 else math.atan2(vy, vx)
        target_theta = math.atan2(dest_y-py, dest_x-px )   # target angle
    
        speed_candidates = np.linspace(0, 1.2 * v_pref, 7)
        angle_candidates = np.linspace(curr_theta - math.radians(60),
                                       curr_theta + math.radians(60), 9)
        
        best_utility = -1e9
        best_v = curr_speed
        best_th = curr_theta
    
        def subjective_value(v_, th_):
            align = max(0.0, math.cos(th_ - target_theta))
            sp_ratio = v_ / max(1e-9, v_pref)
            base = eta * align
            if base <= 0:
                return 0.0
            exponent = (sp_ratio ** xi)
            return base ** exponent
    
        def collision_cost(px2, py2, nei_xs, nei_ys, nei_types, wall_xs, wall_ys):
            # among wall and ped find the nearest collision
            if len(nei_xs)==0:
                dists_to_nei = np.array([1000000000.0, 1000000000.0]) # ghost neighbor far away
                nei_types = np.array([-1.0, -2.0])
            else:
                dists_to_nei = np.sqrt((px2-nei_xs)**2+(py2-nei_ys)**2)
            dists_to_wall = np.sqrt((px2-wall_xs)**2+(py2-wall_ys)**2)
            
            min_dist_to_nei_idx =  np.argmin(dists_to_nei - 2*half_w*(nei_types<0)) # now assume vehicles and peds are the same
            
            nearest_neigh_type = nei_types[min_dist_to_nei_idx]
            # IMPORTANT: compute clearance against the *nearest* neighbor only (scalar).
            # The previous code used the whole `nei_types` array which made `min_dist_to_nei`
            # an array and later `and/or` comparisons crashed with "truth value ... ambiguous".
            extra_clearance = half_w
            if nearest_neigh_type == -2.0:
                extra_clearance += bike_radius
            elif nearest_neigh_type == -1.0:
                extra_clearance += ped_radius
            min_dist_to_nei = dists_to_nei[min_dist_to_nei_idx] - extra_clearance
            
            # Use np.min for robust scalar reduction (Python `min` can misbehave on non-1D arrays).
            min_dist_to_wall = np.min(dists_to_wall) - half_w
    
            if min_dist_to_wall>0 and min_dist_to_nei>0: # no collision found
                return 0.0
            elif min_dist_to_wall>min_dist_to_nei:
                if nearest_neigh_type==-1: # closer to ped
                    return Wc_p
                else: # closer to bike
                    return Wc_b
            else:
                return Wc_bar
            
            
        for sp_index in prange(len(speed_candidates)):
            sp = speed_candidates[sp_index]
            for ag in angle_candidates:
                px2 = px + sp * math.cos(ag) * dt
                py2 = py + sp * math.sin(ag) * dt
                val = subjective_value(sp, ag)
                
                w_c = collision_cost(px2, py2, x_neis, y_neis, type_neis, bound_xs, bound_ys)
                
                if w_c >0:
                    utility = - w_c
                else:
                    utility = val         
                    
                if utility > best_utility:
                    best_utility = utility
                    best_v = sp
                    best_th = ag
        #print("u", best_utility, best_v, best_th)
    
        vx_fin = best_v * math.cos(best_th)
        vy_fin = best_v * math.sin(best_th)
        ax = (vx_fin - vx) / dt
        ay = (vy_fin - vy) / dt
    
        return ax, ay
    ########################################### Convert Angle function ##############################################
    # in sumo, 0 deg is facing north
    def rad_to_sumo_angle(rad):
        raw_deg = np.degrees(rad)
        sumo_angle = 90.0 - raw_deg
        sumo_angle = np.fmod(sumo_angle + 360.0, 360.0)
        return sumo_angle
    
    def sumo_angle_to_rad(sumo_deg):
        phi_deg = 90.0 - sumo_deg
        phi_deg = np.fmod(phi_deg + 360.0, 360.0)
        return np.radians(phi_deg)

    ################ Pos constraints for ped so that they are not oob #############################
    pos_constraints_by_ped_od = {} # x_min x_max, y_min y_max 
    
    for ped_o in ped_route_xbounds_by_od:
        pos_constraints_by_ped_od[ped_o]={}
        for ped_d in ped_route_xbounds_by_od[ped_o]:
            # (Debug plot removed: plt.scatter/show must not run from a worker thread)
            if "EB" in ped_o or "WB" in ped_o:
                min_y = min(ped_route_ybounds_by_od[ped_o][ped_d][0][0], ped_route_ybounds_by_od[ped_o][ped_d][1][0])
                max_y = max(ped_route_ybounds_by_od[ped_o][ped_d][0][0], ped_route_ybounds_by_od[ped_o][ped_d][1][0])
                min_x = min(min(ped_route_xbounds_by_od[ped_o][ped_d][0]), min(ped_route_xbounds_by_od[ped_o][ped_d][1]))
                max_x = max(max(ped_route_xbounds_by_od[ped_o][ped_d][0]), max(ped_route_xbounds_by_od[ped_o][ped_d][1]))
                
            if "NB" in ped_o or "SB" in ped_o:
                min_x = min(ped_route_xbounds_by_od[ped_o][ped_d][0][0], ped_route_xbounds_by_od[ped_o][ped_d][1][0])
                max_x = max(ped_route_xbounds_by_od[ped_o][ped_d][0][0], ped_route_xbounds_by_od[ped_o][ped_d][1][0])
                min_y = min(min(ped_route_ybounds_by_od[ped_o][ped_d][0]), min(ped_route_ybounds_by_od[ped_o][ped_d][1]))
                max_y = max(max(ped_route_ybounds_by_od[ped_o][ped_d][0]), max(ped_route_ybounds_by_od[ped_o][ped_d][1]))
            
            pos_constraints_by_ped_od[ped_o][ped_d] = [min_x, max_x, min_y, max_y]
    ################################################# Run simulation Function ################################################
    print("Starting Sim")
    tree = ET.parse(config_file)
    root = tree.getroot()
    veh_is_initialized={}
    all_evidences_left = {}
    all_evidences_right = {}
    for vid in veh_origins_by_id:
        veh_is_initialized[vid] = False
        all_evidences_left[vid] = []
        all_evidences_right[vid] = []
    
    atm_is_initialized={}
    for id in ped_ods:
        atm_is_initialized[id] = False
    for id in bike_ods:
        atm_is_initialized[id] = False
        
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


    # ---- Communication config (for connected vehicles: CAV + CAHV) ----
    comm_cfg = user_input_data.get("Comm_Params", {"Range": 30.0, "Lookahead": 5, "Latency": 0, "Loss": 0.0})
    if isinstance(comm_cfg, dict):
        comm_cfg = {
            "Range": comm_cfg.get("Range", 30.0),
            "Lookahead": comm_cfg.get("Lookahead", 5),
            "Latency": comm_cfg.get("Latency", 0),
            "Loss": comm_cfg.get("Loss", 0.0),
        }
    bus = ns.CommunicationBus(
        range_m=float(comm_cfg["Range"]),
        m_max=int(comm_cfg["Lookahead"]),
        latency_steps=int(comm_cfg["Latency"]),
        loss_rate=float(comm_cfg["Loss"]),
        connected_tech=("CAV", "CAHV"),
    )

    last_speed = {}
    last_acc = {}

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
    speed_mode_set = {}

    # ----------------------------------------------------------------------
    # SUMO/TraCI compatibility note (IMPORTANT):
    # On older SUMO versions (notably 1.2.0 on Windows), TraCI commands that
    # read or move persons (e.g., person.getPosition / person.moveToXY) can
    # crash SUMO with: "Error: Should not get here!".
    #
    # We therefore disable ALL person TraCI interactions for these versions
    # and let SUMO simulate pedestrians/bikes natively (they will not be
    # included in `all_data` in that case).
    # ----------------------------------------------------------------------
    
    
    person_traci_enabled = True
    try:
        _api_ver, _sumo_ver = traci.getVersion()  # e.g. (20, 'SUMO 1.2.0')
        _sumo_ver_str = str(_sumo_ver)
        if _sumo_ver_str.startswith("SUMO "):
            _ver = _sumo_ver_str.split("SUMO ", 1)[1].strip()
            parts = _ver.split(".")
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            if major == 1 and minor <= 2:
                person_traci_enabled = False
    except Exception:
        # If version detection fails, keep person TraCI enabled.
        person_traci_enabled = True
    t = 0
    step_count = 0
    last_progress = -1
    progress_every = max(1, int(round(1.0 / t_step))) if t_step > 0 else 1
    lc_every_steps = max(1, int(user_input_data.get("LC_Every_Steps", 10)))
    # Effective dt between LC decisions (used inside DDM SDE since evidence updates every N steps now)
    lc_dt = lc_every_steps * t_step
    collect_every_steps = 1
    if collect_data:
        sample_freq = float(user_input_data.get("Sim_DataFreq", t_step))
        collect_every_steps = max(1, int(round(sample_freq / t_step))) if t_step > 0 else 1
    # need to initialize evidence for ddm
    
    if LC_model_name == "DDM":
        A_left_by_id={}
        A_right_by_id={}

    # First car lane index on each approach (single-inter templates: ped=0, bike=1, then vehicles).
    EW_VEH_LANE0 = 2
    NS_VEH_LANE0 = 2

    def next_intersection_distance(road, x, y):
        """Distance (m) to the intersection along the current inbound approach (center at origin)."""
        if road == "EB_West":
            return 0.0 - x
        if road == "WB_East":
            return x - 0.0
        if road == "NB_South":
            return 0.0 - y
        if road.startswith("SB_North"):
            return y - 0.0
        return 1e9

    def get_next_route_edge(vid):
        """Next non-internal edge on the vehicle route after the current road."""
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
        j = idx + 1
        while j < len(r):
            ne = str(r[j])
            if not ne.startswith(":"):
                return ne
            j += 1
        return None

    def required_lane_for_next_move(road, next_edge):
        """
        Dedicated SUMO lane index for the upcoming turn, or None if straight / any lane.
        Uses vehicle-lane offset (EW_VEH_LANE0 / NS_VEH_LANE0) for ped/bike lanes in templates.
        """
        if next_edge is None:
            return None
        ne = str(next_edge)
        if road == "EB_West":
            if ne == "NB_North":
                return EW_VEH_LANE0 + num_EW_lanes - 1
            if ne == "SB_South":
                return EW_VEH_LANE0
            return None
        if road == "WB_East":
            if ne == "NB_North":
                return EW_VEH_LANE0
            if ne == "SB_South":
                return EW_VEH_LANE0 + num_EW_lanes - 1
            return None
        if road == "NB_South":
            if ne.startswith("WB"):
                return NS_VEH_LANE0 + num_NS_lanes - 1
            if ne.startswith("EB"):
                return NS_VEH_LANE0
            return None
        if road.startswith("SB_North"):
            if ne.startswith("EB"):
                return NS_VEH_LANE0 + num_NS_lanes - 1
            if ne.startswith("WB"):
                return NS_VEH_LANE0
            return None
        return None
    
    # now create min and max_bounds so that they do not move oob
    #ped_route_xbounds_by_od[ped_o][ped_d]
    
    #print(pos_constraints_by_ped_od)   
    
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

    while t < sim_time:  # run 100 steps as demo

        if is_running_check is not None and not is_running_check():
            break

        traci.simulationStep()
        t += t_step
        t = np.round(t, 1)

        step_count += 1
        do_lc_step = (step_count % lc_every_steps == 0)
        if progress_cb is not None and (step_count % progress_every == 0 or t >= sim_time):
            pct = int(min(99, max(0, (t / sim_time) * 100)))
            if pct != last_progress:
                progress_cb(pct)
                last_progress = pct

        # save first just for computation purpose
        all_veh_ids = traci.vehicle.getIDList()
        all_veh_id_set = set(all_veh_ids)
        new_veh_ids = all_veh_id_set.difference(veh_subscribed_ids)
        for _vid in new_veh_ids:
            try:
                traci.vehicle.subscribe(_vid, veh_sub_vars)
                veh_subscribed_ids.add(_vid)
            except Exception:
                pass
        veh_subscribed_ids.intersection_update(all_veh_id_set)

        atm_ids = traci.person.getIDList() if person_traci_enabled else [] # here it is only peds
        num_vehs = len(all_veh_ids) # here bikes are also treated as vehs
        num_atms = len(atm_ids)
        tot_num_objs =  num_vehs + num_atms
        
        ids = np.zeros(tot_num_objs).astype(str)
        types = np.zeros(tot_num_objs)
        xs= np.zeros(tot_num_objs)
        ys= np.zeros(tot_num_objs)
        vs= np.zeros(tot_num_objs)
        thetas= np.zeros(tot_num_objs)
        roads= np.zeros(tot_num_objs).astype(str)
        lanes= np.zeros(tot_num_objs).astype(str)
        lengths= np.zeros(tot_num_objs)
        lane_poses=np.zeros(tot_num_objs)
        global_poses=np.zeros(tot_num_objs) # map the pos within lane to a global frame
        lane_indexes = np.zeros(tot_num_objs)  # from the lane name to lane_id

        for i_id in range(len(all_veh_ids)):
            vid = all_veh_ids[i_id]
            sub = traci.vehicle.getSubscriptionResults(vid) or {}
            # Optional: enforce SUMO safety checks even under setSpeed (OFF by default).
            if SET_SPEEDMODE_SAFETY and vid not in speed_mode_set:
                try:
                    traci.vehicle.setSpeedMode(vid, 31)
                except Exception:
                    pass
                speed_mode_set[vid] = True
            veh_key = str(vid)
            if veh_key not in agent_type_by_id:
                # vehicle not from our generated routes (e.g. SUMO duplicate id); mark to skip in motion loop
                types[i_id] = -999
                ids[i_id] = vid
                x, y = sub.get(tc.VAR_POSITION, traci.vehicle.getPosition(vid))
                xs[i_id], ys[i_id] = x, y
                vs[i_id] = float(sub.get(tc.VAR_SPEED, traci.vehicle.getSpeed(vid)))
                thetas[i_id] = float(sub.get(tc.VAR_ANGLE, traci.vehicle.getAngle(vid)))
                edge = sub.get(tc.VAR_ROAD_ID, traci.vehicle.getRoadID(vid))
                lane = sub.get(tc.VAR_LANE_ID, traci.vehicle.getLaneID(vid))
                lane_pos = float(sub.get(tc.VAR_LANEPOSITION, traci.vehicle.getLanePosition(vid)))
                roads[i_id] = edge
                lanes[i_id] = lane
                lengths[i_id] = float(sub.get(tc.VAR_LENGTH, traci.vehicle.getLength(vid)))
                lane_poses[i_id] = lane_pos
                global_poses[i_id] = lane_pos + pos_adjust_by_lanes.get(lane, 0)
                lane_indexes[i_id] = int(sub.get(tc.VAR_LANE_INDEX, int(lane[-1]) if lane else 0))
                continue
            veh_origins_by_id[vid] = 1
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

            # ---- Color once when vehicle is first seen ----
            if vid not in color_set:
                tech_lbl = tech_type_by_id.get(str(vid), "SV")
                try:
                    traci.vehicle.setColor(vid, COLOR_MAP.get(tech_lbl, (255, 255, 255, 255)))
                except Exception:
                    pass
                color_set[vid] = True
            #print(vid, vtype)
            
            global_pos = lane_pos + pos_adjust_by_lanes.get(lane, 0) 
            # for those within the intersection turning return itself (treating turning lanes as its own kind)
            #lane_idx = lane_segment_to_idx[lane]
            lane_idx = int(sub.get(tc.VAR_LANE_INDEX, int(lane[-1])))
            
            # Then append to lists
            ids[i_id] = (vid)
            types[i_id] = (int(agent_type_by_id[veh_key]))
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

        speed_by_vid = {str(all_veh_ids[i]): float(vs[i]) for i in range(len(all_veh_ids))}
        lane_index_by_vid = {str(all_veh_ids[i]): int(lane_indexes[i]) for i in range(len(all_veh_ids))}

        # When controlling ATMs via TraCI, maintain our own speed/heading state
        # to avoid SUMO simultaneously advancing the <walk> stage.
        if person_traci_enabled:
            if 'atm_speed_by_id' not in locals():
                atm_speed_by_id = {}
                atm_theta_by_id = {}

        for i_id in range(len(atm_ids)):
            atm_id = atm_ids[i_id]
            ped_key = str(atm_id)
            if ped_key not in agent_type_by_id and atm_id not in agent_type_by_id:
                continue  # skip agents not from our generated routes
            agent_type = int(agent_type_by_id.get(ped_key, agent_type_by_id.get(atm_id)))
            if VRU_USE_SUMO_DEFAULT:
                x, y = traci.person.getPosition(atm_id)
                speed = float(traci.person.getSpeed(atm_id))
                theta = float(traci.person.getAngle(atm_id))
                edge = traci.person.getRoadID(atm_id)
                lane = traci.person.getLaneID(atm_id)
                lane_pos = traci.person.getLanePosition(atm_id)
                ids[i_id + num_vehs] = atm_id
                types[i_id + num_vehs] = agent_type
                xs[i_id + num_vehs] = x
                ys[i_id + num_vehs] = y
                vs[i_id + num_vehs] = speed
                thetas[i_id + num_vehs] = theta
                roads[i_id + num_vehs] = edge
                lanes[i_id + num_vehs] = lane
                if agent_type == -1:
                    lengths[i_id + num_vehs] = ped_radius * 2
                else:
                    lengths[i_id + num_vehs] = bike_radius * 2
                lane_poses[i_id + num_vehs] = lane_pos
                global_poses[i_id + num_vehs] = np.nan
                lane_indexes[i_id + num_vehs] = 0
                continue
            x, y = traci.person.getPosition(atm_id)
            #print("raw_x_y",pid, x, y)
            if atm_is_initialized[atm_id]==False:
                atm_is_initialized[atm_id]=True
                if agent_type  == -1: # ped
                    speed = np.random.uniform(0.5, 3.0)
                else:
                    speed = np.random.uniform(3, 8) # bike

                # initialize heading towards destination boundary point
                if agent_type == -1:
                    if ped_key not in ped_ods:
                        continue
                    dest_x = ped_ods[ped_key][2]
                    dest_y = ped_ods[ped_key][3]
                else:
                    if ped_key not in bike_ods:
                        continue
                    dest_x = bike_ods[ped_key][2]
                    dest_y = bike_ods[ped_key][3]
                init_theta = rad_to_sumo_angle(math.atan2(dest_y - y, dest_x - x))
                atm_speed_by_id[atm_id] = float(speed)
                atm_theta_by_id[atm_id] = float(init_theta)
                
            else:
                speed = float(atm_speed_by_id.get(atm_id, traci.person.getSpeed(atm_id)))
            
            edge = traci.person.getRoadID(atm_id)
            lane = traci.person.getLaneID(atm_id)
            lane_pos = traci.person.getLanePosition(atm_id)
            theta = float(atm_theta_by_id.get(atm_id, traci.person.getAngle(atm_id)))


            ids[i_id+num_vehs]=atm_id
            # IMPORTANT: use the current ATM id (not last vehicle id)
            types[i_id+num_vehs]=agent_type
            xs[i_id+num_vehs]=(x)
            ys[i_id+num_vehs]=(y)
            vs[i_id+num_vehs]=(speed)
            thetas[i_id+num_vehs]=(theta)
            roads[i_id+num_vehs]=(edge)
            lanes[i_id+num_vehs]=(lane)
            if types[i_id+num_vehs]==-1:
                lengths[i_id+num_vehs]= ped_radius*2
            else:
                lengths[i_id+num_vehs]= bike_radius*2
            lane_poses[i_id+num_vehs]=(lane_pos)
            global_poses[i_id+num_vehs] = np.nan # not useful for peds
            lane_indexes[i_id+num_vehs] = 0
            
        # ---- Share connected states (CAV + CAHV) ----
        states_conn = {}
        for i, vid in enumerate(all_veh_ids):
            if int(types[i]) < 0:
                continue
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

        # loop over all the ids and then update their motion
        #'''
        for i_id in range(len(all_veh_ids)):
            # IDM
            
            # now do ddm
            vid = all_veh_ids[i_id]
            veh_key = str(vid)
             
            #bike_ods[agent_id] = [origin, destination, destination_boundary_xs[random_dest_idx], destination_boundary_ys[random_dest_idx]]
                
            #dest = veh_destinations_by_id[vid]
            x = xs[i_id]
            y = ys[i_id]
            v = vs[i_id]
            lane_idx = lane_indexes[i_id]
            global_pos = global_poses[i_id]
            lane =  str(lanes[i_id])
            length = lengths[i_id]
            road = roads[i_id]
            old_lane_idx = lane_index_by_vid.get(veh_key)
            if old_lane_idx is None:
                old_lane_idx = traci.vehicle.getLaneIndex(vid)
            theta = thetas[i_id]

            if int(types[i_id]) == -999:
                continue  # unknown vehicle not from our routes

            if int(types[i_id]) != -2:
                dest = veh_destinations_by_id.get(veh_key) or veh_destinations_by_id.get(vid)
                if dest is None:
                    continue

                d2i = next_intersection_distance(road, x, y)

                if str(lane).startswith(":"):
                    continue

                if road.startswith("NB") or road.startswith("SB"):
                    _sumo_near_inter = abs(d2i) < HANDOFF_DIST_M
                else:
                    _sumo_near_inter = d2i < HANDOFF_DIST_M
                if _sumo_near_inter:
                    continue

                JUNCTION_CORE = 8.0
                near_core = abs(x) < JUNCTION_CORE and abs(y) < JUNCTION_CORE
                if near_core:
                    continue

                SIGNAL_ZONE = 20.0
                tls_list = traci.vehicle.getNextTLS(vid)
                if tls_list:
                    _tls_id, _tls_link, tls_dist, _tls_state = tls_list[0]
                    if tls_dist < SIGNAL_ZONE:
                        traci.vehicle.setSpeed(vid, -1)
                        continue

                next_e = get_next_route_edge(vid)
                req_lane = required_lane_for_next_move(road, next_e)
                if do_lc_step and req_lane is not None and old_lane_idx != req_lane:
                    traci.vehicle.changeLane(vid, int(req_lane), TRACI_CHANGE_LANE_DURATION_S)
                use_mobil_lc = req_lane is None

                left_lane_exists = 0
                right_lane_exists = 0
                MLC_left = 0
                MLC_right = 0

                li = int(lane[-1])

                if road == "EB_West":
                    if next_e == "NB_North" and li < EW_VEH_LANE0 + num_EW_lanes - 1:
                        MLC_left = 1
                    elif next_e == "SB_South" and li > EW_VEH_LANE0:
                        MLC_right = 1
                elif road == "WB_East":
                    if next_e == "NB_North" and li > EW_VEH_LANE0:
                        MLC_right = 1
                    elif next_e == "SB_South" and li < EW_VEH_LANE0 + num_EW_lanes - 1:
                        MLC_left = 1
                elif road == "NB_South":
                    if next_e is not None and str(next_e).startswith("EB") and li > NS_VEH_LANE0:
                        MLC_right = 1
                    elif next_e is not None and str(next_e).startswith("WB") and li < NS_VEH_LANE0 + num_NS_lanes - 1:
                        MLC_left = 1
                elif road.startswith("SB_North"):
                    if next_e is not None and str(next_e).startswith("EB") and li < NS_VEH_LANE0 + num_NS_lanes - 1:
                        MLC_left = 1
                    elif next_e is not None and str(next_e).startswith("WB") and li > NS_VEH_LANE0:
                        MLC_right = 1

                if "NB" in lane or "SB" in lane:
                    if li > NS_VEH_LANE0:
                        right_lane_exists = 1
                    if li < NS_VEH_LANE0 + num_NS_lanes - 1 and MLC_right == 0:
                        left_lane_exists = 1
                elif "EB" in lane or "WB" in lane:
                    if li > EW_VEH_LANE0:
                        right_lane_exists = 1
                    if li < EW_VEH_LANE0 + num_EW_lanes - 1 and MLC_right == 0:
                        left_lane_exists = 1

                left_lane = lane[:-1] + str(int(lane[-1]) + 1)
                right_lane = lane[:-1] + str(int(lane[-1]) - 1)

                leader_exists, leader_len, leader_global_x, leader_v = find_leader(global_pos, lane_idx, lane, lanes, global_poses, vs, lengths)
                follower_exists, follower_len, follower_global_x, follower_v = find_follower(global_pos, lane_idx, lane, lanes, global_poses, vs, lengths)

                left_leader_exists, left_leader_len, left_leader_global_x, left_leader_v = find_leader(global_pos, lane_idx + 1, left_lane, lanes, global_poses, vs, lengths)
                left_follower_exists, left_follower_len, left_follower_global_x, left_follower_v = find_follower(global_pos, lane_idx + 1, left_lane, lanes, global_poses, vs, lengths)
                right_leader_exists, right_leader_len, right_leader_global_x, right_leader_v = find_leader(global_pos, lane_idx - 1, right_lane, lanes, global_poses, vs, lengths)
                right_follower_exists, right_follower_len, right_follower_global_x, right_follower_v = find_follower(global_pos, lane_idx - 1, right_lane, lanes, global_poses, vs, lengths)

                # ===================== CONNECTED COOPERATIVE CONTROL (CAV + CAHV) =====================
                tech_label = tech_type_by_id.get(str(vid), "SV")

                if tech_label in ("CAV", "CAHV"):
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

                    pcs = bus.get_preceding_connected(ego_state)
                    CF_params = IDM_params_by_id[str(vid)]
                    cidm_cfg = user_input_data.get("CIDM_Params", {"K_v": 0.2, "K_a": 0.05, "s_ref": 50.0})
                    acc = coop.c_idm_accel(
                        CF_params, ego_state, leader_state, pcs, bus,
                        K_v=float(cidm_cfg.get("K_v", 0.2)),
                        K_a=float(cidm_cfg.get("K_a", 0.05)),
                        s_ref=float(cidm_cfg.get("s_ref", 50.0)),
                    )

                    new_v = max(0.0, float(v) + float(acc) * float(t_step))
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

                    if do_lc_step and use_mobil_lc:
                        cm_cfg = user_input_data.get("CMOBIL_Params", {"kappa": 0.1, "gamma": 1.0})
                        kappa = float(cm_cfg.get("kappa", 0.1))
                        gamma = float(cm_cfg.get("gamma", 1.0))

                        Mobil_params = MOBIL_params_by_id[str(vid)]
                        lane_choice = coop.c_mobil_decision(
                            Mobil_params, CF_params,
                            ego_state, leader_state,
                            ns.VehicleState("f", ego_state.lane, float(follower_global_x), float(follower_v), 0.0, float(follower_len), "SV"),
                            ns.VehicleState("ll", str(left_lane), float(left_leader_global_x), float(left_leader_v), 0.0, float(left_leader_len), "SV"),
                            ns.VehicleState("lf", str(left_lane), float(left_follower_global_x), float(left_follower_v), 0.0, float(left_follower_len), "SV"),
                            ns.VehicleState("rl", str(right_lane), float(right_leader_global_x), float(right_leader_v), 0.0, float(right_leader_len), "SV"),
                            ns.VehicleState("rf", str(right_lane), float(right_follower_global_x), float(right_follower_v), 0.0, float(right_follower_len), "SV"),
                            left_lane_exists, right_lane_exists, MLC_left, MLC_right,
                            bus=bus, kappa=kappa, gamma_lc=gamma,
                            left_lane_str=str(left_lane), right_lane_str=str(right_lane),
                        )
                        if lane_choice != 0:
                            traci.vehicle.changeLane(vid, old_lane_idx + lane_choice, TRACI_CHANGE_LANE_DURATION_S)

                    continue
                # ===============================================================================

                # first do car following, then do lane changing
                
                if CF_model_name == "IDM":
                    CF_params = IDM_params_by_id[str(vid)]
                    #print(CF_params)
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
                    #print(acc)
                # test why it goes so slowly
                '''
                if v+acc*t_step < 2 :
                    print( vid, v, acc , x, y, global_pos, leader_global_x )
                    
                    #print( CF_params)
                    PT_plot(CF_params, v, leader_v, global_pos, leader_global_x, length , leader_len)
                '''
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
                        # Use the vehicle's own desired minGap (s0) plus a small buffer
                        try:
                            s0_self = float(IDM_params_by_id[str(vid)][4])
                        except Exception:
                            s0_self = 2.0
                        min_gap_bb = max(0.5, s0_self + float(SAFETY_EXTRA_GAP))
                        # Ensure after one step the bumper-to-bumper gap does not drop below min_gap_bb:
                        # gap_next = gap_bb + leader_speed*dt - new_v*dt >= min_gap_bb
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

                    traci.vehicle.changeLane(vid, old_lane_idx + lane_choice, 1.0)

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
                            traci.vehicle.changeLane(vid, old_lane_idx-1 , 0)
                            del A_right_by_id[vid]

                    elif  A_right_by_id[vid]*right_lane_exists < A_left_by_id[vid]*left_lane_exists and A_left_by_id[vid]>20:
                        if left_follow_gap>min_gap and left_leader_global_x - global_pos - left_leader_len >min_gap:
                            traci.vehicle.changeLane(vid, old_lane_idx+1, 0 )
                            del A_left_by_id[vid]
            else:
                # it is a bike use PT or SF (or SUMO default if VRU_USE_SUMO_DEFAULT)
                if VRU_USE_SUMO_DEFAULT:
                    try:
                        traci.vehicle.setSpeed(vid, -1)
                    except Exception:
                        pass
                    continue
                bike_key = str(vid)
                if bike_key not in bike_ods:
                    continue  # skip bikes not from our generated routes
                dest = bike_ods[bike_key][1] 
                theta_rad = sumo_angle_to_rad(theta)
                # here we already know it is not at an intersection
                bike_o = bike_ods[bike_key][0]
                bike_d = bike_ods[bike_key][1]
                #print(bike_key,bike_o, bike_d)
                #bike_dest_bounds[agent_id] = {origin:[origin_edge_boundary_xs[random_dest_idx0], origin_edge_boundary_ys[random_dest_idx1]], destination:[destination_boundary_xs[random_dest_idx], destination_boundary_ys[random_dest_idx]]}
                
                dest_x = bike_dest_bounds[bike_key][road][0]#bike_ods[bike_key][2]
                dest_y = bike_dest_bounds[bike_key][road][1]#bike_ods[bike_key][3]
                final_dest_x = bike_ods[bike_key][2]
                final_dest_y = bike_ods[bike_key][3]
                #print(road, dest_x, dest_y)

                vx = speed * np.cos(theta_rad)
                vy = speed * np.sin(theta_rad)
                # road can be lane ID (e.g. SB_North_5); bike_* dicts are keyed by edge (e.g. SB_North)
                bike_edge_key = road if road in bike_xbounds_by_edge else (road.rsplit("_", 1)[0] if "_" in road else road)
                if bike_edge_key not in bike_xbounds_by_edge:
                    continue
                bound1_xs, bound2_xs = bike_xbounds_by_edge[bike_edge_key]
                bound1_ys, bound2_ys = bike_ybounds_by_edge[bike_edge_key]

                bound_xs = np.concatenate((bound1_xs, bound2_xs))
                bound_ys = np.concatenate((bound1_ys, bound2_ys))
                center_xs = bike_xcenters_by_edge[bike_edge_key]
                center_ys = bike_ycenters_by_edge[bike_edge_key]

                # consider those on the same lane (match by road for neighbor filter)
                rel_idxes = ((roads == road) & (types == -2))

                rel_nei_ind = (((xs[rel_idxes] - x) ** 2 + (ys[rel_idxes] - y) ** 2 <= cutoff_radius ** 2) &
                           (ids[rel_idxes] != vid))
                v_neis = vs[rel_idxes][rel_nei_ind]
                # thetas are SUMO degrees; convert to radians before cos/sin
                theta_neis_rad = sumo_angle_to_rad(thetas[rel_idxes][rel_nei_ind])
                vx_neis = v_neis * np.cos(theta_neis_rad)
                vy_neis = v_neis * np.sin(theta_neis_rad)
    
                x_neis = xs[rel_idxes][rel_nei_ind]
                y_neis = ys[rel_idxes][rel_nei_ind]
                type_neis = types[rel_idxes][rel_nei_ind].astype(float)
                neighbors = np.array([x_neis, y_neis, vx_neis, vy_neis, type_neis])
                
                if bike_key not in bike_SF_params_by_id or bike_key not in bike_PT_params_by_id:
                    continue
                if Bike_model_name == "SF":
                    bike_params = np.array(bike_SF_params_by_id[bike_key])
                    acc_x, acc_y = compute_bike_accel_SF(
                        x, y, vx, vy, dest_x, dest_y, neighbors,
                        bound1_xs, bound1_ys, bound2_xs, bound2_ys,
                        center_xs, center_ys, bike_params
                    )
                else:  # "PT"
                    pvec = np.array(bike_PT_params_by_id[bike_key])
                    acc_x, acc_y = choose_pt_acceleration(
                        x, y, vx, vy, dest_x, dest_y, pvec,
                        neighbors, bound_xs, bound_ys, 0.1, -2
                    )

                acc_x = np.clip(acc_x, -2, 2)
                acc_y = np.clip(acc_y, -2, 2)
                new_vx = t_step * acc_x + vx
                new_vy = t_step * acc_y + vy
                #new_theta = rad_to_sumo_angle(np.arctan2(new_vy, new_vx))
                new_x = x + vx * t_step + 0.5 * acc_x * t_step ** 2
                new_y = y + vy * t_step + 0.5 * acc_y * t_step ** 2
                new_v = float(np.clip(np.sqrt(new_vx ** 2 + new_vy ** 2), 0, 12))

                destination_reached = False
                if "North" in bike_d and new_y > final_dest_y - dest_tol:
                    destination_reached = True
                elif "South" in bike_d and new_y < final_dest_y + dest_tol:
                    destination_reached = True
                elif "East" in bike_d and new_x > final_dest_x - dest_tol:
                    destination_reached = True
                elif "West" in bike_d and new_x < final_dest_x + dest_tol:
                    destination_reached = True
    
                if destination_reached:
                    if vid in traci.vehicle.getIDList():
                        traci.vehicle.remove( vid )
                else:
                    new_theta = rad_to_sumo_angle(np.arctan2(new_vy, new_vx))
                    #try:
                    traci.vehicle.moveToXY(vid, "", 1, new_x, new_y, angle=new_theta, keepRoute=0)
                    #except Exception as e:
                    #print(f"moveToXY skip for vehicle {vid}: {e}")

        for i_id in range(len(atm_ids)):
            atm_id = atm_ids[i_id]
            idx = i_id + num_vehs
            ped_key = str(atm_id)

            agent_type = int(types[idx])  # -1 ped, -2 bike
            if agent_type not in (-1, -2):
                continue
            # Only peds are in person list (bikes are vehicles); skip if not in our ped routes
            if ped_key not in ped_ods:
                continue

            if VRU_USE_SUMO_DEFAULT:
                try:
                    traci.person.setSpeed(atm_id, -1)
                except Exception:
                    pass
                continue

            speed = vs[idx]
            x = xs[idx]
            y = ys[idx]
            theta = thetas[idx]
            theta_rad = sumo_angle_to_rad(theta)
            road = roads[idx]

            # if it is very close to the cross walk / intersection area, use SUMO
            if x > intersection_x_min and x < intersection_x_max and y > intersection_y_min and y < intersection_y_max:
                # Let SUMO handle movement inside the intersection.
                try:
                    traci.person.setSpeed(atm_id, -1)
                    atm_speed_by_id[atm_id] = float(traci.person.getSpeed(atm_id))
                    atm_theta_by_id[atm_id] = float(traci.person.getAngle(atm_id))
                except Exception:
                    pass
                continue
            else:
                # Outside intersection, fully take control by freezing SUMO walk speed.
                try:
                    traci.person.setSpeed(atm_id, 0)
                except Exception:
                    pass

            
            ped_o = ped_ods[ped_key][0]
            ped_d = ped_ods[ped_key][1]
            dest_x = ped_ods[ped_key][2]
            dest_y = ped_ods[ped_key][3]
                

            vx = speed * np.cos(theta_rad)
            vy = speed * np.sin(theta_rad)
            bound1_xs, bound2_xs = ped_route_xbounds_by_od[ped_o][ped_d]
            bound1_ys, bound2_ys = ped_route_ybounds_by_od[ped_o][ped_d]

            bound_xs = np.concatenate((bound1_xs, bound2_xs))
            bound_ys = np.concatenate((bound1_ys, bound2_ys))
            center_xs = ped_route_xcenters_by_od[ped_o][ped_d]
            center_ys = ped_route_ycenters_by_od[ped_o][ped_d]

            # Step 1: compute squared distances
            dists_sq = (xs[num_vehs:] - x)**2 + (ys[num_vehs:] - y)**2
            
            # Step 2: apply mask
            valid_indices = np.where(
                (dists_sq <= cutoff_radius**2) & (ids[num_vehs:] != atm_id)
            )[0]
            
            # Step 3: select closest 5
            rel_nei_ind = valid_indices#[np.argsort(dists_sq[valid_indices])[:5]]


            v_neis = vs[num_vehs:][rel_nei_ind]
            # thetas are SUMO degrees; convert to radians before cos/sin
            theta_neis_rad = sumo_angle_to_rad(thetas[num_vehs:][rel_nei_ind])
            vx_neis = v_neis * np.cos(theta_neis_rad)
            vy_neis = v_neis * np.sin(theta_neis_rad)

            x_neis = xs[num_vehs:][rel_nei_ind]
            y_neis = ys[num_vehs:][rel_nei_ind]
            type_neis = types[num_vehs:][rel_nei_ind].astype(float)
            neighbors = np.array([x_neis, y_neis, vx_neis, vy_neis, type_neis])

            
            if ped_key not in ped_SF_params_by_id or ped_key not in ped_PT_params_by_id:
                continue
            if Ped_model_name == "SF":
                    ped_params = np.array(ped_SF_params_by_id[ped_key])
                    acc_x, acc_y = compute_ped_accel_SF(
                        x, y, vx, vy, dest_x, dest_y, neighbors,
                        bound1_xs, bound1_ys, bound2_xs, bound2_ys,
                        center_xs, center_ys, ped_params
                    )
            else:  # "PT"
                    pvec = np.array(ped_PT_params_by_id[ped_key])
                    acc_x, acc_y = choose_pt_acceleration(
                        x, y, vx, vy, dest_x, dest_y, pvec,
                        neighbors, bound_xs, bound_ys, 0.1, -1
                    )

            acc_x = np.clip(acc_x, -2, 2)
            acc_y = np.clip(acc_y, -2, 2)
            new_vx = t_step * acc_x + vx
            new_vy = t_step * acc_y + vy
            # restrict the direction here if needed: 
            #if "East" in 
                # NumPy doesn't always expose `np.atan2` (but does expose `np.arctan2`).
            new_theta = rad_to_sumo_angle(np.arctan2(new_vy, new_vx))
            new_x = x + vx * t_step + 0.5 * acc_x * t_step ** 2
            new_y = y + vy * t_step + 0.5 * acc_y * t_step ** 2
            new_v = float(np.clip(np.sqrt(new_vx ** 2 + new_vy ** 2), 0, 5))
           

            destination_reached = False
            if "North" in ped_d and new_y > dest_y - dest_tol:
                destination_reached = True
            elif "South" in ped_d and new_y < dest_y + dest_tol:
                destination_reached = True
            elif "East" in ped_d and new_x > dest_x - dest_tol:
                destination_reached = True
            elif "West" in ped_d and new_x < dest_x + dest_tol:
                destination_reached = True

            if destination_reached:
                if atm_id in traci.person.getIDList():
                    traci.person.remove(atm_id)
            else:
                # keepRoute must be 0/1; 7 can crash SUMO.
                # Use current edge (road) if available to avoid invalid edge matching.
                
                # make sure the ped is within the edge
                # road or 
                #'''
                # out of boundary cases: clip to the bound
                #'''
                ped_min_x, ped_max_x, ped_min_y, ped_max_y = pos_constraints_by_ped_od[ped_o][ped_d]
                
                #print(ped_min_x, ped_max_x, ped_min_y, ped_max_y)
                #'''
                pos_tolerance = 0.2
                if new_x>ped_max_x - pos_tolerance :
                    #print(ped_o, ped_d, "issue x max", new_x, new_y)
                    new_x = ped_max_x - pos_tolerance 
                elif new_x<ped_min_x + pos_tolerance :
                    #print(ped_o, ped_d, "issue x min", new_x, new_y)
                    new_x = ped_min_x + pos_tolerance 
                
                if new_y>ped_max_y - pos_tolerance :
                    #print(ped_o, ped_d, "issue y max", new_x, new_y)
                    new_y = ped_max_y - pos_tolerance 
                elif new_y<ped_min_y + pos_tolerance :
                    #print(ped_o, ped_d, "issue y min", new_x, new_y)
                    new_y = ped_min_y + pos_tolerance 
                #'''
                new_theta_rad = np.arctan2(new_y-y, new_x-x)
                new_theta = rad_to_sumo_angle(new_theta_rad)
                #print(atm_id, new_theta, new_y-y, new_x-x)
                # edgeID must be edge (road), not lane; wrong value can crash SUMO
                #try:
                 # (road or "")
                traci.person.moveToXY(personID=atm_id, edgeID= "", x=new_x, y=new_y, keepRoute=0, angle=new_theta)
                
                #traci.person.setAngle(personID=atm_id, angle=new_theta)
                #print("aCTUAL", traci.person.getAngle(atm_id))
                
                atm_speed_by_id[atm_id] = new_v
                atm_theta_by_id[atm_id] = float(new_theta)
                #except Exception as e:
                    #print(f"moveToXY skip for person {atm_id}: {e}")

        
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
    
