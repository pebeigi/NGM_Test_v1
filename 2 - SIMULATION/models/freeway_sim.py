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
from . import ns
from . import coop_models as coop
import subprocess
import random
from .paths import (
    MODEL_PARAMS_DIR,
    MODELS_DIR,
    TEMPLATES_DIR,
    load_mobil_results_csv,
    mobil_params_from_csv_row,
)

# SUMO TraCI: traci.vehicle.changeLane(..., duration_s): time horizon SUMO uses to pursue /
# mandate the lane (see SUMO TraCI docs). Not the same as LC_NO_RETURN_SEC.
TRACI_CHANGE_LANE_DURATION_S = 0.6
# After moving from lane A to lane B on the same edge, lane A is not a valid LC target for this many seconds.
LC_NO_RETURN_SEC = 3.0


def _downstream_off_dests(exit_key: str, exit_order: list, exit_defs: dict) -> set:
    out = set()
    if exit_key not in exit_order:
        return out
    i = exit_order.index(exit_key)
    for k in exit_order[i + 1 :]:
        out.add(str(exit_defs[k].get("off_dest", "")))
    return {x for x in out if x}


# --- on_off: single off-ramp (Off_Ramp), links Weaving_Area_*, :Weaving_End_0/1, Off_Ramp, Input, On_Ramp, Output ---
def freeway_reroute_on_off(
    off_dest: str,
    dest: str,
    edge: str,
    lane: str,
    edge_lane_i: int,
    signed_d: float,
    v: float,
    t: float,
    vid: str,
    stuck_low_speed_since: dict,
    rerouted_f: bool,
    impos_lookahead: float,
    impos_sec_per_lc: float,
    stuck_time_req: float,
    stuck_low_mps: float,
    weave_reroute: float,
    near_end_of_current_edge: bool,
) -> tuple:
    r, L = edge, lane
    li = int(edge_lane_i)
    on_off_side = on_main_side = False
    if "Output" in r:
        on_main_side = True
    elif "Off_Ramp" in r:
        on_off_side = True
    elif ":Weaving_End_0" in r or ":Weaving_Start_0" in r:
        on_off_side = True
    elif ":Weaving_End_1" in r or ":Weaving_Start_1" in r:
        on_main_side = True
    elif "Weaving_Area" in L:
        try:
            k = int(L.rsplit("_", 1)[-1])
        except Exception:
            k = -1
        on_off_side = k == 0
        on_main_side = k > 0
    elif "Input" in r:
        on_main_side = True
    elif "On_Ramp" in r:
        on_off_side = True
    wrong_exit_lane = (dest == off_dest) and on_main_side
    wrong_through_lane = (dest == "Output") and on_off_side
    wrong_downstream_exit_lane = False
    wrong_lane = wrong_exit_lane or wrong_through_lane
    if rerouted_f or not wrong_lane or not off_dest:
        return (False, "", wrong_exit_lane, wrong_through_lane, wrong_downstream_exit_lane, wrong_lane)
    # Rule 1: impossible
    should = False
    reason = ""
    n_lc = max(1, int(edge_lane_i)) if wrong_exit_lane else 1
    T_need = float(n_lc) * float(impos_sec_per_lc)
    v_now = float(v)
    T_budget = 0.0 if v_now < 1e-3 else max(0.0, signed_d) / v_now
    if signed_d <= 0.0:
        should, reason = True, "past_commit"
    elif signed_d <= impos_lookahead and T_need > T_budget:
        should, reason = True, "infeasible"
    # Rule 2: stuck (names for this net only)
    in_we = (
        ("Weaving_Area" in r)
        or ("Weaving_Area" in L)
        or (":Weaving_End" in r)
        or (":Weaving_Start" in r)
    )
    # Long weaving edges: distance-to-gore (signed_d) can still be > Weave_Reroute_Dist while stuck in idx 0.
    _d_stuck = max(float(weave_reroute), float(impos_lookahead) * 0.5)
    near_stuck = ((signed_d > 0) and (signed_d <= _d_stuck) and in_we) or (
        near_end_of_current_edge and (("Off_Ramp" in r) or in_we)
    )
    try:
        w = float(traci.vehicle.getWaitingTime(vid))
    except Exception:
        w = 0.0
    vs = float(v)
    if vs < float(stuck_low_mps):
        if vid not in stuck_low_speed_since:
            stuck_low_speed_since[vid] = float(t)
        dwell = float(t) - float(stuck_low_speed_since[vid])
    else:
        stuck_low_speed_since.pop(vid, None)
        dwell = 0.0
    sig2 = (w >= float(stuck_time_req)) or (dwell >= float(stuck_time_req))
    # Rule 2 is independent of rule 1 in the same step: stuck can add/combine with infeasible/past_commit.
    if near_stuck and sig2:
        should = True
        if not reason:
            reason = "sumo_waiting"
        elif "sumo_waiting" not in reason:
            reason = reason + "+sumo_waiting"
    return (should, reason, wrong_exit_lane, wrong_through_lane, wrong_downstream_exit_lane, wrong_lane)


# --- on_off_on_off: Off_Ramp1 / Off_Ramp2, Weaving_Area1/2, Between_Weaving, :Weaving1/2_*, etc. ---
def freeway_reroute_on_off_on_off(
    exit_key: str,
    off_dest: str,
    dest: str,
    edge: str,
    lane: str,
    edge_lane_i: int,
    signed_d: float,
    v: float,
    t: float,
    vid: str,
    stuck_low_speed_since: dict,
    rerouted_f: bool,
    exit_order: list,
    exit_defs: dict,
    impos_lookahead: float,
    impos_sec_per_lc: float,
    stuck_time_req: float,
    stuck_low_mps: float,
    weave_reroute: float,
    near_end_of_current_edge: bool,
) -> tuple:
    r, L = edge, lane
    li = int(edge_lane_i)
    on_off_side = on_main_side = False
    if exit_key == "E1":
        if "Between_Weaving" in r or "Between_Weaving" in L:
            on_main_side = True
        elif "Input" in r:
            on_main_side = True
        elif "On_Ramp1" in r:
            on_off_side = True
        elif "Off_Ramp1" in r or ":Weaving1_End_0" in r or ":Weaving1_Start_0" in r:
            on_off_side = True
        elif ":Weaving1_End_1" in r or ":Weaving1_Start_1" in r:
            on_main_side = True
        elif "Weaving_Area1" in L:
            on_off_side = li == 0
            on_main_side = li >= 1
        # Weaving_Area2_0 = E2 off/decision (right); 1+ = through — do not lump all of Area2 as "main".
        elif "Weaving_Area2" in L:
            on_off_side = li == 0
            on_main_side = li >= 1
        elif "Output" in r:
            on_main_side = True
    elif exit_key == "E2":
        if "Between_Weaving" in r or "Between_Weaving" in L:
            on_main_side = True
        elif "Output" in r:
            on_main_side = True
        elif "On_Ramp2" in r:
            on_off_side = True
        elif "Off_Ramp2" in r or ":Weaving2_End_0" in r or ":Weaving2_Start_0" in r:
            on_off_side = True
        elif ":Weaving2_End_1" in r or ":Weaving2_Start_1" in r:
            on_main_side = True
        elif "Weaving_Area2" in L:
            on_off_side = li == 0
            on_main_side = li >= 1
    wrong_exit_lane = (dest == off_dest) and on_main_side
    wrong_through_lane = (dest == "Output") and on_off_side
    ds = _downstream_off_dests(exit_key, exit_order, exit_defs)
    wrong_downstream_exit_lane = on_off_side and (dest in ds)
    wrong_lane = wrong_exit_lane or wrong_through_lane or wrong_downstream_exit_lane
    if rerouted_f or not wrong_lane or not off_dest:
        return (False, "", wrong_exit_lane, wrong_through_lane, wrong_downstream_exit_lane, wrong_lane)
    should = False
    reason = ""
    n_lc = max(1, int(edge_lane_i)) if wrong_exit_lane else 1
    T_need = float(n_lc) * float(impos_sec_per_lc)
    v_now = float(v)
    T_budget = 0.0 if v_now < 1e-3 else max(0.0, signed_d) / v_now
    if signed_d <= 0.0:
        should, reason = True, "past_commit"
    elif signed_d <= impos_lookahead and T_need > T_budget:
        should, reason = True, "infeasible"
    in_we = (
        ("Weaving_Area1" in r)
        or ("Weaving_Area2" in r)
        or ("Weaving_Area1" in L)
        or ("Weaving_Area2" in L)
        or ("Between_Weaving" in r)
        or ("Between_Weaving" in L)
        or (":Weaving1_" in r)
        or (":Weaving2_" in r)
    )
    _d_stuck = max(float(weave_reroute), float(impos_lookahead) * 0.5)
    near_stuck = ((signed_d > 0) and (signed_d <= _d_stuck) and in_we) or (
        near_end_of_current_edge
        and (("Off_Ramp1" in r) or ("Off_Ramp2" in r) or in_we)
    )
    try:
        w = float(traci.vehicle.getWaitingTime(vid))
    except Exception:
        w = 0.0
    vs = float(v)
    if vs < float(stuck_low_mps):
        if vid not in stuck_low_speed_since:
            stuck_low_speed_since[vid] = float(t)
        dwell = float(t) - float(stuck_low_speed_since[vid])
    else:
        stuck_low_speed_since.pop(vid, None)
        dwell = 0.0
    sig2 = (w >= float(stuck_time_req)) or (dwell >= float(stuck_time_req))
    if near_stuck and sig2:
        should = True
        if not reason:
            reason = "sumo_waiting"
        elif "sumo_waiting" not in reason:
            reason = reason + "+sumo_waiting"
    return (should, reason, wrong_exit_lane, wrong_through_lane, wrong_downstream_exit_lane, wrong_lane)


# --- on_on_off_off: Weaving_Area, After_Weaving, Off_Ramp2_Taper, :Off_Ramp2_*, :Weaving_End_*, etc. ---
def freeway_reroute_on_on_off_off(
    exit_key: str,
    off_dest: str,
    dest: str,
    edge: str,
    lane: str,
    edge_lane_i: int,
    signed_d: float,
    v: float,
    t: float,
    vid: str,
    stuck_low_speed_since: dict,
    rerouted_f: bool,
    exit_order: list,
    exit_defs: dict,
    impos_lookahead: float,
    impos_sec_per_lc: float,
    stuck_time_req: float,
    stuck_low_mps: float,
    weave_reroute: float,
    near_end_of_current_edge: bool,
) -> tuple:
    r, L = edge, lane
    li = int(edge_lane_i)
    on_off_side = on_main_side = False
    if exit_key == "E1":
        if "Output" in r and "Off_Ramp2" not in r:
            on_main_side = True
        elif "Off_Ramp1" in r and "Off_Ramp2" not in r:
            on_off_side = True
        elif ":Weaving_End_0" in r or ":Weaving_Start_0" in r:
            on_off_side = True
        elif ":Weaving_End_1" in r or ":Weaving_Start_1" in r:
            on_main_side = True
        elif "Weaving_Area" in L and "After_Weaving" not in L:
            on_off_side = li == 0
            on_main_side = li >= 1
        elif "Before_Weaving" in L or "On_Ramp1" in r or "On_Ramp2" in r:
            on_main_side = True
    elif exit_key == "E2":
        if "Output" in r:
            on_main_side = True
        # Shared weave inner edges (same :Weaving_* IDs as E1). Replaces the old "Off_Ramp1" substring branch.
        elif ":Weaving_Start_0" in r or ":Weaving_End_0" in r:
            on_off_side = True
        elif ":Weaving_Start_1" in r or ":Weaving_End_1" in r:
            on_main_side = True
        elif "After_Weaving" in L:
            on_off_side = li == 0
            on_main_side = li >= 1
        elif ":Off_Ramp2_Start_1" in r:
            on_main_side = True
        elif ":Off_Ramp2_Start_0" in r or ("Off_Ramp2" in r and "Taper" not in r) or (L and "Off_Ramp2" in L and "Off_Ramp1" not in L and "Taper" not in L):
            on_off_side = True
        elif "Off_Ramp2_Taper" in L or ":Off_Ramp2_Taper_Start" in r:
            on_off_side = li == 0
            on_main_side = li >= 1
        elif "Weaving_Area" in L and "After_Weaving" not in L and "Off_Ramp2" not in L:
            # Match E1 above: lane 0 is the E1 off-ramp side; 1+ is mainline (was incorrectly all "main" so
            # wrong_through_lane was never set for Output vehicles stuck in _0).
            on_off_side = li == 0
            on_main_side = li >= 1
    wrong_exit_lane = (dest == off_dest) and on_main_side
    wrong_through_lane = (dest == "Output") and on_off_side
    ds = _downstream_off_dests(exit_key, exit_order, exit_defs)
    wrong_downstream_exit_lane = on_off_side and (dest in ds)
    wrong_lane = wrong_exit_lane or wrong_through_lane or wrong_downstream_exit_lane
    if rerouted_f or not wrong_lane or not off_dest:
        return (False, "", wrong_exit_lane, wrong_through_lane, wrong_downstream_exit_lane, wrong_lane)
    should = False
    reason = ""
    n_lc = max(1, int(edge_lane_i)) if wrong_exit_lane else 1
    T_need = float(n_lc) * float(impos_sec_per_lc)
    v_now = float(v)
    T_budget = 0.0 if v_now < 1e-3 else max(0.0, signed_d) / v_now
    if signed_d <= 0.0:
        should, reason = True, "past_commit"
    elif signed_d <= impos_lookahead and T_need > T_budget:
        should, reason = True, "infeasible"
    in_we = (
        ("Weaving_Area" in r)
        or ("Weaving_Area" in L)
        or (":Weaving_End" in r)
        or (":Weaving_Start" in r)
        or ("After_Weaving" in L)
        or ("Before_Weaving" in L)
        or ("Off_Ramp2_Taper" in L)
        or (":Off_Ramp2" in r)
    )
    _d_stuck = max(float(weave_reroute), float(impos_lookahead) * 0.5)
    near_stuck = ((signed_d > 0) and (signed_d <= _d_stuck) and in_we) or (
        near_end_of_current_edge
        and (("Off_Ramp1" in r) or ("Off_Ramp2" in r) or in_we)
    )
    try:
        w = float(traci.vehicle.getWaitingTime(vid))
    except Exception:
        w = 0.0
    vs = float(v)
    if vs < float(stuck_low_mps):
        if vid not in stuck_low_speed_since:
            stuck_low_speed_since[vid] = float(t)
        dwell = float(t) - float(stuck_low_speed_since[vid])
    else:
        stuck_low_speed_since.pop(vid, None)
        dwell = 0.0
    sig2 = (w >= float(stuck_time_req)) or (dwell >= float(stuck_time_req))
    if near_stuck and sig2:
        should = True
        if not reason:
            reason = "sumo_waiting"
        elif "sumo_waiting" not in reason:
            reason = reason + "+sumo_waiting"
    return (should, reason, wrong_exit_lane, wrong_through_lane, wrong_downstream_exit_lane, wrong_lane)


def _sumo_diverge_unreachable(cur_edge: str, target_edge: str) -> bool:
    """True when the network has no path between sibling diverge branches (typical freeway gore)."""
    if not cur_edge or not target_edge:
        return False
    out = (cur_edge == "Output") or cur_edge.startswith("Output")
    off = "Off_Ramp" in cur_edge
    tgt_out = (target_edge == "Output") or target_edge.startswith("Output")
    tgt_off = target_edge.startswith("Off_Ramp")
    if out and tgt_off:
        return True
    if off and tgt_out:
        return True
    return False


def safe_change_target(vid: str, target_edge: str) -> bool:
    cur_edge = traci.vehicle.getRoadID(vid)
    # Even if we're already on the target edge, still try to set it so SUMO
    # recomputes the route and the vehicle can "arrive" properly.
    if cur_edge == target_edge:
        try:
            traci.vehicle.changeTarget(vid, target_edge)
        except traci.TraCIException:
            pass
        return True

    if _sumo_diverge_unreachable(cur_edge, target_edge):
        return False

    # Must route from the edge the vehicle is actually on (including ":" internal).
    # Do not substitute r[routeIndex]: on a connector toward Output, findRoute(Weaving, Off_Ramp)
    # can succeed while changeTarget(Off_Ramp) fails and spams SUMO errors every step.
    rr = traci.simulation.findRoute(cur_edge, target_edge)
    if rr is None:
        return False
    edges = getattr(rr, "edges", None)
    if not edges or len(edges) == 0:
        return False

    try:
        traci.vehicle.changeTarget(vid, target_edge)
        return True
    except traci.TraCIException:
        return False

#Fix Progress Bar
#def run_sim_freeway(user_input_data):
def run_sim_freeway(user_input_data, progress_cb=None, is_running_check=None):
    # added here geometry type
    print(user_input_data)
    freeway_type= user_input_data["Geometry"]["Freeway_Type"] # "on_off", "on_off_on_off", or "on_on_off_off"
    
    # these things below are not affected by geometry type
    num_lanes = user_input_data["Geometry"]["Num_Lanes"]
    
    ramp_length = user_input_data["Geometry"]["Ramp_Length"] # for simplicity we assume the lengths of the ramps are all the same
    # MUST be defined before new_node_xs uses it
    ramp_x_range = ramp_length * np.cos(np.pi / 6)
    ramp_y = -ramp_length * np.sin(np.pi / 6)

        

    # these are names of the saved files and do not change
    new_file_name = str(MODELS_DIR / "freeway.net.xml")
    net_file_name = new_file_name
    route_file_name = str(MODELS_DIR / "all_trips.trips.xml")
    config_file = str(MODELS_DIR / "freeway.sumocfg")

    if freeway_type == "on_off":
        freeway_template_file = str(TEMPLATES_DIR / f"freeway_{num_lanes}lane_template.net.xml")
        Input_to_Weaving_Length = user_input_data["Geometry"]["Input_to_Weaving_Length"] 
        Weaving_Length = user_input_data["Geometry"]["Weaving_Length"] 
        Weaving_to_Output_Length = user_input_data["Geometry"]["Weaving_to_Output_Length"] 
        new_node_xs = {
            "Start": 0,
            "Weaving_Start": Input_to_Weaving_Length,
            "Weaving_End": Input_to_Weaving_Length + Weaving_Length ,
            "End": Input_to_Weaving_Length + Weaving_Length + Weaving_to_Output_Length,
            "On_Ramp_Start": Input_to_Weaving_Length - ramp_x_range,
            "Off_Ramp_End": Input_to_Weaving_Length + Weaving_Length + ramp_x_range
        }
        
        new_node_ys={
            "Start": 0,
            "Weaving_Start": 0 ,
            "Weaving_End": 0,
            "End": 0,
            "On_Ramp_Start": ramp_y,
            "Off_Ramp_End": ramp_y
        }
        
    elif freeway_type == "on_on_off_off":
        freeway_template_file = str(TEMPLATES_DIR / f"freeway_{num_lanes}lane_template_on_on_off_off.net.xml")
        Input_to_Onramp1_Length = user_input_data["Geometry"]["Input_to_Onramp1_Length"] 
        Onramp1_Taper_Length = user_input_data["Geometry"]["Onramp1_Taper_Length"] 
        Onramp1_Taper_to_Weaving_Length = user_input_data["Geometry"]["Onramp1_Taper_to_Weaving_Length"] 
        Weaving_Length = user_input_data["Geometry"]["Weaving_Length"] 
        Weaving_to_Offramp2_Taper_Length = user_input_data["Geometry"]["Weaving_to_Offramp2_Taper_Length"]
        Offramp2_Taper_Length = user_input_data["Geometry"]["Offramp2_Taper_Length"]
        Offramp2_to_Output_Length = user_input_data["Geometry"]["Offramp2_to_Output_Length"]

        new_node_xs = {
            "Start": 0,
            "On_Ramp1_End": Input_to_Onramp1_Length,
            "On_Ramp1_Taper_End": Input_to_Onramp1_Length + Onramp1_Taper_Length ,
            "Weaving_Start": Input_to_Onramp1_Length + Onramp1_Taper_Length + Onramp1_Taper_to_Weaving_Length,
            "Weaving_End": Input_to_Onramp1_Length + Onramp1_Taper_Length + Onramp1_Taper_to_Weaving_Length + Weaving_Length ,
            "Off_Ramp2_Taper_Start": Input_to_Onramp1_Length + Onramp1_Taper_Length + Onramp1_Taper_to_Weaving_Length + Weaving_Length + Weaving_to_Offramp2_Taper_Length,
            "Off_Ramp2_Start": Input_to_Onramp1_Length + Onramp1_Taper_Length + Onramp1_Taper_to_Weaving_Length + Weaving_Length + Weaving_to_Offramp2_Taper_Length + Offramp2_Taper_Length ,
            "End": Input_to_Onramp1_Length + Onramp1_Taper_Length + Onramp1_Taper_to_Weaving_Length + Weaving_Length + Weaving_to_Offramp2_Taper_Length + Offramp2_Taper_Length + Offramp2_to_Output_Length, 
            
            "On_Ramp1_Start": Input_to_Onramp1_Length - ramp_x_range,
            "On_Ramp2_Start": Input_to_Onramp1_Length + Onramp1_Taper_Length + Onramp1_Taper_to_Weaving_Length - ramp_x_range,
            "Off_Ramp1_End": Input_to_Onramp1_Length + Onramp1_Taper_Length + Onramp1_Taper_to_Weaving_Length + Weaving_Length + ramp_x_range,
            "Off_Ramp2_End": Input_to_Onramp1_Length + Onramp1_Taper_Length + Onramp1_Taper_to_Weaving_Length + Weaving_Length + Weaving_to_Offramp2_Taper_Length + Offramp2_Taper_Length + ramp_x_range
            }
            
        new_node_ys={
            "Start": 0,
            "On_Ramp1_End": 0,
            "On_Ramp1_Taper_End": 0,
            "Weaving_Start": 0,
            "Weaving_End": 0 ,
            "Off_Ramp2_Taper_Start": 0,
            "Off_Ramp2_Start": 0,
            "End": 0 , 
            "On_Ramp1_Start": ramp_y,
            "On_Ramp2_Start": ramp_y,
            "Off_Ramp1_End": ramp_y,
            "Off_Ramp2_End":ramp_y
            }
        
    elif freeway_type == "on_off_on_off":
        freeway_template_file = str(TEMPLATES_DIR / f"freeway_{num_lanes}lane_template_on_off_on_off.net.xml")
        Input_to_Onramp1_Length = user_input_data["Geometry"]["Input_to_Onramp1_Length"] 
        Weaving1_Length = user_input_data["Geometry"]["Weaving1_Length"] 
        Between_Weaving_Length = user_input_data["Geometry"]["Between_Weaving_Length"] 
        Weaving2_Length = user_input_data["Geometry"]["Weaving2_Length"] 
        Offramp2_to_Output_Length = user_input_data["Geometry"]["Offramp2_to_Output_Length"] 
        
        new_node_xs={
            "Start": 0,
            "Weaving1_Start": Input_to_Onramp1_Length, 
            "Weaving1_End": Input_to_Onramp1_Length + Weaving1_Length, 
            "Weaving2_Start": Input_to_Onramp1_Length + Weaving1_Length + Between_Weaving_Length,
            "Weaving2_End": Input_to_Onramp1_Length + Weaving1_Length + Between_Weaving_Length + Weaving2_Length,
            "End": Input_to_Onramp1_Length + Weaving1_Length + Between_Weaving_Length + Weaving2_Length +  Offramp2_to_Output_Length , 
            "On_Ramp1_Start": Input_to_Onramp1_Length - ramp_x_range ,
            "On_Ramp2_Start": Input_to_Onramp1_Length + Weaving1_Length + Between_Weaving_Length - ramp_x_range,
            "Off_Ramp1_End": Input_to_Onramp1_Length + Weaving1_Length + ramp_x_range,
            "Off_Ramp2_End": Input_to_Onramp1_Length + Weaving1_Length + Between_Weaving_Length + Weaving2_Length + + ramp_x_range
        }
        new_node_ys={
            "Start": 0,
            "Weaving1_Start": 0,
            "Weaving1_End": 0,
            "Weaving2_Start": 0,
            "Weaving2_End": 0,
            "End": 0 , 
            "On_Ramp1_Start": ramp_y,
            "On_Ramp2_Start": ramp_y,
            "Off_Ramp1_End": ramp_y,
            "Off_Ramp2_End":ramp_y
        }
        
        
    ############# NOTE proportion not needed any more
    #proportion_between = user_input_data["Geometry"]["Prop_between_Ramps"] / 100.0

    # NEW start proportion logic (safe defaults)
    #default_start_prop = (1.0 - proportion_between) / 2.0
    #onramp_start_prop = user_input_data["Geometry"].get("OnRamp_Start_Prop", default_start_prop * 100.0) / 100.0

    #weave_len = total_length * proportion_between
    #max_start_prop = max(0.0, 1.0 - proportion_between)
    #onramp_start_prop = min(max(onramp_start_prop, 0.0), max_start_prop)

    #weaving_start = total_length * onramp_start_prop
    #weaving_end = weaving_start + weave_len

    
    def construct_geometry(new_file_name):
        tree = ET.parse(freeway_template_file)
        root = tree.getroot()
        for junction in root.findall("junction"):
            junction.set("shape", "")
            junction_id = junction.get("id")
            if junction_id in new_node_ys:
                junction.set("x", str(np.round(new_node_xs[junction_id],2)))
                junction.set("y", str(np.round(new_node_ys[junction_id],2)))
        for edge in root.findall("edge"):
            if "length" in edge.attrib:
                del edge.attrib["length"]
            if "shape" in edge.attrib:
                del edge.attrib["shape"]
        tree.write(new_file_name)
        try:
            subprocess.run([
                "netconvert",
                "--sumo-net-file", new_file_name,
                "-o", new_file_name
            ], check=True)
        except subprocess.CalledProcessError as e:
            pass
    construct_geometry(new_file_name)
    # now time to connect the lanes
    num_weave_lanes = num_lanes+1 # the num_lanes is the num of lanes in non-weaving segment
    if freeway_type == "on_off":
        lane6_names = ["Input_5", ":Weaving_Start_1_5", "Weaving_Area_6", ":Weaving_End_1_5", "Output_5" ]
        lane5_names = ["Input_4", ":Weaving_Start_1_4", "Weaving_Area_5", ":Weaving_End_1_4", "Output_4" ]
        lane4_names = ["Input_3", ":Weaving_Start_1_3", "Weaving_Area_4", ":Weaving_End_1_3", "Output_3" ]
        lane3_names = ["Input_2", ":Weaving_Start_1_2", "Weaving_Area_3", ":Weaving_End_1_2", "Output_2" ]
        lane2_names = ["Input_1", ":Weaving_Start_1_1", "Weaving_Area_2", ":Weaving_End_1_1", "Output_1" ]
        lane1_names = ["Input_0", ":Weaving_Start_1_0", "Weaving_Area_1", ":Weaving_End_1_0", "Output_0" ]
        lane0_names = ["On_Ramp_0", ":Weaving_Start_0_0", "Weaving_Area_0", ":Weaving_End_0_0", "Off_Ramp_0" ]
        all_lane_names = [lane0_names, lane1_names, lane2_names, lane3_names, lane4_names, lane5_names, lane6_names]
        
    elif freeway_type == "on_on_off_off":
        lane6_names = ["Input_5", ":On_Ramp1_End_1_5", "On_Ramp1_Taper_6", ":On_Ramp1_Taper_End_0_5", "Before_Weaving_5", ":Weaving_Start_1_5", "Weaving_Area_6", ":Weaving_End_1_5", "After_Weaving_5", ":Off_Ramp2_Taper_Start_0_6", "Off_Ramp2_Taper_6", ":Off_Ramp2_Start_1_5", "Output_5"]
        lane5_names = ["Input_4", ":On_Ramp1_End_1_4", "On_Ramp1_Taper_5", ":On_Ramp1_Taper_End_0_4", "Before_Weaving_4", ":Weaving_Start_1_4", "Weaving_Area_5", ":Weaving_End_1_4", "After_Weaving_4", ":Off_Ramp2_Taper_Start_0_5", "Off_Ramp2_Taper_5", ":Off_Ramp2_Start_1_4", "Output_4"]
        lane4_names = ["Input_3", ":On_Ramp1_End_1_3", "On_Ramp1_Taper_4", ":On_Ramp1_Taper_End_0_3", "Before_Weaving_3", ":Weaving_Start_1_3", "Weaving_Area_4", ":Weaving_End_1_3", "After_Weaving_3", ":Off_Ramp2_Taper_Start_0_4", "Off_Ramp2_Taper_4", ":Off_Ramp2_Start_1_3", "Output_3"]
        lane3_names = ["Input_2", ":On_Ramp1_End_1_2", "On_Ramp1_Taper_3", ":On_Ramp1_Taper_End_0_2", "Before_Weaving_2", ":Weaving_Start_1_2", "Weaving_Area_3", ":Weaving_End_1_2", "After_Weaving_2", ":Off_Ramp2_Taper_Start_0_3", "Off_Ramp2_Taper_3", ":Off_Ramp2_Start_1_2", "Output_2"]
        lane2_names = ["Input_1", ":On_Ramp1_End_1_1", "On_Ramp1_Taper_2", ":On_Ramp1_Taper_End_0_1", "Before_Weaving_1", ":Weaving_Start_1_1", "Weaving_Area_2", ":Weaving_End_1_1", "After_Weaving_1", ":Off_Ramp2_Taper_Start_0_2", "Off_Ramp2_Taper_2", ":Off_Ramp2_Start_1_1", "Output_1"]
        lane1_names = ["Input_0", ":On_Ramp1_End_1_0", "On_Ramp1_Taper_1", ":On_Ramp1_Taper_End_0_0", "Before_Weaving_0", ":Weaving_Start_1_0", "Weaving_Area_1", ":Weaving_End_1_0", "After_Weaving_0", ":Off_Ramp2_Taper_Start_0_1", "Off_Ramp2_Taper_1", ":Off_Ramp2_Start_1_0", "Output_0"]
        lane0_onramp1_names = ["On_Ramp1_0", ":On_Ramp1_End_0_0", "On_Ramp1_Taper_0", ":On_Ramp1_Taper_End_0_0"]
        lane0_weaving_names = [ "On_Ramp2_0", ":Weaving_Start_0_0", "Weaving_Area_0", ":Weaving_End_0_0", "Off_Ramp1_0" ] 
        lane0_offramp2_names = [":Off_Ramp2_Taper_Start_0_0", "Off_Ramp2_Taper_0", ":Off_Ramp2_Start_0_0", "Off_Ramp2_0" ]
        all_lane_names = [lane0_onramp1_names, lane0_weaving_names, lane0_offramp2_names  , lane1_names, lane2_names, lane3_names, lane4_names, lane5_names, lane6_names]

    elif freeway_type == "on_off_on_off":
        
        lane6_names = ["Input_5", ":Weaving1_Start_1_5", "Weaving_Area1_6", ":Weaving1_End_1_5", "Between_Weaving_5", ":Weaving2_Start_1_5", "Weaving_Area2_6", ":Weaving2_End_1_5", "Output_5"]

        lane5_names = ["Input_4", ":Weaving1_Start_1_4", "Weaving_Area1_5", ":Weaving1_End_1_4", "Between_Weaving_4", ":Weaving2_Start_1_4", "Weaving_Area2_5", ":Weaving2_End_1_4", "Output_4"]
        
        lane4_names = ["Input_3", ":Weaving1_Start_1_3", "Weaving_Area1_4", ":Weaving1_End_1_3", "Between_Weaving_3", ":Weaving2_Start_1_3", "Weaving_Area2_4", ":Weaving2_End_1_3", "Output_3"]
        
        lane3_names = ["Input_2", ":Weaving1_Start_1_2", "Weaving_Area1_3", ":Weaving1_End_1_2", "Between_Weaving_2", ":Weaving2_Start_1_2", "Weaving_Area2_3", ":Weaving2_End_1_2", "Output_2"]
        
        lane2_names = ["Input_1", ":Weaving1_Start_1_1", "Weaving_Area1_2", ":Weaving1_End_1_1", "Between_Weaving_1", ":Weaving2_Start_1_1", "Weaving_Area2_2", ":Weaving2_End_1_1", "Output_1"]
        
        lane1_names = ["Input_0", ":Weaving1_Start_1_0", "Weaving_Area1_1", ":Weaving1_End_1_0", "Between_Weaving_0", ":Weaving2_Start_1_0", "Weaving_Area2_1", ":Weaving2_End_1_0", "Output_0"]

        lane0_weave1_names = ["On_Ramp1_0", ":Weaving1_Start_0_0", "Weaving_Area1_0", ":Weaving1_End_0_0", "Off_Ramp1_0"]
        lane0_weave2_names = ["On_Ramp2_0", ":Weaving2_Start_0_0", "Weaving_Area2_0", ":Weaving2_End_0_0", "Off_Ramp2_0"]
        all_lane_names = [lane0_weave1_names, lane0_weave2_names , lane1_names, lane2_names, lane3_names, lane4_names, lane5_names, lane6_names]


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
            return {}
        for edge in root.findall('edge'):
            for lane in edge.findall('lane'):
                lane_id = lane.get('id')
                length_str = lane.get('length')
                if lane_id and length_str:
                    try:
                        lane_seg_lengths[lane_id] = float(length_str)
                    except ValueError:
                        pass
        return lane_seg_lengths
    lane_lengths = get_lane_lengths(net_file_name)
    print(lane_lengths)

    # --- Exit behavior tuning ---
    EXIT_PREP_DIST = float(user_input_data.get("Exit_Prep_Dist", 400.0))  # meters
    EXIT_MIN_WEIGHT = float(user_input_data.get("Exit_Min_Intent", 0.15))  # minimum intent weight

    # --- Deadline + reroute failsafe tuning ---
    WEAVE_REROUTE_DIST = float(user_input_data.get("Weave_Reroute_Dist", 30.0))  # m to end of weaving
    STUCK_TIME_REQUIRED = float(user_input_data.get("Stuck_Time_Required", 4.0))  # sec (typ. 4–5 s stuck law)
    # "Stuck" also if speed stays below this (m/s) for STUCK_TIME_REQUIRED s (creep; SUMO waitingTime is often 0 then).
    STUCK_LOW_SPEED_MPS = float(user_input_data.get("Stuck_Low_Speed_m_s", 0.75))
    EDGE_END_BUFFER = float(user_input_data.get("Edge_End_Buffer", 4.0))  # m from end of edge
    # Impossible-exit reroute: wrong-lane vehicles only; compare remaining distance vs. time needed for lane changes.
    IMPOSSIBLE_EXIT_LOOKAHEAD = float(user_input_data.get("Impossible_Exit_Lookahead_m", 250.0))
    IMPOSSIBLE_EXIT_SEC_PER_LC = float(user_input_data.get("Impossible_Exit_Sec_Per_Lane", 2.75))  # s per LC
    # "Exit deadline" positions in global coordinates (supports all geometries / multiple exits)
    # NOTE: global_pos is computed as lane_pos + pos_adjust_by_lanes[lane].
    # We compute EXIT_DEFS after pos_adjust_by_lanes is available.
    EXIT_MID_FRAC = float(user_input_data.get('Exit_Mid_Frac', 0.2))
    # Print limited diagnostics about reroute decisions (can be noisy).
    REROUTE_DEBUG = bool(user_input_data.get("Reroute_Debug", False))

    # --- Collision / rear-end safety tuning (parameters only) ---
    # Keep models (IDM/PT/MOBIL/DDM) unchanged; only constrain parameters for realism/safety.
    #
    # Optional guards (OFF by default to avoid changing the model logic):
    # - Enable_Safety_SpeedCap: post-process speed to guarantee a minimum gap (changes control logic)
    # - Set_SpeedMode_Safety: forces SUMO safety checks even under setSpeed (changes behavior)
    ENABLE_SAFETY_SPEED_CAP = bool(user_input_data.get("Enable_Safety_SpeedCap", False))
    SET_SPEEDMODE_SAFETY = bool(user_input_data.get("Set_SpeedMode_Safety", False))
    # Extra bumper-to-bumper gap (m) required beyond the vehicle's own minGap (s0) if safety cap is enabled.
    SAFETY_EXTRA_GAP = float(user_input_data.get("Safety_Extra_Gap", 0.5))
    # Clamp extremely aggressive IDM samples (used by our custom longitudinal model)
    IDM_MIN_T = float(user_input_data.get("IDM_Min_T", 0.9))      # s
    IDM_MIN_B = float(user_input_data.get("IDM_Min_b", 2.5))      # m/s^2
    IDM_MIN_S0 = float(user_input_data.get("IDM_Min_s0", 2.0))    # m
    # Optional: extra Gaussian noise on PT acceleration (default off; separate from PT softmax)
    PT_STOCHASTIC_ENABLE = bool(user_input_data.get("PT_Stochastic_Enable", False))
    PT_STOCHASTIC_SIGMA = float(user_input_data.get("PT_Stochastic_Sigma", 0.0))
    # NOTE: We intentionally do NOT guard vType tau/decel/emergencyDecel here.
    # Safety is handled via IDM parameter clamps + optional runtime safety guards below.



    lane_segment_to_lane_list ={}
    for lanes in all_lane_names:
        for lane in lanes:
            lane_segment_to_lane_list[lane]=lanes
    pos_adjust_by_lanes = {}
    if freeway_type == "on_off":
        for lanes in all_lane_names:
            for lane in lanes:
                if "Input" in lane:
                    pos_adjust_by_lanes[lane] = - lane_lengths[":Weaving_Start_1_0"] - lane_lengths["Input_0"]
                if "On_Ramp" in lane:
                    pos_adjust_by_lanes[lane] = - lane_lengths[":Weaving_Start_0_0"] - lane_lengths["On_Ramp_0"]
                if ":Weaving_Start_1" in lane:
                    pos_adjust_by_lanes[lane] = - lane_lengths[":Weaving_Start_1_0"]
                if ":Weaving_Start_0" in lane:
                    pos_adjust_by_lanes[lane] = - lane_lengths[":Weaving_Start_0_0"]
                if "Weaving_Area" in lane:
                    pos_adjust_by_lanes[lane] = 0
                if ":Weaving_End" in lane:
                    pos_adjust_by_lanes[lane] = lane_lengths["Weaving_Area_0"]
                if "Output" in lane:
                    pos_adjust_by_lanes[lane] = lane_lengths["Weaving_Area_0"] + lane_lengths[":Weaving_End_1_0"]
                if "Off_Ramp" in lane:
                    pos_adjust_by_lanes[lane] = lane_lengths["Weaving_Area_0"] + lane_lengths[":Weaving_End_0_0"]
    elif freeway_type == "on_on_off_off":
        all_main_lanes = [lane1_names, lane2_names, lane3_names, lane4_names, lane5_names, lane6_names]
        
        rel_lanes = [all_main_lanes[l_idx] for l_idx in range(num_lanes) ]

        for lanes in rel_lanes:
            for lane_seg_i in range(len(lanes)): # do iteratively, the origin is 0
                lane =  lanes[lane_seg_i]
                if lane_seg_i == 0:
                    pos_adjust_by_lanes[lane] = 0 
                else:
                    pos_adjust_by_lanes[lane] = pos_adjust_by_lanes[lanes[lane_seg_i-1]] + lane_lengths[lanes[lane_seg_i-1]]
        # the other three separated lanes would be trickier
        
        pos_adjust_by_lanes["On_Ramp1_0"] = pos_adjust_by_lanes["Before_Weaving_0"] - lane_lengths[":On_Ramp1_Taper_End_0_0"] - pos_adjust_by_lanes[":On_Ramp1_Taper_End_0_0"] - lane_lengths["On_Ramp1_Taper_0"] - lane_lengths[":On_Ramp1_End_0_0"] - lane_lengths["On_Ramp1_0"]
        pos_adjust_by_lanes["On_Ramp2_0"] = pos_adjust_by_lanes["Weaving_Area_1"] - lane_lengths[":Weaving_Start_0_0"] - lane_lengths["On_Ramp2_0"]
        pos_adjust_by_lanes[":Off_Ramp2_Taper_Start_0_0"] = pos_adjust_by_lanes[":Off_Ramp2_Taper_Start_0_1"]
        for lanes in [lane0_onramp1_names, lane0_weaving_names, lane0_offramp2_names ]:
            for lane_seg_i in range(1, len(lanes)):
                lane = lanes[lane_seg_i]
                pos_adjust_by_lanes[lane] = pos_adjust_by_lanes[lanes[lane_seg_i-1]] + lane_lengths[lanes[lane_seg_i-1]]

    elif freeway_type == "on_off_on_off":
        all_main_lanes = [lane1_names, lane2_names, lane3_names, lane4_names, lane5_names, lane6_names]
    
        rel_lanes = [all_main_lanes[l_idx] for l_idx in range(num_lanes) ]
        
        for lanes in rel_lanes:
            for lane_seg_i in range(len(lanes)): # do iteratively, the origin is 0
                lane =  lanes[lane_seg_i]
                if lane_seg_i == 0:
                    pos_adjust_by_lanes[lane] = 0 
                else:
                    pos_adjust_by_lanes[lane] = pos_adjust_by_lanes[lanes[lane_seg_i-1]] + lane_lengths[lanes[lane_seg_i-1]]
        
        # the other two separated lanes would be trickier  
        pos_adjust_by_lanes["On_Ramp1_0"] = pos_adjust_by_lanes["Weaving_Area1_1"] -  lane_lengths[":Weaving1_Start_0_0"] - lane_lengths["On_Ramp1_0"]
        pos_adjust_by_lanes["On_Ramp2_0"] = pos_adjust_by_lanes["Weaving_Area2_1"] -  lane_lengths[":Weaving2_Start_0_0"] - lane_lengths["On_Ramp2_0"]
        
        for lanes in [lane0_weave1_names, lane0_weave2_names ]:

            for lane_seg_i in range(1, len(lanes)): 
                lane = lanes[lane_seg_i]
                
                pos_adjust_by_lanes[lane] = pos_adjust_by_lanes[lanes[lane_seg_i-1]] + lane_lengths[lanes[lane_seg_i-1]]

    
    # ---- Build per-exit global coordinate definitions (now that pos_adjust_by_lanes is known) ----


    
    def _end_x_of_lane(lane_id: str) -> float:

    
        return float(pos_adjust_by_lanes.get(lane_id, 0.0)) + float(lane_lengths.get(lane_id, 0.0))


    
    def _start_x_of_lane(lane_id: str) -> float:

    
        return float(pos_adjust_by_lanes.get(lane_id, 0.0))


    
    def build_exit_defs():

    
        '''Return dict exit_key -> info used for exit logic across geometries.


    
        Each info has:

    
          off_dest: destination string for the off-ramp at this exit

    
          start_x, end_x: global coordinate interval for the critical zone (weaving or taper)

    
          zone_tags: substrings that identify lanes/edges belonging to this exit zone

    
          mid_tags: substrings that identify lanes where midpoint lane-change rules apply

    
        '''

    
        defs = {}

    
        if freeway_type == 'on_off':

    
            weave0 = 'Weaving_Area_0'

    
            defs['E1'] = {

    
                'off_dest': 'Off_Ramp',

    
                'start_x': _start_x_of_lane(weave0),

    
                'end_x': _end_x_of_lane(weave0),

    
                'zone_tags': ['Weaving_Area', ':Weaving_Start', ':Weaving_End', 'Off_Ramp'],

    
                'mid_tags': ['Weaving_Area'],

    
            }

    
        elif freeway_type == 'on_off_on_off':

    
            weave1 = 'Weaving_Area1_0'

    
            weave2 = 'Weaving_Area2_0'

    
            defs['E1'] = {

    
                'off_dest': 'Off_Ramp1',

    
                'start_x': _start_x_of_lane(weave1),

    
                'end_x': _end_x_of_lane(weave1),

    
                'zone_tags': ['Weaving_Area1', ':Weaving1_Start', ':Weaving1_End', 'Off_Ramp1'],

    
                'mid_tags': ['Weaving_Area1'],

    
            }

    
            defs['E2'] = {

    
                'off_dest': 'Off_Ramp2',

    
                'start_x': _start_x_of_lane(weave2),

    
                'end_x': _end_x_of_lane(weave2),

    
                # Between_Weaving links exit 1 to exit 2; include it so E2 reroute / sumo_waiting apply there.
                'zone_tags': ['Weaving_Area2', ':Weaving2_Start', ':Weaving2_End', 'Off_Ramp2', 'Between_Weaving'],

    
                'mid_tags': ['Weaving_Area2'],

    
            }

    
        elif freeway_type == 'on_on_off_off':

    
            weave = 'Weaving_Area_0'

    
            after0 = 'After_Weaving_0'

    
            defs['E1'] = {

    
                'off_dest': 'Off_Ramp1',

    
                'start_x': _start_x_of_lane(weave),

    
                'end_x': _end_x_of_lane(weave),

    
                'zone_tags': ['Weaving_Area', ':Weaving_Start', ':Weaving_End', 'Off_Ramp1'],

    
                'mid_tags': ['Weaving_Area'],

    
            }

    
            defs['E2'] = {

    
                'off_dest': 'Off_Ramp2',

    
                'start_x': _start_x_of_lane(after0),

    
                'end_x': _end_x_of_lane(after0),

    
                # Include After_Weaving so reroute/zone checks work before taper begins
                'zone_tags': ['After_Weaving', 'Off_Ramp2_Taper', ':Off_Ramp2_Taper_Start', ':Off_Ramp2_Start', 'Off_Ramp2'],

    
                'mid_tags': ['Off_Ramp2_Taper', 'After_Weaving'],

    
            }

    
        return defs


    
    EXIT_DEFS = build_exit_defs()

    
    EXIT_X = float(next(iter(EXIT_DEFS.values()))['end_x']) if len(EXIT_DEFS) else 0.0

    # Exits sorted by downstream position (used for "next exit ahead" logic)
    EXIT_ORDER = sorted(list(EXIT_DEFS.keys()), key=lambda k: float(EXIT_DEFS[k].get('end_x', 0.0)))

    def current_exit_key_for_position(lane_id: str, edge_id: str, global_pos: float):
        """Pick the exit key whose zone the vehicle is currently in.

        This is intentionally independent of the vehicle destination, so the reroute
        logic is consistent for both:
        - exit-bound vehicles stuck on mainline side (missed exit), and
        - through-bound vehicles stuck on exit side (forced exit).
        """
        if not EXIT_ORDER:
            return None
        cands = [k for k in EXIT_ORDER if in_exit_zone(k, lane_id, edge_id)]
        if not cands:
            return None
        # Prefer the next downstream end_x if multiple zones match
        after = [k for k in cands if float(EXIT_DEFS[k].get('end_x', 0.0)) >= float(global_pos)]
        if after:
            return min(after, key=lambda k: float(EXIT_DEFS[k].get('end_x', 0.0)))
        # Otherwise, pick the most downstream candidate
        return max(cands, key=lambda k: float(EXIT_DEFS[k].get('end_x', 0.0)))


    
    def exit_key_for_vehicle(dest: str, lane_id: str, edge_id: str,

    
                             on_off1_side: bool=False, on_off2_side: bool=False):

    
        if freeway_type == 'on_off':

    
            return 'E1'

    
        if freeway_type == 'on_off_on_off':

    
            if dest == 'Off_Ramp1' or 'Weaving_Area1' in lane_id or 'Weaving1_' in lane_id or 'Off_Ramp1' in edge_id or 'Off_Ramp1' in lane_id:

    
                return 'E1'

    
            if dest == 'Off_Ramp2' or 'Weaving_Area2' in lane_id or 'Weaving2_' in lane_id or 'Off_Ramp2' in edge_id or 'Off_Ramp2' in lane_id:

    
                return 'E2'

    
            if on_off1_side:

    
                return 'E1'

    
            if on_off2_side:

    
                return 'E2'

    
            return None

    
        if freeway_type == 'on_on_off_off':

    
            if dest == 'Off_Ramp1' or 'Weaving_Area' in lane_id or 'Weaving_' in lane_id or 'Off_Ramp1' in edge_id or 'Off_Ramp1' in lane_id:

    
                return 'E1'

    
            if dest == 'Off_Ramp2' or 'Off_Ramp2' in lane_id or 'Off_Ramp2' in edge_id or 'Off_Ramp2_Taper' in lane_id:

    
                return 'E2'

    
            if on_off1_side:

    
                return 'E1'

    
            if on_off2_side:

    
                return 'E2'

    
            return None

    
        return None


    
    def in_exit_zone(exit_key: str, lane_id: str, edge_id: str) -> bool:

    
        info = EXIT_DEFS.get(exit_key)

    
        if not info:

    
            return False

    
        return any((tag in lane_id) or (tag in edge_id) for tag in info.get('zone_tags', []))


    
    def in_mid_section(exit_key: str, lane_id: str) -> bool:

    
        info = EXIT_DEFS.get(exit_key)

    
        if not info:

    
            return False

    
        return any(tag in lane_id for tag in info.get('mid_tags', []))


    
    lane_segment_to_idx={}
    for lane_idx in range(len(all_lane_names)):
        lanes = all_lane_names[lane_idx]
        for lane in lanes:
            lane_segment_to_idx[lane] = lane_idx

    sim_visualization = user_input_data["Sim_Visualization"]
    sim_time =  user_input_data["Sim_Time"]
    min_t = 0
    t_step = user_input_data["Sim_StepSize"]
    # here separate case again depending on geometry
    if freeway_type == "on_off":
        flow_main_main = user_input_data["Vehicle_Flows"]["Main-Main"]
        flow_onramp_main = user_input_data["Vehicle_Flows"]["OnRamp-Main"]
        flow_main_offramp = user_input_data["Vehicle_Flows"]["Main-OffRamp"]
        flow_onramp_offramp = user_input_data["Vehicle_Flows"]["OnRamp-OffRamp"]
        in_flows = [flow_main_main, flow_onramp_main, flow_main_offramp, flow_onramp_offramp]
        
    elif freeway_type == "on_off_on_off":
        flow_main_main = user_input_data["Vehicle_Flows"]["Main-Main"]
        flow_main_offramp1 = user_input_data["Vehicle_Flows"]["Main-OffRamp1"]
        flow_main_offramp2 = user_input_data["Vehicle_Flows"]["Main-OffRamp2"]

        flow_onramp1_main = user_input_data["Vehicle_Flows"]["OnRamp1-Main"]
        flow_onramp1_offramp1 = user_input_data["Vehicle_Flows"]["OnRamp1-OffRamp1"]
        flow_onramp1_offramp2 = user_input_data["Vehicle_Flows"]["OnRamp1-OffRamp2"]

        
        flow_onramp2_main = user_input_data["Vehicle_Flows"]["OnRamp2-Main"]
        flow_onramp2_offramp2 = user_input_data["Vehicle_Flows"]["OnRamp2-OffRamp2"]

        in_flows = [flow_main_main, flow_main_offramp1, flow_main_offramp2, 
                    flow_onramp1_main, flow_onramp1_offramp1, flow_onramp1_offramp2,
                    flow_onramp2_main ,flow_onramp2_offramp2 ]
        
    elif freeway_type == "on_on_off_off":
        flow_main_main = user_input_data["Vehicle_Flows"]["Main-Main"]
        flow_main_offramp1 = user_input_data["Vehicle_Flows"]["Main-OffRamp1"]
        flow_main_offramp2 = user_input_data["Vehicle_Flows"]["Main-OffRamp2"]
        
        flow_onramp1_main = user_input_data["Vehicle_Flows"]["OnRamp1-Main"]
        flow_onramp1_offramp1 = user_input_data["Vehicle_Flows"]["OnRamp1-OffRamp1"]
        flow_onramp1_offramp2 = user_input_data["Vehicle_Flows"]["OnRamp1-OffRamp2"]
        
        flow_onramp2_main = user_input_data["Vehicle_Flows"]["OnRamp2-Main"]
        flow_onramp2_offramp1 = user_input_data["Vehicle_Flows"]["OnRamp2-OffRamp1"]
        flow_onramp2_offramp2 = user_input_data["Vehicle_Flows"]["OnRamp2-OffRamp2"]

        in_flows = [flow_main_main, flow_main_offramp1, flow_main_offramp2, 
                    flow_onramp1_main, flow_onramp1_offramp1, flow_onramp1_offramp2,
                    flow_onramp2_main ,flow_onramp2_offramp1,flow_onramp2_offramp2 ]

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
    LC_model_name = user_input_data["LC_Model"]
    CF_model_name = user_input_data["CF_Model"]
    CF_default = user_input_data["CF_Default_Params"]
    LC_default = user_input_data["LC_Default_Params"]
    min_hw = 1.5
    min_gap = 2
    model_folder = str(MODEL_PARAMS_DIR) + os.sep
    # Merged CF priors: 0=S, 1=A, 2=L; 4=CAHV (connected heavy) uses CL, not L.
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
    MOBIL_data = load_mobil_results_csv()
    veh_class_names = {0: "Small Vehicle", 1: "Automated Vehicle", 2: "Heavy Vehicle", 3: "CAV", 4: "CAHV"}
    def sample_IDM(veh_class=None):
        base_class = 1 if veh_class == 3 else 2 if veh_class == 4 else veh_class
        if CF_default == False and CF_model_name == "IDM":
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
        else:
            # CAV(3)->merged A; CAHV(4)->merged CL; SV/AV/HV use own bucket.
            _idm_csv_key = 4 if veh_class == 4 else (1 if veh_class == 3 else veh_class)
            samples = IDM_param_data[_idm_csv_key]
            samples = samples[samples["T"]>0]
            picked_row = samples.sample(n=1).iloc[0]
            driving_params = np.array(picked_row[["T", "a", "b", "v0", "so", "delta"]].values, dtype = np.float64 )
            return driving_params
    def sample_PT(veh_class=None):
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
            mean_Wm = user_input_data["CF_Parameters"]["W_m"][veh_class_names[base_class]]["Mean"]
            std_Wm = user_input_data["CF_Parameters"]["W_m"][veh_class_names[base_class]]["Std"]
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
    def sample_MOBIL(veh_class=None):
        base_class = 1 if veh_class == 3 else 2 if veh_class == 4 else veh_class
        if LC_default == False and LC_model_name == "MOBIL":
            mean_p_disc = user_input_data["LC_Parameters"]['Disc: p_opt'][veh_class_names[base_class]]["Mean"]
            std_p_disc  = user_input_data["LC_Parameters"]['Disc: p_opt'][veh_class_names[base_class]]["Std"]
            mean_ath_disc = user_input_data["LC_Parameters"]['Disc: a_th'][veh_class_names[base_class]]["Mean"]
            std_ath_disc  = user_input_data["LC_Parameters"]['Disc: a_th'][veh_class_names[base_class]]["Std"]

            mean_b_disc = user_input_data["LC_Parameters"]['Disc: b_safe'][veh_class_names[base_class]]["Mean"]
            std_b_disc  = user_input_data["LC_Parameters"]['Disc: b_safe'][veh_class_names[base_class]]["Std"]
            mean_b_mand = user_input_data["LC_Parameters"]['Mand: b_safe'][veh_class_names[base_class]]["Mean"]
            std_b_mand  = user_input_data["LC_Parameters"]['Mand: b_safe'][veh_class_names[base_class]]["Std"]

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
    def sample_DDM(veh_class=None):
        base_class = 1 if veh_class == 3 else 2 if veh_class == 4 else veh_class
        if LC_default == False and LC_model_name == "DDM":
            mean_alpha_h = user_input_data["LC_Parameters"]["α_h"][veh_class_names[base_class]]["Mean"]
            std_alpha_h  = user_input_data["LC_Parameters"]["α_h"][veh_class_names[base_class]]["Std"]
            mean_beta0_left = user_input_data["LC_Parameters"]["β_0_left"][veh_class_names[base_class]]["Mean"]
            std_beta0_left  = user_input_data["LC_Parameters"]["β_0_left"][veh_class_names[base_class]]["Std"]
            mean_beta0_right = user_input_data["LC_Parameters"]["β_0_right"][veh_class_names[base_class]]["Mean"]
            std_beta0_right  = user_input_data["LC_Parameters"]["β_0_right"][veh_class_names[base_class]]["Std"]
            mean_beta_G = user_input_data["LC_Parameters"]["β_G"][veh_class_names[base_class]]["Mean"]
            std_beta_G  = user_input_data["LC_Parameters"]["β_G"][veh_class_names[base_class]]["Std"]
            mean_G0 = user_input_data["LC_Parameters"]["G_0"][veh_class_names[base_class]]["Mean"]
            std_G0  = user_input_data["LC_Parameters"]["G_0"][veh_class_names[base_class]]["Std"]
            mean_beta_V = user_input_data["LC_Parameters"]["β_V"][veh_class_names[base_class]]["Mean"]
            std_beta_V  = user_input_data["LC_Parameters"]["β_V"][veh_class_names[base_class]]["Std"]
            mean_beta_MLC = user_input_data["LC_Parameters"]["β_MLC"][veh_class_names[base_class]]["Mean"]
            std_beta_MLC  = user_input_data["LC_Parameters"]["β_MLC"][veh_class_names[base_class]]["Std"]
            mean_sigma = user_input_data["LC_Parameters"]["σ"][veh_class_names[base_class]]["Mean"]
            std_sigma  = user_input_data["LC_Parameters"]["σ"][veh_class_names[base_class]]["Std"]

            random_alpha_h = random.gauss(mean_alpha_h, std_alpha_h)
            random_beta0_left = random.gauss(mean_beta0_left, std_beta0_left)
            random_beta0_right = random.gauss(mean_beta0_right, std_beta0_right)
            random_beta_G = random.gauss(mean_beta_G, std_beta_G)
            random_G0 = random.gauss(mean_G0, std_G0)
            random_beta_V = max(0.001, random.gauss(mean_beta_V, std_beta_V))
            random_beta_MLC = random.gauss(mean_beta_MLC, std_beta_MLC)
            random_sigma = max(0.001, random.gauss(mean_sigma, std_sigma))
            return np.array([random_alpha_h, random_beta0_left, random_beta0_right, random_beta_G, random_G0, random_beta_V, random_beta_MLC, random_sigma], dtype=np.float64)
        else:
            return np.array([0.08, -3.5, -4.2, 0.2737, 8.69, 0.6808, 87.0, 8.458], dtype=np.float64)

    def generate_agents(output_file="all_trips.trips.xml"):
        tech_type_by_id = {}
        all_veh_generation_times={}
        agent_type_by_id={}
        veh_origins_by_id={}
        veh_destinations_by_id={}
        IDM_params_by_id = {}
        PT_params_by_id = {}
        MOBIL_params_by_id ={}
        DDM_params_by_id = {}
        gen_times = []
        origins_list = []
        destinations_list = []

        vf = user_input_data.get("Vehicle_Flows", {})

        # OD pairs explicitly tied to the GUI keys, dependent on the edges
        if freeway_type == "on_off":
            od_pairs = [
                ("Input", "Output", float(vf.get("Main-Main", 0.0))),
                ("On_Ramp", "Output", float(vf.get("OnRamp-Main", 0.0))),  
                ("Input", "Off_Ramp", float(vf.get("Main-OffRamp", 0.0))),
                ("On_Ramp", "Off_Ramp", float(vf.get("OnRamp-OffRamp", 0.0))),
            ]
        elif freeway_type == "on_off_on_off":
            od_pairs = [
                 ("Input",    "Output",     flow_main_main),
                 ("Input",    "Off_Ramp1",  flow_main_offramp1),
                 ("Input",    "Off_Ramp2",  flow_main_offramp2),
            
                 ("On_Ramp1", "Output",     flow_onramp1_main),
                 ("On_Ramp1", "Off_Ramp1",  flow_onramp1_offramp1),
                 ("On_Ramp1", "Off_Ramp2",  flow_onramp1_offramp2),
            
                 ("On_Ramp2", "Output",     flow_onramp2_main),
                 ("On_Ramp2", "Off_Ramp2",  flow_onramp2_offramp2),
            ]
            
        elif freeway_type == "on_on_off_off":
            od_pairs = [
                 ("Input",    "Output",     flow_main_main),
                 ("Input",    "Off_Ramp1",  flow_main_offramp1),
                 ("Input",    "Off_Ramp2",  flow_main_offramp2),
            
                 ("On_Ramp1", "Output",     flow_onramp1_main),
                 ("On_Ramp1", "Off_Ramp1",  flow_onramp1_offramp1),
                 ("On_Ramp1", "Off_Ramp2",  flow_onramp1_offramp2),
            
                 ("On_Ramp2", "Output",     flow_onramp2_main),
                 ("On_Ramp2", "Off_Ramp1",  flow_onramp2_offramp1),
                 ("On_Ramp2", "Off_Ramp2",  flow_onramp2_offramp2),
            ]
            

        

        for in_name, out_name, demand in od_pairs:
            if demand <= 0:
                generation_times = np.array([])
            else:
                generation_times = np.cumsum(
                    np.maximum(np.random.exponential(3600 / demand, 100000), min_hw)
                )
                generation_times = np.round(
                    generation_times[(generation_times >= min_t) & (generation_times <= sim_time)],
                    1
                )

            if in_name not in all_veh_generation_times:
                all_veh_generation_times[in_name] = {}
            all_veh_generation_times[in_name][out_name] = generation_times

            for gen_t in generation_times:
                origins_list.append(in_name)
                destinations_list.append(out_name)
                gen_times.append(gen_t)

        origins_arr=np.array(origins_list)
        destinations_arr=np.array(destinations_list)
        gen_times_arr=np.array(gen_times)
        sort_indices=np.argsort(gen_times_arr)
        gen_times_arr=gen_times_arr[sort_indices]
        origins_arr=origins_arr[sort_indices]
        destinations_arr=destinations_arr[sort_indices]
        agent_ids=np.array([idx+1 for idx in range(len(gen_times_arr))])
        root = ET.Element("routes")
        for i in range(len(agent_ids)):
            veh_class = np.random.choice([0, 1, 2, 3, 4], p=[SV_rate, AV_rate, HV_rate, CAV_rate, CAHV_rate])
            agent_type_by_id[str(agent_ids[i])] = veh_class
            tech_type_by_id[str(agent_ids[i])] = (
                "SV" if veh_class == 0 else
                "AV" if veh_class == 1 else
                "HV" if veh_class == 2 else
                "CAV" if veh_class == 3 else
                "CAHV"
            )
            veh_origins_by_id[str(agent_ids[i])] = str(origins_arr[i])
            veh_destinations_by_id[str(agent_ids[i])] = str(destinations_arr[i])
            idm_params = sample_IDM(veh_class)
            PT_params = sample_PT(veh_class)
            MOBIL_params = sample_MOBIL(veh_class)
            DDM_params = sample_DDM(veh_class)
            # Clamp aggressive samples (prevents rear-end crashes / unrealistic following)
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
            guishape = "truck" if veh_len > 10 else "passenger"
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
                "vClass": "truck" if veh_len > 10 else "passenger"
            })
            # departLane "best": SUMO picks the lane on `from` that fits the routed path to `to`
            # (strategic default). "random" often starts vehicles on the wrong side for weaves.
            _depart_lane = str(user_input_data.get("Depart_Lane", "best")).strip()
            if _depart_lane not in ("best", "random", "free", "allowed", "first"):
                _depart_lane = "best"
            trip = ET.SubElement(root, "trip", {
                "id": str(agent_ids[i]),
                "type": str(agent_ids[i]) ,
                "depart": str(gen_times_arr[i]),
                "from": str(origins_arr[i]),
                "to": str(destinations_arr[i]),
                "departLane": _depart_lane,
                "departSpeed": "max"
            })
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        return agent_type_by_id, tech_type_by_id, veh_origins_by_id, veh_destinations_by_id, IDM_params_by_id, PT_params_by_id, MOBIL_params_by_id, DDM_params_by_id
    agent_type_by_id, tech_type_by_id, veh_origins_by_id, veh_destinations_by_id, IDM_params_by_id, PT_params_by_id, MOBIL_params_by_id , DDM_params_by_id =generate_agents(output_file=route_file_name)
    
    def _normalize_lane_seg(lane_seg: str):
        """
        Normalize lane segment ids that can show up with '-1'/'1' indices on
        internal connector edges. Returns None if the segment should be treated
        as out-of-network for leader/follower queries.
        """
        if lane_seg is None:
            return None
        lane_seg = str(lane_seg)

        if freeway_type == "on_off":
            remap = {
                "Input_-1": "On_Ramp_0",
                "Output_-1": "Off_Ramp_0",
                ":Weaving_Start_1_-1": ":Weaving_Start_0_0",
                ":Weaving_End_1_-1": ":Weaving_End_0_0",
                ":Weaving_Start_0_1": ":Weaving_Start_1_0",
                ":Weaving_End_0_1": ":Weaving_End_1_0",
            }
            lane_seg = remap.get(lane_seg, lane_seg)
            if "-1" in lane_seg or ("Ramp" in lane_seg and "1" in lane_seg):
                return None

        elif freeway_type == "on_off_on_off":
            remap = {
                "Input_-1": "On_Ramp1_0",
                "Output_-1": "Off_Ramp2_0",
                ":Weaving1_Start_1_-1": ":Weaving1_Start_0_0",
                ":Weaving1_End_1_-1": ":Weaving1_End_0_0",
                ":Weaving1_Start_0_1": ":Weaving1_Start_1_0",
                ":Weaving1_End_0_1": ":Weaving1_End_1_0",
                ":Weaving2_Start_1_-1": ":Weaving2_Start_0_0",
                ":Weaving2_End_1_-1": ":Weaving2_End_0_0",
                ":Weaving2_Start_0_1": ":Weaving2_Start_1_0",
                ":Weaving2_End_0_1": ":Weaving2_End_1_0",
            }
            lane_seg = remap.get(lane_seg, lane_seg)
            if "-1" in lane_seg or ("Ramp" in lane_seg and "1" in lane_seg):
                return None

        elif freeway_type == "on_on_off_off":
            remap = {
                "Input_-1": "On_Ramp1_0",
                "Output_-1": "Off_Ramp2_0",
                ":On_Ramp1_End_1_-1": ":On_Ramp1_End_0_0",
                ":On_Ramp1_End_0_1": ":On_Ramp1_End_1_0",
                ":Weaving_Start_1_-1": ":Weaving_Start_0_0",
                ":Weaving_End_1_-1": ":Weaving_End_0_0",
                ":Weaving_Start_0_1": ":Weaving_Start_1_0",
                ":Weaving_End_0_1": ":Weaving_End_1_0",
                # Needed for the on_on_off_off geometry (previously commented out)
                ":Off_Ramp2_Taper_Start_0_1": ":Off_Ramp2_Taper_Start_1_0",
                ":Off_Ramp2_Taper_Start_1_-1": ":Off_Ramp2_Taper_Start_0_0",
                ":Off_Ramp2_Start_0_1": ":Off_Ramp2_Start_1_0",
                ":Off_Ramp2_Start_1_-1": ":Off_Ramp2_Start_0_0",
            }
            lane_seg = remap.get(lane_seg, lane_seg)
            if "-1" in lane_seg or lane_seg in ('On_Ramp1_1', 'On_Ramp2_1', 'Off_Ramp1_1', 'Off_Ramp2_1'):
                return None

        return lane_seg

    def find_leader(global_pos, lane_idx_, lane_seg, lanes,  global_poses, vs, lengths):
        lane_seg_in = str(lane_seg) if lane_seg is not None else ""
        lane_seg_norm = _normalize_lane_seg(lane_seg)
        if lane_seg_norm is None or lane_seg_norm not in lane_segment_to_lane_list:
            return 0, 5, global_pos+500, 50
        # Same SUMO lane as caller (raw TraCI id); rel_lane_segs would mix parallel lanes (bad for PT/CF).
        lead_inds = np.where((global_poses>global_pos+0.001) & (lanes == lane_seg_in) )[0]
        if len(lead_inds)==0:
            return 0, 5, global_pos+500, 50
        else:
            lead_xs = global_poses[lead_inds]
            lead_vs = vs[lead_inds]
            lead_lens = lengths[lead_inds]
            nearest_idx = np.argmin(lead_xs)
            return 1, lead_lens[nearest_idx], lead_xs[nearest_idx], lead_vs[nearest_idx]
    def find_follower(global_pos, lane_idx_, lane_seg, lanes, global_poses, vs, lengths):
        lane_seg_in = str(lane_seg) if lane_seg is not None else ""
        lane_seg_norm = _normalize_lane_seg(lane_seg)
        if lane_seg_norm is None or lane_seg_norm not in lane_segment_to_lane_list:
            return 0, 5, global_pos-500, 50
        follow_inds = np.where((global_poses<global_pos-0.001) & (lanes == lane_seg_in) )[0]
        if len(follow_inds)==0:
            return 0, 5, global_pos-500, 0.1
        else:
            follow_xs = follow_xs = global_poses[follow_inds]
            follow_vs = vs[follow_inds]
            follow_lens = lengths[follow_inds]
            nearest_idx = np.argmax(follow_xs)
            return 1, follow_lens[nearest_idx], follow_xs[nearest_idx], follow_vs[nearest_idx]
    @njit
    def DDM_LC(DDM_params, direction, is_MLC, adj_gap, adj_lead_v, lead_v ):
        alpha_h, beta_0_left, beta_0_right, beta_G, G_0, beta_V, beta_MLC, sigma =  DDM_params
        beta_0 = beta_0_left if direction == 1 else beta_0_right
        mu = beta_0 + beta_G*np.arctan(adj_gap-G_0) + beta_V*np.arctan(adj_lead_v - lead_v) + beta_MLC*is_MLC
        return mu

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
        """
        Prospect-style utility on *deviations* from IDM reference acceleration; discrete softmax only
        over a narrow delta grid. Collision term uses candidate a = a_ref + delta.
        """
        tau_max, alpha_v, beta_PT, wc, gamma1, gamma2, wm = PT_params
        a_ref = IDM(IDM_params, v, v_leader, x, x_leader, length, length_leader)
        # Discrete choice only on delta around IDM (not full physical accel range)
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

        # Calculate accelerations (IDM or PT-relative-to-IDM)
        no_lc_acc_self = _cf_accel(use_pt, CF_params, idm_params, v_self, v_leader, x_self, x_leader, length_self, length_leader)
        no_lc_acc_follower = _cf_accel(use_pt, CF_params, idm_params, v_follower, v_self, x_follower, x_self, length_follower, length_self)

        # Left Lane Calculations
        lc_left_acc_self = -100.0  # Default unsafe
        lc_left_acc_follower = -100.0
        lc_left_acc_left_follower = -100.0

        if left_lane_exists:
            lc_left_acc_self = _cf_accel(use_pt, CF_params, idm_params, v_self, v_left_leader, x_self, x_left_leader, length_self,
                                        length_left_leader)
            lc_left_acc_follower = _cf_accel(use_pt, CF_params, idm_params, v_follower, v_leader, x_follower, x_leader, length_follower,
                                            length_leader)
            lc_left_acc_left_follower = _cf_accel(use_pt, CF_params, idm_params, v_left_follower, v_self, x_left_follower, x_self,
                                                 length_left_follower, length_self)

        # Right Lane Calculations
        lc_right_acc_self = -100.0
        lc_right_acc_follower = -100.0  # Note: this is actually the old follower in current lane
        lc_right_acc_right_follower = -100.0

        if right_lane_exists:
            lc_right_acc_self = _cf_accel(use_pt, CF_params, idm_params, v_self, v_right_leader, x_self, x_right_leader, length_self,
                                         length_right_leader)
            lc_right_acc_follower = lc_left_acc_follower  # Same old follower situation
            lc_right_acc_right_follower = _cf_accel(use_pt, CF_params, idm_params, v_right_follower, v_self, x_right_follower, x_self,
                                                   length_right_follower, length_self)

            # --- FIXED SAFETY CHECK ---
            # Discretionary: Standard comfort (-4.0)
            # Mandatory (MLC): Panic braking allowed (-9.0) to force merges in traffic

        # Calculate safety for Left
        b_thresh_left = b_safe_mand if MLC_left == 1 else b_safe_disc
        left_safe = left_lane_exists and (lc_left_acc_left_follower > -b_thresh_left) and (
                        lc_left_acc_self > -b_thresh_left)

        # Calculate safety for Right
        b_thresh_right = b_safe_mand if MLC_right == 1 else b_safe_disc
        right_safe = right_lane_exists and (lc_right_acc_right_follower > -b_thresh_right) and (
                        lc_right_acc_self > -b_thresh_right)

        # Mandatory Lane Change (MLC) Logic - Prioritize Safety only, ignore Incentive
        if MLC_left == 1:
            return 1 if left_safe else 0
        if MLC_right == 1:
            return -1 if right_safe else 0

        # Discretionary Lane Change Logic - Check Incentive AND Safety
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

        # Decision
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

    acc_max = 5
    acc_min = -10
    acc_step_size = 0.1
    tree = ET.parse(config_file)
    root = tree.getroot()
    all_evidences_left = {}
    all_evidences_right = {}
    for vid in veh_origins_by_id:
        all_evidences_left[vid] = []
        all_evidences_right[vid] = []
    for inp in root.findall("input"):
        net = inp.find("net-file")
        net.set("value", net_file_name)
        route = inp.find("route-files")
        route.set("value", route_file_name)
    time_elem = root.find("time")
    if time_elem is not None:
        end = time_elem.find("end")
        end.set("value", str(sim_time))
        step = time_elem.find("step-length")
        step.set("value", str(t_step))
    tree.write(config_file)
    # Data collection: honor Sim_DataCollection (on/off) and Sim_DataFreq (sec between samples).
    # If disabled, keep all_data as None so the append path is skipped entirely.
    collect_data = bool(user_input_data.get("Sim_DataCollection", True))
    all_data = {"time":[], "id":[], "type":[], "x":[], "y":[], "v":[], "theta":[], "road":[], "length":[] , "lane":[], "lane_pos":[] } if collect_data else None
    collect_every_steps = 1
    if collect_data:
        sample_freq = float(user_input_data.get("Sim_DataFreq", t_step))
        collect_every_steps = max(1, int(round(sample_freq / t_step))) if t_step > 0 else 1
    sumo_binary = checkBinary('sumo-gui') if sim_visualization else checkBinary('sumo')

    # Reduce SUMO console spam unless explicitly requested
    sumo_step_log = bool(user_input_data.get("Sumo_StepLog", False))
    sumo_suppress_warnings = bool(user_input_data.get("Sumo_Suppress_Warnings", True))
    sumo_args = [
        sumo_binary,
        "-c", config_file,
        "--lanechange.duration", "1.1",
    ]
    if not sumo_step_log:
        sumo_args += ["--no-step-log", "true", "--duration-log.disable", "true"]
    if sumo_suppress_warnings:
        sumo_args.append("--no-warnings")
    traci.start(sumo_args)

    comm_cfg = user_input_data.get("Comm_Params", {"Range": 30.0, "Lookahead": 5, "Latency": 0, "Loss": 0.0})

    # tolerate extra keys like "Default"

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

    rerouted_flag = {}  # prevent repeated reroutes
    # Per-vehicle: sim time when current episode of v < STUCK_LOW_SPEED_MPS started (stuck / slow queue).
    stuck_low_speed_since = {}
    # Per-vehicle: last TraCI (road_id, lane_index) after each step; lane-changes update no-return map.
    last_lane_road_by_vid = {}
    last_lane_idx_by_vid = {}
    # vid -> {(road_id, lane_index): until_sim_time} — cannot LC into that lane index on that edge until then.
    no_return_until_by_vid = {}

    COLOR_MAP = {
        # Requested palette:
        # - CAV & CAHV: blue
        # - AV: green
        # - SV & HV: gray
        "SV":  (160, 160, 160, 255),
        "HV":  (160, 160, 160, 255),
        "AV":  (0, 180, 0, 255),
        "CAV": (0, 102, 204, 255),
        "CAHV": (0, 102, 204, 255),
    }
    color_set = {}
    speed_mode_set = {}
    t = 0

    # Fix progress update throttle (define once before the loop)
    last_progress = -1
    progress_every = max(1, int(round(1.0 / t_step)))  # update about every 1.0s of sim time
    step_count = 0
    # Throttle LC decisions (the expensive part: MOBIL_LC / DDM_LC / c_mobil_decision)
    # *and* their SUMO actuation to every N steps. Car-following, routing, and the
    # simulation step itself still run every step.
    # For DDM, the evidence SDE now integrates with lc_dt = lc_every_steps * t_step
    # (equivalent to stepping the same SDE over the longer interval).
    lc_every_steps = max(1, int(user_input_data.get("LC_Every_Steps", 10)))
    lc_dt = lc_every_steps * t_step

    if LC_model_name == "DDM":
        A_left_by_id={}
        A_right_by_id={}

    # --- TraCI subscriptions (batched state pulls; ~1 TraCI call per vehicle per step
    # instead of ~10). Mirrors single_inter_sim.py so both scenarios share the pattern.
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
        t += t_step
        t = np.round(t, 1)

        step_count += 1
        do_lc_step = (step_count % lc_every_steps == 0)
        if progress_cb is not None and (step_count % progress_every == 0 or t >= sim_time):
            pct = int(min(99, max(0, (t / sim_time) * 100)))
            if pct != last_progress:
                progress_cb(pct)
                last_progress = pct

        all_ids = traci.vehicle.getIDList()
        all_veh_id_set = set(all_ids)
        # Subscribe newly spawned vehicles; drop departed from the tracked set.
        for _vid in all_veh_id_set.difference(veh_subscribed_ids):
            try:
                traci.vehicle.subscribe(_vid, veh_sub_vars)
                veh_subscribed_ids.add(_vid)
            except traci.TraCIException:
                pass
        veh_subscribed_ids.intersection_update(all_veh_id_set)
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
        global_poses=np.zeros(len(all_ids))
        lane_indexes = np.zeros(len(all_ids))
        # edge_lane_indexes[i] = TraCI per-edge lane index (tc.VAR_LANE_INDEX); distinct from
        # lane_indexes[i] which is the *global* lane index via lane_segment_to_idx[lane].
        edge_lane_indexes = np.zeros(len(all_ids), dtype=int)
        for i_id in range(len(all_ids)):
            vid = all_ids[i_id]
            if vid not in color_set:
                tech_lbl = tech_type_by_id.get(str(vid), "SV")
                traci.vehicle.setColor(vid, COLOR_MAP.get(tech_lbl, (255, 255, 255, 255)))
                color_set[vid] = True
            # Optional: enforce SUMO safety checks even under setSpeed (OFF by default).
            if SET_SPEEDMODE_SAFETY and vid not in speed_mode_set:
                try:
                    traci.vehicle.setSpeedMode(vid, 31)
                except:
                    pass
                speed_mode_set[vid] = True
            # One batched pull per vehicle (TraCI subscriptions). Fall back to direct
            # getters per-field in the rare case a variable isn't in the result dict
            # yet (e.g., the vehicle just spawned in this step).
            sub = traci.vehicle.getSubscriptionResults(vid) or {}
            x, y = sub.get(tc.VAR_POSITION, traci.vehicle.getPosition(vid))
            speed = float(sub.get(tc.VAR_SPEED, traci.vehicle.getSpeed(vid)))
            prev_v = last_speed.get(vid, speed)
            a_est = (speed - prev_v) / (t_step if t_step>0 else 1.0)
            last_speed[vid] = speed
            last_acc[vid] = a_est

            edge = sub.get(tc.VAR_ROAD_ID, traci.vehicle.getRoadID(vid))
            lane = sub.get(tc.VAR_LANE_ID, traci.vehicle.getLaneID(vid))

            lane_pos = float(sub.get(tc.VAR_LANEPOSITION, traci.vehicle.getLanePosition(vid)))
            length = float(sub.get(tc.VAR_LENGTH, traci.vehicle.getLength(vid)))
            theta = float(sub.get(tc.VAR_ANGLE, traci.vehicle.getAngle(vid)))
            global_pos = lane_pos + pos_adjust_by_lanes[lane]
            lane_idx = lane_segment_to_idx[lane]
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
            # Fallback parses lane-id suffix, matching the previous direct-call contract.
            try:
                _eli = int(sub.get(tc.VAR_LANE_INDEX, int(lane.rsplit("_", 1)[-1])))
            except Exception:
                _eli = 0
            edge_lane_indexes[i_id] = _eli
        states_cav = {}
        for _i, _vid in enumerate(all_ids):
            _tech = tech_type_by_id.get(str(_vid), "SV")
            if _tech not in ("CAV", "CAHV"):
                continue
            states_cav[str(_vid)] = ns.VehicleState(
                vid=str(_vid),
                lane=str(lanes[_i]),
                global_pos=float(global_poses[_i]),
                v=float(vs[_i]),
                a=float(last_acc.get(str(_vid), 0.0)),
                length=float(lengths[_i]),
                tech=str(_tech),
            )
        bus.step(states_cav)

        for i_id in range(len(all_ids)):
            vid = all_ids[i_id]

            dest = veh_destinations_by_id[vid]
            v = vs[i_id]
            lane_idx = lane_indexes[i_id]
            global_pos = global_poses[i_id]
            lane =  lanes[i_id]

            edge = roads[i_id]
            lane_pos = lane_poses[i_id]

            # --- Robust lane index parsing + neighbor lane ids (define early) ---
            lane_i = int(lane.rsplit("_", 1)[-1])  # safer than lane[-1]

            # --- Post-step lane change: forbid returning to the lane index just left (same edge) for LC_NO_RETURN_SEC ---
            # Use this-step subscription snapshot (roads / edge_lane_indexes) instead of direct TraCI calls.
            _nr_road = edge
            _nr_lane_i = int(edge_lane_indexes[i_id])
            if vid not in no_return_until_by_vid:
                no_return_until_by_vid[vid] = {}
            _nrt = no_return_until_by_vid[vid]
            for _nk in list(_nrt.keys()):
                if float(_nrt[_nk]) <= float(t):
                    del _nrt[_nk]
            if vid in last_lane_idx_by_vid:
                _pr = last_lane_road_by_vid.get(vid)
                _pl = last_lane_idx_by_vid[vid]
                if _pr == _nr_road and _pl != _nr_lane_i:
                    _nrt[(_nr_road, _pl)] = float(t) + float(LC_NO_RETURN_SEC)
            last_lane_road_by_vid[vid] = _nr_road
            last_lane_idx_by_vid[vid] = _nr_lane_i

            # Hi Pedram, here I defined the wrong exits separately for all cases, and hopefully it is useful for future derivation.
            # dest here differs depending on the datasets

            # Again here we consider three geometries
            if freeway_type == "on_off":
                # “Off-ramp side” means: on off-ramp edge OR in lane index 0 in weaving/off-ramp connector lanes
                on_offramp_side = ("Off_Ramp" in edge) or ("Off_Ramp" in lane) or (("Weaving_Area" in lane) and lane_i == 0)
    
                # “Mainline side” means: on output edge OR lane index >=1 in weaving
                on_mainline_side = ("Output" in edge) or ("Output" in lane) or (("Weaving_Area" in lane) and lane_i >= 1)
    
                wrong_exit_lane = (dest == "Off_Ramp") and on_mainline_side
                wrong_through_lane = (dest == "Output") and on_offramp_side
                wrong_lane = wrong_exit_lane or wrong_through_lane

            if freeway_type == "on_on_off_off":
                # the lanes at which if you keep going without lane change you could lead to the exit
                on_offramp1_side = ("Off_Ramp1" in edge) or ("Weaving_End_0" in lane) or (("Weaving_Area" in lane ) and lane_i == 0)
                # NOTE: bugfix: this is the Off_Ramp2 side, not Off_Ramp1
                on_offramp2_side = ("Off_Ramp2" in edge) \
                                   or (("After_Weaving" in lane) and lane_i == 0) \
                                   or ("Off_Ramp2_Taper_Start_0" in lane) \
                                   or (("Off_Ramp2_Taper" in lane) and (lane_i == 0)) \
                                   or (("Off_Ramp2_Start_0" in lane))
                # if you keep going through, you would still not get off the freeway (only keep those after weaving)
                # these are not relevant ("input" in edge) or ("On_Ramp1_End_1" in edge) or (("On_Ramp1_Taper" in edge) and (lane_i>=1)) or ("On_Ramp1_Taper_End_1"in edge) or ("Before_Weaving" in lane) or ("Weaving_Start_1" in lane) or ("Before_Weaving" in lane) or
                on_mainline_side1 = (("Weaving_Area" in lane) and (lane_i>=1)) or ("Weaving_End_1" in lane) or ("After_Weaving" in lane) or ("Off_Ramp2_Taper_Start_1" in lane) or (("Off_Ramp2_Taper" in lane) and (lane_i>=1)) or ("Off_Ramp2_Start_1" in lane) or ("Output" in lane)
                # For the 2nd exit, being in After_Weaving (lane_i>=1) is still "mainline side"
                # and is wrong for vehicles that need Off_Ramp2 (they must be in lane 0).
                on_mainline_side2 = ((("After_Weaving" in lane) and (lane_i >= 1))
                                     or (("Off_Ramp2_Taper" in lane) and (lane_i >= 1))
                                     or ("Off_Ramp2_Start_1" in lane)
                                     or ("Output" in lane))

                wrong_exit_lane1 = (dest == "Off_Ramp1") and on_mainline_side1 
                wrong_exit_lane2 = (dest == "Off_Ramp2") and on_mainline_side2
                wrong_through_lane = (dest == "Output") and (on_offramp1_side or on_offramp2_side) 
                wrong_lane = wrong_exit_lane1 or wrong_exit_lane2 or wrong_through_lane 

            if freeway_type == "on_off_on_off":
                on_offramp1_side = (("Weaving_Area1" in lane) and (lane_i==0)) or ("Weaving1_End_0" in lane) or ("Off_Ramp1" in lane)
                on_offramp2_side = (("Weaving_Area2" in lane) and (lane_i==0)) or ("Weaving2_End_0" in lane) or ("Off_Ramp2" in lane)

                on_mainline_side1 = (("Weaving_Area1" in lane) and (lane_i>=1)) or ("Weaving1_End_1" in lane) or ("Between_Weaving" in lane) or ("Weaving2_Start_1" in lane) or (("Weaving_Area2" in lane) and (lane_i>=1)) or ("Weaving2_End" in lane) or ("Output" in lane)
                on_mainline_side2 = ("Between_Weaving" in lane) or (("Weaving_Area2" in lane) and (lane_i>=1)) or ("Weaving2_End" in lane) or ("Output" in lane)

                wrong_exit_lane1 = (dest == "Off_Ramp1") and on_mainline_side1 
                wrong_exit_lane2 = (dest == "Off_Ramp2") and on_mainline_side2
                wrong_through_lane = (dest == "Output") and (on_offramp1_side or on_offramp2_side) 
                wrong_lane = wrong_exit_lane1 or wrong_exit_lane2 or wrong_through_lane 
         
            base = lane.rsplit("_", 1)[0] + "_"
            left_lane_str = base + str(lane_i + 1)
            right_lane_str = base + str(lane_i - 1)

            # --- diverge proximity + wrong-lane detection ---
            dist_to_end = float(EXIT_X) - float(global_pos)

            # Current lane length (from parsed net, fallback to TraCI if missing)
            lane_len_cur = float(lane_lengths.get(lane, 0.0))
            if lane_len_cur <= 0.0:
                try:
                    lane_len_cur = float(traci.lane.getLength(lane))
                except:
                    lane_len_cur = 0.0

            near_end_of_current_edge = (lane_len_cur > 0.0) and (lane_pos >= lane_len_cur - EDGE_END_BUFFER)
            # Hi Pedram
            
            # I am not sure about the rewrouting part here. Please revise as needed. I commented this out for now.
            # Suggestions: You will likely have to do weaving1 and weaving 2 based on the exit_number
            # The three cases of freeway geometry are freeway_type == "on_off_on_off" , "on_off" (The original case), and "on_on_off_off" 
            '''
            in_weave_or_end = ("Weaving_Area" in edge) or (":Weaving_End" in edge) or (":Weaving_Start" in edge)
            near_end_zone = ((dist_to_end > 0) and (dist_to_end <= WEAVE_REROUTE_DIST) and in_weave_or_end) \
                            or (near_end_of_current_edge and (("Off_Ramp" in edge) or in_weave_or_end))

            # --- stuck detector: only reroute if actually stuck for >= STUCK_TIME_REQUIRED ---
            # One-time reroute guard
            if vid not in rerouted_flag:
                rerouted_flag[vid] = False

            # “Stuck for 2 seconds” using SUMO waiting time
            # (Active reroute also uses v < Stuck_Low_Speed_m_s for Stuck_Time_Required; see below.)
            try:
                wait_t = float(traci.vehicle.getWaitingTime(vid))
            except:
                wait_t = 0.0

            if (not rerouted_flag[vid]) and near_end_zone and wrong_lane and (wait_t >= STUCK_TIME_REQUIRED):

                # Exit-bound in wrong lane => missed exit => reroute to Output
                if wrong_exit_lane:
                    ok = safe_change_target(vid, "Output")
                    if ok:
                        veh_destinations_by_id[vid] = "Output"
                        rerouted_flag[vid] = True

                # Through-bound still on off-ramp side => forced exit => reroute to Off_Ramp
                elif wrong_through_lane:
                    ok = safe_change_target(vid, "Off_Ramp")
                    if ok:
                        veh_destinations_by_id[vid] = "Off_Ramp"
                        rerouted_flag[vid] = True

            '''

            # --- generalized stuck detector + one-time reroute (all geometries / exits) ---
            _exit_key = current_exit_key_for_position(lane, edge, global_pos)
            if _exit_key is not None and _exit_key in EXIT_DEFS:
                _off_dest = str(EXIT_DEFS[_exit_key].get('off_dest', ''))
                # TraCI per-edge lane index (more reliable than parsing lane id suffixes on internal edges).
                _edge_lane_i = int(edge_lane_indexes[i_id])
                _exit_end_x = float(EXIT_DEFS[_exit_key]['end_x'])
                dist_to_end = _exit_end_x - float(global_pos)
                signed_d = float(dist_to_end)
                if vid not in rerouted_flag:
                    rerouted_flag[vid] = False
                _rf = bool(rerouted_flag[vid])
                # Reroute rules 1+2: geometry-specific (edge/lane names differ); see freeway_reroute_* at module top.
                if freeway_type == "on_off":
                    (
                        should_reroute,
                        reroute_reason,
                        wrong_exit_lane,
                        wrong_through_lane,
                        wrong_downstream_exit_lane,
                        _,
                    ) = freeway_reroute_on_off(
                        _off_dest,
                        dest,
                        edge,
                        lane,
                        _edge_lane_i,
                        signed_d,
                        v,
                        t,
                        vid,
                        stuck_low_speed_since,
                        _rf,
                        IMPOSSIBLE_EXIT_LOOKAHEAD,
                        IMPOSSIBLE_EXIT_SEC_PER_LC,
                        STUCK_TIME_REQUIRED,
                        STUCK_LOW_SPEED_MPS,
                        WEAVE_REROUTE_DIST,
                        near_end_of_current_edge,
                    )
                elif freeway_type == "on_off_on_off":
                    (
                        should_reroute,
                        reroute_reason,
                        wrong_exit_lane,
                        wrong_through_lane,
                        wrong_downstream_exit_lane,
                        _,
                    ) = freeway_reroute_on_off_on_off(
                        str(_exit_key),
                        _off_dest,
                        dest,
                        edge,
                        lane,
                        _edge_lane_i,
                        signed_d,
                        v,
                        t,
                        vid,
                        stuck_low_speed_since,
                        _rf,
                        EXIT_ORDER,
                        EXIT_DEFS,
                        IMPOSSIBLE_EXIT_LOOKAHEAD,
                        IMPOSSIBLE_EXIT_SEC_PER_LC,
                        STUCK_TIME_REQUIRED,
                        STUCK_LOW_SPEED_MPS,
                        WEAVE_REROUTE_DIST,
                        near_end_of_current_edge,
                    )
                elif freeway_type == "on_on_off_off":
                    (
                        should_reroute,
                        reroute_reason,
                        wrong_exit_lane,
                        wrong_through_lane,
                        wrong_downstream_exit_lane,
                        _,
                    ) = freeway_reroute_on_on_off_off(
                        str(_exit_key),
                        _off_dest,
                        dest,
                        edge,
                        lane,
                        _edge_lane_i,
                        signed_d,
                        v,
                        t,
                        vid,
                        stuck_low_speed_since,
                        _rf,
                        EXIT_ORDER,
                        EXIT_DEFS,
                        IMPOSSIBLE_EXIT_LOOKAHEAD,
                        IMPOSSIBLE_EXIT_SEC_PER_LC,
                        STUCK_TIME_REQUIRED,
                        STUCK_LOW_SPEED_MPS,
                        WEAVE_REROUTE_DIST,
                        near_end_of_current_edge,
                    )
                else:
                    should_reroute, reroute_reason = False, ""
                    wrong_exit_lane = False
                    wrong_through_lane = False
                    wrong_downstream_exit_lane = False

                if should_reroute and (not rerouted_flag[vid]) and len(_off_dest) > 0:
                    # Wrong-exit: on mainline but dest is the off-ramp. If we are past commit / infeasible,
                    # the only reachable destination is Output; accept missing the exit.
                    if wrong_exit_lane and dest.startswith("Off_Ramp"):
                        _target = "Output"
                    # Wrong-through: on the off-ramp side but dest is Output. Once committed to the off
                    # branch there is no route back (diverge unreachable). Accept the forced exit by
                    # targeting this exit's off-ramp edge (not "Output" — that would be a no-op that SUMO
                    # also rejects via _sumo_diverge_unreachable).
                    elif wrong_through_lane and (dest == "Output"):
                        _target = _off_dest
                    # Wrong-downstream: on this exit's off-ramp side but dest is a later off-ramp. Same
                    # logic: take *this* off-ramp (not Output, which isn't reachable from the off branch).
                    elif wrong_downstream_exit_lane:
                        _target = _off_dest
                    else:
                        _target = _off_dest
                    ok = safe_change_target(vid, _target)
                    # Only fall back to Output for the wrong-exit case (mainline side is reachable).
                    # For wrong_through / wrong_downstream, Output is unreachable — don't retry with it.
                    if (not ok) and wrong_exit_lane and _target.startswith("Off_Ramp"):
                        ok = safe_change_target(vid, "Output")
                        if ok:
                            _target = "Output"
                    if REROUTE_DEBUG and (str(_exit_key) in ('E1', 'E2') or str(_off_dest).startswith('Off_Ramp')):
                        cur_edge_dbg = edge
                        cur_lane_dbg = lane
                        cur_lane_i_dbg = _edge_lane_i
                        _dbg_td = ""
                        if "past_commit" in reroute_reason or "infeasible" in reroute_reason:
                            _nl = max(1, int(_edge_lane_i)) if wrong_exit_lane else 1
                            _tn = float(_nl) * float(IMPOSSIBLE_EXIT_SEC_PER_LC)
                            _vdbg = float(v)
                            _tb = 0.0 if _vdbg < 1e-3 else max(0.0, signed_d) / _vdbg
                            _dbg_td = f" d={signed_d:.1f} T_need={_tn:.2f} T_budget={_tb:.2f}"
                        if "sumo_waiting" in reroute_reason:
                            try:
                                _wt = float(traci.vehicle.getWaitingTime(vid))
                            except Exception:
                                _wt = 0.0
                            _sd = 0.0
                            if float(v) < float(STUCK_LOW_SPEED_MPS) and vid in stuck_low_speed_since:
                                _sd = float(t) - float(stuck_low_speed_since[vid])
                            _dbg2 = f" wait_t={_wt:.1f}s v={float(v):.2f} slow_dwell={_sd:.1f}s (<{STUCK_LOW_SPEED_MPS})"
                            _dbg_td = f"{_dbg_td} {_dbg2}" if _dbg_td else _dbg2
                        print(f"[REROUTE] vid={vid} exit={_exit_key} dest={dest} reason={reroute_reason}{_dbg_td} wrongExit={wrong_exit_lane} wrongThrough={wrong_through_lane} wrongDownstream={wrong_downstream_exit_lane} laneI={cur_lane_i_dbg} edge={cur_edge_dbg} lane={cur_lane_dbg} -> {_target} ok={ok}")
                    if ok:
                        veh_destinations_by_id[vid] = _target
                        rerouted_flag[vid] = True
                    else:
                        # Stop retrying every step (avoids TraCI/SUMO spam if routing still disagrees).
                        rerouted_flag[vid] = True
                        if _target.startswith("Off_Ramp"):
                            veh_destinations_by_id[vid] = "Output"

            # Reroute updates veh_destinations_by_id above; local `dest` was read at loop start. Without
            # syncing, MLC / midpoint rules / MOBIL still act on the *old* destination for the rest of this step
            # (e.g. still forcing MLC_right toward exit right after changeTarget to Output).
            dest = veh_destinations_by_id[vid]

            length = lengths[i_id]
            old_lane_idx = int(edge_lane_indexes[i_id])
            left_lane_exists = 0
            right_lane_exists = 0
            MLC_left = 0
            MLC_right = 0

            # --- Early cooperative "make room" for Off_Ramp vehicles (CAV only) ---
            # If this vehicle needs to exit and is not yet in the rightmost lane,
            # broadcast intent with urgency increasing as it approaches EXIT_X.
            # HI Pedram. 
            # I have not touched any code that it related to CAV since I am not 100% sure what your true intended logic is
            # I suppose based on the scenario type, you would have to select the threshold . For the on-on-off-off case, the off-ramps are attached to tapers. I am not sure the threshold in this case
            '''
            if dest == "Off_Ramp" and lane_i >= 1:
                dist_to_exit = EXIT_X - float(global_pos)

                # If we're within the preparation zone, urgency ramps from 0 -> 1
                if dist_to_exit < EXIT_PREP_DIST:
                    urgency = 1.0 - max(0.0, dist_to_exit) / EXIT_PREP_DIST
                    urgency = min(1.0, max(EXIT_MIN_WEIGHT, urgency))

                    tech_label = tech_type_by_id.get(str(vid), "SV")
                    if tech_label == "CAV":
                        ego_state = states_cav.get(
                            str(vid),
                            ns.VehicleState(str(vid), str(lane), float(global_pos), float(v),
                                            float(last_acc.get(str(vid), 0.0)), float(length), "CAV")
                        )
                        # Broadcast intent to the immediate right lane segment (same segment name, lane index - 1)
                        bus.broadcast_intent(ego_state, target_lane=str(right_lane_str), weight=float(urgency))
            '''

            # --- generalized CAV exit intent broadcast (all geometries / exits) ---
            tech_label = tech_type_by_id.get(str(vid), 'SV')
            if tech_label in ('CAV', 'CAHV') and lane_i >= 1 and dest.startswith('Off_Ramp'):
                _ek = exit_key_for_vehicle(dest, lane, edge)
                if _ek is not None and _ek in EXIT_DEFS and dest == EXIT_DEFS[_ek]['off_dest']:
                    dist_to_exit = float(EXIT_DEFS[_ek]['end_x']) - float(global_pos)
                    if dist_to_exit < EXIT_PREP_DIST:
                        urgency = 1.0 - max(0.0, dist_to_exit) / EXIT_PREP_DIST
                        urgency = min(1.0, max(EXIT_MIN_WEIGHT, urgency))
                        ego_state = states_cav.get(
                            str(vid),
                            ns.VehicleState(str(vid), str(lane), float(global_pos), float(v),
                                            float(last_acc.get(str(vid), 0.0)), float(length), str(tech_label))
                        )
                        bus.broadcast_intent(ego_state, target_lane=str(right_lane_str), weight=float(urgency))



            # --- Mandatory LC (MOBIL/DDM): push exit-bound traffic toward lane index 0 (exit side) ---
            # MOBIL_LC uses relaxed safety (b_safe_mand) when MLC_right==1 so merges are attempted in denser traffic.
            # Keep lane_i>=1 so we do not ask for a non-existent move from the rightmost lane.
            if freeway_type == "on_off":
                if dest == "Off_Ramp" and lane_i >= 1 and (
                    ("Weaving_Area" in lane)
                    or ("Input" in lane)
                    or ("On_Ramp" in edge)
                    or ("On_Ramp" in lane)
                ):
                    MLC_right = 1
            elif freeway_type == "on_off_on_off":
                if dest == "Off_Ramp1" and lane_i >= 1 and (
                    ("Weaving_Area1" in lane)
                    or ("Input" in lane)
                ):
                    MLC_right = 1
                # Off_Ramp2: include Input so mainline traffic can start merging before weaving area 2.
                elif dest == "Off_Ramp2" and lane_i >= 1 and (
                    ("Weaving_Area2" in lane)
                    or ("Between_Weaving" in lane)
                    or ("Input" in lane)
                ):
                    MLC_right = 1
            elif freeway_type == "on_on_off_off":
                # Mainline approach (Input) + pre-weave + weave (lane_i>=1): merge toward exit 1 side.
                if dest == "Off_Ramp1" and lane_i >= 1 and (
                    ("Input" in lane)
                    or ("Before_Weaving" in lane)
                    or ("Weaving_Area" in lane)
                ):
                    MLC_right = 1
                # Early merge on mainline; After_Weaving/Taper only while still not in lane 0.
                elif dest == "Off_Ramp2" and lane_i >= 1 and (
                    ("Input" in lane)
                    or ("Before_Weaving" in lane)
                    or ("Weaving_Area" in lane)
                    or ("After_Weaving" in lane)
                    or ("Off_Ramp2_Taper" in lane)
                ):
                    MLC_right = 1

            # --- neighbor existence flags (use lane_i, not lane[-1]) ---
            if lane_i >= 1 and MLC_left == 0: # if MLC left not even consider the right lane
                right_lane_exists = 1

            if freeway_type == "on_off":
                if dest == "Output" and ("Weaving_Area" in lane) and lane_i == 1 and MLC_left == 0:
                    right_lane_exists = 0
            if freeway_type == "on_on_off_off":
                # in the merge taper we treat the second most right lane as having no right lane
                if "On_Ramp1_Taper" in lane and lane_i == 1:
                    right_lane_exists = 0
                elif ((dest == "Output") or (dest == "Off_Ramp2")) and (("Weaving_Area" in lane) and (lane_i  == 1)):
                    right_lane_exists = 0
                elif (dest == "Output") and (("Off_Ramp2_Taper" in lane) and (lane_i  == 1)):
                    right_lane_exists = 0
                    

            if freeway_type == "on_off_on_off":
                if ((dest == "Output") or ( dest == "Off_Ramp2")) and (("Weaving_Area1" in lane) and (lane_i == 1)):
                    right_lane_exists = 0
                elif (dest == "Output" ) and (("Weaving_Area2" in lane) and (lane_i == 1)):
                    right_lane_exists = 0

            # --- default left-lane availability (same spirit as your current logic) ---
            left_lane_exists = 0
            if lane_i < num_lanes - 1 and MLC_right == 0 and ("Off_Ramp" not in lane) and ("On_Ramp" not in lane):
                left_lane_exists = 1
            if lane_i < num_weave_lanes - 1 and MLC_right == 0 and (("Weaving_Area" in lane) or ("On_Ramp1_Taper" in lane) or ("Off_Ramp2_Taper" in lane)):
                left_lane_exists = 1

            # --- YOUR REQUESTED RULE: no left moves in the 2nd half of the weaving section ---
            # Midpoint of weaving section in global coordinates
            # --- Midpoint of weaving section in global coordinates ---

            ## Hi Pedram,
            # I am still not sure about the following logic here: it appears EXIT_X is undefined in the code. I will leave this to you
            '''
            EXIT_MID_X = 0.2 * EXIT_X

            # (1) Off_Ramp vehicles: after midpoint, do NOT move left
            if dest == "Off_Ramp" and ("Weaving_Area" in lane) and (float(global_pos) >= EXIT_MID_X):
                left_lane_exists = 0

            # (2) Output vehicles: after midpoint, do NOT enter the exit-adjacent lane (_0)
            #     and if already in lane 0, force an escape left (mandatory).
            if dest == "Output" and ("Weaving_Area" in lane) and (float(global_pos) >= EXIT_MID_X):

                # Prevent moving right into lane 0 late in the weaving section
                if lane_i >= 1:
                    right_lane_exists = 0

                # If stuck in lane 0 late, force escape left
                if lane_i == 0:
                    MLC_left = 1
                    left_lane_exists = 1

                    # Optional (recommended): if CAV, broadcast strong intent to move left so others make a gap
                    if tech_type_by_id.get(str(vid), "SV") == "CAV":
                        ego_state = states_cav.get(
                            str(vid),
                            ns.VehicleState(str(vid), str(lane), float(global_pos), float(v),
                                            float(last_acc.get(str(vid), 0.0)), float(length), "CAV")
                        )
                        bus.broadcast_intent(ego_state, target_lane=str(left_lane_str), weight=1.0)
            '''

            # --- generalized midpoint lane-change restrictions (per-exit) ---
            _mid_exit_key = exit_key_for_vehicle(dest, lane, edge,
                                                on_off1_side=locals().get('on_offramp1_side', False),
                                                on_off2_side=locals().get('on_offramp2_side', False))
            if _mid_exit_key is not None and _mid_exit_key in EXIT_DEFS and in_mid_section(_mid_exit_key, lane):
                _sx = float(EXIT_DEFS[_mid_exit_key]['start_x'])
                _ex = float(EXIT_DEFS[_mid_exit_key]['end_x'])
                _mid_x = _sx + EXIT_MID_FRAC * (_ex - _sx)

                if dest == EXIT_DEFS[_mid_exit_key]['off_dest'] and float(global_pos) >= _mid_x:
                    left_lane_exists = 0

                if dest == 'Output' and float(global_pos) >= _mid_x:
                    if lane_i >= 1:
                        right_lane_exists = 0
                    if lane_i == 0:
                        MLC_left = 1
                        left_lane_exists = 1
                        _t = tech_type_by_id.get(str(vid), 'SV')
                        if _t in ('CAV', 'CAHV'):
                            ego_state = states_cav.get(
                                str(vid),
                                ns.VehicleState(str(vid), str(lane), float(global_pos), float(v),
                                                float(last_acc.get(str(vid), 0.0)), float(length), str(tech_label))
                            )
                            bus.broadcast_intent(ego_state, target_lane=str(left_lane_str), weight=1.0)

            # --- No immediate return to lane just left (MOBIL / C-MOBIL / DDM use left_lane_exists & right_lane_exists) ---
            # Cached from this step's subscription snapshot at loop top; no extra TraCI round trip.
            _rn = edge
            _ln = int(edge_lane_indexes[i_id])
            _nrt_map = no_return_until_by_vid.get(vid, {})

            def _no_return_blocks_target(tgt_lane: int) -> bool:
                return float(_nrt_map.get((_rn, int(tgt_lane)), 0.0)) > float(t)

            if left_lane_exists and _no_return_blocks_target(_ln + 1):
                left_lane_exists = 0
            if right_lane_exists and _ln > 0 and _no_return_blocks_target(_ln - 1):
                right_lane_exists = 0

            leader_exists, leader_len, leader_global_x, leader_v = find_leader(global_pos, lane_idx, lane, lanes, global_poses, vs, lengths)
            follower_exists, follower_len, follower_global_x, follower_v = find_follower(global_pos, lane_idx, lane, lanes, global_poses, vs, lengths)
            left_leader_exists, left_leader_len, left_leader_global_x, left_leader_v = find_leader(global_pos, lane_idx+1, left_lane_str, lanes, global_poses, vs, lengths)
            left_follower_exists, left_follower_len, left_follower_global_x, left_follower_v = find_follower(global_pos, lane_idx+1, left_lane_str, lanes, global_poses, vs, lengths)
            right_leader_exists, right_leader_len, right_leader_global_x, right_leader_v = find_leader(global_pos, lane_idx-1, right_lane_str, lanes, global_poses, vs, lengths)
            right_follower_exists, right_follower_len, right_follower_global_x, right_follower_v = find_follower(global_pos, lane_idx-1, right_lane_str, lanes, global_poses, vs, lengths)
            tech_label = tech_type_by_id.get(str(vid), "SV")

            # ===================== CONNECTED COOPERATIVE CONTROL (CAV + CAHV) =====================
            if tech_label in ("CAV", "CAHV"):
                ego_state = states_cav.get(
                    str(vid),
                    ns.VehicleState(
                        str(vid),
                        str(lane),
                        float(global_pos),
                        float(v),
                        float(last_acc.get(str(vid), 0.0)),
                        float(length),
                        str(tech_label),
                    ),
                )
                leader_state = ns.VehicleState(
                    "leader",
                    ego_state.lane,
                    float(leader_global_x),
                    float(leader_v),
                    0.0,
                    float(leader_len),
                    "SV",
                )
                pcs = bus.get_preceding_connected(ego_state)

                # C-IDM always uses IDM parameters (CAV -> merged_IDM_A, CAHV -> merged_IDM_CL via sampling)
                CF_params = IDM_params_by_id[str(vid)]
                cidm_cfg = user_input_data.get("CIDM_Params", {"K_v": 0.2, "K_a": 0.05, "s_ref": 50.0})
                acc = coop.c_idm_accel(
                    CF_params, ego_state, leader_state, pcs, bus,
                    K_v=float(cidm_cfg.get("K_v", 0.2)),
                    K_a=float(cidm_cfg.get("K_a", 0.05)),
                    s_ref=float(cidm_cfg.get("s_ref", 50.0)),
                )

                new_v = max(0.0, v + acc * t_step)
                traci.vehicle.setSpeed(vid, new_v)

                # C-MOBIL (always for connected vehicles) — expensive decision gated to LC steps.
                if do_lc_step:
                    Mobil_params = MOBIL_params_by_id[str(vid)]
                    cm_cfg = user_input_data.get("CMOBIL_Params", {"kappa": 0.1, "gamma": 1.0})
                    kappa = float(cm_cfg.get("kappa", 0.1))
                    gamma = float(cm_cfg.get("gamma", 1.0))
                    lane_choice = coop.c_mobil_decision(
                        Mobil_params, CF_params,
                        ego_state, leader_state,
                        ns.VehicleState("f", ego_state.lane, float(follower_global_x), float(follower_v), 0.0, float(follower_len), "SV"),
                        ns.VehicleState("ll", str(left_lane_str), float(left_leader_global_x), float(left_leader_v), 0.0, float(left_leader_len), "SV"),
                        ns.VehicleState("lf", str(left_lane_str), float(left_follower_global_x), float(left_follower_v), 0.0, float(left_follower_len), "SV"),
                        ns.VehicleState("rl", str(right_lane_str), float(right_leader_global_x), float(right_leader_v), 0.0, float(right_leader_len), "SV"),
                        ns.VehicleState("rf", str(right_lane_str), float(right_follower_global_x), float(right_follower_v), 0.0, float(right_follower_len), "SV"),
                        left_lane_exists, right_lane_exists, MLC_left, MLC_right,
                        bus=bus, kappa=kappa, gamma_lc=gamma,
                        left_lane_str=str(left_lane_str), right_lane_str=str(right_lane_str),
                    )
                    if lane_choice != 0:
                        traci.vehicle.changeLane(vid, old_lane_idx + lane_choice, TRACI_CHANGE_LANE_DURATION_S)

                continue
            # ===============================================================================
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
            acc = float(np.clip(acc, acc_min, acc_max))
            new_v = max(0.0, float(v) + acc * float(t_step))
            # --- Anti rear-end safety cap using TraCI leader (more robust than our global leader mapping) ---
            if ENABLE_SAFETY_SPEED_CAP:
                try:
                    leader_info = traci.vehicle.getLeader(vid, 200.0)  # (leaderID, gap in m) or None
                except:
                    leader_info = None
                if leader_info is not None:
                    leader_id, gap_bb = leader_info
                    try:
                        leader_speed = float(traci.vehicle.getSpeed(leader_id))
                    except:
                        leader_speed = float(leader_v)
                    # Use the vehicle's own desired minGap (s0) plus a small buffer
                    try:
                        s0_self = float(IDM_params_by_id[str(vid)][4])
                    except:
                        s0_self = 2.0
                    min_gap_bb = max(0.5, s0_self + float(SAFETY_EXTRA_GAP))
                    # Ensure after one step the bumper-to-bumper gap does not drop below min_gap_bb:
                    # gap_next = gap_bb + leader_speed*dt - new_v*dt >= min_gap_bb
                    v_cap = leader_speed + (float(gap_bb) - min_gap_bb) / float(t_step)
                    new_v = float(min(new_v, max(0.0, v_cap)))
            traci.vehicle.setSpeed(vid, new_v)
            
            # Non-CAV LC: gate the whole decision (MOBIL_LC / DDM_LC) to LC steps only.
            if do_lc_step and LC_model_name == "MOBIL":
                self_info = np.array([global_pos, v, length]); leader_info = np.array([leader_global_x, leader_v, leader_len] ); follower_info = np.array([follower_global_x, follower_v, follower_len] ); left_leader_info = np.array([left_leader_global_x, left_leader_v, left_leader_len] ); left_follower_info = np.array([left_follower_global_x, left_follower_v, left_follower_len] ); right_leader_info = np.array([right_leader_global_x, right_leader_v, right_leader_len] ); right_follower_info = np.array([right_follower_global_x, right_follower_v, right_follower_len] )
                Mobil_params = MOBIL_params_by_id[str(vid)]
                _idm_m = IDM_params_by_id[str(vid)]
                _use_pt = 1 if CF_model_name == "PT" else 0
                _cf_m = PT_params_by_id[str(vid)] if CF_model_name == "PT" else _idm_m
                if not (("Ramp" in lane) or ('0_0' in lane )) : # Note if "0_0" is in lane, that means the lane connects a ramp to main freeway and has no lane change options
                    lane_choice = MOBIL_LC(Mobil_params, _cf_m, _use_pt, _idm_m, left_lane_exists, right_lane_exists , MLC_left, MLC_right, self_info, leader_info, follower_info, left_leader_info, left_follower_info, right_leader_info, right_follower_info )
                    if lane_choice != 0:
                        traci.vehicle.changeLane(vid, old_lane_idx + lane_choice, TRACI_CHANGE_LANE_DURATION_S)
            elif do_lc_step and LC_model_name == "DDM":
                hw =  min(20, (leader_global_x-global_pos)/(v+0.01) )
                DDM_params = DDM_params_by_id[str(vid)]; alpha_h = DDM_params[0]; sigma = DDM_params[-1]
                if not vid in A_left_by_id: A_left_by_id[vid] = 10 - alpha_h*hw
                if not vid in A_right_by_id: A_right_by_id[vid] = 10 - alpha_h*hw
                all_evidences_left[vid].append(A_left_by_id[vid] ); all_evidences_right[vid].append(A_right_by_id[vid] )
                # SDE integrates over lc_dt = lc_every_steps * t_step since we now update evidence only every N steps.
                if left_lane_exists:
                    left_follow_gap = global_pos - left_follower_global_x - length
                    left_mu = DDM_LC(DDM_params, 1, MLC_left, left_follow_gap, left_leader_v, leader_v )
                    A_left_by_id[vid]  = A_left_by_id[vid]  + left_mu*lc_dt + np.random.normal()*sigma*np.sqrt(lc_dt)
                if right_lane_exists:
                    right_follow_gap = global_pos - right_follower_global_x - length
                    right_mu = DDM_LC(DDM_params, -1, MLC_right, right_follow_gap, right_leader_v, leader_v )
                    A_right_by_id[vid]  = A_right_by_id[vid]  + right_mu*lc_dt + np.random.normal()*sigma*np.sqrt(lc_dt)
                if A_right_by_id[vid]*right_lane_exists > A_left_by_id[vid]*left_lane_exists and right_lane_exists and A_right_by_id[vid]>20:
                    if right_follow_gap>min_gap and right_leader_global_x - global_pos - right_leader_len >min_gap:
                        traci.vehicle.changeLane(vid, old_lane_idx - 1, TRACI_CHANGE_LANE_DURATION_S)
                        del A_right_by_id[vid] # clear the evidence
                elif  A_right_by_id[vid]*right_lane_exists < A_left_by_id[vid]*left_lane_exists and left_lane_exists and A_left_by_id[vid]>20 and (not "Ramp" in lane) and (not "0_0" in lane):
                    if left_follow_gap>min_gap and left_leader_global_x - global_pos - left_leader_len >min_gap:
                        traci.vehicle.changeLane(vid, old_lane_idx + 1, TRACI_CHANGE_LANE_DURATION_S)
                        del A_left_by_id[vid] # evidence
        if collect_data and (step_count % collect_every_steps == 0):
            all_data["time"].append(np.ones(len(ids)) * t)
            all_data["id"].append(ids); all_data["type"].append(types); all_data["x"].append(xs); all_data["y"].append(ys); all_data["v"].append(vs); all_data["theta"].append(thetas); all_data["length"].append(lengths); all_data["road"].append(roads); all_data["lane"].append(lanes); all_data["lane_pos"].append(lane_poses)
    if collect_data and all_data is not None:
        for key in all_data:
            all_data[key] = np.concatenate(all_data[key]) if len(all_data[key]) else np.array([])
        all_data = pd.DataFrame(all_data)
    else:
        all_data = pd.DataFrame()
    traci.close()
    return all_data