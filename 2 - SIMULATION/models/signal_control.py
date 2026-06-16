"""
Signal control module: Webster's formula and SUMO phase generation.

- Default: compute optimal cycle length and green times from volumes (Webster).
- User can override with manual cycle and green times via GUI / user_input_data["Signal_Control"].
"""

from __future__ import division
import math
import xml.etree.ElementTree as ET


# ---------- Webster's formula constants ----------
DEFAULT_SATURATION_FLOW_PER_LANE = 1800.0   # veh/h per lane
DEFAULT_LOST_TIME_PER_PHASE = 4.0            # seconds (amber + all-red equivalent per phase)
MIN_CYCLE = 30
MAX_CYCLE = 150
ROUND_CYCLE_TO = 5


def webster_optimal_cycle_length(Y, L):
    """
    Webster's formula: C0 = (1.5*L + 5) / (1 - Y).
    Y = sum of critical flow ratios (y_i = q_i / s_i).
    L = total lost time per cycle (seconds).
    """
    if Y >= 1.0:
        return MAX_CYCLE
    c0 = (1.5 * L + 5.0) / (1.0 - Y)
    c0 = max(MIN_CYCLE, min(MAX_CYCLE, c0))
    # Round to nearest ROUND_CYCLE_TO
    c0 = round(c0 / ROUND_CYCLE_TO) * ROUND_CYCLE_TO
    return int(c0)


def webster_effective_greens(y_list, C, L):
    """
    Allocate effective green time to each phase: g_i = (y_i / Y) * (C - L).
    Returns list of effective green times in seconds (can be rounded).
    """
    Y = sum(y_list)
    if Y <= 0:
        n = len(y_list)
        return [max(5, (C - L) // n)] * n
    return [max(5, int(round((yi / Y) * (C - L)))) for yi in y_list]


# ---------- Single intersection ----------
# Phase order in template: EW green, EW yellow(5), EW yellow(3), all-red(1), NS green, NS yellow(5), NS yellow(3), all-red(1).
# We only change the two green durations (first and fifth phase); others stay fixed.
SINGLE_FIXED_PHASES = [
    (5, "gGggGGGgrrrrrrrrgGggGGGgrrrrrrrrrrrr"),   # yellow
    (3, "yyyyyyyyrrrrrrrryyyyyyyyrrrrrrrrrrrr"),
    (1, "rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"),
    (5, "rrrrrrrrgGggGGGgrrrrrrrrgGggGGGgrrrr"),
    (3, "rrrrrrrryyyyyyyyrrrrrrrryyyyyyyyrrrr"),
    (1, "rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"),
]
# State strings for EW green and NS green (must match template lane count; template EW3_NS3 has 35 chars)
def _single_ew_state(n_ew, n_ns):
    # EW: first n_ew*2 chars for EB+WB, then NS red, etc. Template uses g/G for green, r for red.
    ne = n_ew
    nn = n_ns
    # Simplified: use same pattern as template "gGggGGGgrrrrrrrrgGggGGGgrrrrrrrrrGrG" -> EW green, NS red, then EW red, NS green
    s_ew_green = "g" + "G" * (ne - 1) + "g" * ne + "r" * (nn * 2) + "g" + "G" * (ne - 1) + "g" * ne + "r" * (nn * 2) + "rGrG"
    s_ns_green = "r" * (ne * 2) + "g" + "G" * (nn - 1) + "g" * nn + "r" * (ne * 2) + "g" + "G" * (nn - 1) + "g" * nn + "GrGr"
    return s_ew_green[:35], s_ns_green[:35]  # trim to 35 if needed


def get_single_intersection_phase_states(num_ew_lanes, num_ns_lanes):
    """Return (ew_green_state, ns_green_state) for the template; state length may vary by template (35 for EW3_NS3)."""
    # Match template: EW3_NS3 has 35 chars
    # gGggGGGgrrrrrrrrgGggGGGgrrrrrrrrrGrG = EW green (EB+WB), NS red, EW green (cont), NS red, end
    # rrrrrrrrgGggGGGgrrrrrrrrgGggGGGgGrGr = EW red, NS green, EW red, NS green
    base_ew = "gGggGGGgrrrrrrrrgGggGGGgrrrrrrrrrGrG"
    base_ns = "rrrrrrrrgGggGGGgrrrrrrrrgGggGGGgGrGr"
    return base_ew, base_ns


def compute_single_webster(vehicle_flows, num_ew_lanes, num_ns_lanes,
                          saturation_per_lane=DEFAULT_SATURATION_FLOW_PER_LANE,
                          lost_time_per_phase=DEFAULT_LOST_TIME_PER_PHASE):
    """
    Compute Webster-based signal plan for a single 4-arm intersection.
    vehicle_flows: dict with keys "East-Bound", "West-Bound", "North-Bound", "South-Bound",
                   each with "volume" (veh/h).
    Returns: dict with cycle_length, phases list of (duration, state).
    """
    q_e = float(vehicle_flows.get("East-Bound", {}).get("volume", 0))
    q_w = float(vehicle_flows.get("West-Bound", {}).get("volume", 0))
    q_n = float(vehicle_flows.get("North-Bound", {}).get("volume", 0))
    q_s = float(vehicle_flows.get("South-Bound", {}).get("volume", 0))

    s_ew = saturation_per_lane * num_ew_lanes
    s_ns = saturation_per_lane * num_ns_lanes
    if s_ew <= 0:
        s_ew = DEFAULT_SATURATION_FLOW_PER_LANE
    if s_ns <= 0:
        s_ns = DEFAULT_SATURATION_FLOW_PER_LANE

    y_ew = (q_e + q_w) / s_ew if s_ew else 0
    y_ns = (q_n + q_s) / s_ns if s_ns else 0
    Y = y_ew + y_ns
    L = 2 * lost_time_per_phase  # two main phases
    C = webster_optimal_cycle_length(Y, L)
    greens = webster_effective_greens([y_ew, y_ns], C, L)
    g_ew, g_ns = greens[0], greens[1]

    ew_state, ns_state = get_single_intersection_phase_states(num_ew_lanes, num_ns_lanes)
    phases = [
        (g_ew, ew_state),
        (5, "gGggGGGgrrrrrrrrgGggGGGgrrrrrrrrrrrr"),
        (3, "yyyyyyyyrrrrrrrryyyyyyyyrrrrrrrrrrrr"),
        (1, "rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"),
        (g_ns, ns_state),
        (5, "rrrrrrrrgGggGGGgrrrrrrrrgGggGGGgrrrr"),
        (3, "rrrrrrrryyyyyyyyrrrrrrrryyyyyyyyrrrr"),
        (1, "rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"),
    ]
    return {"cycle_length": C, "phases": phases}


def get_single_signal_plan(user_input_data, num_ew_lanes, num_ns_lanes):
    """
    Resolve signal plan for single intersection: either Webster from volumes or manual override.
    user_input_data["Signal_Control"] can be:
      - None / missing / {"use_webster": True}: use Webster from Vehicle_Flows.
      - {"use_webster": False, "cycle_length": C, "green_ew": g1, "green_ns": g2}: manual.
    Returns dict with cycle_length and phases (list of (duration, state)).
    """
    sig = user_input_data.get("Signal_Control") or {}
    use_webster = sig.get("use_webster", True)

    if use_webster:
        flows = user_input_data.get("Vehicle_Flows") or {}
        return compute_single_webster(flows, num_ew_lanes, num_ns_lanes)

    # Manual
    C = int(sig.get("cycle_length", 90))
    g_ew = int(sig.get("green_ew", 36))
    g_ns = int(sig.get("green_ns", 36))
    ew_state, ns_state = get_single_intersection_phase_states(num_ew_lanes, num_ns_lanes)
    phases = [
        (g_ew, ew_state),
        (5, "gGggGGGgrrrrrrrrgGggGGGgrrrrrrrrrrrr"),
        (3, "yyyyyyyyrrrrrrrryyyyyyyyrrrrrrrrrrrr"),
        (1, "rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"),
        (g_ns, ns_state),
        (5, "rrrrrrrrgGggGGGgrrrrrrrrgGggGGGgrrrr"),
        (3, "rrrrrrrryyyyyyyyrrrrrrrryyyyyyyyrrrr"),
        (1, "rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"),
    ]
    return {"cycle_length": C, "phases": phases}


# ---------- Multi (arterial) intersection ----------
# Each intersection has 4 phases: green EW, yellow 3, green NS, yellow 3.
# Template state strings (e.g. 24 chars): "GGGGggrrrrrrGGGGggrrrrrr", "yyyyyyrrrrrryyyyyyrrrrrr", etc.
MULTI_FIXED_YELLOW = 3


def _multi_phase_states(num_ew_lanes, num_ns_lanes):
    """Return (ew_green_state, ew_yellow_state, ns_green_state, ns_yellow_state) for multi template (24 chars)."""
    # Template exact: "GGGGggrrrrrrGGGGggrrrrrr" (EW green), "rrrrrrGGGGggrrrrrrGGGGgg" (NS green)
    ew_g = "GGGGggrrrrrrGGGGggrrrrrr"
    ew_y = "yyyyyyrrrrrryyyyyyrrrrrr"
    ns_g = "rrrrrrGGGGggrrrrrrGGGGgg"
    ns_y = "rrrrrryyyyyyrrrrrryyyyyy"
    return ew_g, ew_y, ns_g, ns_y


def compute_multi_webster_per_intersection(in_flows, origins_, destinations_, user_edge_names,
                                          in_names, num_ew_lanes, num_ns_lanes,
                                          saturation_per_lane=DEFAULT_SATURATION_FLOW_PER_LANE,
                                          lost_time_per_phase=DEFAULT_LOST_TIME_PER_PHASE):
    """
    Compute approach volumes for each of the 3 arterial intersections and return Webster plan per intersection.
    in_flows: list of flow values in same order as origins_/destinations_ (OD pairs).
    Returns list of 3 dicts, each with cycle_length and phases (4 phases: g_ew, y, g_ns, y).
    """
    # Map OD to approach volumes per intersection.
    # OD origins: EB01 (West), WB34 (East), NB1_South, NB2_South, NB3_South, SB1_North, SB2_North, SB3_North.
    # OD destinations: WB01, EB34, SB1_South, SB2_South, SB3_South, NB1_North, NB2_North, NB3_North.
    # No vehicles start on mid-link edges (WB12, EB12, WB23, EB23), so EW approach volumes must be
    # derived from (origin, destination): which OD pairs pass through each intersection on the mainline.
    # Int1 west approach = from EB01; Int1 east approach = from WB34 to West/North West/South West (WB01, NB1_North, SB1_South).
    # Int2 = from EB01 to East/North East/South East + from WB34 to West/North West/South West.
    # Int3 west approach = from EB01 to East/North East/South East; Int3 east approach = from WB34.
    def approach_volumes_for_intersection(k):
        """k in {1,2,3}. Return (q_ew, q_ns) for intersection k."""
        q_ew = 0.0
        q_ns = 0.0
        # Destinations west of Int2 (approach Int1 from east): WB01, NB1_North, SB1_South
        dest_west_of_int2 = ("WB01", "NB1_North", "SB1_South")
        # Destinations east of Int2 (approach Int3 from west): EB34, NB3_North, SB3_South
        dest_east_of_int2 = ("EB34", "NB3_North", "SB3_South")
        for i in range(len(origins_)):
            o = origins_[i]
            d = destinations_[i]
            f = in_flows[i] if i < len(in_flows) else 0
            # NS: approaches are origin edges (side-street entries)
            if k == 1:
                if o == "NB1_South": q_ns += f
                if o == "SB1_North": q_ns += f
            elif k == 2:
                if o == "NB2_South": q_ns += f
                if o == "SB2_North": q_ns += f
            else:
                if o == "NB3_South": q_ns += f
                if o == "SB3_North": q_ns += f
            # EW: mainline approaches from (origin, destination) pairs
            if k == 1:
                if o == "EB01": q_ew += f  # west approach to Int1
                if o == "WB34" and d in dest_west_of_int2: q_ew += f  # east approach to Int1
            elif k == 2:
                if o == "EB01" and d in dest_east_of_int2: q_ew += f  # west approach to Int2
                if o == "WB34" and d in dest_west_of_int2: q_ew += f  # east approach to Int2
            else:
                if o == "EB01" and d in dest_east_of_int2: q_ew += f  # west approach to Int3
                if o == "WB34": q_ew += f  # east approach to Int3
        return q_ew, q_ns

    s_ew = saturation_per_lane * num_ew_lanes
    s_ns = saturation_per_lane * num_ns_lanes
    if s_ew <= 0:
        s_ew = DEFAULT_SATURATION_FLOW_PER_LANE
    if s_ns <= 0:
        s_ns = DEFAULT_SATURATION_FLOW_PER_LANE

    plans = []
    ew_g, ew_y, ns_g, ns_y = _multi_phase_states(num_ew_lanes, num_ns_lanes)
    for k in (1, 2, 3):
        q_ew, q_ns = approach_volumes_for_intersection(k)
        y_ew = q_ew / s_ew
        y_ns = q_ns / s_ns
        Y = y_ew + y_ns
        L = 2 * lost_time_per_phase
        C = webster_optimal_cycle_length(Y, L)
        greens = webster_effective_greens([y_ew, y_ns], C, L)
        g_ew, g_ns = greens[0], greens[1]
        phases = [
            (g_ew, ew_g),
            (MULTI_FIXED_YELLOW, ew_y),
            (g_ns, ns_g),
            (MULTI_FIXED_YELLOW, ns_y),
        ]
        plans.append({"cycle_length": C, "phases": phases})
    return plans


def get_webster_display_values(user_input_data, scenario):
    """
    Compute Webster-based cycle and green times for UI display.
    user_input_data: dict with Vehicle_Flows and Geometry (Num_Lanes_EW, Num_Lanes_NS).
    scenario: "Single Intersection" or "Arterial".
    Returns:
      - Single: {"cycle_length": int, "green_ew": int, "green_ns": int} or None
      - Arterial: [{"cycle_length": int, "green_ew": int, "green_ns": int}, ...] (list of 3) or None
    """
    geom = user_input_data.get("Geometry") or {}
    num_ew = int(geom.get("Num_Lanes_EW", 3))
    num_ns = int(geom.get("Num_Lanes_NS", 3))
    flows = user_input_data.get("Vehicle_Flows") or {}

    if scenario == "Single Intersection":
        plan = compute_single_webster(flows, num_ew, num_ns)
        phases = plan["phases"]
        return {
            "cycle_length": plan["cycle_length"],
            "green_ew": phases[0][0],
            "green_ns": phases[4][0],
        }

    if scenario == "Arterial":
        in_names = ["EB01", "WB34", "NB1_South", "NB2_South", "NB3_South", "SB1_North", "SB2_North", "SB3_North"]
        out_names = ["WB01", "EB34", "SB1_South", "SB2_South", "SB3_South", "NB1_North", "NB2_North", "NB3_North"]
        user_edge_names = ["West", "East", "South West", "South", "South East", "North West", "North", "North East"]
        origins_ = []
        destinations_ = []
        in_flows = []
        for i in range(len(in_names)):
            in_user = user_edge_names[i]
            for j in range(len(out_names)):
                if j == i:
                    continue
                out_user = user_edge_names[j]
                try:
                    val = flows.get(in_user, {}).get(out_user)
                    in_flows.append(int(val) if val is not None else 0)
                except (TypeError, ValueError):
                    in_flows.append(0)
                origins_.append(in_names[i])
                destinations_.append(out_names[j])
        plans = compute_multi_webster_per_intersection(
            in_flows, origins_, destinations_, user_edge_names, in_names, num_ew, num_ns
        )
        return [
            {
                "cycle_length": p["cycle_length"],
                "green_ew": p["phases"][0][0],
                "green_ns": p["phases"][2][0],
            }
            for p in plans
        ]

    return None


def get_multi_signal_plans(user_input_data, in_flows, origins_, destinations_, user_edge_names,
                           in_names, num_ew_lanes, num_ns_lanes):
    """
    Resolve signal plans for all 3 arterial intersections.
    Returns list of 3 dicts, each with cycle_length and phases.
    """
    sig = user_input_data.get("Signal_Control") or {}
    use_webster = sig.get("use_webster", True)

    ew_g, ew_y, ns_g, ns_y = _multi_phase_states(num_ew_lanes, num_ns_lanes)

    if use_webster:
        return compute_multi_webster_per_intersection(
            in_flows, origins_, destinations_, user_edge_names, in_names,
            num_ew_lanes, num_ns_lanes
        )

    # Manual: per-intersection plans from manual_plans (list of 3), or fallback to single set for all
    manual_plans = sig.get("manual_plans")
    if manual_plans and len(manual_plans) >= 3:
        result = []
        for p in manual_plans[:3]:
            C = int(p.get("cycle_length", 90))
            g_ew = int(p.get("green_ew", 42))
            g_ns = int(p.get("green_ns", 42))
            phases = [
                (g_ew, ew_g),
                (MULTI_FIXED_YELLOW, ew_y),
                (g_ns, ns_g),
                (MULTI_FIXED_YELLOW, ns_y),
            ]
            result.append({"cycle_length": C, "phases": phases})
        return result
    C = int(sig.get("cycle_length", 90))
    g_ew = int(sig.get("green_ew", 42))
    g_ns = int(sig.get("green_ns", 42))
    phases = [
        (g_ew, ew_g),
        (MULTI_FIXED_YELLOW, ew_y),
        (g_ns, ns_g),
        (MULTI_FIXED_YELLOW, ns_y),
    ]
    return [{"cycle_length": C, "phases": phases} for _ in range(3)]


# ---------- Apply plan to net file ----------
# Single-intersection net template: phase 0 = NS (minor) green, phase 4 = EW (mainline) green (link order SB, WB, NB, EB).
# Our plan: phases[0]=EW green, phases[4]=NS green. So we swap when writing: net phase 0 <- plan[4], net phase 4 <- plan[0].
def apply_single_signal_to_net(net_file_path, signal_plan):
    """Update green phase durations in tlLogic. Net template phase 0=NS green, phase 4=EW green; our plan has [0]=EW, [4]=NS, so we swap."""
    tree = ET.parse(net_file_path)
    root = tree.getroot()
    phases = signal_plan["phases"]
    for tll in root.findall("tlLogic"):
        if tll.get("programID") == "0":
            plist = tll.findall("phase")
            for i, p in enumerate(plist):
                if i >= len(phases):
                    continue
                if i == 0:
                    # Net phase 0 = NS green -> use plan g_ns (phases[4])
                    p.set("duration", str(phases[4][0] if len(phases) > 4 else phases[0][0]))
                elif i == 4:
                    # Net phase 4 = EW green -> use plan g_ew (phases[0])
                    p.set("duration", str(phases[0][0]))
                else:
                    p.set("duration", str(phases[i][0]))
            break
    tree.write(net_file_path, encoding="utf-8", xml_declaration=True, default_namespace="")


def apply_multi_signal_to_net(net_file_path, signal_plans, offset_1_2=0, offset_2_3=0):
    """
    signal_plans: list of 3 dicts (cycle_length, phases) for intersection 1, 2, 3.
    Update duration for each phase and set offset for EW (main road) arterial coordination.
    Offsets are for the East-West green/cycle: Int1 (west) reference 0, then Int2 (mid), then Int3 (east).
    Direction: West → East (left to right). Int2's cycle starts offset_1_2 s after Int1's; Int3 starts offset_2_3 s after Int2's.
    tl_ids in file: Intersection1, Intersection3, J3 (J3 = middle = Int2).
    Plan mapping: Intersection1->plans[0], J3->plans[1], Intersection3->plans[2].
    Multi net template phase order: phase 0 = NS (minor) green, phase 1 = NS yellow, phase 2 = EW (mainline) green, phase 3 = EW yellow.
    Our plan order: phases[0]=EW green, phases[2]=NS green. So we map net phase 0 <- plan phase 2 (g_ns), net phase 2 <- plan phase 0 (g_ew).
    """
    tree = ET.parse(net_file_path)
    root = tree.getroot()
    tl_ids = ["Intersection1", "Intersection3", "J3"]
    offsets_by_id = {"Intersection1": 0, "J3": offset_1_2, "Intersection3": offset_1_2 + offset_2_3}
    plan_idx_by_id = {"Intersection1": 0, "J3": 1, "Intersection3": 2}
    # Net file: phase 0 = NS green, phase 2 = EW green. Our plan: index 0 = EW, index 2 = NS.
    plan_phase_to_net_phase = (2, 1, 0, 3)  # net phase i gets duration from plan phase plan_phase_to_net_phase[i]
    tll_list = list(root.findall("tlLogic"))
    for tll in tll_list:
        tlid = tll.get("id")
        if tlid not in tl_ids:
            continue
        plan = signal_plans[plan_idx_by_id[tlid]] if plan_idx_by_id[tlid] < len(signal_plans) else signal_plans[0]
        tll.set("offset", str(offsets_by_id[tlid]))
        plist = tll.findall("phase")
        for i, p in enumerate(plist):
            plan_idx = plan_phase_to_net_phase[i] if i < len(plan_phase_to_net_phase) else i
            if plan_idx < len(plan["phases"]):
                p.set("duration", str(plan["phases"][plan_idx][0]))
    tree.write(net_file_path, encoding="utf-8", xml_declaration=True, default_namespace="")
