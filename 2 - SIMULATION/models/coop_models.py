from __future__ import annotations
from typing import List, Optional
import numpy as np
from numba import njit
from . import ns


@njit
def _omega_from_gap_numba(gaps: np.ndarray, s_ref: float) -> np.ndarray:
    w = np.empty_like(gaps)
    s = 0.0
    for i in range(gaps.shape[0]):
        g = gaps[i]
        if g < 1e-3: g = 1e-3
        wi = np.exp(-g / s_ref)
        w[i] = wi
        s += wi
    if s > 0.0:
        for i in range(w.shape[0]):
            w[i] = w[i] / s
    return w


@njit
def idm_accel(IDM_params: np.ndarray, v: float, v_lead: float, x: float, x_lead: float, len_self: float,
              len_lead: float) -> float:
    T, a, b, v0, s0, delta = IDM_params
    gap = x_lead - x - 0.5 * (len_self + len_lead)
    if gap < 1e-3: gap = 1e-3
    dv = v - v_lead
    ab = a * b
    if ab < 1e-6: ab = 1e-6
    s_star = s0 + v * T + v * dv / (2.0 * np.sqrt(ab))
    acc = a * (1.0 - (v / max(1e-3, v0)) ** delta - (s_star / gap) ** 2)
    return acc


@njit
def c_idm_kernel(
        IDM_params: np.ndarray, v: float, a_self: float, x: float,
        v_lead: float, x_lead: float, len_self: float, len_lead: float,
        gaps: np.ndarray, vks: np.ndarray, aks: np.ndarray, K_v: float, K_a: float, s_ref: float,
        has_vv_ahead: int, vv_gap_ahead: float, vv_v_ahead: float, vv_len_ahead: float, vv_weight_ahead: float,
        has_vv_behind: int, vv_gap_behind: float, c_const: float
) -> float:
    T, a, b, v0, s0, delta = IDM_params

    # --- FIX 1: REAR VIRTUAL VEHICLE LOGIC ---
    # Only increase speed if the car behind is TOO CLOSE (gap < c_const).
    # Never decrease v0 (never create negative speed) based on rear gap.
    v0_adj = v0
    if has_vv_behind == 1:
        if vv_gap_behind < c_const:
            # We are blocking someone close; speed up.
            v0_adj = v0 * (1.0 + (c_const - vv_gap_behind) / c_const)
        # Else: Gap is large enough, do nothing (keep v0 standard)

    # Calculate Standard IDM (ACC)
    dv = v - v_lead
    s_star = s0 + v * T + v * dv / (2.0 * np.sqrt(a * b))
    eff_gap = x_lead - x - 0.5 * (len_self + len_lead)
    acc = a * (1.0 - (v / max(1e-3, v0_adj)) ** delta - (s_star / max(1e-3, eff_gap)) ** 2)

    # Apply CACC (Platoon) modifications if applicable
    if gaps.shape[0] > 0:
        w = _omega_from_gap_numba(gaps, s_ref)
        for i in range(gaps.shape[0]):
            acc += w[i] * (K_v * (vks[i] - v) + K_a * (aks[i] - a_self))

    # --- FIX 2: FRONT VIRTUAL VEHICLE LOGIC ---
    # Use interpolation instead of direct scaling to avoid capping acceleration.
    if has_vv_ahead == 1:
        # Cooperative buffer on top of s0 for virtual merge vehicle (C-IDM).
        s0_coop = s0 + 3.0

        # Use s0_coop instead of s0
        s_star_vv = s0_coop + v * T + v * (v - vv_v_ahead) / (2.0 * np.sqrt(a * b))
        acc_vv = a * (1.0 - (v / max(1e-3, v0_adj)) ** delta - (s_star_vv / max(1e-3, vv_gap_ahead)) ** 2)

        # Interpolate between our current plan (acc) and the virtual requirement (acc_vv)
        # If weight is 1.0 (committed), we fully respect acc_vv.
        # If weight is 0.1 (just signaling), we mostly ignore it.
        # However, for safety, we usually take the MINIMUM if the virtual car requires braking.

        if acc_vv < acc:
            # Virtual car requires braking (or less accel). We blend towards it.
            acc = (1.0 - vv_weight_ahead) * acc + vv_weight_ahead * acc_vv
        # If acc_vv > acc, the virtual car is far ahead/fast, so it doesn't restrict us. We ignore it.

    return acc


def c_idm_accel(
        IDM_params: np.ndarray, ego: ns.VehicleState, leader: ns.VehicleState,
        preceding_connected: List[ns.VehicleState], bus: ns.CommunicationBus,
        K_v: float = 0.2, K_a: float = 0.05, s_ref: float = 50.0, c_const: float = 15.0
) -> float:
    if preceding_connected:
        gaps = np.array([pc.global_pos - ego.global_pos for pc in preceding_connected], dtype=np.float64)
        vks = np.array([pc.v for pc in preceding_connected], dtype=np.float64)
        aks = np.array([pc.a for pc in preceding_connected], dtype=np.float64)
    else:
        gaps = np.zeros(0, dtype=np.float64)
        vks = np.zeros(0, dtype=np.float64)
        aks = np.zeros(0, dtype=np.float64)

    vv_a = bus.nearest_virtual_ahead(ego)
    has_vv_a = 0;
    vga = 0.0;
    vva = 0.0;
    vla = 0.0;
    vwa = 0.0

    if vv_a:
        raw_gap = float(vv_a.global_pos - ego.global_pos - 0.5 * (ego.length + vv_a.length))
        # only accept if it is truly ahead with positive bumper gap
        if raw_gap > 1.0:
            has_vv_a = 1
            vga = raw_gap
            vva = float(vv_a.v)
            vla = float(vv_a.length)
            vwa = float(vv_a.weight)
        else:
            has_vv_a = 0

    vv_b = bus.nearest_virtual_behind(ego)
    has_vv_b = 0;
    vgb = 0.0
    if vv_b:
        has_vv_b = 1
        vgb = float(ego.global_pos - vv_b.global_pos)

    return float(c_idm_kernel(
        IDM_params.astype(np.float64), float(ego.v), float(ego.a), float(ego.global_pos),
        float(leader.v), float(leader.global_pos), float(ego.length), float(leader.length),
        gaps, vks, aks, float(K_v), float(K_a), float(s_ref),
        has_vv_a, vga, vva, vla, vwa, has_vv_b, vgb, float(c_const)
    ))


# (Note: c_mobil_decision should be the updated version you already have)
def c_mobil_decision(
        Mobil_params: np.ndarray, IDM_params: np.ndarray, ego: ns.VehicleState,
        leader: ns.VehicleState, follower: ns.VehicleState,
        left_leader: ns.VehicleState, left_follower: ns.VehicleState,
        right_leader: ns.VehicleState, right_follower: ns.VehicleState,
        left_lane_exists: int, right_lane_exists: int, MLC_left: int, MLC_right: int,
        bus: ns.CommunicationBus, kappa: float = 0.1, gamma_lc: float = 1.0,
        left_lane_str: str = "", right_lane_str: str = ""
) -> int:
    politeness = float(Mobil_params[0])
    a_thresh = float(Mobil_params[1])
    b_safe_disc = float(Mobil_params[2])
    b_safe_mand = float(Mobil_params[3])
    T, alpha, b, v0, s0, delta = IDM_params

    # Low-speed IDM safety slack (m/s²); zero above threshold so calibrated b_safe applies at highway speeds.
    LOW_SPEED_SLACK_THRESH_MPS = 3.0
    DISC_DECEL_SLACK = 1.5
    MAND_DECEL_SLACK = 2.5

    def safety_ok(nf: ns.VehicleState, nl: ns.VehicleState, is_mandatory: bool = False) -> bool:
        b_safe = b_safe_mand if is_mandatory else b_safe_disc
        min_gap = 0.05 if is_mandatory else 0.2

        a_nf = idm_accel(IDM_params, nf.v, nl.v, nf.global_pos, nl.global_pos, nf.length, nl.length)
        # IDM over-estimates braking demand at short gaps when both vehicles are slow; relax only then.
        if min(float(nf.v), float(nl.v)) < LOW_SPEED_SLACK_THRESH_MPS:
            decel_slack = MAND_DECEL_SLACK if is_mandatory else DISC_DECEL_SLACK
        else:
            decel_slack = 0.0
        if a_nf < -(b_safe + decel_slack):
            return False

        s = nl.global_pos - nf.global_pos - 0.5 * (nf.length + nl.length)
        return s > min_gap

        # ... (inside c_mobil_decision) ...

    # Check discretionary safety first (False flag)
    is_left_safe = left_lane_exists and safety_ok(left_follower, ego, False) and safety_ok(ego, left_leader, False)
    is_right_safe = right_lane_exists and safety_ok(right_follower, ego, False) and safety_ok(ego, right_leader,
                                                                                                  False)
    if MLC_left == 1:
        # FIX: Check safety with is_mandatory=True
        if left_lane_exists and safety_ok(left_follower, ego, True) and safety_ok(ego, left_leader, True):
            return 1
        else:
            return 0

    if MLC_right == 1:
        # FIX: Check safety with is_mandatory=True
        if right_lane_exists and safety_ok(right_follower, ego, True) and safety_ok(ego, right_leader, True):
            return -1
        else:
            return 0


    a_self0 = idm_accel(IDM_params, ego.v, leader.v, ego.global_pos, leader.global_pos, ego.length, leader.length)
    a_fol0 = idm_accel(IDM_params, follower.v, ego.v, follower.global_pos, ego.global_pos, follower.length, ego.length)

    best_dir = 0
    best_gain = -999.0

    if is_left_safe:
        a_selfL = idm_accel(IDM_params, ego.v, left_leader.v, ego.global_pos, left_leader.global_pos, ego.length,
                            left_leader.length)
        a_lf0 = idm_accel(IDM_params, left_follower.v, left_leader.v, left_follower.global_pos, left_leader.global_pos,
                          left_follower.length, left_leader.length)
        a_lf1 = idm_accel(IDM_params, left_follower.v, ego.v, left_follower.global_pos, ego.global_pos,
                          left_follower.length, ego.length)
        gain_left = (a_selfL - a_self0) + politeness * (
                    a_lf1 - a_lf0 + a_fol0 - idm_accel(IDM_params, follower.v, leader.v, follower.global_pos,
                                                       leader.global_pos, follower.length, leader.length))
        if gain_left > a_thresh:
            best_gain = gain_left;
            best_dir = 1

    if is_right_safe:
        a_selfR = idm_accel(IDM_params, ego.v, right_leader.v, ego.global_pos, right_leader.global_pos, ego.length,
                            right_leader.length)
        a_rf0 = idm_accel(IDM_params, right_follower.v, right_leader.v, right_follower.global_pos,
                          right_leader.global_pos, right_follower.length, right_leader.length)
        a_rf1 = idm_accel(IDM_params, right_follower.v, ego.v, right_follower.global_pos, ego.global_pos,
                          right_follower.length, ego.length)
        gain_right = (a_selfR - a_self0) + politeness * (
                    a_rf1 - a_rf0 + a_fol0 - idm_accel(IDM_params, follower.v, leader.v, follower.global_pos,
                                                       leader.global_pos, follower.length, leader.length))
        if gain_right > a_thresh and gain_right > best_gain:
            best_dir = -1

    if best_dir != 0:
        target_lane = left_lane_str if best_dir == 1 else right_lane_str
        s_to_leader = leader.global_pos - ego.global_pos - 0.5 * (ego.length + leader.length)
        w_v = min(1.0, max(0.0, kappa * (bus.range_m - s_to_leader)))
        bus.broadcast_intent(ego, target_lane=target_lane, weight=w_v)

    return int(best_dir)