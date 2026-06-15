from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

@dataclass(frozen=True)
class VehicleState:
    vid: str
    lane: str
    global_pos: float
    v: float
    a: float
    length: float
    tech: str

@dataclass
class VirtualVehicle:
    owner_vid: str
    lane: str
    global_pos: float
    v: float
    a: float
    length: float
    weight: float

class CommunicationBus:
    def __init__(
            self,
            range_m: float = 30.0,
            m_max: int = 5,
            latency_steps: int = 0,
            loss_rate: float = 0.0,
            connected_tech: Tuple[str, ...] = ("CV", "CAV"),
    ):
        self.range_m = float(range_m)
        self.m_max = int(m_max)
        self.latency_steps = int(max(0, latency_steps))
        self.loss_rate = float(loss_rate)
        self.connected_tech = tuple(connected_tech)
        self._buffers: List[Dict[str, VehicleState]] = [dict() for _ in range(self.latency_steps + 1)]
        self._virtuals: Dict[str, VirtualVehicle] = {}

    def step(self, states: Dict[str, VehicleState]) -> None:
        transmitted = {}
        for vid, state in states.items():
            if random.random() >= self.loss_rate:
                transmitted[vid] = state

        if self.latency_steps > 0:
            self._buffers = [transmitted] + self._buffers[:-1]
        else:
            self._buffers[0] = transmitted

        present = set(states.keys())

        # --- FIX: keep virtual vehicles synchronized with their owners ---
        for owner_vid, vv in list(self._virtuals.items()):
            if owner_vid in states:
                s = states[owner_vid]
                vv.global_pos = s.global_pos
                vv.v = s.v
                vv.a = s.a
                vv.length = s.length

        # existing cleanup (keep)
        for owner in list(self._virtuals.keys()):
            if owner not in present:
                del self._virtuals[owner]

    def visible_states(self) -> Dict[str, VehicleState]:
        return self._buffers[-1]

    def broadcast_intent(
        self,
        owner_state: VehicleState,
        target_lane: str,
        weight: float,
        v_project: Optional[float] = None,
        a_project: Optional[float] = None,
    ) -> None:
        w = float(max(0.0, min(1.0, weight)))
        self._virtuals[owner_state.vid] = VirtualVehicle(
            owner_vid=owner_state.vid,
            lane=target_lane,
            global_pos=owner_state.global_pos,
            v=owner_state.v if v_project is None else float(v_project),
            a=owner_state.a if a_project is None else float(a_project),
            length=owner_state.length,
            weight=w,
        )

    def get_preceding_connected(self, ego: VehicleState) -> List[VehicleState]:
        snap = self.visible_states()
        out: List[VehicleState] = []
        for s in snap.values():
            if s.vid == ego.vid: continue
            if s.tech not in self.connected_tech: continue
            if s.lane != ego.lane: continue
            gap = s.global_pos - ego.global_pos
            if gap <= 0.0 or gap > self.range_m: continue
            out.append(s)
        out.sort(key=lambda st: st.global_pos)
        return out[: self.m_max]

    def get_virtuals_on_lane(self, lane: str) -> List[VirtualVehicle]:
        return [vv for vv in self._virtuals.values() if vv.lane == lane and vv.weight > 0.0]

    def nearest_virtual_ahead(self, ego: VehicleState) -> Optional[VirtualVehicle]:
        cands = []
        for vv in self.get_virtuals_on_lane(ego.lane):
            gap = vv.global_pos - ego.global_pos
            if gap <= 0.0 or gap > self.range_m: continue
            cands.append((gap, vv))
        if not cands: return None
        cands.sort(key=lambda x: x[0])
        return cands[0][1]

    def nearest_virtual_behind(self, ego: VehicleState) -> Optional[VirtualVehicle]:
        cands = []
        for vv in self.get_virtuals_on_lane(ego.lane):
            gap = ego.global_pos - vv.global_pos
            if gap <= 0.0 or gap > self.range_m: continue
            cands.append((gap, vv))
        if not cands: return None
        cands.sort(key=lambda x: x[0])
        return cands[0][1]
