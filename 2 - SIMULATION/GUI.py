import sys
import time
import os
import csv
import pandas as pd
import sys
import traci
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
from models.freeway_sim import run_sim_freeway
from models.multi_inter_sim import run_sim_multi_inter
from models.single_inter_sim import run_sim_single_inter
from models.tgsim_sim import run_sim_tgsim
from models import signal_control
from models.paths import MODEL_PARAMS_DIR, mobil_default_stats_from_csv

import subprocess
import random
from sklearn.cluster import KMeans

from PyQt5.QtWidgets import (
    QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QRadioButton, QButtonGroup,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QFileDialog, QFrame, QGridLayout, QSlider, QSizePolicy, QGroupBox,
    QMessageBox, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QObject, QTimer
)
from PyQt5.QtGui import QFont

# --- Constants (Used in the original code, defined here for the example) ---
PAGE_WIDTH = 600
PAGE_HEIGHT = 400
FONT_SIZE = 12
responses = {}  # Placeholder for the responses dictionary


def _param_info_icon(font_size=None):
    """Small circled 'i' label; hover shows QToolTip text set by caller."""
    fz = FONT_SIZE if font_size is None else font_size
    ico = QLabel("i")
    ico.setFixedSize(18, 18)
    ico.setAlignment(Qt.AlignCenter)
    font = QFont()
    font.setPointSize(max(8, fz - 1))
    font.setBold(True)
    ico.setFont(font)
    ico.setStyleSheet(
        "QLabel { color: #1565c0; background-color: #e8f4fc; border: 1px solid #90caf9; border-radius: 9px; }"
    )
    ico.setCursor(Qt.PointingHandCursor)
    return ico


def _add_label_with_info(layout, label_widget, tooltip_text):
    """Append label_widget and an info icon with tooltip to a QHBoxLayout (no stretch)."""
    info = _param_info_icon()
    info.setToolTip(tooltip_text)
    layout.addWidget(label_widget)
    layout.addWidget(info)
    return info


# --- Hover tooltips for model parameter rows (exact keys match GUI labels) ---
_CF_PARAM_TOOLTIPS = {
    # Intelligent Driver Model (IDM)
    "T": (
        "Desired (minimum) time headway in seconds. Larger values mean drivers keep more "
        "spacing at a given speed and react more conservatively to the leader."
    ),
    "a": "Maximum comfortable acceleration (m/s²). Caps how quickly a vehicle can speed up.",
    "b": (
        "Comfortable deceleration (m/s²). Controls braking strength when approaching a slower "
        "vehicle; higher values imply sharper braking."
    ),
    "v_0": "Free-flow / desired speed (m/s). The speed drivers target when the road ahead is clear.",
    "s_0": (
        "Minimum bumper-to-bumper gap at standstill (m). Sets the jam density and close-quarters "
        "spacing in the IDM gap equation."
    ),
    # Prospect Theory (PT) car-following
    "T_max": (
        "Upper bound on effective time headway used in the PT formulation. Together with other "
        "terms it limits how aggressive following can become."
    ),
    "α": "Shape / sensitivity parameter in the prospect-theory value weighting for gains vs losses.",
    "β": "Second shape parameter balancing risk and reward in the PT utility for acceleration choices.",
    "W_c": "Weight on cumulative prospect components tied to spacing / closing speed in the PT model.",
    "W_m": "Weight on momentum-style prospect terms (continuation of previous acceleration decisions).",
    "Gamma1": "Curvature exponent for the gains portion of the prospect value function.",
    "Gamma2": "Curvature exponent for the losses portion of the prospect value function.",
}

_CIDM_TOOLTIPS = {
    "K_v": (
        "Cooperative gain on relative speed / velocity matching for CAVs using the C-IDM extension. "
        "Higher values react more strongly to differences from a reference motion."
    ),
    "K_a": (
        "Cooperative gain on acceleration alignment. Larger values make the CAV coordinate "
        "acceleration more strongly with the cooperative reference."
    ),
    "s_ref": (
        "Reference spacing (m) used in cooperative car-following. Vehicles steer toward this "
        "target gap when coordinating with neighbors."
    ),
}

_COMM_PARAM_TOOLTIPS = {
    "Communication Range (m)": (
        "Maximum distance (m) over which a vehicle can receive V2V messages. Beyond this range, "
        "updates from that neighbor are ignored."
    ),
    "Maximum Lookahead (Vehs)": (
        "How many vehicles ahead along the lane are considered when building the V2V awareness set. "
        "Higher values include more distant leaders but add computation."
    ),
    "Network Latency (Steps)": (
        "End-to-end message delay expressed in simulation steps. Non-zero latency shifts "
        "when a vehicle acts on received cooperative data."
    ),
    "Packet Loss Rate (0.0-1.0)": (
        "Probability that a V2V packet is dropped (0 = reliable, 1 = no messages get through). "
        "Stochastic losses mimic imperfect wireless channels."
    ),
}

_LC_MOBIL_TOOLTIPS = {
    "Disc: p_opt": (
        "Politeness for discretionary lane changes. Higher values impose a larger perceived "
        "cost on disturbing the follower in the target lane."
    ),
    "Disc: a_th": (
        "Acceleration threshold (m/s²) for discretionary changes. Lane change proceeds only if "
        "the MOBIL incentive exceeds this value."
    ),
    "Disc: b_safe": (
        "Maximum comfortable braking (m/s²) for discretionary lane changes. Safety requires "
        "predicted accelerations above -b_safe for ego and the new follower."
    ),
    "Mand: b_safe": (
        "Maximum braking allowed (m/s²) for mandatory lane changes. Higher values permit harder "
        "deceleration so merges/weaves can occur in denser traffic (MLC uses safety only, not p/a)."
    ),
}

_LC_DDM_TOOLTIPS = {
    "α_h": "Scale of the systematic (deterministic) part of the lane-change latent utility.",
    "β_0_left": "Alternative-specific constant for choosing a left lane change vs staying.",
    "β_0_right": "Alternative-specific constant for choosing a right lane change vs staying.",
    "β_G": "Coefficient on acceptable gap size; larger values reward bigger gaps when changing lanes.",
    "G_0": "Reference gap scale (m) used to normalize observed gaps in the gap-acceptance term.",
    "β_V": "Weight on relative speed advantage (e.g., passing a slower leader) in the utility.",
    "β_MLC": "Urgency weight for mandatory lane-changing needs (e.g., forced exit or blockage).",
    "σ": "Scale parameter of the random utility error (dispersion of lane-change propensity).",
}

_CMOBIL_TOOLTIPS = {
    "kappa": (
        "Intent urgency weight in C-MOBIL. Higher κ makes cooperative lane changes respond "
        "more strongly to strategic intent (e.g., mandatory need)."
    ),
    "gamma": (
        "Safety weight on lane-change duration / completion time. Larger γ penalizes changes "
        "that take longer to execute, favoring quicker, clearer gaps."
    ),
}

_ATM_SF_TOOLTIPS = {
    "Ped: v_α": "Desired walking speed (m/s) for pedestrians in the social-force formulation.",
    "Ped: τ_α": "Relaxation time (s): how quickly a pedestrian adjusts actual speed toward desired speed.",
    "Ped: A_pp": "Amplitude of pedestrian–pedestrian repulsion (social force magnitude scale).",
    "Ped: B_pp": "Range of pedestrian–pedestrian repulsion (decay distance in the exponential repulsion).",
    "Ped: A_wall": "Amplitude of repulsion from walls and obstacles.",
    "Ped: B_wall": "Spatial decay range for wall repulsion.",
    "Bike: τ_γ": "Relaxation time (s) for cyclists adjusting toward desired cycling speed.",
    "Bike: v_γ": "Desired cycling speed (m/s) when unobstructed.",
    "Bike: a_γ": "Maximum acceleration capability for bikes (m/s²).",
    "Bike: b_γ": "Comfortable braking deceleration for bikes (m/s²).",
    "Bike: η_γ": "Noise or heterogeneity scale affecting bike maneuver variability.",
    "Bike: ε_m": "Small smoothing parameter in bike lateral/longitudinal coupling (numerical stability).",
    "Bike: A_w": "Amplitude of bike interaction with walls or barriers.",
    "Bike: B_w": "Decay range for bike–wall repulsion.",
    "Bike: A_s": "Amplitude of bike–static obstacle or special-field interaction.",
    "Bike: B_s": "Decay range for the static interaction term.",
    "Bike: τ": "Time headway or reaction time scale (s) for bike car-following / gap keeping.",
}

_ATM_PT_TOOLTIPS = {
    "w_c b-b": "Prospect-theory weight for bike–bike interaction costs in the ATM utility.",
    "w_c p-p": "Prospect-theory weight for pedestrian–pedestrian interactions.",
    "w_c p-b": "Cross-weight for pedestrian–bike conflicts (ped perspective).",
    "w_c b-p": "Cross-weight for bike–pedestrian conflicts (bike perspective).",
    "w_c p_bar": "Aggregated or normalized prospect weight for pedestrian bulk / crowd effects.",
    "w_c b_bar": "Aggregated or normalized prospect weight for bike bulk / group effects.",
    "η_ped": "Pedestrian risk-sensitivity / loss-aversion style parameter in the PT formulation.",
    "ξ_ped": "Pedestrian curvature or scaling parameter on prospect components.",
    "τ_ped": "Pedestrian time-scale (s) for prospect accumulation or memory over recent outcomes.",
    "v_desired_ped": "Target walking speed (m/s) used in pedestrian prospect evaluation.",
    "η_bike": "Cyclist risk-sensitivity parameter (parallel role to η for pedestrians).",
    "ξ_bike": "Cyclist scaling parameter for prospect terms.",
    "τ_bike": "Cyclist time scale (s) for prospect dynamics.",
    "v_desired_bike": "Target cycling speed (m/s) in the bike PT utility.",
}


# --- End of Constants ---


# ====================================================================
# 1. Simulation Thread (Handles the background task)
# ====================================================================
class SimulationThread(QThread):
    """
    A QThread to run the simulation function in the background,
    keeping the main UI responsive.
    """
    # Define signals to communicate with the main thread
    simulation_finished = pyqtSignal(bool)  # Emits True on success, False on termination/failure
    progress_update = pyqtSignal(int)  # Emits current progress (0-100)

    def __init__(self, target_function, *args, **kwargs):
        super().__init__()
        self.target_function = target_function
        self.args = args
        self.kwargs = kwargs
        self._is_running = True

    def run(self):
        """The main entry point for the thread."""
        if not self._is_running:
            self.simulation_finished.emit(False)
            return

        try:
            # Call the actual simulation function
            self.target_function(
                self.progress_update,  # Pass the progress signal to the function
                lambda: self._is_running,  # Pass a check for termination
                *self.args, **self.kwargs
            )
            # If the function finishes normally, emit success
            if self._is_running:
                self.simulation_finished.emit(True)
        except Exception as e:
            print(f"Simulation error: {e}")
            self.simulation_finished.emit(False)  # Emit failure

    def stop(self):
        """Called to request the thread to stop gracefully."""
        self._is_running = False
        self.wait()  # Wait for the run method to finish its current loop


# ====================================================================
# 2. Toy Simulation Function (f(..))
# ====================================================================


def run_simulation(progress_signal, is_running_check):
    """
    run sumimulation
    - It must accept the progress_signal and the is_running_check.
    - It saves a file when done.
    """
    user_input_data = responses.all_responses
    print(user_input_data)
    file_path = user_input_data["Data_Folder"] + "/test_sim.csv"
    print(f"Starting simulation. Output file: {os.path.abspath(file_path)}")

    # Persist the current GUI config so automation scripts (e.g. run_case_studies.py)
    # can reuse it as a baseline. Written to <project>/results/.
    try:
        _sim_root = os.path.dirname(os.path.abspath(__file__))
        _baselines_dir = os.path.join(_sim_root, "results")
        os.makedirs(_baselines_dir, exist_ok=True)
        with open(os.path.join(_baselines_dir, "last_gui_config.pkl"), "wb") as _fh:
            pickle.dump(user_input_data, _fh)
        # Also stash per-scenario baselines (freeway/arterial/single_intersection) so
        # automation can pick the right template without depending on run order.
        _scenario_slug = str(user_input_data.get("Scenario", "unknown")).strip().lower().replace(" ", "_")
        with open(os.path.join(_baselines_dir, f"baseline_{_scenario_slug}.pkl"), "wb") as _fh:
            pickle.dump(user_input_data, _fh)
        # For freeway, the geometry and Vehicle_Flows key structure depend on the
        # freeway_type (on_off / on_off_on_off / on_on_off_off). Save a separate
        # baseline per freeway_type so automation can load the right template.
        if _scenario_slug == "freeway":
            _fwy_type = str(user_input_data.get("Geometry", {}).get("Freeway_Type", "")).strip().lower()
            if _fwy_type:
                with open(os.path.join(_baselines_dir, f"baseline_freeway_{_fwy_type}.pkl"), "wb") as _fh:
                    pickle.dump(user_input_data, _fh)
    except Exception as _ex:
        print(f"[config-dump] could not save baseline pickle: {_ex}")

    progress_signal.emit(0)

    try:
        if user_input_data['Scenario'] == 'Single Intersection':
            data = run_sim_single_inter(
                user_input_data,
                progress_cb=progress_signal.emit,
                is_running_check=is_running_check
            )
        elif user_input_data['Scenario'] == "Freeway":
            data = run_sim_freeway(
                user_input_data,
                progress_cb=progress_signal.emit,
                is_running_check=is_running_check
            )
        elif user_input_data['Scenario'] == "TGSIM":
            # Real-world fixed network (e.g. I90/94). Uses templates/I90_94_simple.net.xml
            # as-is; see models/tgsim_sim.py for details / limitations.
            data = run_sim_tgsim(
                user_input_data,
                progress_cb=progress_signal.emit,
                is_running_check=is_running_check
            )
        else:
            data = run_sim_multi_inter(
                user_input_data,
                progress_cb=progress_signal.emit,
                is_running_check=is_running_check
            )
    except Exception as e:
        # SUMO was closed by user or TraCI connection lost - clean up so we can run again
        import traceback
        print("Simulation error (full traceback):")
        traceback.print_exc()
        try:
            traci.close()
        except Exception:
            pass
        raise

    collect_data = bool(user_input_data.get("Sim_DataCollection", True))
    collected_data = data if (collect_data and not data.empty) else pd.DataFrame()

    # Final progress update to ensure 100% is shown
    progress_signal.emit(100)

    # Simulation complete: Save the result file
    try:
        # only save if user wants
        if collect_data and not collected_data.empty:
            collected_data.to_csv(file_path)
            postviz = user_input_data.get(
                "PostSim_Viz", ["trajectory_xy", "time_space", "flow_density"]
            )
            if isinstance(postviz, str):
                postviz = [postviz]
            try:
                _sim_root = os.path.dirname(os.path.abspath(__file__))
                if _sim_root not in sys.path:
                    sys.path.insert(0, _sim_root)
                from results.post_sim_plots import run_post_sim_plots

                if postviz:
                    fd_bin = float(
                        user_input_data.get("PostSim_FlowDensity_TimeBin_s", 30.0)
                    )
                    run_post_sim_plots(
                        file_path,
                        postviz,
                        flow_density_time_bin_s=fd_bin,
                    )
            except Exception as ex:
                print(f"Post-simulation plots failed: {ex}")
    except Exception as e:
        print(f"Error saving file: {e}")


# ------------------ Globals ------------------
FONT_SIZE = 14
PAGE_WIDTH = 1000
PAGE_HEIGHT = 800  # Increased slightly to fit long parameter lists (Social Force)

STEPS = [
    "Geometry",
    "Network",
    "Volume",
    "Signal Control",
    "ATM Demand",
    "Models",
    "Visualization",
    "Simulation",
]


# ------------------ Utility ------------------
class Responses:
    def __init__(self):
        self.all_responses = {}


responses = Responses()


def clear_layout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                clear_layout(child_layout)


def make_separator():
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Sunken)
    sep.setStyleSheet("color: #dddddd;")
    return sep


def create_progress_bar(current_step_name):
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    layout.setAlignment(Qt.AlignLeft)

    try:
        curr_idx = STEPS.index(current_step_name)
    except ValueError:
        curr_idx = -1

    for i, step in enumerate(STEPS):
        lbl = QLabel(step)
        f = QFont()
        f.setPointSize(10)

        if i < curr_idx:
            lbl.setStyleSheet("color: #555555;")
            f.setBold(False)
        elif i == curr_idx:
            lbl.setStyleSheet("color: #007AFF; font-weight: bold;")
            f.setBold(True)
            f.setPointSize(11)
        else:
            lbl.setStyleSheet("color: #AAAAAA;")
            f.setBold(False)

        lbl.setFont(f)
        layout.addWidget(lbl)

        if i < len(STEPS) - 1:
            arrow = QLabel(">")
            arrow.setStyleSheet("color: #CCCCCC;")
            arrow.setFont(QFont("", 10))
            layout.addWidget(arrow)

    return container


# ------------------ Pages ------------------

class WelcomePage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setAlignment(Qt.AlignCenter)
        title = QLabel("Welcome to NGM Simulation!")
        tf = title.font();
        tf.setPointSize(24);
        tf.setBold(True)
        title.setFont(tf)
        title.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title)

        layout.addStretch()
        layout.addWidget(container)
        layout.addStretch()

        nav = QHBoxLayout()
        quit_btn = QPushButton("Quit")
        quit_btn_font = quit_btn.font();
        quit_btn_font.setPointSize(FONT_SIZE);
        quit_btn.setFont(quit_btn_font)
        next_btn = QPushButton("Start >");
        next_btn.setFont(quit_btn_font)

        quit_btn.clicked.connect(stacked_widget.close)
        next_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        nav.addStretch()
        nav.addWidget(quit_btn)
        nav.addWidget(next_btn)
        layout.addLayout(nav)


class GeometrySelectionPage(QWidget):
    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.responses = responses
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)

        main = QVBoxLayout(self)
        main.setAlignment(Qt.AlignTop)

        self.title_label = QLabel("Select Geometry")
        tf = self.title_label.font();
        tf.setPointSize(20);
        tf.setBold(True)
        self.title_label.setFont(tf)
        main.addWidget(self.title_label)

        main.addWidget(create_progress_bar("Geometry"))
        main.addSpacing(10)
        main.addWidget(make_separator())
        main.addSpacing(10)

        sub = QLabel("Choose the scenario:")
        sf = sub.font();
        sf.setPointSize(FONT_SIZE)
        sub.setFont(sf)
        main.addWidget(sub)
        main.addSpacing(12)

        content = QVBoxLayout()
        content.setAlignment(Qt.AlignLeft)
        self.freeway_rb = QRadioButton("Freeway")
        self.arterial_rb = QRadioButton("Arterial")
        self.intersection_rb = QRadioButton("Single Intersection")
        self.tgsim_rb = QRadioButton("TGSIM")
        for rb in (self.freeway_rb, self.arterial_rb, self.intersection_rb, self.tgsim_rb):
            rb.setFont(sf)
            content.addWidget(rb)
        self.freeway_rb.setChecked(True)

        # TGSIM sub-option: dataset/network selector (indented under the TGSIM radio).
        tgsim_sub_row = QHBoxLayout()
        tgsim_sub_row.setContentsMargins(28, 0, 0, 0)  # indent under the radio
        tgsim_sub_label = QLabel("Network:")
        tgsim_sub_label.setFont(sf)
        self.tgsim_network_combo = QComboBox()
        self.tgsim_network_combo.setFont(sf)
        self.tgsim_network_combo.addItems(["I90/94"])
        self.tgsim_network_combo.setCurrentText("I90/94")
        self.tgsim_network_combo.setEnabled(False)  # only enabled when TGSIM is selected
        tgsim_sub_row.addWidget(tgsim_sub_label)
        tgsim_sub_row.addWidget(self.tgsim_network_combo)
        tgsim_sub_row.addStretch()
        content.addLayout(tgsim_sub_row)

        # Enable/disable the TGSIM sub-combo based on the radio selection.
        def _on_tgsim_toggled(checked):
            self.tgsim_network_combo.setEnabled(bool(checked))
        self.tgsim_rb.toggled.connect(_on_tgsim_toggled)

        main.addLayout(content)
        main.addStretch()

        nav = QHBoxLayout()
        quit_btn = QPushButton("Quit");
        back_btn = QPushButton("<Back");
        next_btn = QPushButton("Next>")
        for b in (quit_btn, back_btn, next_btn):
            f = b.font();
            f.setPointSize(FONT_SIZE);
            b.setFont(f)
        quit_btn.clicked.connect(stacked_widget.close)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        next_btn.clicked.connect(self.go_next)

        nav.addStretch()
        nav.addWidget(quit_btn)
        nav.addWidget(back_btn)
        nav.addWidget(next_btn)
        main.addLayout(nav)

    def go_next(self):
        nt = "Freeway"
        if self.arterial_rb.isChecked():
            nt = "Arterial"
        elif self.intersection_rb.isChecked():
            nt = "Single Intersection"
        elif self.tgsim_rb.isChecked():
            nt = "TGSIM"
        self.responses.all_responses["Scenario"] = nt

        # Persist the TGSIM sub-network selection so downstream pages / the
        # simulation runner can resolve which template to load.
        if nt == "TGSIM":
            self.responses.all_responses["TGSIM_Network"] = self.tgsim_network_combo.currentText()

        net_page = self.stacked_widget.widget(2)
        net_page.set_network_type(nt)
        self.stacked_widget.setCurrentIndex(2)

class NetworkConfigurationPage(QWidget):
    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.responses = responses
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)

        self.title_label = QLabel("Network Configuration")
        tf = self.title_label.font()
        tf.setPointSize(20)
        tf.setBold(True)
        self.title_label.setFont(tf)
        self.main_layout.addWidget(self.title_label)

        self.main_layout.addWidget(create_progress_bar("Network"))
        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(make_separator())
        self.main_layout.addSpacing(10)

        self.sub_label = QLabel("Select Lane number and Road Length")
        sf = self.sub_label.font()
        sf.setPointSize(FONT_SIZE)
        self.sub_label.setFont(sf)
        self.main_layout.addWidget(self.sub_label)
        self.main_layout.addSpacing(8)

        self.inputs_layout = QVBoxLayout()
        self.main_layout.addLayout(self.inputs_layout)
        self.main_layout.addStretch()

        nav = QHBoxLayout()
        quit_btn = QPushButton("Quit")
        back_btn = QPushButton("<Back")
        next_btn = QPushButton("Next>")
        for b in (quit_btn, back_btn, next_btn):
            f = b.font()
            f.setPointSize(FONT_SIZE)
            b.setFont(f)
        quit_btn.clicked.connect(stacked_widget.close)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        next_btn.clicked.connect(self.go_next)
        nav.addStretch()
        nav.addWidget(quit_btn)
        nav.addWidget(back_btn)
        nav.addWidget(next_btn)
        self.main_layout.addLayout(nav)

        self.set_network_type("Freeway")

    def clear_inputs(self):
        clear_layout(self.inputs_layout)

    # -------------------------
    # Small UI helpers
    # -------------------------
    def _add_label(self, text, font):
        lbl = QLabel(text)
        lbl.setFont(font)
        self.inputs_layout.addWidget(lbl)
        return lbl

    def _add_combo(self, items, current, font):
        cb = QComboBox()
        cb.setFont(font)
        cb.addItems(items)
        cb.setCurrentText(current)
        self.inputs_layout.addWidget(cb)
        return cb

    def _add_spin(self, minv, maxv, step, val, font):
        sp = QSpinBox()
        sp.setRange(minv, maxv)
        sp.setSingleStep(step)
        sp.setFont(font)
        sp.setValue(val)
        self.inputs_layout.addWidget(sp)
        return sp

    def _clear_layout_only(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
            else:
                sub = item.layout()
                if sub is not None:
                    self._clear_layout_only(sub)
    # Add this method to your class
    def update_lane_options_for_control(self, control_text):
        """Update lane number options based on intersection control type"""
        
        if control_text == "Signal":
            # Signal: allow 2-3 lanes
            self.ns_lanes_combo.clear()
            self.ns_lanes_combo.addItems([str(i) for i in range(2, 4)])
            self.ns_lanes_combo.setCurrentText("3")
            
            self.ew_lanes_combo.clear()
            self.ew_lanes_combo.addItems([str(i) for i in range(2, 4)])
            self.ew_lanes_combo.setCurrentText("3")
            
        else:  # All Way Stop Sign
            # Stop sign: only allow 1-2 lanes (combinations: 1,1; 1,2; 2,1; 2,2)
            self.ns_lanes_combo.clear()
            self.ns_lanes_combo.addItems([str(i) for i in range(1, 3)])  # 1-2 lanes
            self.ns_lanes_combo.setCurrentText("1")
            
            self.ew_lanes_combo.clear()
            self.ew_lanes_combo.addItems([str(i) for i in range(1, 3)])  # 1-2 lanes
            self.ew_lanes_combo.setCurrentText("1")

    # -------------------------
    # Freeway geometry rendering
    # (enforces EXACT order)
    # -------------------------
    def _render_freeway_geometry(self, scenario: str):
        # clear geometry sub-layout only
        self._clear_layout_only(self.geom_layout)

        # insert widgets in exact required order
        for key in self._geom_keys_by_scenario[scenario]:
            lbl, sp = self._geom_widgets[key]
            self.geom_layout.addWidget(lbl)
            self.geom_layout.addWidget(sp)

    # -------------------------
    # Main switch
    # -------------------------
    def set_network_type(self, network_type):
        self.network_type = network_type
        self.title_label.setText(f"{network_type} Scenario - Network Configuration")
        self.clear_inputs()
        font = self.sub_label.font()

        # ==========================================================
        # Freeway (UPDATED with scenario + ordered geometry)
        # ==========================================================
        if network_type == "Freeway":
            # --- basic freeway controls ---
            self._add_label("Number of Lanes:", font)
            self.lanes_combo = self._add_combo([str(i) for i in range(1, 7)], "3", font)

            # NEW: Ramp Length (rolling box / spinbox)
            self._add_label("Ramp Length (m):", font)
            self.ramp_length_spin = self._add_spin(50, 1000, 10, 200, font)

            self._add_label("Freeway Scenario Type:", font)
            # IMPORTANT: dropdown order as requested
            self.freeway_type_combo = self._add_combo(
                ["on_off", "on_on_off_off", "on_off_on_off"],
                "on_off",
                font
            )

            # --- define exact geometry order (as requested) ---
            self._geom_keys_by_scenario = {
                "on_off": [
                    "Input_to_Weaving_Length",
                    "Weaving_Length",
                    "Weaving_to_Output_Length",
                ],
                "on_on_off_off": [
                    "Input_to_Onramp1_Length",
                    "Onramp1_Taper_Length",
                    "Onramp1_Taper_to_Weaving_Length",
                    "Weaving_Length",
                    "Weaving_to_Offramp2_Taper_Length",
                    "Offramp2_Taper_Length",
                    "Offramp2_to_Output_Length",
                ],
                "on_off_on_off": [
                    "Input_to_Onramp1_Length",
                    "Weaving1_Length",
                    "Between_Weaving_Length",
                    "Weaving2_Length",
                    "Offramp2_to_Output_Length",
                ],
            }

            # --- create ALL geometry widgets once (order doesn't matter here) ---
            self._geom_widgets = {}

            def make_geom(key, label, minv, maxv, step, default):
                lbl = QLabel(label)
                lbl.setFont(font)
                sp = QSpinBox()
                sp.setRange(minv, maxv)
                sp.setSingleStep(step)
                sp.setFont(font)
                sp.setValue(default)
                self._geom_widgets[key] = (lbl, sp)

            # on_off
            make_geom("Input_to_Weaving_Length",  "Input → Weaving Length (m):",   0, 20000, 10, 800)
            make_geom("Weaving_Length",           "Weaving Length (m):",           50, 5000, 10, 300)
            make_geom("Weaving_to_Output_Length", "Weaving → Output Length (m):",  0, 20000, 10, 500)

            # on_on_off_off
            make_geom("Input_to_Onramp1_Length",          "Input → OnRamp1 Length (m):",                  0, 20000, 10, 600)
            make_geom("Onramp1_Taper_Length",             "OnRamp1 Taper Length (m):",                    50, 5000, 10, 200)
            make_geom("Onramp1_Taper_to_Weaving_Length",  "OnRamp1 Taper → Weaving Length (m):",          0, 20000, 10, 200)
            make_geom("Weaving_to_Offramp2_Taper_Length", "Weaving → OffRamp2 Taper Length (m):",         0, 20000, 10, 200)
            make_geom("Offramp2_Taper_Length",            "OffRamp2 Taper Length (m):",                   50, 5000, 10, 100)
            make_geom("Offramp2_to_Output_Length",        "OffRamp2 → Output Length (m):",                0, 20000, 10, 300)

            # on_off_on_off
            make_geom("Weaving1_Length",          "Weaving 1 Length (m):",          50, 5000, 10, 250)
            make_geom("Between_Weaving_Length",   "Between Weavings Length (m):",   0, 20000, 10, 400)
            make_geom("Weaving2_Length",          "Weaving 2 Length (m):",          50, 5000, 10, 250)

            # --- geometry sub-layout (kept separate so we can reorder) ---
            self.geom_layout = QVBoxLayout()
            self.inputs_layout.addLayout(self.geom_layout)

            # hook switch (ordered render)
            self.freeway_type_combo.currentTextChanged.connect(self._render_freeway_geometry)

            # initial draw
            self._render_freeway_geometry(self.freeway_type_combo.currentText())

        # ==========================================================
        # TGSIM (real-world fixed network, e.g. I90/94)
        # The geometry is fixed by the chosen template, so this page only
        # echoes the dataset selection and offers a minimal placeholder.
        # ==========================================================
        elif network_type == "TGSIM":
            tgsim_net = self.responses.all_responses.get("TGSIM_Network", "I90/94")

            self._add_label(f"TGSIM Network: {tgsim_net}", font)
            info_lbl = QLabel(
                "Network geometry is fixed by the dataset template "
                "(no procedural geometry adjustments)."
            )
            info_lbl.setFont(font)
            info_lbl.setWordWrap(True)
            info_lbl.setStyleSheet("color: #555555;")
            self.inputs_layout.addWidget(info_lbl)

        # ==========================================================
        # Single Intersection (UPDATED with Control & Walkway)
        # ==========================================================
        elif network_type == "Single Intersection":
            # NEW: Intersection Control
            lbl = QLabel("Intersection Control:")
            lbl.setFont(font)
            self.inputs_layout.addWidget(lbl)
            self.int_control_combo = QComboBox()
            self.int_control_combo.setFont(font)
            self.int_control_combo.addItems(["Signal", "All Way Stop Sign"])
            self.int_control_combo.setCurrentText("Signal")
            self.inputs_layout.addWidget(self.int_control_combo)
            
            lbl = QLabel("Number of lanes (North-South):")
            lbl.setFont(font)
            self.inputs_layout.addWidget(lbl)
            self.ns_lanes_combo = QComboBox()
            self.ns_lanes_combo.setFont(font)
            self.ns_lanes_combo.addItems([str(i) for i in range(2, 4)])  # Initially 2-3
            self.ns_lanes_combo.setCurrentText("3")
            self.inputs_layout.addWidget(self.ns_lanes_combo)
        
            lbl = QLabel("Number of lanes (East-West):")
            lbl.setFont(font)
            self.inputs_layout.addWidget(lbl)
            self.ew_lanes_combo = QComboBox()
            self.ew_lanes_combo.setFont(font)
            self.ew_lanes_combo.addItems([str(i) for i in range(2, 4)])  # Initially 2-3
            self.ew_lanes_combo.setCurrentText("3")
            self.inputs_layout.addWidget(self.ew_lanes_combo)
        
            lbl = QLabel("Road length between input and intersection (m):")
            lbl.setFont(font)
            self.inputs_layout.addWidget(lbl)
            self.road_length_spin = QSpinBox()
            self.road_length_spin.setRange(50, 1000)
            self.road_length_spin.setSingleStep(10)
            self.road_length_spin.setFont(font)
            self.road_length_spin.setValue(200)
            self.inputs_layout.addWidget(self.road_length_spin)
            
            # Connect the combo box change to update lane options
            self.int_control_combo.currentTextChanged.connect(self.update_lane_options_for_control)
        
            
        
            # NEW: Pedestrian Walkway Width
            lbl = QLabel("Pedestrian Walkway Width (m):")
            lbl.setFont(font)
            self.inputs_layout.addWidget(lbl)
            self.walkway_width_spin = QDoubleSpinBox()
            self.walkway_width_spin.setRange(0.0, 10.0)
            self.walkway_width_spin.setSingleStep(0.5)
            self.walkway_width_spin.setFont(font)
            self.walkway_width_spin.setValue(2.0)
            self.inputs_layout.addWidget(self.walkway_width_spin)



        # ==========================================================
        # Corridor / Multi-intersection (UNCHANGED)
        # ==========================================================
        else:
            lbl = QLabel("Number of lanes (North-South):")
            lbl.setFont(font)
            self.inputs_layout.addWidget(lbl)
            self.ns_lanes_combo = QComboBox()
            self.ns_lanes_combo.setFont(font)
            self.ns_lanes_combo.addItems([str(i) for i in range(2, 4)])
            self.ns_lanes_combo.setCurrentText("3")
            self.inputs_layout.addWidget(self.ns_lanes_combo)

            lbl = QLabel("Number of lanes (East-West):")
            lbl.setFont(font)
            self.inputs_layout.addWidget(lbl)
            self.ew_lanes_combo = QComboBox()
            self.ew_lanes_combo.setFont(font)
            self.ew_lanes_combo.addItems([str(i) for i in range(2, 4)])
            self.ew_lanes_combo.setCurrentText("3")
            self.inputs_layout.addWidget(self.ew_lanes_combo)

            lbl = QLabel("Distance between West and Central Intersections (m):")
            lbl.setFont(font)
            self.inputs_layout.addWidget(lbl)
            self.wc_distance_spin = QSpinBox()
            self.wc_distance_spin.setRange(50, 2500)
            self.wc_distance_spin.setSingleStep(10)
            self.wc_distance_spin.setFont(font)
            self.wc_distance_spin.setValue(800)
            self.inputs_layout.addWidget(self.wc_distance_spin)

            lbl = QLabel("Distance between Central and East Intersections (m):")
            lbl.setFont(font)
            self.inputs_layout.addWidget(lbl)
            self.ce_distance_spin = QSpinBox()
            self.ce_distance_spin.setRange(50, 2500)
            self.ce_distance_spin.setSingleStep(10)
            self.ce_distance_spin.setFont(font)
            self.ce_distance_spin.setValue(800)
            self.inputs_layout.addWidget(self.ce_distance_spin)

    def go_next(self):
        # -------------------------
        # Save geometry responses
        # -------------------------
        self.responses.all_responses["Geometry"] = {}

        if self.network_type == "Freeway":
            self.responses.all_responses["Geometry"]["Num_Lanes"] = int(self.lanes_combo.currentText())
            self.responses.all_responses["Geometry"]["Ramp_Length"] = self.ramp_length_spin.value()
            self.responses.all_responses["Geometry"]["Freeway_Type"] = self.freeway_type_combo.currentText()

            ft = self.freeway_type_combo.currentText()
            for key in self._geom_keys_by_scenario[ft]:
                self.responses.all_responses["Geometry"][key] = self._geom_widgets[key][1].value()

        elif self.network_type == "Single Intersection":
            self.responses.all_responses["Geometry"]["Num_Lanes_NS"] = int(self.ns_lanes_combo.currentText())
            self.responses.all_responses["Geometry"]["Num_Lanes_EW"] = int(self.ew_lanes_combo.currentText())
            self.responses.all_responses["Geometry"]["Road_Length"] = self.road_length_spin.value()
            # NEW Data saving
            self.responses.all_responses["Geometry"]["Intersection_Control"] = self.int_control_combo.currentText()
            self.responses.all_responses["Geometry"]["Walkway_Width"] = self.walkway_width_spin.value()

        elif self.network_type == "TGSIM":
            # Geometry is fixed by the template; just record which dataset was chosen.
            self.responses.all_responses["Geometry"]["TGSIM_Network"] = (
                self.responses.all_responses.get("TGSIM_Network", "I90/94")
            )

        else:
            self.responses.all_responses["Geometry"]["Num_Lanes_NS"] = int(self.ns_lanes_combo.currentText())
            self.responses.all_responses["Geometry"]["Num_Lanes_EW"] = int(self.ew_lanes_combo.currentText())
            self.responses.all_responses["Geometry"]["West_Central_Length"] = self.wc_distance_spin.value()
            self.responses.all_responses["Geometry"]["Central_East_Length"] = self.ce_distance_spin.value()

        vol_page = self.stacked_widget.widget(3)
        vol_page.set_network_type(self.network_type)
        self.stacked_widget.setCurrentIndex(3)


class VolumeConfigurationPage(QWidget):
    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.responses = responses
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)

        self.title_label = QLabel("Volume Configuration")
        tf = self.title_label.font()
        tf.setPointSize(20)
        tf.setBold(True)
        self.title_label.setFont(tf)
        self.main_layout.addWidget(self.title_label)

        self.main_layout.addWidget(create_progress_bar("Volume"))
        self.main_layout.addSpacing(6)
        self.main_layout.addWidget(make_separator())
        self.main_layout.addSpacing(12)

        self.sub_label = QLabel("Enter traffic volumes (veh/h)")
        sf = self.sub_label.font()
        sf.setPointSize(FONT_SIZE)
        self.sub_label.setFont(sf)
        self.main_layout.addWidget(self.sub_label)
        self.main_layout.addSpacing(6)

        self.inputs_layout = QVBoxLayout()
        self.main_layout.addLayout(self.inputs_layout)
        self.main_layout.addStretch()

        self.nav = QHBoxLayout()
        quit_btn = QPushButton("Quit")
        back_btn = QPushButton("<Back")
        next_btn = QPushButton("Next>")
        for b in (quit_btn, back_btn, next_btn):
            f = b.font()
            f.setPointSize(FONT_SIZE)
            b.setFont(f)
        quit_btn.clicked.connect(stacked_widget.close)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        next_btn.clicked.connect(self.go_next)
        self.nav.addStretch()
        self.nav.addWidget(quit_btn)
        self.nav.addWidget(back_btn)
        self.nav.addWidget(next_btn)
        self.main_layout.addLayout(self.nav)

    def clear_inputs(self):
        clear_layout(self.inputs_layout)

    def set_network_type(self, network_type):
        """
        Called by previous page.
        For Freeway, we also read the selected freeway type from:
            self.responses.all_responses["Geometry"]["Freeway_Type"]
        """
        self.network_type = network_type
        self.title_label.setText(f"{network_type} Scenario - Volume Configuration")
        self.clear_inputs()

        font = self.sub_label.font()
        small_font = QFont(font)
        small_font.setPointSize(10)

        if network_type == "Freeway":
            # get freeway type chosen on the previous page
            geom = self.responses.all_responses.get("Geometry", {})
            self.freeway_type = geom.get("Freeway_Type", "on_off")
            self.setup_freeway_inputs(font, self.freeway_type)
        elif network_type == "Single Intersection":
            self.setup_intersection_inputs(font, small_font)   # UNCHANGED
        elif network_type == "TGSIM":
            self.setup_tgsim_inputs(font)
        else:
            self.setup_arterial_inputs(font, small_font)       # UNCHANGED

        # keep vehicle mix controls unchanged
        self.add_penetration_controls()

    # -------------------------
    # FREEWAY (UPDATED)
    # -------------------------
    def setup_freeway_inputs(self, font, freeway_type: str):
        self.veh_flows = {}

        # Route definitions by scenario (order matters for UI + saving)
        if freeway_type == "on_off":
            self.routes = ["Main-Main", "OnRamp-Main", "Main-OffRamp", "OnRamp-OffRamp"]
            items = [
                ("Main-Main Volume (Veh/h):",        0, 8000, 4500),
                ("OnRamp-Main Volume (Veh/h):",      0, 2000, 300),
                ("Main-OffRamp Volume (Veh/h):",     0, 2000, 300),
                ("OnRamp-OffRamp Volume (Veh/h):",   0, 2000, 100),
            ]

        elif freeway_type == "on_off_on_off":
            # Based on your in_flows list (exact OD pairs)
            self.routes = [
                "Main-Main",
                "Main-OffRamp1",
                "Main-OffRamp2",
                "OnRamp1-Main",
                "OnRamp1-OffRamp1",
                "OnRamp1-OffRamp2",
                "OnRamp2-Main",
                "OnRamp2-OffRamp2",
            ]
            items = [
                ("Main-Main Volume (Veh/h):",           10, 8000, 4500),
                ("Main-OffRamp1 Volume (Veh/h):",       0,  4000, 300),
                ("Main-OffRamp2 Volume (Veh/h):",       0,  4000, 300),
                ("OnRamp1-Main Volume (Veh/h):",        0,  3000, 300),
                ("OnRamp1-OffRamp1 Volume (Veh/h):",    0,  3000, 100),
                ("OnRamp1-OffRamp2 Volume (Veh/h):",    0,  3000, 100),
                ("OnRamp2-Main Volume (Veh/h):",        0,  3000, 300),
                ("OnRamp2-OffRamp2 Volume (Veh/h):",    0,  3000, 100),
            ]

        elif freeway_type == "on_on_off_off":
            self.routes = [
                "Main-Main",
                "Main-OffRamp1",
                "Main-OffRamp2",
                "OnRamp1-Main",
                "OnRamp1-OffRamp1",
                "OnRamp1-OffRamp2",
                "OnRamp2-Main",
                "OnRamp2-OffRamp1",
                "OnRamp2-OffRamp2",
            ]
            items = [
                ("Main-Main Volume (Veh/h):",           10, 8000, 4500),
                ("Main-OffRamp1 Volume (Veh/h):",       0,  4000, 300),
                ("Main-OffRamp2 Volume (Veh/h):",       0,  4000, 300),
                ("OnRamp1-Main Volume (Veh/h):",        0,  3000, 300),
                ("OnRamp1-OffRamp1 Volume (Veh/h):",    0,  3000, 100),
                ("OnRamp1-OffRamp2 Volume (Veh/h):",    0,  3000, 100),
                ("OnRamp2-Main Volume (Veh/h):",        0,  3000, 300),
                ("OnRamp2-OffRamp1 Volume (Veh/h):",    0,  3000, 100),
                ("OnRamp2-OffRamp2 Volume (Veh/h):",    0,  3000, 100),
            ]

        else:
            # fallback: behave like on_off
            self.routes = ["Main-Main", "OnRamp-Main", "Main-OffRamp", "OnRamp-OffRamp"]
            items = [
                ("Main-Main Volume (Veh/h):",        10, 8000, 4500),
                ("OnRamp-Main Volume (Veh/h):",      10, 2000, 300),
                ("Main-OffRamp Volume (Veh/h):",     10, 2000, 300),
                ("OnRamp-OffRamp Volume (Veh/h):",   10, 2000, 100),
            ]

        # UI build (same style as old code)
        self.spin_boxes = []
        for label_text, lo, hi, default in items:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFont(font)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(10)
            spin.setFont(font)
            spin.setValue(default)

            row.addWidget(lbl)
            row.addSpacing(10)
            row.addWidget(spin)
            self.inputs_layout.addLayout(row)
            self.inputs_layout.addSpacing(8)

            self.spin_boxes.append(spin)

    # -------------------------
    # TGSIM (real-world fixed networks, e.g. I90/94)
    # Routes are determined by the chosen TGSIM dataset.
    # -------------------------
    def setup_tgsim_inputs(self, font):
        self.veh_flows = {}
        tgsim_net = self.responses.all_responses.get("TGSIM_Network", "I90/94")

        if tgsim_net == "I90/94":
            # I90_94_simple.net.xml: Main (6 lanes, 0=rightmost) -> diverge to NB / SB.
            # Demand is specified per *starting Main lane* x destination (12 OD pairs).
            # At the gore Main lanes 0/1/2 connect to NB and 3/4/5 connect to SB, so
            # trips whose starting lane matches the destination side just go straight,
            # and trips whose starting lane is on the *other* side weave across Main
            # before the diverge (handled by MLC + the 2 reroute heuristics).
            # Defaults seed a plausible split: heavier flow on the "natural" side per
            # starting lane (matches the diverge connections), lighter cross-flow.
            lane_count = 6
            defaults = {
                # lane: (volume_to_NB, volume_to_SB) in veh/h
                0: (1450, 0),
                1: (1450, 0),
                2: (315, 190),
                3: (15, 890),
                4: (0, 1150),
                5: (0, 1150),
            }
            self.routes = []
            for k in range(lane_count):
                self.routes.append(f"MainL{k}-NB")
                self.routes.append(f"MainL{k}-SB")
            grid = QGridLayout()
            grid.setHorizontalSpacing(20)
            grid.setVerticalSpacing(6)
            hdr_lane = QLabel("Start Lane")
            hdr_nb = QLabel("→ NB (Veh/h)")
            hdr_sb = QLabel("→ SB (Veh/h)")
            for w in (hdr_lane, hdr_nb, hdr_sb):
                w.setFont(font)
            grid.addWidget(hdr_lane, 0, 0)
            grid.addWidget(hdr_nb, 0, 1)
            grid.addWidget(hdr_sb, 0, 2)

            self.spin_boxes = []
            for k in range(lane_count):
                lbl_text = f"Lane {k}" + (" (rightmost)" if k == 0 else " (leftmost)" if k == lane_count - 1 else "")
                lbl = QLabel(lbl_text)
                lbl.setFont(font)
                grid.addWidget(lbl, k + 1, 0)
                for col, dest_idx in ((1, 0), (2, 1)):
                    spin = QSpinBox()
                    spin.setRange(0, 8000)
                    spin.setSingleStep(10)
                    spin.setFont(font)
                    spin.setValue(int(defaults[k][dest_idx]))
                    grid.addWidget(spin, k + 1, col)
                    self.spin_boxes.append(spin)
            self.inputs_layout.addLayout(grid)
            self.inputs_layout.addSpacing(8)
            return

        # Fallback for unknown TGSIM networks: single aggregate flow.
        self.routes = ["Main-Main"]
        items = [("Main-Main Volume (Veh/h):", 0, 8000, 4000)]
        self.spin_boxes = []
        for label_text, lo, hi, default in items:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFont(font)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(10)
            spin.setFont(font)
            spin.setValue(default)
            row.addWidget(lbl)
            row.addSpacing(10)
            row.addWidget(spin)
            self.inputs_layout.addLayout(row)
            self.inputs_layout.addSpacing(8)
            self.spin_boxes.append(spin)

    # -------------------------
    # Intersection / arterial parts below are UNCHANGED
    # -------------------------
    def setup_intersection_inputs(self, font, small_font):
        bounds = ["East-Bound", "West-Bound", "North-Bound", "South-Bound"]
        self.bound_inputs = {}
        self.veh_flows = {}

        for b_idx, bound in enumerate(bounds):
            self.veh_flows[bound] = {}
            hdr = QLabel(bound)
            hdr.setFont(font)
            self.inputs_layout.addWidget(hdr)
            self.inputs_layout.addSpacing(6)

            row = QHBoxLayout()
            # Volume
            vol_col = QVBoxLayout()
            vol_lbl = QLabel("Volume (Veh/h)")
            vol_lbl.setFont(font)
            vol_spin = QSpinBox()
            vol_spin.setRange(10, 8000)
            vol_spin.setSingleStep(10)
            vol_spin.setFont(font)
            vol_spin.setValue(600)
            vol_col.addWidget(vol_lbl)
            vol_col.addWidget(vol_spin)
            row.addLayout(vol_col)
            row.addSpacing(30)

            # Left Turn Ratio
            lt_col = QVBoxLayout()
            lt_label_row = QHBoxLayout()
            lt_label = QLabel("Left Turn Ratio")
            lt_label.setFont(font)
            lt_pct = QLabel("10%")
            lt_pct.setFont(small_font)
            lt_label_row.addWidget(lt_label)
            lt_label_row.addSpacing(8)
            lt_label_row.addWidget(lt_pct)
            lt_label_row.addStretch()
            lt_col.addLayout(lt_label_row)
            lt_slider = QSlider(Qt.Horizontal)
            lt_slider.setRange(0, 100)
            lt_slider.setValue(10)
            lt_col.addWidget(lt_slider)
            row.addLayout(lt_col)
            row.addSpacing(30)

            # Right Turn Ratio
            rt_col = QVBoxLayout()
            rt_label_row = QHBoxLayout()
            rt_label = QLabel("Right Turn Ratio")
            rt_label.setFont(font)
            rt_pct = QLabel("10%")
            rt_pct.setFont(small_font)
            rt_label_row.addWidget(rt_label)
            rt_label_row.addSpacing(8)
            rt_label_row.addWidget(rt_pct)
            rt_label_row.addStretch()
            rt_col.addLayout(rt_label_row)
            rt_slider = QSlider(Qt.Horizontal)
            rt_slider.setRange(0, 100)
            rt_slider.setValue(10)
            rt_col.addWidget(rt_slider)
            row.addLayout(rt_col)

            self.bound_inputs[bound] = {
                "volume": vol_spin,
                "left_slider": lt_slider, "left_pct": lt_pct,
                "right_slider": rt_slider, "right_pct": rt_pct
            }

            def make_wiring(lt_s, rt_s, lt_p, rt_p):
                def on_lt(v):
                    lt_p.setText(f"{v}%")
                    max_rt = 100 - v
                    rt_s.setMaximum(max_rt)
                    if rt_s.value() > max_rt:
                        rt_s.setValue(max_rt)
                        rt_p.setText(f"{rt_s.value()}%")

                def on_rt(v):
                    rt_p.setText(f"{v}%")
                    max_lt = 100 - v
                    lt_s.setMaximum(max_lt)
                    if lt_s.value() > max_lt:
                        lt_s.setValue(max_lt)
                        lt_p.setText(f"{lt_s.value()}%")

                lt_s.valueChanged.connect(on_lt)
                rt_s.valueChanged.connect(on_rt)

            make_wiring(lt_slider, rt_slider, lt_pct, rt_pct)
            # Initial values do not emit valueChanged; enforce left+right ≤ 100% for default 10%/10%.
            rt_slider.setMaximum(100 - lt_slider.value())
            lt_slider.setMaximum(100 - rt_slider.value())

            self.inputs_layout.addLayout(row)

            if b_idx < len(bounds) - 1:
                self.inputs_layout.addSpacing(8)
                sep = make_separator()
                self.inputs_layout.addWidget(sep)
                self.inputs_layout.addSpacing(8)

    def setup_arterial_inputs(self, font, small_font):
        od_nodes = ["West", "North West", "South West", "North",
                    "South", "North East", "South East", "East"]

        DEFAULT_OD = [
            [0, 30, 30, 40, 40, 30, 30, 500],
            [40, 0, 200, 15, 15, 5, 5, 80],
            [40, 200, 0, 15, 15, 5, 5, 80],
            [80, 15, 15, 0, 200, 15, 15, 80],
            [80, 15, 15, 200, 0, 15, 15, 80],
            [80, 5, 5, 15, 15, 0, 200, 40],
            [80, 5, 5, 15, 15, 200, 0, 40],
            [500, 30, 30, 40, 40, 30, 30, 0],
        ]

        self.veh_flows = {}

        self.csv_radio = QRadioButton("Upload CSV")
        self.manual_radio = QRadioButton("Manual Input")
        self.manual_radio.setChecked(True)
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.csv_radio)
        self.radio_group.addButton(self.manual_radio)
        row = QHBoxLayout()
        row.addWidget(self.manual_radio)
        row.addWidget(self.csv_radio)
        self.inputs_layout.addLayout(row)

        self.csv_radio.setFont(font)
        self.manual_radio.setFont(font)

        self.upload_btn = QPushButton("Select CSV file")
        self.upload_btn.setEnabled(False)
        self.upload_btn.clicked.connect(self.upload_csv)
        self.upload_btn.setFont(small_font)
        self.error_label = QLabel("")
        self.error_label.setFont(font)

        self.inputs_layout.addWidget(self.upload_btn)
        self.inputs_layout.addWidget(self.error_label)

        def on_radio_change():
            self.upload_btn.setEnabled(self.csv_radio.isChecked())

        self.csv_radio.toggled.connect(on_radio_change)

        self.od_matrix = {}
        grid = QGridLayout()
        grid.setSpacing(6)
        corner = QLabel("Origin / Destination")
        corner.setFont(small_font)
        grid.addWidget(corner, 0, 0)

        for j, dest in enumerate(od_nodes, start=1):
            lbl = QLabel(dest)
            lbl.setFont(small_font)
            grid.addWidget(lbl, 0, j)

        for i, origin in enumerate(od_nodes, start=1):
            row_lbl = QLabel(origin)
            row_lbl.setFont(small_font)
            grid.addWidget(row_lbl, i, 0)
            self.od_matrix[origin] = {}
            for j, dest in enumerate(od_nodes, start=1):
                if i == j:
                    grid.addWidget(QLabel("—"), i, j)
                    self.od_matrix[origin][dest] = None
                else:
                    spin = QSpinBox()
                    spin.setRange(0, 5000)
                    spin.setSingleStep(10)
                    spin.setFont(small_font)
                    spin.setValue(DEFAULT_OD[i - 1][j - 1])
                    grid.addWidget(spin, i, j)
                    self.od_matrix[origin][dest] = spin

        self.inputs_layout.addLayout(grid)

    def upload_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            try:
                df = pd.read_csv(file_path, header=None, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, header=None, encoding='utf-16')
            if df.shape[0] < 9 or df.shape[1] < 9:
                self.error_label.setText("Invalid CSV: must be at least 9x9 including headers")
                return
            values = df.iloc[1:9, 1:9]
            if not values.applymap(lambda x: str(x).isdigit() and int(x) >= 0).all().all():
                self.error_label.setText("Invalid CSV: all values in B2:I9 must be non-negative integers")
                return
            diag_nonzero = any(int(values.iloc[i, i]) != 0 for i in range(8))
            if diag_nonzero:
                self.error_label.setText("Warning: Origin and destination cannot be the same!")
            else:
                self.error_label.setText("CSV loaded successfully! ")

            od_nodes = list(self.od_matrix.keys())
            for i, origin in enumerate(od_nodes):
                for j, dest in enumerate(od_nodes):
                    if i != j:
                        self.od_matrix[origin][dest].setValue(int(values.iloc[i, j]))
        except Exception as e:
            self.error_label.setText("Error reading CSV: " + str(e))

    # -------------------------
    # Vehicle mix controls (UNCHANGED)
    # -------------------------
    def add_penetration_controls(self):
        font = self.sub_label.font()
        small_font = QFont(font)
        small_font.setPointSize(10)

        row = QHBoxLayout()

        def _make_slider(title: str, default=0):
            col = QVBoxLayout()
            label_row = QHBoxLayout()
            lbl = QLabel(title)
            lbl.setFont(font)
            pct = QLabel(f"{default}%")
            pct.setFont(small_font)
            label_row.addWidget(lbl)
            label_row.addSpacing(8)
            label_row.addWidget(pct)
            label_row.addStretch()
            col.addLayout(label_row)
            s = QSlider(Qt.Horizontal)
            s.setRange(0, 100)
            s.setValue(default)
            col.addWidget(s)
            return col, s, pct

        sv_col, self.sv_slider, sv_pct = _make_slider("SV:", 80)
        av_col, self.av_slider, av_pct = _make_slider("AV:", 10)
        hv_col, self.hv_slider, hv_pct = _make_slider("HV:", 5)
        cav_col, self.cav_slider, cav_pct = _make_slider("CAV:", 5)
        chv_col, self.chv_slider, chv_pct = _make_slider("CAHV:", 0)

        row.addLayout(sv_col)
        row.addSpacing(15)
        row.addLayout(av_col)
        row.addSpacing(15)
        row.addLayout(hv_col)
        row.addSpacing(15)
        row.addLayout(chv_col)
        row.addSpacing(15)
        row.addLayout(cav_col)

        self.total_mix_lbl = QLabel("")
        self.total_mix_lbl.setFont(font)

        def _update_view():
            sv_pct.setText(f"{self.sv_slider.value()}%")
            av_pct.setText(f"{self.av_slider.value()}%")
            hv_pct.setText(f"{self.hv_slider.value()}%")
            chv_pct.setText(f"{self.chv_slider.value()}%")
            cav_pct.setText(f"{self.cav_slider.value()}%")
            total = (self.sv_slider.value() + self.av_slider.value() + self.hv_slider.value() +
                     self.chv_slider.value() + self.cav_slider.value())
            rem = 100 - total
            if rem == 0:
                self.total_mix_lbl.setText("Total = 100% ✓")
                self.total_mix_lbl.setStyleSheet("color: #2E7D32; font-weight: bold;")
            else:
                self.total_mix_lbl.setText(f"Total = {total}% (Remaining {rem}%)")
                self.total_mix_lbl.setStyleSheet("color: #B00020; font-weight: bold;")

        def _enforce_bounds(changed=None):
            sv = self.sv_slider.value()
            av = self.av_slider.value()
            hv = self.hv_slider.value()
            chv = self.chv_slider.value()
            cav = self.cav_slider.value()
            total = sv + av + hv + chv + cav
            if total > 100:
                overflow = total - 100
                order = ['sv', 'av', 'hv', 'chv', 'cav']
                if changed in order:
                    order.remove(changed)
                    order.append(changed)
                for k in order:
                    if overflow <= 0:
                        break
                    slider = {
                        'sv': self.sv_slider,
                        'av': self.av_slider,
                        'hv': self.hv_slider,
                        'chv': self.chv_slider,
                        'cav': self.cav_slider
                    }[k]
                    d = min(slider.value(), overflow)
                    slider.setValue(slider.value() - d)
                    overflow -= d

            sv = self.sv_slider.value()
            av = self.av_slider.value()
            hv = self.hv_slider.value()
            chv = self.chv_slider.value()
            cav = self.cav_slider.value()
            self.sv_slider.setMaximum(100 - (av + hv + chv + cav))
            self.av_slider.setMaximum(100 - (sv + hv + chv + cav))
            self.hv_slider.setMaximum(100 - (sv + av + chv + cav))
            self.chv_slider.setMaximum(100 - (sv + av + hv + cav))
            self.cav_slider.setMaximum(100 - (sv + av + hv + chv))
            _update_view()

        self.sv_slider.valueChanged.connect(lambda v: _enforce_bounds('sv'))
        self.av_slider.valueChanged.connect(lambda v: _enforce_bounds('av'))
        self.hv_slider.valueChanged.connect(lambda v: _enforce_bounds('hv'))
        self.chv_slider.valueChanged.connect(lambda v: _enforce_bounds('chv'))
        self.cav_slider.valueChanged.connect(lambda v: _enforce_bounds('cav'))

        self.inputs_layout.addSpacing(12)
        sep = make_separator()
        self.inputs_layout.addWidget(sep)
        self.inputs_layout.addSpacing(6)
        vm_subtitle = QLabel("Vehicle Mix")
        vm_subtitle.setFont(font)
        self.inputs_layout.addWidget(vm_subtitle)
        self.inputs_layout.addSpacing(6)
        self.inputs_layout.addLayout(row)
        self.inputs_layout.addSpacing(6)
        self.inputs_layout.addWidget(self.total_mix_lbl)
        self.inputs_layout.addSpacing(4)

        # Color legend for SUMO GUI (must match COLOR_MAP in sim code)
        legend = QLabel(
            "<span style='font-size:10pt;'>"
            "<b>Color legend in the Simulation:</b><br>"
            "<span style='color:#A0A0A0;'>SV: Small Vehicle</span><br>"
            "<span style='color:#A0A0A0;'>HV: Heavy Vehicle</span><br>"
            "<span style='color:#00B400;'>AV: Automated Vehicle</span><br>"
            "<span style='color:#0066CC;'>CAV: Connected Autonomous Vehicle</span><br>"
            "<span style='color:#0066CC;'>CAHV: Connected Autonomous Heavy Vehicle</span>"
            "</span>"
        )
        legend.setFont(small_font)
        legend.setWordWrap(True)
        self.inputs_layout.addWidget(legend)

        _enforce_bounds(None)

    def go_next(self):
        sv_p = self.sv_slider.value()
        av_p = self.av_slider.value()
        hv_p = self.hv_slider.value()
        chv_p = self.chv_slider.value()
        cav_p = self.cav_slider.value()
        total = sv_p + av_p + hv_p + chv_p + cav_p
        if total != 100:
            QMessageBox.warning(
                self,
                "Warning",
                f"Vehicle mix must sum to 100%. Current total = {total}%. Please adjust sliders."
            )
            return

        self.veh_flows["SV_rate"] = sv_p / 100.0
        self.veh_flows["AV_rate"] = av_p / 100.0
        self.veh_flows["HV_rate"] = hv_p / 100.0
        self.veh_flows["CAHV_rate"] = chv_p / 100.0
        self.veh_flows["CAV_rate"] = cav_p / 100.0

        if getattr(self, "network_type", "") == "Single Intersection":
            for bound, widgets in self.bound_inputs.items():
                self.veh_flows[bound] = {
                    "volume": widgets["volume"].value(),
                    "LT_Ratio": widgets["left_slider"].value() / 100.0,
                    "RT_Ratio": widgets["right_slider"].value() / 100.0
                }
        elif getattr(self, "network_type", "") in ("Freeway", "TGSIM"):
            for route, spin in zip(self.routes, self.spin_boxes):
                self.veh_flows[route] = spin.value()
        else:
            for origin, dests in self.od_matrix.items():
                self.veh_flows[origin] = {}
                for dest, widget in dests.items():
                    if widget is None:
                        self.veh_flows[origin][dest] = None
                    else:
                        self.veh_flows[origin][dest] = widget.value()

        self.responses.all_responses["Vehicle_Flows"] = self.veh_flows

        if getattr(self, "network_type", "") == "Single Intersection":
            if self.responses.all_responses["Geometry"]["Intersection_Control"] == "Signal":
                signal_page = self.stacked_widget.widget(4)
                signal_page.set_network_type(self.network_type)
                self.stacked_widget.setCurrentIndex(4)
            else:
                atm_page = self.stacked_widget.widget(5)
                atm_page.set_network_type(self.network_type)
                self.stacked_widget.setCurrentIndex(5)
        elif getattr(self, "network_type", "") == "Arterial":
            signal_page = self.stacked_widget.widget(4)
            signal_page.set_network_type(self.network_type)
            self.stacked_widget.setCurrentIndex(4)
        else:
            car_following_page = self.stacked_widget.widget(6)
            car_following_page.set_network_type(self.network_type)
            self.stacked_widget.setCurrentIndex(6)






class SignalControlPage(QWidget):
    """Separate page for signal control (Webster or manual). Shown after Volume for Single Intersection and Arterial."""
    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.responses = responses
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)
        self.network_type = None

        main = QVBoxLayout(self)
        main.setAlignment(Qt.AlignTop)

        self.title_label = QLabel("Signal Control")
        tf = self.title_label.font()
        tf.setPointSize(20)
        tf.setBold(True)
        self.title_label.setFont(tf)
        main.addWidget(self.title_label)

        main.addWidget(create_progress_bar("Signal Control"))
        main.addSpacing(10)
        main.addWidget(make_separator())
        main.addSpacing(12)

        font = QFont()
        font.setPointSize(FONT_SIZE)
        self.sub_label = QLabel("Traffic signal timing (Webster from volumes or manual). Values below show the Webster-calculated default from the previous Volume page.")
        self.sub_label.setFont(font)
        self.sub_label.setWordWrap(True)
        main.addWidget(self.sub_label)
        main.addSpacing(12)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        self.signal_webster_rb = QRadioButton("Use Webster's formula (from volumes – default)")
        self.signal_webster_rb.setFont(font)
        self.signal_webster_rb.setChecked(True)
        
        self.content_layout.addWidget(self.signal_webster_rb)
        self.signal_manual_rb = QRadioButton("Manual timing")
        self.signal_manual_rb.setFont(font)
        self.content_layout.addWidget(self.signal_manual_rb)
        self.content_layout.addSpacing(12)

        # Single intersection: one row (cycle, green EW, green NS)
        self.single_timing_widget = QWidget()
        single_layout = QVBoxLayout(self.single_timing_widget)
        single_layout.setContentsMargins(0, 0, 0, 0)
        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Cycle length (s):"))
        self.signal_cycle_spin = QSpinBox()
        self.signal_cycle_spin.setRange(30, 150)
        self.signal_cycle_spin.setSingleStep(5)
        self.signal_cycle_spin.setValue(90)
        self.signal_cycle_spin.setFont(font)
        self.signal_cycle_spin.setEnabled(False)
        manual_row.addWidget(self.signal_cycle_spin)
        manual_row.addSpacing(20)
        manual_row.addWidget(QLabel("Green EW (s):"))
        self.signal_green_ew_spin = QSpinBox()
        self.signal_green_ew_spin.setRange(5, 120)
        self.signal_green_ew_spin.setSingleStep(1)
        self.signal_green_ew_spin.setValue(36)
        self.signal_green_ew_spin.setFont(font)
        self.signal_green_ew_spin.setEnabled(False)
        manual_row.addWidget(self.signal_green_ew_spin)
        manual_row.addSpacing(20)
        manual_row.addWidget(QLabel("Green NS (s):"))
        self.signal_green_ns_spin = QSpinBox()
        self.signal_green_ns_spin.setRange(5, 120)
        self.signal_green_ns_spin.setSingleStep(1)
        self.signal_green_ns_spin.setValue(36)
        self.signal_green_ns_spin.setFont(font)
        self.signal_green_ns_spin.setEnabled(False)
        manual_row.addWidget(self.signal_green_ns_spin)
        manual_row.addStretch()
        single_layout.addLayout(manual_row)
        self.content_layout.addWidget(self.single_timing_widget)

        # Arterial: 3 intersections (Int1 West, Int2 Mid, Int3 East), each with cycle, green EW, green NS
        self.arterial_timing_widget = QWidget()
        arterial_layout = QVBoxLayout(self.arterial_timing_widget)
        arterial_layout.setContentsMargins(0, 0, 0, 0)
        self.arterial_spins = []  # list of 3 dicts: {"cycle": spin, "green_ew": spin, "green_ns": spin}
        for idx, (label, short) in enumerate([("Int1 (West)", "Int1"), ("Int2 (Mid)", "Int2"), ("Int3 (East)", "Int3")]):
            grp = QGroupBox(label)
            grp.setFont(font)
            row = QHBoxLayout()
            row.addWidget(QLabel("Cycle (s):"))
            sp_c = QSpinBox()
            sp_c.setRange(30, 150)
            sp_c.setSingleStep(5)
            sp_c.setValue(90)
            sp_c.setFont(font)
            sp_c.setEnabled(False)
            row.addWidget(sp_c)
            row.addSpacing(12)
            row.addWidget(QLabel("Green EW (s):"))
            sp_ew = QSpinBox()
            sp_ew.setRange(5, 120)
            sp_ew.setSingleStep(1)
            sp_ew.setValue(36)
            sp_ew.setFont(font)
            sp_ew.setEnabled(False)
            row.addWidget(sp_ew)
            row.addSpacing(12)
            row.addWidget(QLabel("Green NS (s):"))
            sp_ns = QSpinBox()
            sp_ns.setRange(5, 120)
            sp_ns.setSingleStep(1)
            sp_ns.setValue(36)
            sp_ns.setFont(font)
            sp_ns.setEnabled(False)
            row.addWidget(sp_ns)
            row.addStretch()
            grp.setLayout(row)
            arterial_layout.addWidget(grp)
            self.arterial_spins.append({"cycle": sp_c, "green_ew": sp_ew, "green_ns": sp_ns})
        self.arterial_timing_widget.setVisible(False)
        self.content_layout.addWidget(self.arterial_timing_widget)
        self.content_layout.addSpacing(16)

        # Arterial: offsets for EW (main road) green/cycle coordination, West → East
        self.offset_section_widget = QWidget()
        offset_section_layout = QVBoxLayout(self.offset_section_widget)
        offset_section_layout.setContentsMargins(0, 0, 0, 0)
        self.offset_label = QLabel("EW (main road) coordination – cycle offset along West → East (s):")
        self.offset_label.setFont(font)
        offset_section_layout.addWidget(self.offset_label)
        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Offset West→Mid (Int1→Int2) (s):"))
        self.offset_1_2_spin = QSpinBox()
        self.offset_1_2_spin.setRange(0, 150)
        self.offset_1_2_spin.setSingleStep(5)
        self.offset_1_2_spin.setValue(0)
        self.offset_1_2_spin.setFont(font)
        offset_row.addWidget(self.offset_1_2_spin)
        offset_row.addSpacing(20)
        offset_row.addWidget(QLabel("Offset Mid→East (Int2→Int3) (s):"))
        self.offset_2_3_spin = QSpinBox()
        self.offset_2_3_spin.setRange(0, 150)
        self.offset_2_3_spin.setSingleStep(5)
        self.offset_2_3_spin.setValue(0)
        self.offset_2_3_spin.setFont(font)
        offset_row.addWidget(self.offset_2_3_spin)
        offset_row.addStretch()
        offset_section_layout.addLayout(offset_row)
        self.offset_hint = QLabel("Offsets align EW green start times (Int1=0, then Int2, then Int3). Default 0 = no coordination. Typical: quarter-cycle (≈22 s for 90 s) for progression.")
        self.offset_hint.setStyleSheet("color: #555; font-size: 11px;")
        self.offset_hint.setWordWrap(True)
        offset_section_layout.addWidget(self.offset_hint)
        self.content_layout.addWidget(self.offset_section_widget)

        main.addWidget(self.content_widget)

        def on_signal_mode():
            manual = self.signal_manual_rb.isChecked()
            self.signal_cycle_spin.setEnabled(manual)
            self.signal_green_ew_spin.setEnabled(manual)
            self.signal_green_ns_spin.setEnabled(manual)
            for d in self.arterial_spins:
                d["cycle"].setEnabled(manual)
                d["green_ew"].setEnabled(manual)
                d["green_ns"].setEnabled(manual)
        self.signal_webster_rb.toggled.connect(on_signal_mode)
        self.signal_manual_rb.toggled.connect(on_signal_mode)

        main.addStretch()
        nav = QHBoxLayout()
        quit_btn = QPushButton("Quit")
        back_btn = QPushButton("<Back")
        next_btn = QPushButton("Next>")
        for b in (quit_btn, back_btn, next_btn):
            f = b.font()
            f.setPointSize(FONT_SIZE)
            b.setFont(f)
        quit_btn.clicked.connect(stacked_widget.close)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        next_btn.clicked.connect(self.go_next)
        nav.addStretch()
        nav.addWidget(quit_btn)
        nav.addWidget(back_btn)
        nav.addWidget(next_btn)
        main.addLayout(nav)

    def set_network_type(self, network_type):
        self.network_type = network_type
        self.title_label.setText(f"{network_type} Scenario - Signal Control")
        is_arterial = network_type == "Arterial"
        self.single_timing_widget.setVisible(not is_arterial)
        self.arterial_timing_widget.setVisible(is_arterial)
        self.offset_section_widget.setVisible(is_arterial)
        # Populate with Webster-calculated values from Volume page
        try:
            disp = signal_control.get_webster_display_values(self.responses.all_responses, network_type)
            if disp and not is_arterial:
                self.signal_cycle_spin.setValue(max(30, min(150, disp["cycle_length"])))
                self.signal_green_ew_spin.setValue(max(5, min(120, disp["green_ew"])))
                self.signal_green_ns_spin.setValue(max(5, min(120, disp["green_ns"])))
            elif is_arterial and isinstance(disp, list) and len(disp) >= 3:
                for i, d in enumerate(disp[:3]):
                    self.arterial_spins[i]["cycle"].setValue(max(30, min(150, d["cycle_length"])))
                    self.arterial_spins[i]["green_ew"].setValue(max(5, min(120, d["green_ew"])))
                    self.arterial_spins[i]["green_ns"].setValue(max(5, min(120, d["green_ns"])))
        except Exception:
            pass

    def go_next(self):
        if self.signal_webster_rb.isChecked():
            self.responses.all_responses["Signal_Control"] = {"use_webster": True}
        else:
            if self.network_type == "Arterial":
                manual_plans = []
                for d in self.arterial_spins:
                    manual_plans.append({
                        "cycle_length": d["cycle"].value(),
                        "green_ew": d["green_ew"].value(),
                        "green_ns": d["green_ns"].value(),
                    })
                self.responses.all_responses["Signal_Control"] = {
                    "use_webster": False,
                    "manual_plans": manual_plans,
                }
            else:
                self.responses.all_responses["Signal_Control"] = {
                    "use_webster": False,
                    "cycle_length": self.signal_cycle_spin.value(),
                    "green_ew": self.signal_green_ew_spin.value(),
                    "green_ns": self.signal_green_ns_spin.value(),
                }
        if self.network_type == "Arterial":
            sig = self.responses.all_responses["Signal_Control"]
            sig["offset_1_2"] = self.offset_1_2_spin.value()
            sig["offset_2_3"] = self.offset_2_3_spin.value()

        if self.network_type == "Single Intersection":
            atm_page = self.stacked_widget.widget(5)
            atm_page.set_network_type(self.network_type)
            self.stacked_widget.setCurrentIndex(5)
        else:
            car_following_page = self.stacked_widget.widget(6)
            car_following_page.set_network_type(self.network_type)
            self.stacked_widget.setCurrentIndex(6)


class ATMDemandPage(QWidget):
    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.responses = responses
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)

        main = QVBoxLayout(self)
        main.setAlignment(Qt.AlignTop)

        self.title_label = QLabel("ATM Demand")
        tf = self.title_label.font();
        tf.setPointSize(20);
        tf.setBold(True)
        self.title_label.setFont(tf)
        main.addWidget(self.title_label)

        main.addWidget(create_progress_bar("ATM Demand"))
        main.addWidget(make_separator())
        main.addSpacing(8)

        subtitle = QLabel("Enter Pedestrian Demand")
        sf = subtitle.font();
        sf.setPointSize(FONT_SIZE)
        subtitle.setFont(sf)
        main.addWidget(subtitle)
        main.addSpacing(8)

        desc_label = QLabel("Pedestrians are initialized with randomized origins and destinations.")
        desc_font = desc_label.font()
        desc_font.setPointSize(12)
        desc_label.setFont(desc_font)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #555555;")
        main.addWidget(desc_label)
        main.addSpacing(12)

        row_ped = QHBoxLayout()
        self.allow_ped = QCheckBox("Allow Pedestrian")
        self.allow_ped.setFont(sf)
        self.allow_ped.setChecked(True)
        row_ped.addWidget(self.allow_ped)
        row_ped.addSpacing(20)

        ped_label = QLabel("Pedestrian Volume (Pedestrians/h):")
        ped_label.setFont(sf)
        self.ped_spin = QSpinBox()
        self.ped_spin.setRange(0, 10000)
        self.ped_spin.setSingleStep(50)
        self.ped_spin.setFont(sf)
        self.ped_spin.setEnabled(True)
        self.ped_spin.setValue(500)

        row_bike = QHBoxLayout()
        self.allow_bike = QCheckBox("Allow Bike")
        self.allow_bike.setFont(sf)
        self.allow_bike.setChecked(True)
        row_bike.addWidget(self.allow_bike)
        row_bike.addSpacing(20)

        bike_label = QLabel("Bike Volume (Bikes/h):")
        bike_label.setFont(sf)
        self.bike_spin = QSpinBox()
        self.bike_spin.setRange(0, 10000)
        self.bike_spin.setSingleStep(50)
        self.bike_spin.setFont(sf)
        self.bike_spin.setEnabled(True)
        self.bike_spin.setValue(200)

        row_ped.addWidget(ped_label)
        row_ped.addWidget(self.ped_spin)
        row_bike.addWidget(bike_label)
        row_bike.addWidget(self.bike_spin)

        row_ped.addStretch()
        row_bike.addStretch()

        main.addLayout(row_ped)
        main.addLayout(row_bike)

        self.allow_ped.stateChanged.connect(lambda s: self.ped_spin.setEnabled(s == Qt.Checked))
        self.allow_bike.stateChanged.connect(lambda s: self.bike_spin.setEnabled(s == Qt.Checked))

        main.addStretch()

        nav = QHBoxLayout()
        quit_btn = QPushButton("Quit");
        back_btn = QPushButton("<Back");
        next_btn = QPushButton("Next>")
        for b in (quit_btn, back_btn, next_btn):
            f = b.font();
            f.setPointSize(FONT_SIZE);
            b.setFont(f)
        quit_btn.clicked.connect(stacked_widget.close)
        back_btn.clicked.connect(self.go_back)
        next_btn.clicked.connect(self.go_next)

        nav.addStretch()
        nav.addWidget(quit_btn)
        nav.addWidget(back_btn)
        nav.addWidget(next_btn)
        main.addLayout(nav)

    def set_network_type(self, network_type):
        self.network_type = network_type
        self.title_label.setText(f"{network_type} Scenario - ATM Demand")
    def go_back(self):
        if getattr(self, "network_type", "") == "Single Intersection" and self.responses.all_responses["Geometry"]["Intersection_Control"] == "Signal":
            self.stacked_widget.setCurrentIndex(4)
        else:
            self.stacked_widget.setCurrentIndex(3)
    def go_next(self):
        self.responses.all_responses["Allow_Ped"] = self.allow_ped.isChecked()
        self.responses.all_responses["Allow_Bike"] = self.allow_bike.isChecked()
        if self.allow_ped.isChecked():
            self.responses.all_responses["Ped_Volume"] = self.ped_spin.value()
        else:
            self.responses.all_responses["Ped_Volume"] = 0

        if self.allow_bike.isChecked():
            self.responses.all_responses["Bike_Volume"] = self.bike_spin.value()
        else:
            self.responses.all_responses["Bike_Volume"] = 0

        car_following_page = self.stacked_widget.widget(6)
        car_following_page.set_network_type(self.network_type)
        car_following_page.ped_allowed = self.allow_ped.isChecked()
        car_following_page.bike_allowed = self.allow_bike.isChecked()
        self.stacked_widget.setCurrentIndex(6)


class CFModelsPage(QWidget):
    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.responses = responses
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)

        self.network_type = None
        self.param_rows = []
        self.ped_allowed = False
        self.bike_allowed = False

        # Parameter names / defaults
        self.IDM_PARAMS = ["T", "a", "b", "v_0", "s_0"]
        self.PT_PARAMS = ["T_max", "α", "β", "W_c", "W_m", "Gamma1", "Gamma2"]

        # Default parameters should reflect what's in the calibrated CSV sample sets.
        # We'll compute mean/std by vehicle type from the merged CSVs and use those
        # whenever "Default Parameters" is checked.
        self._default_cf_stats = self._compute_default_cf_stats()

        # Communication defaults
        self.COMM_DEFAULTS = {
            "Range": 30.0,
            "Lookahead": 3,
            "Latency": 0,
            "Loss": 0.0,
        }

        self._build_ui()

    def _compute_default_cf_stats(self):
        """
        Compute per-vehicle-type Mean/Std defaults from the merged calibration CSVs.

        Returns a structure like:
          {
            "IDM": { "T": {cls: {"Mean":..,"Std":..}, ...}, ...},
            "PT":  { "T_max": {...}, ...}
          }
        """
        cls_files_idm = {
            "Small Vehicle": MODEL_PARAMS_DIR / "merged_IDM_S.csv",
            "Automated Vehicle": MODEL_PARAMS_DIR / "merged_IDM_A.csv",
            "Heavy Vehicle": MODEL_PARAMS_DIR / "merged_IDM_L.csv",
        }
        cls_files_pt = {
            "Small Vehicle": MODEL_PARAMS_DIR / "merged_PT_S.csv",
            "Automated Vehicle": MODEL_PARAMS_DIR / "merged_PT_A.csv",
            "Heavy Vehicle": MODEL_PARAMS_DIR / "merged_PT_L.csv",
        }

        idm_col_map = {"T": "T", "a": "a", "b": "b", "v_0": "v0", "s_0": "so"}
        pt_col_map = {"T_max": "Tmax", "α": "Alpha", "β": "Beta", "W_c": "Wc", "W_m": "Wm", "Gamma1": "Gamma1", "Gamma2": "Gamma2"}

        out = {"IDM": {}, "PT": {}}

        # IDM stats
        for p in self.IDM_PARAMS:
            out["IDM"][p] = {}
            for cls, path in cls_files_idm.items():
                try:
                    df = pd.read_csv(str(path))
                    if "T" in df.columns:
                        df = df[df["T"] > 0]
                    col = idm_col_map[p]
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    out["IDM"][p][cls] = {"Mean": float(s.mean()), "Std": float(s.std(ddof=1))}
                except Exception:
                    # Fallback to something sane if the CSV cannot be read/parsed.
                    out["IDM"][p][cls] = {"Mean": 0.0, "Std": 0.0}

        # PT stats
        for p in self.PT_PARAMS:
            out["PT"][p] = {}
            for cls, path in cls_files_pt.items():
                try:
                    df = pd.read_csv(str(path))
                    col = pt_col_map[p]
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    out["PT"][p][cls] = {"Mean": float(s.mean()), "Std": float(s.std(ddof=1))}
                except Exception:
                    out["PT"][p][cls] = {"Mean": 0.0, "Std": 0.0}

        return out

    def set_network_type(self, network_type):
        self.network_type = network_type
        self.title_label.setText(f"{network_type} Scenario - Driving Models (CF)")

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setAlignment(Qt.AlignTop)

        # Title
        self.title_label = QLabel("Driving Models (CF)")
        tf = self.title_label.font()
        tf.setPointSize(20)
        tf.setBold(True)
        self.title_label.setFont(tf)
        main.addWidget(self.title_label)

        # Progress + separator
        main.addWidget(create_progress_bar("Models"))
        main.addWidget(make_separator())
        main.addSpacing(12)

        # Consistent font (same as other pages)
        sf = QFont("", FONT_SIZE)

        # Model selection
        sub = QLabel("Car-Following Model Selection")
        sub.setFont(sf)
        main.addWidget(sub)

        model_row = QHBoxLayout()
        model_row.setSpacing(18)
        self.idm_rb = QRadioButton("IDM")
        self.pt_rb = QRadioButton("Prospect Theory (PT)")
        self.idm_rb.setFont(sf)
        self.pt_rb.setFont(sf)
        self.idm_rb.setChecked(True)
        model_row.addWidget(self.idm_rb)
        model_row.addWidget(self.pt_rb)
        model_row.addStretch()
        main.addLayout(model_row)

        self._model_group = QButtonGroup(self)
        self._model_group.addButton(self.idm_rb)
        self._model_group.addButton(self.pt_rb)

        self.default_cb = QCheckBox("Default Parameters")
        self.default_cb.setFont(sf)
        self.default_cb.setChecked(True)
        main.addWidget(self.default_cb)
        main.addSpacing(6)

        # --- Parameter table in a scroll area (keeps layout stable) ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        param_container = QWidget()
        param_layout = QVBoxLayout(param_container)
        param_layout.setContentsMargins(8, 8, 8, 8)
        param_layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        empty_lbl = QLabel("")
        empty_lbl.setFixedWidth(158)  # parameter name + info icon
        header_row.addWidget(empty_lbl)

        header_font = QFont("", FONT_SIZE)
        header_font.setBold(True)

        for cls_name in ("Small Vehicle", "Automated Vehicle", "Heavy Vehicle"):
            lbl = QLabel(cls_name)
            lbl.setFont(header_font)
            lbl.setAlignment(Qt.AlignCenter)
            header_row.addWidget(lbl, stretch=1)

        spacer = QLabel("")
        spacer.setFixedWidth(12)
        header_row.addWidget(spacer)

        param_layout.addLayout(header_row)

        ms_row = QHBoxLayout()
        ms_row.setSpacing(12)
        ms_empty = QLabel("")
        ms_empty.setFixedWidth(158)
        ms_row.addWidget(ms_empty)
        for _ in range(3):
            mean_lbl = QLabel("Mean")
            mean_lbl.setFont(QFont("", FONT_SIZE))
            mean_lbl.setAlignment(Qt.AlignCenter)
            std_lbl = QLabel("Std")
            std_lbl.setFont(QFont("", FONT_SIZE))
            std_lbl.setAlignment(Qt.AlignCenter)
            ms_row.addWidget(mean_lbl)
            ms_row.addWidget(std_lbl)
        spacer2 = QLabel("")
        spacer2.setFixedWidth(12)
        ms_row.addWidget(spacer2)
        param_layout.addLayout(ms_row)
        param_layout.addSpacing(6)

        # Max rows = 7 (PT has 7, IDM has 5)
        self.param_rows = []
        for r in range(7):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(10)

            name_cell = QWidget()
            name_cell.setFixedWidth(158)
            name_lay = QHBoxLayout(name_cell)
            name_lay.setContentsMargins(0, 0, 0, 0)
            name_lay.setSpacing(4)
            var_label = QLabel(f"Param{r + 1}")
            var_label.setFont(QFont("", FONT_SIZE))
            var_label.setFixedWidth(132)
            info_icon = _param_info_icon()
            name_lay.addWidget(var_label)
            name_lay.addWidget(info_icon)
            name_lay.addStretch()
            row_layout.addWidget(name_cell)

            spinboxes = []
            for _ in range(3):
                mean_sb = QDoubleSpinBox()
                # PT includes W_c which can be ~1e5+, so we need a wider numeric range.
                mean_sb.setRange(-1e9, 1e9)
                mean_sb.setDecimals(2)
                mean_sb.setSingleStep(0.1)
                mean_sb.setFixedWidth(80)
                mean_sb.setKeyboardTracking(False)
                mean_sb.setSpecialValueText("")
                mean_sb.setValue(0.0)

                std_sb = QDoubleSpinBox()
                std_sb.setRange(0.0, 1e9)
                std_sb.setDecimals(2)
                std_sb.setSingleStep(0.1)
                std_sb.setFixedWidth(80)
                std_sb.setKeyboardTracking(False)
                std_sb.setSpecialValueText("")
                std_sb.setValue(0.0)

                row_layout.addWidget(mean_sb)
                row_layout.addWidget(std_sb)
                spinboxes.extend([mean_sb, std_sb])

            self.param_rows.append((var_label, info_icon, spinboxes))
            param_layout.addLayout(row_layout)

        scroll.setWidget(param_container)
        main.addWidget(scroll)

        # --- CAV C-IDM parameters (with Default/Manual toggle like other pages) ---
        main.addSpacing(8)
        cidm_box = QGroupBox("CAV (C-IDM) parameters")
        cidm_layout = QVBoxLayout(cidm_box)

        self.cidm_default_cb = QCheckBox("Default C-IDM parameters")
        self.cidm_default_cb.setFont(sf)
        self.cidm_default_cb.setChecked(True)
        cidm_layout.addWidget(self.cidm_default_cb)

        r1 = QHBoxLayout()
        lbl_kv = QLabel("K_v")
        lbl_kv.setFont(sf)
        self.cidm_kv = QDoubleSpinBox()
        self.cidm_kv.setRange(0.0, 10.0)
        self.cidm_kv.setDecimals(4)
        self.cidm_kv.setValue(0.1)
        _add_label_with_info(r1, lbl_kv, _CIDM_TOOLTIPS["K_v"])
        r1.addWidget(self.cidm_kv)
        r1.addStretch()
        cidm_layout.addLayout(r1)

        r2 = QHBoxLayout()
        lbl_ka = QLabel("K_a")
        lbl_ka.setFont(sf)
        self.cidm_ka = QDoubleSpinBox()
        self.cidm_ka.setRange(0.0, 10.0)
        self.cidm_ka.setDecimals(4)
        self.cidm_ka.setValue(0.03)
        _add_label_with_info(r2, lbl_ka, _CIDM_TOOLTIPS["K_a"])
        r2.addWidget(self.cidm_ka)
        r2.addStretch()
        cidm_layout.addLayout(r2)

        r3 = QHBoxLayout()
        lbl_sref = QLabel("s_ref (m)")
        lbl_sref.setFont(sf)
        self.cidm_sref = QDoubleSpinBox()
        self.cidm_sref.setRange(0.0, 1000.0)
        self.cidm_sref.setDecimals(2)
        self.cidm_sref.setValue(35.0)
        _add_label_with_info(r3, lbl_sref, _CIDM_TOOLTIPS["s_ref"])
        r3.addWidget(self.cidm_sref)
        r3.addStretch()
        cidm_layout.addLayout(r3)

        def _cidm_toggle():
            enabled = not self.cidm_default_cb.isChecked()
            for w in (self.cidm_kv, self.cidm_ka, self.cidm_sref):
                w.setEnabled(enabled)
                w.setStyleSheet("" if enabled else "color:#888;")

        self.cidm_default_cb.stateChanged.connect(_cidm_toggle)
        _cidm_toggle()
        main.addWidget(cidm_box)

        # --- V2V Communication parameters (Default/Manual toggle like other pages) ---
        comm_box = QGroupBox("V2V Communication Configuration")
        comm_layout = QVBoxLayout(comm_box)

        self.comm_default_cb = QCheckBox("Default communication parameters")
        self.comm_default_cb.setFont(sf)
        self.comm_default_cb.setChecked(True)
        comm_layout.addWidget(self.comm_default_cb)

        self.comm_range = QDoubleSpinBox()
        self.comm_range.setRange(0, 10000)
        self.comm_range.setDecimals(1)
        self.comm_range.setValue(self.COMM_DEFAULTS["Range"])

        self.comm_look = QSpinBox()
        self.comm_look.setRange(1, 50)
        self.comm_look.setValue(self.COMM_DEFAULTS["Lookahead"])

        self.comm_lat = QSpinBox()
        self.comm_lat.setRange(0, 500)
        self.comm_lat.setValue(self.COMM_DEFAULTS["Latency"])

        self.comm_loss = QDoubleSpinBox()
        self.comm_loss.setRange(0.0, 1.0)
        self.comm_loss.setDecimals(3)
        self.comm_loss.setSingleStep(0.01)
        self.comm_loss.setValue(self.COMM_DEFAULTS["Loss"])

        cr1 = QHBoxLayout()
        lbl_cr = QLabel("Communication Range (m)")
        lbl_cr.setFont(sf)
        _add_label_with_info(cr1, lbl_cr, _COMM_PARAM_TOOLTIPS["Communication Range (m)"])
        cr1.addWidget(self.comm_range)
        cr1.addStretch()
        comm_layout.addLayout(cr1)

        cr2 = QHBoxLayout()
        lbl_look = QLabel("Maximum Lookahead (Vehs)")
        lbl_look.setFont(sf)
        _add_label_with_info(cr2, lbl_look, _COMM_PARAM_TOOLTIPS["Maximum Lookahead (Vehs)"])
        cr2.addWidget(self.comm_look)
        cr2.addStretch()
        comm_layout.addLayout(cr2)

        cr3 = QHBoxLayout()
        lbl_lat = QLabel("Network Latency (Steps)")
        lbl_lat.setFont(sf)
        _add_label_with_info(cr3, lbl_lat, _COMM_PARAM_TOOLTIPS["Network Latency (Steps)"])
        cr3.addWidget(self.comm_lat)
        cr3.addStretch()
        comm_layout.addLayout(cr3)

        cr4 = QHBoxLayout()
        lbl_loss = QLabel("Packet Loss Rate (0.0-1.0)")
        lbl_loss.setFont(sf)
        _add_label_with_info(cr4, lbl_loss, _COMM_PARAM_TOOLTIPS["Packet Loss Rate (0.0-1.0)"])
        cr4.addWidget(self.comm_loss)
        cr4.addStretch()
        comm_layout.addLayout(cr4)

        def _comm_toggle():
            enabled = not self.comm_default_cb.isChecked()
            for w in (self.comm_range, self.comm_look, self.comm_lat, self.comm_loss):
                w.setEnabled(enabled)
                w.setStyleSheet("" if enabled else "color:#888;")
            if not enabled:
                self.comm_range.setValue(float(self.COMM_DEFAULTS["Range"]))
                self.comm_look.setValue(int(self.COMM_DEFAULTS["Lookahead"]))
                self.comm_lat.setValue(int(self.COMM_DEFAULTS["Latency"]))
                self.comm_loss.setValue(float(self.COMM_DEFAULTS["Loss"]))

        self.comm_default_cb.stateChanged.connect(_comm_toggle)
        _comm_toggle()
        main.addWidget(comm_box)

        # Wiring
        self.idm_rb.toggled.connect(self._on_model_toggled)
        self.pt_rb.toggled.connect(self._on_model_toggled)
        self.default_cb.stateChanged.connect(self._on_default_toggled)

        self._on_model_toggled()
        self._on_default_toggled()

        # Navigation
        main.addStretch()
        nav = QHBoxLayout()
        nav.addStretch()
        quit_btn = QPushButton("Quit")
        back_btn = QPushButton("<Back")
        next_btn = QPushButton("Next>")
        for b in (quit_btn, back_btn, next_btn):
            f = b.font()
            f.setPointSize(FONT_SIZE)
            b.setFont(f)
        quit_btn.clicked.connect(self.stacked_widget.close)
        back_btn.clicked.connect(self.go_back)
        next_btn.clicked.connect(self.go_next)
        nav.addWidget(quit_btn)
        nav.addWidget(back_btn)
        nav.addWidget(next_btn)
        main.addLayout(nav)

    def _on_model_toggled(self):
        use_default = self.default_cb.isChecked()
        is_idm = self.idm_rb.isChecked()
        labels = self.IDM_PARAMS if is_idm else self.PT_PARAMS
        model_key = "IDM" if is_idm else "PT"
        classes = ["Small Vehicle", "Automated Vehicle", "Heavy Vehicle"]

        for i, (lbl, info_icon, spinboxes) in enumerate(self.param_rows):
            if i < len(labels):
                lbl.setText(labels[i])
                lbl.setVisible(True)
                info_icon.setVisible(True)
                info_icon.setToolTip(_CF_PARAM_TOOLTIPS.get(labels[i], ""))
                for j, sb in enumerate(spinboxes):
                    sb.setVisible(True)
                    sb.setEnabled(not use_default)
                    sb.setStyleSheet("" if not use_default else "color:#888;")

                if use_default:
                    p = labels[i]
                    # Fill Mean/Std for each vehicle class from the CSV-derived defaults.
                    for c_idx, cls in enumerate(classes):
                        stats = (
                            self._default_cf_stats.get(model_key, {})
                            .get(p, {})
                            .get(cls, {"Mean": 0.0, "Std": 0.0})
                        )
                        spinboxes[c_idx * 2].setValue(float(stats.get("Mean", 0.0)))
                        spinboxes[c_idx * 2 + 1].setValue(float(stats.get("Std", 0.0)))
            else:
                lbl.setVisible(False)
                info_icon.setVisible(False)
                for sb in spinboxes:
                    sb.setVisible(False)

    def _on_default_toggled(self):
        self._on_model_toggled()

    def _find_page_index_by_classname(self, classname: str):
        for i in range(self.stacked_widget.count()):
            w = self.stacked_widget.widget(i)
            if w.__class__.__name__ == classname:
                return i
        return None

    def go_back(self):
        self.responses.all_responses["Bike_Allowed"] = False
        self.responses.all_responses["Ped_Allowed"] = False
        self.responses.all_responses["Ped_Volume"] = 0
        self.responses.all_responses["Bike_Volume"] = 0

        if getattr(self, "network_type", "") == "Single Intersection":
            idx = self._find_page_index_by_classname("ATMDemandPage")
            if idx is not None:
                self.stacked_widget.setCurrentIndex(idx)
                return
        idx = self._find_page_index_by_classname("VolumeConfigurationPage")
        if idx is not None:
            self.stacked_widget.setCurrentIndex(idx)

    def go_next(self):
        model_name = "IDM" if self.idm_rb.isChecked() else "PT"
        self.responses.all_responses["CF_Model"] = model_name
        self.responses.all_responses["CF_Default_Params"] = self.default_cb.isChecked()

        param_data = {}
        for lbl, _info_icon, spinboxes in self.param_rows:
            if not lbl.isVisible():
                continue
            row_label = lbl.text().strip()
            cls_data = {}
            for i, cls_name in enumerate(["Small Vehicle", "Automated Vehicle", "Heavy Vehicle"]):
                cls_data[cls_name] = {
                    "Mean": float(spinboxes[i * 2].value()),
                    "Std": float(spinboxes[i * 2 + 1].value()),
                }
            param_data[row_label] = cls_data
        self.responses.all_responses["CF_Parameters"] = param_data

        self.responses.all_responses["CIDM_Params"] = {
            "Default": bool(self.cidm_default_cb.isChecked()),
            "K_v": float(self.cidm_kv.value()),
            "K_a": float(self.cidm_ka.value()),
            "s_ref": float(self.cidm_sref.value()),
        }

        self.responses.all_responses["Comm_Params"] = {
            "Default": bool(self.comm_default_cb.isChecked()),
            "Range": float(self.comm_range.value()),
            "Lookahead": int(self.comm_look.value()),
            "Latency": int(self.comm_lat.value()),
            "Loss": float(self.comm_loss.value()),
        }

        lane_changing_page = self.stacked_widget.widget(7)
        lane_changing_page.set_network_type(self.network_type)
        lane_changing_page.ped_allowed = self.ped_allowed
        lane_changing_page.bike_allowed = self.bike_allowed
        self.stacked_widget.setCurrentIndex(7)



class LCModelsPage(QWidget):
    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.responses = responses
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)

        self.network_type = None
        self.param_rows = []
        self.ped_allowed = False
        self.bike_allowed = False

        # Corrected Parameter Names
        self.MOBIL_PARAMS = [
            "Disc: p_opt", "Disc: a_th",
            "Disc: b_safe", "Mand: b_safe",
        ]
        self.DDM_PARAMS = ["α_h", "β_0_left", "β_0_right", "β_G", "G_0", "β_V", "β_MLC", "σ"]
        self.DDM_DEFAULTS = [0.08, -3.5, -4.2, 0.2737, 8.69, 0.6808, 87.0, 8.458]

        # Default MOBIL parameters should reflect what's in the calibrated sample set.
        # MOBIL_results.csv is not class-specific, so we apply the same Mean/Std to
        # all three vehicle-class columns in the UI when Default is checked.
        self._default_mobil_stats = self._compute_default_mobil_stats()

        self._build_ui()

    def _compute_default_mobil_stats(self):
        """Mean/Std defaults for MOBIL from models/model_params/MOBIL_results.csv."""
        return mobil_default_stats_from_csv(MODEL_PARAMS_DIR / "MOBIL_results.csv")

    def set_network_type(self, network_type):
        self.network_type = network_type
        self.title_label.setText(f"{network_type} Scenario - Driving Models (LC)")

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setAlignment(Qt.AlignTop)

        self.title_label = QLabel("Driving Models")
        tf = self.title_label.font();
        tf.setPointSize(20);
        tf.setBold(True)
        self.title_label.setFont(tf)
        main.addWidget(self.title_label)

        main.addWidget(create_progress_bar("Models"))
        main.addWidget(make_separator())
        main.addSpacing(12)

        sub = QLabel("Lane-Changing Model Selection")
        sf = sub.font();
        sf.setPointSize(FONT_SIZE);
        sub.setFont(sf)
        main.addWidget(sub)

        model_row = QHBoxLayout()
        model_row.setSpacing(18)
        self.mobil_rb = QRadioButton("MOBIL")
        self.ddm_rb = QRadioButton("Drift Diffusion Model (DDM)")
        self.mobil_rb.setFont(sf)
        self.ddm_rb.setFont(sf)
        self.mobil_rb.setChecked(True)
        model_row.addWidget(self.mobil_rb)
        model_row.addWidget(self.ddm_rb)
        model_row.addStretch()
        main.addLayout(model_row)

        self._model_group = QButtonGroup(self)
        self._model_group.addButton(self.mobil_rb)
        self._model_group.addButton(self.ddm_rb)

        self.default_cb = QCheckBox("Default Parameters")
        self.default_cb.setFont(sf)
        self.default_cb.setChecked(True)
        self.default_cb.stateChanged.connect(self._on_default_toggled)
        main.addWidget(self.default_cb)

        param_box = QGroupBox()
        param_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        param_layout = QVBoxLayout(param_box)
        param_layout.setContentsMargins(8, 8, 8, 8)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        empty_lbl = QLabel("")
        empty_lbl.setFixedWidth(158)
        header_row.addWidget(empty_lbl)
        header_font = QFont();
        header_font.setPointSize(FONT_SIZE);
        header_font.setBold(True)
        for cls_name in ("Small Vehicle", "Automated Vehicle", "Heavy Vehicle"):
            lbl = QLabel(cls_name)
            lbl.setFont(header_font)
            lbl.setAlignment(Qt.AlignCenter)
            header_row.addWidget(lbl, stretch=1)
            spacer = QLabel("")
            spacer.setFixedWidth(12)
            header_row.addWidget(spacer)
        param_layout.addLayout(header_row)

        ms_row = QHBoxLayout()
        ms_row.setSpacing(12)
        ms_empty = QLabel("")
        ms_empty.setFixedWidth(158)
        ms_row.addWidget(ms_empty)
        for _ in range(3):
            mean_lbl = QLabel("Mean");
            mean_lbl.setFont(QFont("", FONT_SIZE));
            mean_lbl.setAlignment(Qt.AlignCenter)
            std_lbl = QLabel("Std");
            std_lbl.setFont(QFont("", FONT_SIZE));
            std_lbl.setAlignment(Qt.AlignCenter)
            ms_row.addWidget(mean_lbl);
            ms_row.addWidget(std_lbl)
            spacer = QLabel("");
            spacer.setFixedWidth(12);
            ms_row.addWidget(spacer)
        param_layout.addLayout(ms_row)
        param_layout.addSpacing(6)

        self.param_rows = []
        # Max rows = 8 (MOBIL)
        for r in range(8):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(10)
            name_cell = QWidget()
            name_cell.setFixedWidth(158)
            name_lay = QHBoxLayout(name_cell)
            name_lay.setContentsMargins(0, 0, 0, 0)
            name_lay.setSpacing(4)
            var_label = QLabel(f"Param{r + 1}")
            var_label.setFont(QFont("", FONT_SIZE))
            var_label.setFixedWidth(132)
            info_icon = _param_info_icon()
            name_lay.addWidget(var_label)
            name_lay.addWidget(info_icon)
            name_lay.addStretch()
            row_layout.addWidget(name_cell)

            spinboxes = []
            for _ in range(3):
                mean_sb = QDoubleSpinBox()
                mean_sb.setRange(-999.0, 9999.0)
                mean_sb.setDecimals(2)
                mean_sb.setSingleStep(0.1)
                mean_sb.setFixedWidth(80)
                mean_sb.setKeyboardTracking(False)
                mean_sb.setSpecialValueText("")
                mean_sb.setValue(0.0)

                std_sb = QDoubleSpinBox()
                std_sb.setRange(0.0, 9999.0)
                std_sb.setDecimals(2)
                std_sb.setSingleStep(0.1)
                std_sb.setFixedWidth(80)
                std_sb.setKeyboardTracking(False)
                std_sb.setSpecialValueText("")
                std_sb.setValue(0.0)

                row_layout.addWidget(mean_sb)
                row_layout.addWidget(std_sb)
                spinboxes.extend([mean_sb, std_sb])

            self.param_rows.append((var_label, info_icon, spinboxes))
            param_layout.addLayout(row_layout)

        main.addWidget(param_box)

        # --- CAV Cooperative Lane-Changing (C-MOBIL) parameters ---
        cmobil_box = QGroupBox("CAV (C-MOBIL) parameters")
        cmobil_layout = QVBoxLayout(cmobil_box)
        self.cmobil_default_cb = QCheckBox("Default C-MOBIL parameters")
        self.cmobil_default_cb.setFont(sf)
        self.cmobil_default_cb.setChecked(True)
        cmobil_layout.addWidget(self.cmobil_default_cb)

        r1 = QHBoxLayout();
        lbl_kappa = QLabel("kappa (intent urgency weight)");
        lbl_kappa.setFont(sf)
        self.cmobil_kappa = QDoubleSpinBox();
        self.cmobil_kappa.setRange(0.0, 10.0);
        self.cmobil_kappa.setDecimals(3);
        self.cmobil_kappa.setValue(0.1)
        _add_label_with_info(r1, lbl_kappa, _CMOBIL_TOOLTIPS["kappa"])
        r1.addWidget(self.cmobil_kappa);
        r1.addStretch();
        cmobil_layout.addLayout(r1)

        r2 = QHBoxLayout();
        lbl_gamma = QLabel("gamma (lane-change time safety)");
        lbl_gamma.setFont(sf)
        self.cmobil_gamma = QDoubleSpinBox();
        self.cmobil_gamma.setRange(0.0, 10.0);
        self.cmobil_gamma.setDecimals(3);
        self.cmobil_gamma.setValue(1.00)
        _add_label_with_info(r2, lbl_gamma, _CMOBIL_TOOLTIPS["gamma"])
        r2.addWidget(self.cmobil_gamma);
        r2.addStretch();
        cmobil_layout.addLayout(r2)

        def _cmobil_toggle():
            enabled = (not self.cmobil_default_cb.isChecked())
            for w in (self.cmobil_kappa, self.cmobil_gamma):
                w.setEnabled(enabled)
                w.setStyleSheet("" if enabled else "color:#888;")

        self.cmobil_default_cb.stateChanged.connect(_cmobil_toggle)
        _cmobil_toggle()

        main.addWidget(cmobil_box)

        self._param_box = param_box

        self.mobil_rb.toggled.connect(self._on_model_toggled)
        self.ddm_rb.toggled.connect(self._on_model_toggled)

        self._on_model_toggled()
        self._on_default_toggled()
        main.addStretch()

        nav = QHBoxLayout()
        nav.addStretch()
        quit_btn = QPushButton("Quit");
        back_btn = QPushButton("<Back");
        next_btn = QPushButton("Next>")
        for b in (quit_btn, back_btn, next_btn):
            f = b.font();
            f.setPointSize(FONT_SIZE);
            b.setFont(f)
        quit_btn.clicked.connect(stacked_widget.close)
        back_btn.clicked.connect(self.go_back)
        next_btn.clicked.connect(self.go_next)
        nav.addWidget(quit_btn);
        nav.addWidget(back_btn);
        nav.addWidget(next_btn)
        main.addLayout(nav)

    def _on_model_toggled(self):
        use_default = self.default_cb.isChecked()
        is_mobil = self.mobil_rb.isChecked()
        labels = self.MOBIL_PARAMS if is_mobil else self.DDM_PARAMS
        defaults = self.DDM_DEFAULTS  # only used for DDM
        tip_map = _LC_MOBIL_TOOLTIPS if is_mobil else _LC_DDM_TOOLTIPS

        for i, (lbl, info_icon, spinboxes) in enumerate(self.param_rows):
            if i < len(labels):
                lbl.setText(labels[i])
                lbl.setVisible(True)
                info_icon.setVisible(True)
                info_icon.setToolTip(tip_map.get(labels[i], ""))
                for j, sb in enumerate(spinboxes):
                    sb.setVisible(True)
                    sb.setEnabled(not use_default)
                    sb.setStyleSheet("" if not use_default else "color: #888;")

                if use_default:
                    if is_mobil:
                        # Apply same MOBIL Mean/Std to all three classes (CSV is not class-based).
                        p = labels[i]
                        st = self._default_mobil_stats.get(p, {"Mean": 0.0, "Std": 0.0})
                        for c_idx in range(3):
                            spinboxes[c_idx * 2].setValue(float(st.get("Mean", 0.0)))
                            spinboxes[c_idx * 2 + 1].setValue(float(st.get("Std", 0.0)))
                    else:
                        # DDM has fixed defaults; Std defaults to 0.
                        for c_idx in range(3):
                            spinboxes[c_idx * 2].setValue(float(defaults[i]))
                            spinboxes[c_idx * 2 + 1].setValue(0.0)
            else:
                lbl.setVisible(False)
                info_icon.setVisible(False)
                for sb in spinboxes:
                    sb.setVisible(False)

    def _on_default_toggled(self):
        self._on_model_toggled()

    def go_back(self):
        car_following_page = self.stacked_widget.widget(6)
        car_following_page.set_network_type(self.network_type)
        car_following_page.ped_allowed = self.ped_allowed
        car_following_page.bike_allowed = self.bike_allowed

        self.stacked_widget.setCurrentIndex(6)

    def go_next(self):
        model_name = "MOBIL" if self.mobil_rb.isChecked() else "DDM"
        self.responses.all_responses["LC_Model"] = model_name
        use_default = self.default_cb.isChecked()
        self.responses.all_responses["LC_Default_Params"] = use_default

        param_data = {}
        for lbl, _info_icon, spinboxes in self.param_rows:
            if not lbl.isVisible(): continue
            row_label = lbl.text().strip()
            cls_data = {}
            for i, cls_name in enumerate(["Small Vehicle", "Automated Vehicle", "Heavy Vehicle"]):
                mean_val = spinboxes[i * 2].value()
                std_val = spinboxes[i * 2 + 1].value()
                cls_data[cls_name] = {"Mean": mean_val, "Std": std_val}
            param_data[row_label] = cls_data

        self.responses.all_responses["LC_Parameters"] = param_data

        # Store CAV C-MOBIL parameters
        try:
            self.responses.all_responses["CMOBIL_Params"] = {
                "Default": bool(self.cmobil_default_cb.isChecked()),
                "kappa": float(self.cmobil_kappa.value()),
                "gamma": float(self.cmobil_gamma.value()),
            }
        except Exception:
            self.responses.all_responses["CMOBIL_Params"] = {"Default": True, "kappa": 0.1, "gamma": 1.0}

        if getattr(self, "network_type", "") == "Single Intersection" and (self.ped_allowed or self.bike_allowed):
            atm_model_page = self.stacked_widget.widget(8)
            atm_model_page.set_network_type(self.network_type)
            atm_model_page.ped_allowed = self.ped_allowed
            atm_model_page.bike_allowed = self.bike_allowed
            self.stacked_widget.setCurrentIndex(8)
        else:
            viz_page = self.stacked_widget.widget(9)
            viz_page.set_network_type(self.network_type)
            viz_page.ped_allowed = self.ped_allowed
            viz_page.bike_allowed = self.bike_allowed
            self.stacked_widget.setCurrentIndex(9)


class ATMModelsPage(QWidget):
    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.responses = responses
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)

        self.network_type = None
        self.param_rows = []
        self.ped_allowed = True
        self.bike_allowed = True
        # Corrected Parameters
        self.PT_ATM_PARAMS = [
            "w_c b-b", "w_c p-p", "w_c p-b", "w_c b-p", "w_c p_bar", "w_c b_bar",
            "η_ped", "ξ_ped", "τ_ped", "v_desired_ped", "η_bike", "ξ_bike", "τ_bike", "v_desired_bike"
        ]
        self.SF_ATM_PARAMS = [
            "Ped: v_α", "Ped: τ_α", "Ped: A_pp", "Ped: B_pp", "Ped: A_wall", "Ped: B_wall",
            "Bike: τ_γ", "Bike: v_γ", "Bike: a_γ", "Bike: b_γ", "Bike: η_γ",
            "Bike: ε_m", "Bike: A_w", "Bike: B_w", "Bike: A_s", "Bike: B_s", "Bike: τ"
        ]
        # Default ATM parameters should reflect what's in the calibrated CSV sample sets.
        self._default_atm_stats = self._compute_default_atm_stats()

        self._build_ui()

    def _compute_default_atm_stats(self):
        """
        Compute Mean/Std defaults for ATM models from:
          - models/model_params/SF_atm_params.csv
          - models/model_params/PT_atm_params.csv

        Returns:
          {"SF": {label: {"Mean":..,"Std":..}}, "PT": {...}}
        """
        out = {"SF": {}, "PT": {}}

        # --- SF (Social Force / bike dynamics) ---
        sf_path = MODEL_PARAMS_DIR / "SF_atm_params.csv"
        sf_col_map = {
            "Ped: v_α": "vAlpha0_ped",
            "Ped: τ_α": "tauAlpha_ped",
            "Ped: A_pp": "A_pp",
            "Ped: B_pp": "B_pp",
            "Ped: A_wall": "A_wall",
            "Ped: B_wall": "B_wall",
            "Bike: τ_γ": "tau_gamma",
            "Bike: v_γ": "v_gamma0",
            "Bike: a_γ": "a_gamma_max",
            "Bike: b_γ": "b_gamma",
            "Bike: η_γ": "eta_gamma",
            "Bike: ε_m": "eps_m",
            "Bike: A_w": "A_w",
            "Bike: B_w": "B_w",
            "Bike: A_s": "A_s",
            "Bike: B_s": "B_s",
            "Bike: τ": "T_i",
        }
        try:
            df_sf = pd.read_csv(str(sf_path))
        except Exception:
            df_sf = None

        for label in self.SF_ATM_PARAMS:
            col = sf_col_map.get(label)
            if df_sf is None or (col is None) or (col not in df_sf.columns):
                out["SF"][label] = {"Mean": 0.0, "Std": 0.0}
                continue
            s = pd.to_numeric(df_sf[col], errors="coerce").dropna()
            out["SF"][label] = {"Mean": float(s.mean()), "Std": float(s.std(ddof=1))}

        # --- PT (Prospect Theory) ---
        pt_path = MODEL_PARAMS_DIR / "PT_atm_params.csv"
        pt_col_map = {
            "w_c b-b": "Wc_bb",
            "w_c p-p": "Wc_pp",
            "w_c p-b": "Wc_pb",
            "w_c b-p": "Wc_bp",
            "w_c p_bar": "Wc_pbar",
            "w_c b_bar": "Wc_bbar",
            "η_ped": "eta_ped",
            "ξ_ped": "xi_ped",
            "τ_ped": "tau_ped",
            "v_desired_ped": "v_pref_ped",
            "η_bike": "eta_bike",
            "ξ_bike": "xi_bike",
            "τ_bike": "tau_bike",
            "v_desired_bike": "v_pref_bike",
        }
        try:
            df_pt = pd.read_csv(str(pt_path))
        except Exception:
            df_pt = None

        for label in self.PT_ATM_PARAMS:
            col = pt_col_map.get(label)
            if df_pt is None or (col is None) or (col not in df_pt.columns):
                out["PT"][label] = {"Mean": 0.0, "Std": 0.0}
                continue
            s = pd.to_numeric(df_pt[col], errors="coerce").dropna()
            out["PT"][label] = {"Mean": float(s.mean()), "Std": float(s.std(ddof=1))}

        return out

    def set_network_type(self, network_type):
        self.network_type = network_type
        self.title_label.setText(f"{network_type} Scenario - ATM Models")

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setAlignment(Qt.AlignTop)

        self.title_label = QLabel("Active-Traffic-Mode Models")
        tf = self.title_label.font();
        tf.setPointSize(20);
        tf.setBold(True);
        self.title_label.setFont(tf)
        main.addWidget(self.title_label)

        main.addWidget(create_progress_bar("Models"))
        main.addWidget(make_separator())
        main.addSpacing(12)

        sub = QLabel("Pedestrian Model Selection")
        sf = sub.font();
        sf.setPointSize(FONT_SIZE);
        sub.setFont(sf)
        main.addWidget(sub)

        model_row = QHBoxLayout()
        self.sf_rb = QRadioButton("Social Force (SF)")
        self.pt_rb = QRadioButton("Prospect Theory (PT)")
        self.sf_rb.setFont(sf);
        self.pt_rb.setFont(sf)
        self.sf_rb.setChecked(True)
        model_row.addWidget(self.sf_rb)
        model_row.addWidget(self.pt_rb)
        model_row.addStretch()
        main.addLayout(model_row)

        self._model_group = QButtonGroup(self)
        self._model_group.addButton(self.sf_rb)
        self._model_group.addButton(self.pt_rb)

        self.default_cb = QCheckBox("Default Parameters")
        self.default_cb.setFont(sf)
        self.default_cb.setChecked(True)
        self.default_cb.stateChanged.connect(self._on_default_toggled)
        main.addWidget(self.default_cb)

        param_box = QGroupBox()
        param_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # removed setFixedHeight to allow expansion
        param_layout = QVBoxLayout(param_box)
        param_layout.setContentsMargins(8, 8, 8, 8)

        header_row = QHBoxLayout()
        hdr_param_cell = QWidget()
        hdr_param_cell.setFixedWidth(158)
        hdr_param_lay = QHBoxLayout(hdr_param_cell)
        hdr_param_lay.setContentsMargins(0, 0, 0, 0)
        param_lbl = QLabel("Parameter")
        param_lbl.setFont(QFont("", FONT_SIZE, QFont.Bold))
        param_lbl.setAlignment(Qt.AlignCenter)
        hdr_param_lay.addWidget(param_lbl)
        header_row.addWidget(hdr_param_cell)
        mean_lbl = QLabel("Mean")
        mean_lbl.setFont(QFont("", FONT_SIZE, QFont.Bold))
        mean_lbl.setAlignment(Qt.AlignCenter)
        mean_lbl.setFixedWidth(100)
        std_lbl = QLabel("Std")
        std_lbl.setFont(QFont("", FONT_SIZE, QFont.Bold))
        std_lbl.setAlignment(Qt.AlignCenter)
        std_lbl.setFixedWidth(100)
        header_row.addWidget(mean_lbl)
        header_row.addWidget(std_lbl)
        param_layout.addLayout(header_row)
        param_layout.addSpacing(6)

        self.param_rows = []
        # Max rows = 17 (SF)
        for i in range(17):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(18)
            name_cell = QWidget()
            name_cell.setFixedWidth(158)
            name_lay = QHBoxLayout(name_cell)
            name_lay.setContentsMargins(0, 0, 0, 0)
            name_lay.setSpacing(4)
            var_label = QLabel(f"Param{i + 1}")
            var_label.setFixedWidth(128)
            var_label.setFont(QFont("", 11))  # slightly smaller font to fit vertically
            info_icon = _param_info_icon(11)
            name_lay.addWidget(var_label)
            name_lay.addWidget(info_icon)
            name_lay.addStretch()
            row_layout.addWidget(name_cell)

            mean_sb = QDoubleSpinBox()
            # PT contains W_c terms that can be large; allow a wider numeric range.
            mean_sb.setRange(-1e9, 1e9)
            mean_sb.setSingleStep(0.1)
            mean_sb.setDecimals(2)
            mean_sb.setKeyboardTracking(False)
            mean_sb.setFixedWidth(100)
            mean_sb.setSpecialValueText("")
            mean_sb.setValue(0.0)

            std_sb = QDoubleSpinBox()
            std_sb.setRange(0.0, 1e9)
            std_sb.setSingleStep(0.1)
            std_sb.setDecimals(2)
            std_sb.setKeyboardTracking(False)
            std_sb.setFixedWidth(100)
            std_sb.setSpecialValueText("")
            std_sb.setValue(0.0)

            row_layout.addWidget(mean_sb)
            row_layout.addWidget(std_sb)

            self.param_rows.append((var_label, info_icon, [mean_sb, std_sb]))
            param_layout.addLayout(row_layout)

        main.addWidget(param_box)
        self._param_box = param_box

        self.sf_rb.toggled.connect(self._on_model_toggled)
        self.pt_rb.toggled.connect(self._on_model_toggled)
        self._on_model_toggled()
        self._on_default_toggled()

        main.addStretch()

        nav = QHBoxLayout()
        nav.addStretch()
        quit_btn = QPushButton("Quit");
        back_btn = QPushButton("<Back");
        next_btn = QPushButton("Next>")
        for b in (quit_btn, back_btn, next_btn):
            f = b.font();
            f.setPointSize(FONT_SIZE);
            b.setFont(f)
        quit_btn.clicked.connect(stacked_widget.close)
        back_btn.clicked.connect(self.go_back)
        next_btn.clicked.connect(self.go_next)
        nav.addWidget(quit_btn);
        nav.addWidget(back_btn);
        nav.addWidget(next_btn)
        main.addLayout(nav)

    def _on_model_toggled(self):
        use_default = self.default_cb.isChecked()
        is_sf = self.sf_rb.isChecked()
        labels = self.SF_ATM_PARAMS if is_sf else self.PT_ATM_PARAMS
        model_key = "SF" if is_sf else "PT"
        tip_map = _ATM_SF_TOOLTIPS if is_sf else _ATM_PT_TOOLTIPS

        for i, (lbl, info_icon, spinboxes) in enumerate(self.param_rows):
            if i < len(labels):
                lbl.setText(labels[i])
                lbl.setVisible(True)
                info_icon.setVisible(True)
                info_icon.setToolTip(tip_map.get(labels[i], ""))
                for sb in spinboxes:
                    sb.setVisible(True)
                    sb.setEnabled(not use_default)
                    sb.setStyleSheet("" if not use_default else "color: #888;")
                if use_default:
                    stats = self._default_atm_stats.get(model_key, {}).get(labels[i], {"Mean": 0.0, "Std": 0.0})
                    spinboxes[0].setValue(float(stats.get("Mean", 0.0)))
                    spinboxes[1].setValue(float(stats.get("Std", 0.0)))
            else:
                lbl.setVisible(False)
                info_icon.setVisible(False)
                for sb in spinboxes:
                    sb.setVisible(False)

    def _on_default_toggled(self):
        self._on_model_toggled()

    def go_back(self):
        lane_changing_page = self.stacked_widget.widget(7)
        lane_changing_page.set_network_type(self.network_type)
        lane_changing_page.ped_allowed = self.ped_allowed
        lane_changing_page.bike_allowed = self.bike_allowed
        self.stacked_widget.setCurrentIndex(7)

    def go_next(self):
        model_name = "SF" if self.sf_rb.isChecked() else "PT"
        self.responses.all_responses["ATM_Model"] = model_name
        use_default = self.default_cb.isChecked()
        self.responses.all_responses["ATM_Default_Params"] = use_default

        param_data = {}
        for lbl, _info_icon, spinboxes in self.param_rows:
            if not lbl.isVisible(): continue
            row_label = lbl.text().strip()
            mean_val = spinboxes[0].value()
            std_val = spinboxes[1].value()
            param_data[row_label] = {"Mean": mean_val, "Std": std_val}

        self.responses.all_responses["ATM_Parameters"] = param_data
        self.responses.all_responses["Ped_Allowed"] = self.ped_allowed
        self.responses.all_responses["Bike_Allowed"] = self.bike_allowed
        viz_page = self.stacked_widget.widget(9)
        viz_page.set_network_type(self.network_type)
        viz_page.ped_allowed = self.ped_allowed
        viz_page.bike_allowed = self.bike_allowed
        self.stacked_widget.setCurrentIndex(9)


class VisualizationSelectionPage(QWidget):
    """Choose which post-run plots to generate after the trajectory CSV is saved."""

    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.responses = responses
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)
        self.network_type = None
        self.ped_allowed = False
        self.bike_allowed = False
        self._build_ui()

    def set_network_type(self, network_type):
        self.network_type = network_type
        self.title_label.setText(f"{network_type} Scenario - Result visualization")

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setAlignment(Qt.AlignTop)

        self.title_label = QLabel("Result visualization")
        tf = self.title_label.font()
        tf.setPointSize(20)
        tf.setBold(True)
        self.title_label.setFont(tf)
        main.addWidget(self.title_label)

        main.addWidget(create_progress_bar("Visualization"))
        main.addWidget(make_separator())
        main.addSpacing(12)

        sub = QLabel(
            "Select one or more plot types. They are generated automatically when the trajectory "
            "CSV is saved (after simulation with data collection enabled)."
        )
        sf = sub.font()
        sf.setPointSize(FONT_SIZE)
        sub.setFont(sf)
        sub.setWordWrap(True)
        main.addWidget(sub)
        main.addSpacing(16)

        self.cb_traj = QCheckBox("X–Y trajectory plots (all agents on one plot)")
        self.cb_ts = QCheckBox("Time vs space (one plot per lane)")
        self.cb_fd = QCheckBox("Flow vs density (one plot per lane)")
        for cb in (self.cb_traj, self.cb_ts, self.cb_fd):
            cb.setFont(sf)
        for cb in (self.cb_traj, self.cb_ts, self.cb_fd):
            cb.setChecked(True)

        bulk_row = QHBoxLayout()
        bulk_row.setSpacing(12)
        sel_all = QPushButton("Select all")
        sel_none = QPushButton("Clear all")
        for b in (sel_all, sel_none):
            bf = b.font()
            bf.setPointSize(FONT_SIZE - 1)
            b.setFont(bf)
        sel_all.clicked.connect(self._select_all_plots)
        sel_none.clicked.connect(self._clear_all_plots)
        bulk_row.addWidget(sel_all)
        bulk_row.addWidget(sel_none)
        bulk_row.addStretch()

        main.addWidget(self.cb_traj)
        main.addWidget(self.cb_ts)
        main.addWidget(self.cb_fd)
        main.addSpacing(8)
        main.addLayout(bulk_row)

        fd_row = QHBoxLayout()
        fd_lbl = QLabel("Flow/density aggregation time (s):")
        fd_lbl.setFont(sf)
        self.fd_time_bin = QDoubleSpinBox()
        self.fd_time_bin.setRange(0.5, 3600.0)
        self.fd_time_bin.setSingleStep(0.5)
        self.fd_time_bin.setDecimals(2)
        self.fd_time_bin.setValue(30.0)
        self.fd_time_bin.setToolTip(
            "Time window used to bin trajectory data when estimating flow vs density "
            "(smaller = more points, noisier; larger = smoother)."
        )
        fd_row.addWidget(fd_lbl)
        fd_row.addWidget(self.fd_time_bin)
        fd_row.addStretch()
        main.addSpacing(12)
        main.addLayout(fd_row)
        main.addStretch()

        nav = QHBoxLayout()
        nav.addStretch()
        quit_btn = QPushButton("Quit")
        back_btn = QPushButton("<Back")
        next_btn = QPushButton("Next>")
        for b in (quit_btn, back_btn, next_btn):
            f = b.font()
            f.setPointSize(FONT_SIZE)
            b.setFont(f)
        quit_btn.clicked.connect(self.stacked_widget.close)
        back_btn.clicked.connect(self.go_back)
        next_btn.clicked.connect(self.go_next)
        nav.addWidget(quit_btn)
        nav.addWidget(back_btn)
        nav.addWidget(next_btn)
        main.addLayout(nav)

    def _select_all_plots(self):
        self.cb_traj.setChecked(True)
        self.cb_ts.setChecked(True)
        self.cb_fd.setChecked(True)

    def _clear_all_plots(self):
        self.cb_traj.setChecked(False)
        self.cb_ts.setChecked(False)
        self.cb_fd.setChecked(False)

    def _selected_modes(self):
        """Return ordered list of mode keys for selected checkboxes."""
        order = (
            ("trajectory_xy", self.cb_traj),
            ("time_space", self.cb_ts),
            ("flow_density", self.cb_fd),
        )
        return [key for key, cb in order if cb.isChecked()]

    def go_back(self):
        if getattr(self, "network_type", "") == "Single Intersection" and (
            self.ped_allowed or self.bike_allowed
        ):
            atm_model_page = self.stacked_widget.widget(8)
            atm_model_page.set_network_type(self.network_type)
            atm_model_page.ped_allowed = self.ped_allowed
            atm_model_page.bike_allowed = self.bike_allowed
            self.stacked_widget.setCurrentIndex(8)
        else:
            lane_changing_page = self.stacked_widget.widget(7)
            lane_changing_page.set_network_type(self.network_type)
            lane_changing_page.ped_allowed = self.ped_allowed
            lane_changing_page.bike_allowed = self.bike_allowed
            self.stacked_widget.setCurrentIndex(7)

    def go_next(self):
        modes = self._selected_modes()
        if not modes:
            QMessageBox.warning(
                self,
                "No plot selected",
                "Please select at least one visualization type, or click Cancel and use "
                "\"Select all\" to enable all three.",
            )
            return
        self.responses.all_responses["PostSim_Viz"] = modes
        self.responses.all_responses["PostSim_FlowDensity_TimeBin_s"] = float(
            self.fd_time_bin.value()
        )
        self.responses.all_responses["Ped_Allowed"] = self.ped_allowed
        self.responses.all_responses["Bike_Allowed"] = self.bike_allowed
        sim_config_page = self.stacked_widget.widget(10)
        sim_config_page.set_network_type(self.network_type)
        sim_config_page.ped_allowed = self.ped_allowed
        sim_config_page.bike_allowed = self.bike_allowed
        self.stacked_widget.setCurrentIndex(10)


class SimulationSettingsPage(QWidget):
    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.responses = responses
        self.stacked_widget = stacked_widget
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)

        self.step_size = 0.1
        self.sim_time = 1200
        self.visualization = True
        self.data_collection = True
        self.data_freq = 0.5
        self.output_location = ""

        self.ped_allowed = False
        self.bike_allowed = False
        self._build_ui()

    def set_network_type(self, network_type):
        self.network_type = network_type
        self.title_label.setText(f"{network_type} Scenario - Simulation Settings")

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setAlignment(Qt.AlignTop)

        self.title_label = QLabel("Simulation Setting")
        tf = self.title_label.font()
        tf.setPointSize(20)
        tf.setBold(True)
        self.title_label.setFont(tf)
        main.addWidget(self.title_label)

        main.addWidget(create_progress_bar("Simulation"))
        main.addWidget(make_separator())
        main.addSpacing(12)

        sub = QLabel("Customize your simulation parameters")
        sf = sub.font();
        sf.setPointSize(FONT_SIZE);
        sub.setFont(sf)
        main.addWidget(sub)

        step_layout = QHBoxLayout()
        step_lbl = QLabel("Step size (s):")
        step_lbl.setFont(sf)
        self.step_combo = QComboBox()
        self.step_combo.setFont(sf)
        self.step_combo.addItems(["0.1", "0.2", "0.5", "1.0"])
        self.step_combo.setCurrentText(str(self.step_size))
        step_layout.addWidget(step_lbl)
        step_layout.addWidget(self.step_combo)
        step_layout.addStretch()
        main.addLayout(step_layout)
        main.addSpacing(8)

        sim_layout = QHBoxLayout()
        sim_lbl = QLabel("Simulation time (s):")
        sim_lbl.setFont(sf)
        self.sim_spin = QSpinBox()
        self.sim_spin.setFont(sf)
        self.sim_spin.setRange(10, 18000)
        self.sim_spin.setSingleStep(50)
        self.sim_spin.setValue(self.sim_time)
        sim_layout.addWidget(sim_lbl)
        sim_layout.addWidget(self.sim_spin)
        sim_layout.addStretch()
        main.addLayout(sim_layout)
        main.addSpacing(8)

        self.vis_cb = QCheckBox("Enable Visualization")
        self.vis_cb.setFont(sf)
        self.vis_cb.setChecked(self.visualization)
        main.addWidget(self.vis_cb)
        main.addSpacing(4)

        self.data_cb = QCheckBox("Enable Data Collection")
        self.data_cb.setFont(sf)
        self.data_cb.setChecked(self.data_collection)
        main.addWidget(self.data_cb)
        main.addSpacing(4)

        freq_layout = QHBoxLayout()
        freq_lbl = QLabel("Data Collection Frequency (s):")
        freq_lbl.setFont(sf)
        self.freq_combo = QComboBox()
        self.freq_combo.setFont(sf)
        self.freq_combo.addItems(["0.1", "0.2", "0.5", "1", "5", "10"])
        self.freq_combo.setCurrentText(str(self.data_freq))
        freq_layout.addWidget(freq_lbl)
        freq_layout.addWidget(self.freq_combo)
        freq_layout.addStretch()
        main.addLayout(freq_layout)
        main.addSpacing(8)

        folder_layout = QHBoxLayout()
        folder_lbl = QLabel("Output Location:")
        folder_lbl.setFont(sf)
        self.folder_path_le = QLineEdit()
        self.folder_path_le.setFont(sf)
        self.folder_path_le.setReadOnly(True)
        self.folder_btn = QPushButton("Browse")
        self.folder_btn.setFont(sf)
        self.folder_btn.clicked.connect(self._browse_folder)

        folder_layout.addWidget(folder_lbl)
        folder_layout.addWidget(self.folder_path_le)
        folder_layout.addWidget(self.folder_btn)
        folder_layout.addStretch()
        main.addLayout(folder_layout)
        main.addSpacing(12)

        self.folder_path_le.setEnabled(self.data_cb.isChecked())
        self.folder_btn.setEnabled(self.data_cb.isChecked())
        self.data_cb.stateChanged.connect(self._on_data_collection_toggled)

        main.addStretch()

        nav = QHBoxLayout()
        nav.addStretch()
        quit_btn = QPushButton("Quit");
        back_btn = QPushButton("<Back");
        next_btn = QPushButton("Next>")
        for b in (quit_btn, back_btn, next_btn):
            f = b.font();
            f.setPointSize(FONT_SIZE);
            b.setFont(f)
        quit_btn.clicked.connect(stacked_widget.close)
        back_btn.clicked.connect(self.go_back)
        next_btn.clicked.connect(self.go_next)
        nav.addWidget(quit_btn);
        nav.addWidget(back_btn);
        nav.addWidget(next_btn)
        main.addLayout(nav)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Location")
        if folder:
            self.folder_path_le.setText(folder)
            self.responses.all_responses["Data_Folder"] = folder

    def _on_data_collection_toggled(self):
        enabled = self.data_cb.isChecked()
        self.folder_path_le.setEnabled(enabled)
        self.folder_btn.setEnabled(enabled)
        if not enabled:
            self.folder_path_le.clear()
            self.responses.all_responses["Data_Folder"] = ""

    def go_back(self):
        viz_page = self.stacked_widget.widget(9)
        viz_page.set_network_type(self.network_type)
        viz_page.ped_allowed = self.ped_allowed
        viz_page.bike_allowed = self.bike_allowed
        self.stacked_widget.setCurrentIndex(9)

    def go_next(self):
        if self.data_cb.isChecked() and not self.folder_path_le.text():
            QMessageBox.warning(self, "Warning", "Output location cannot be empty!")
            return

        try:
            step_size_val = float(self.step_combo.currentText())
        except ValueError:
            step_size_val = self.step_size
        self.responses.all_responses["Sim_StepSize"] = step_size_val

        self.responses.all_responses["Sim_Time"] = self.sim_spin.value()
        self.responses.all_responses["Sim_Visualization"] = self.vis_cb.isChecked()
        self.responses.all_responses["Sim_DataCollection"] = self.data_cb.isChecked()
        try:
            freq_val = float(self.freq_combo.currentText())
        except ValueError:
            freq_val = self.data_freq
        self.responses.all_responses["Sim_DataFreq"] = freq_val

        self.responses.all_responses["Ped_Allowed"] = self.ped_allowed
        self.responses.all_responses["Bike_Allowed"] = self.bike_allowed
        self.responses.all_responses["Data_Folder"] = self.folder_path_le.text() if self.data_cb.isChecked() else ""

        sim_running_page = self.stacked_widget.widget(11)
        sim_running_page.set_network_type(self.network_type)
        sim_running_page.ped_allowed = self.ped_allowed
        sim_running_page.bike_allowed = self.bike_allowed
        self.stacked_widget.setCurrentIndex(11)


# ====================================================================
# 3. Simulation Running Page (Manages the thread and UI)
# ====================================================================
class SimulationRunningPage(QWidget):
    def __init__(self, stacked_widget, responses):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.responses = responses
        self.network_type = "Default"
        self.sim_thread = None
        self._sim_running_on_main_thread = False
        self._sim_cancel_requested = False
        self.setFixedSize(PAGE_WIDTH, PAGE_HEIGHT)

        self._build_ui()

    def set_network_type(self, network_type):
        """Sets the network type and updates the title."""
        self.network_type = network_type
        self.title_lbl.setText(f"**{network_type}** Simulation Running ...")

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setAlignment(Qt.AlignCenter)

        # Title Label
        self.title_lbl = QLabel("Simulation Running ...")
        tf = self.title_lbl.font()
        tf.setPointSize(24)
        tf.setBold(True)
        self.title_lbl.setFont(tf)
        self.title_lbl.setAlignment(Qt.AlignCenter)
        main.addWidget(self.title_lbl)

        main.addSpacing(40)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main.addWidget(self.progress_bar)

        main.addStretch()

        # Navigation Buttons
        nav = QHBoxLayout()
        nav.addStretch()

        self.back_btn = QPushButton("<Back")
        self.close_btn = QPushButton("Close")

        for b in (self.back_btn, self.close_btn):
            f = b.font()
            f.setPointSize(FONT_SIZE)
            b.setFont(f)

        self.back_btn.clicked.connect(self.go_back)
        self.close_btn.clicked.connect(self.terminate_and_close)

        nav.addWidget(self.back_btn)
        nav.addWidget(self.close_btn)
        main.addLayout(nav)

    def showEvent(self, event):
        """Called automatically when the page is made visible."""
        super().showEvent(event)
        # Start the simulation when the page is shown
        self.start_simulation()

    def start_simulation(self):
        """Initializes and starts the simulation (thread or main thread for SUMO-GUI)."""
        self.progress_bar.setValue(0)
        self.back_btn.setEnabled(True)
        self._sim_cancel_requested = False

        use_visualization = self.responses.all_responses.get("Sim_Visualization", False)
        if use_visualization:
            # SUMO-GUI must run with TraCI on the main thread on Windows; run sim here and keep UI responsive
            self._sim_running_on_main_thread = True
            self.sim_thread = None
            try:
                def progress_cb(pct):
                    self.progress_bar.setValue(pct)
                    QApplication.processEvents()

                # run_simulation expects a progress_signal with .emit(int); use adapter for plain callable
                class _ProgressAdapter:
                    def __init__(self, fn):
                        self.emit = fn
                run_simulation(_ProgressAdapter(progress_cb), lambda: not self._sim_cancel_requested)
                self.simulation_done(True)
            except Exception as e:
                print(f"Simulation error: {e}")
                self.simulation_done(False)
            finally:
                self._sim_running_on_main_thread = False
            return

        # No visualization: run in background thread
        self.sim_thread = SimulationThread(run_simulation)
        self.sim_thread.progress_update.connect(self.update_progress)
        self.sim_thread.simulation_finished.connect(self.simulation_done)
        self.sim_thread.start()

    def update_progress(self, value):
        """Updates the progress bar based on the thread's signal."""
        self.progress_bar.setValue(value)

    def simulation_done(self, success):
        """Handles the completion or termination of the simulation."""
        self.sim_thread = None
        self._sim_running_on_main_thread = False

        if success:
            self.title_lbl.setText(f"**{self.network_type}** Simulation **Completed!**")
            self.back_btn.setEnabled(False)  # Prevent going back after success
            # Automatically close the page after a short delay (optional)
            QTimer.singleShot(2000, self.stacked_widget.close)
        else:
            self.title_lbl.setText(
                f"**{self.network_type}** Simulation stopped (SUMO closed). Use **Back** to change settings and run again."
            )
            # Back stays enabled so user can go back and rerun

    def terminate_simulation(self):
        """Requests the simulation to stop (thread or main-thread run)."""
        if self._sim_running_on_main_thread:
            print("Requesting simulation stop...")
            self._sim_cancel_requested = True
            return
        if self.sim_thread and self.sim_thread.isRunning():
            print("Terminating simulation...")
            self.sim_thread.stop()
            self.sim_thread = None

    def go_back(self):
        """Terminates simulation and navigates back to the settings page."""
        self.terminate_simulation()

        sim_settings_page = self.stacked_widget.widget(10)
        sim_settings_page.set_network_type(self.network_type)
        self.stacked_widget.setCurrentIndex(10)

    def terminate_and_close(self):
        """Terminates simulation and closes the application."""
        self.terminate_simulation()
        self.stacked_widget.close()


# ====================================================================
# 4. Minimal Setup for Running the Example
# ====================================================================

# Placeholders for required classes and functions

# --- End of Placeholders ---

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # --- Global font: force one simple system font family everywhere (no installs) ---
    # This only sets the font FAMILY globally; widget-specific point sizes/bold remain unchanged.
    from PyQt5.QtGui import QFontDatabase
    preferred = ['Segoe UI', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'Sans Serif']
    families = set(QFontDatabase().families())
    family = next((f for f in preferred if f in families), 'Sans Serif')
    app.setFont(QFont(family, 13))
    app.setStyleSheet(f"QWidget {{ font-family: '{family}'; }}")

    stacked_widget = QStackedWidget()

    # Create page instances
    welcome_page = WelcomePage(stacked_widget)  # 0
    geometry_page = GeometrySelectionPage(stacked_widget, responses)  # 1
    network_page = NetworkConfigurationPage(stacked_widget, responses)  # 2
    volume_page = VolumeConfigurationPage(stacked_widget, responses)  # 3
    signal_control_page = SignalControlPage(stacked_widget, responses)  # 4
    atm_page = ATMDemandPage(stacked_widget, responses)  # 5
    cf_models_page = CFModelsPage(stacked_widget, responses)  # 6
    lc_models_page = LCModelsPage(stacked_widget, responses)  # 7
    atm_models_page = ATMModelsPage(stacked_widget, responses)  # 8
    visualization_page = VisualizationSelectionPage(stacked_widget, responses)  # 9
    sim_setting_page = SimulationSettingsPage(stacked_widget, responses)  # 10
    sim_running_page = SimulationRunningPage(stacked_widget, responses)  # 11

    # Add pages to stacked widget
    stacked_widget.addWidget(welcome_page)  # index 0
    stacked_widget.addWidget(geometry_page)  # index 1
    stacked_widget.addWidget(network_page)  # index 2
    stacked_widget.addWidget(volume_page)  # index 3
    stacked_widget.addWidget(signal_control_page)  # index 4
    stacked_widget.addWidget(atm_page)  # index 5
    stacked_widget.addWidget(cf_models_page)  # index 6
    stacked_widget.addWidget(lc_models_page)  # index 7
    stacked_widget.addWidget(atm_models_page)  # index 8
    stacked_widget.addWidget(visualization_page)  # index 9
    stacked_widget.addWidget(sim_setting_page)  # index 10
    stacked_widget.addWidget(sim_running_page)  # index 11

    # Set the network type for a better demonstration
    # sim_running_page.set_network_type("Highway")

    # Start on the SimulationRunningPage for testing
    stacked_widget.setCurrentIndex(0)
    # stacked_widget.setWindowTitle("Simulation GUI with QThread")
    stacked_widget.show()

    sys.exit(app.exec_())