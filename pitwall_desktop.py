"""
F1 Pit-Wall Live Telemetry & Strategy Matrix
Desktop Standalone Simulation (pitwall_desktop.py)
Featuring Real-Time Seamless Track Switching Across All 17 World Championship Circuits.
Fixed for Python 3.15 + Tkinter Canvas Focus + Matplotlib FuncAnimation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from collections import deque

# Import World Circuits Library
from circuits import get_circuit_data, CIRCUIT_DATABASE, match_circuit_key, catmull_rom_spline_2d

# ==============================================================================
# 1. CORE THEMING & COLOR PALETTE (Enterprise Pit-Wall Dark Mode)
# ==============================================================================
HEX_BG_CANVAS = '#0B0C10'      # Outer app window background
HEX_BG_PANEL  = '#111318'      # Subplot widget backgrounds
HEX_BORDER    = '#232732'      # Structural grid lines and card borders
HEX_TEXT_MAIN = '#EAEAEA'      # Primary data labels
HEX_TEXT_MUTED= '#6A7680'      # Secondary units and axis labels

HEX_SPEED    = '#00E5FF'       # Cyan - High visibility speed index
HEX_THROTTLE = '#00E676'       # Green - Positive acceleration
HEX_BRAKE    = '#FF1744'       # Red - High friction braking
HEX_TRACK    = '#45A29E'       # Teal - Track structural outline
HEX_CAR      = '#FFFFFF'       # White - Telemetry position node

CIRCUIT_ORDER = list(CIRCUIT_DATABASE.keys())

# ==============================================================================
# 2. RESOLVE STARTUP CIRCUIT
# ==============================================================================
initial_circuit_key = "cota"
if len(sys.argv) > 1:
    arg_clean = sys.argv[1].strip().lower()
    if arg_clean.isdigit() and 1 <= int(arg_clean) <= len(CIRCUIT_ORDER):
        initial_circuit_key = CIRCUIT_ORDER[int(arg_clean) - 1]
    else:
        initial_circuit_key = match_circuit_key(arg_clean)

# ==============================================================================
# 3. GLOBAL TELEMETRY STATE & BUFFERS
# ==============================================================================
current_circuit_key = initial_circuit_key
current_circuit = get_circuit_data(current_circuit_key)
target_length = current_circuit["length_m"]
fine_dist = np.linspace(0, target_length, 2000)
smooth_track_x = np.zeros(2000)
smooth_track_y = np.zeros(2000)
current_landmark_artists = []
is_paused = [False]

BUFFER_LEN = 150
time_buffer = deque(np.linspace(-3.0, 0.0, BUFFER_LEN), maxlen=BUFFER_LEN)
speed_buffer = deque([0.0] * BUFFER_LEN, maxlen=BUFFER_LEN)
throttle_buffer = deque([0.0] * BUFFER_LEN, maxlen=BUFFER_LEN)
brake_buffer = deque([0.0] * BUFFER_LEN, maxlen=BUFFER_LEN)

# ==============================================================================
# 4. GRAPHICS INTERFACE & LAYOUT ARCHITECTURE
# ==============================================================================
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['text.color'] = HEX_TEXT_MAIN
plt.rcParams['toolbar'] = 'None'  # Disables default toolbar buttons so they never steal Tab focus

fig = plt.figure(figsize=(15, 9), facecolor=HEX_BG_CANVAS)
gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0, 0.8], figure=fig)

ax_track  = fig.add_subplot(gs[:, 0])
ax_speed  = fig.add_subplot(gs[0, 1])
ax_pedals = fig.add_subplot(gs[1, 1])
ax_status = fig.add_subplot(gs[2, 1])

def style_panel(ax, title=""):
    ax.set_facecolor(HEX_BG_PANEL)
    for spine in ax.spines.values():
        spine.set_color(HEX_BORDER)
    ax.tick_params(colors=HEX_TEXT_MUTED, labelsize=9, length=3)
    ax.grid(True, color=HEX_BORDER, linestyle=':', alpha=0.6)
    if title:
        ax.set_title(title, color=HEX_TEXT_MAIN, fontsize=10, fontweight='bold', loc='left', pad=8)

style_panel(ax_track)
style_panel(ax_speed, "CHANNEL M01: VEHICLE SPEED")
style_panel(ax_pedals, "CHANNEL M02: DRIVER INPUTS (THROTTLE / BRAKE)")
style_panel(ax_status)

# Track Geometry Vectors
line_track_border, = ax_track.plot([], [], color=HEX_BORDER, lw=8, zorder=1)
line_track_inner,  = ax_track.plot([], [], color=HEX_TRACK, lw=2.5, alpha=0.85, zorder=2)
car_node,   = ax_track.plot([], [], marker='o', color=HEX_CAR, ms=8, markeredgecolor=HEX_SPEED, markeredgewidth=1.5, zorder=4)
ghost_node, = ax_track.plot([], [], marker='x', color=HEX_TEXT_MUTED, ms=6, alpha=0.6, zorder=4)
ax_track.axis('equal')
ax_track.set_xticks([])
ax_track.set_yticks([])

# Speed & Pedal Vectors
line_speed, = ax_speed.plot(list(time_buffer), list(speed_buffer), color=HEX_SPEED, lw=1.75)
ax_speed.set_ylabel("V [km/h]", color=HEX_SPEED, fontsize=9, fontweight='bold')
ax_speed.set_xlim(-3.0, 0)

line_throttle, = ax_pedals.plot(list(time_buffer), list(throttle_buffer), color=HEX_THROTTLE, lw=1.5, label='Throttle')
ax_brake = ax_pedals.twinx()
line_brake, = ax_brake.plot(list(time_buffer), list(brake_buffer), color=HEX_BRAKE, lw=1.5, label='Brake')

ax_pedals.set_ylabel("THR [%]", color=HEX_THROTTLE, fontsize=9, fontweight='bold')
ax_brake.set_ylabel("BRK [%]", color=HEX_BRAKE, fontsize=9, fontweight='bold')
ax_pedals.set_ylim(-5, 105)
ax_brake.set_ylim(-5, 105)
ax_pedals.set_xlim(-3.0, 0)
style_panel(ax_brake)
ax_brake.grid(False)

# Live Status Matrix
ax_status.axis('off')
status_header = ax_status.text(0.02, 0.85, "SYS_STATUS: NOMINAL ENGINE STRATEGY 2", color=HEX_THROTTLE, fontsize=10, fontweight='bold')
telemetry_matrix_text = ax_status.text(0.02, 0.15, "", color=HEX_TEXT_MAIN, fontsize=10, fontfamily='monospace', verticalalignment='bottom')

# On-Screen Interactive Menu Banner
menu_banner = fig.text(
    0.5, 0.02, "",
    ha='center', va='bottom', fontsize=9, color=HEX_SPEED,
    bbox=dict(boxstyle='round,pad=0.4', facecolor=HEX_BG_PANEL, edgecolor=HEX_BORDER, alpha=0.95)
)

# ==============================================================================
# 5. DYNAMIC TRACK SWITCHING ENGINE
# ==============================================================================
def load_and_render_circuit(ckey):
    """Recalculates spline trajectory, redraws track lines, apex landmarks, and scales axes."""
    global current_circuit_key, current_circuit, target_length, fine_dist, smooth_track_x, smooth_track_y
    
    current_circuit_key = ckey
    current_circuit = get_circuit_data(ckey)
    target_length = current_circuit["length_m"]
    raw_waypoints = current_circuit["waypoints"]
    
    # Periodic Pure-NumPy Catmull-Rom Spline Generation
    raw_x, raw_y = catmull_rom_spline_2d(raw_waypoints, n_samples=2000)
    dists = np.sqrt(np.diff(raw_x, prepend=raw_x[0])**2 + np.diff(raw_y, prepend=raw_y[0])**2)
    cum_d = np.cumsum(dists)
    scale_factor = target_length / max(1.0, cum_d[-1])
    
    smooth_track_x = raw_x * scale_factor
    smooth_track_y = raw_y * scale_factor
    fine_dist = np.linspace(0, target_length, 2000)
    
    # Update Vector Track Lines
    line_track_border.set_data(smooth_track_x, smooth_track_y)
    line_track_inner.set_data(smooth_track_x, smooth_track_y)
    
    # Clear & Re-plot Landmark Corner Callouts
    for art in current_landmark_artists:
        art.remove()
    current_landmark_artists.clear()
    
    for lm in current_circuit.get("landmarks", []):
        idx = int(lm["dist_ratio"] * len(fine_dist))
        d_art, = ax_track.plot(smooth_track_x[idx], smooth_track_y[idx], marker='d', color='#ffd700', ms=5, zorder=3)
        t_art = ax_track.text(smooth_track_x[idx], smooth_track_y[idx], f" {lm['name']}", color=HEX_TEXT_MUTED, fontsize=7, zorder=3)
        current_landmark_artists.extend([d_art, t_art])
        
    # Scale Track View Bounding Box with Padding
    pad_x = (smooth_track_x.max() - smooth_track_x.min()) * 0.12
    pad_y = (smooth_track_y.max() - smooth_track_y.min()) * 0.12
    ax_track.set_xlim(smooth_track_x.min() - pad_x, smooth_track_x.max() + pad_x)
    ax_track.set_ylim(smooth_track_y.min() - pad_y, smooth_track_y.max() + pad_y)
    ax_track.set_aspect('equal', adjustable='datalim')
    
    # Adjust Speed Axis Range based on circuit type
    ax_speed.set_ylim(max(30, current_circuit.get("min_speed", 50) - 20), current_circuit.get("top_speed", 335) + 20)
    
    # Update Header Titles & Dynamic Menu Status
    curr_num = CIRCUIT_ORDER.index(ckey) + 1
    total_circuits = len(CIRCUIT_ORDER)
    
    ax_track.set_title(
        f"GPS TRACK TRACKER // {current_circuit['flag']} {current_circuit['name'].upper()} ({target_length:.0f}m)",
        color=HEX_TEXT_MAIN, fontsize=10, fontweight='bold', loc='left', pad=8
    )
    fig.suptitle(
        f"F1 PIT-WALL LIVE TELEMETRY // {current_circuit['flag']} {current_circuit['name'].upper()} - {current_circuit['country'].upper()} [{current_circuit.get('track_type', '')}]",
        color=HEX_SPEED, fontsize=11, fontweight='bold', y=0.98
    )
    menu_banner.set_text(
        f"TRACK [{curr_num}/{total_circuits}]: {current_circuit['flag']} {current_circuit['name']}  |  "
        f"[TAB] or [RIGHT-CLICK] Next Track  |  [SHIFT+TAB] Prev  |  [1-9] Direct Jump  |  [SPACE] Pause"
    )
    
    fig.canvas.draw_idle()

# ==============================================================================
# 6. SEAMLESS TRACK SWITCHING & FOCUS MANAGEMENT ENGINE
# ==============================================================================
def toggle_pause():
    """Toggles live simulation pause/resume state."""
    if 'ani' in globals() and ani.event_source:
        if is_paused[0]:
            ani.event_source.start()
            is_paused[0] = False
            status_header.set_text("SYS_STATUS: SIMULATION RESUMED // TELEMETRY ACTIVE")
            status_header.set_color(HEX_THROTTLE)
        else:
            ani.event_source.stop()
            is_paused[0] = True
            status_header.set_text("SYS_STATUS: SIMULATION PAUSED [PRESS SPACE]")
            status_header.set_color(HEX_BRAKE)

def cycle_track(forward=True):
    """Safely transitions to next or previous circuit and immediately preserves canvas focus."""
    global current_circuit_key
    curr_idx = CIRCUIT_ORDER.index(current_circuit_key)
    step = 1 if forward else -1
    next_idx = (curr_idx + step) % len(CIRCUIT_ORDER)
    load_and_render_circuit(CIRCUIT_ORDER[next_idx])
    
    if hasattr(fig.canvas, 'get_tk_widget'):
        try:
            fig.canvas.get_tk_widget().focus_set()
        except Exception:
            pass

def on_key_press(event):
    """Allows instant live switching of tracks using keyboard hotkeys without losing focus."""
    if not event.key:
        return
    key = event.key.lower()
    
    if key.isdigit() and 1 <= int(key) <= min(9, len(CIRCUIT_ORDER)):
        target_idx = int(key) - 1
        load_and_render_circuit(CIRCUIT_ORDER[target_idx])
    elif key in ['tab', 'right', 'd']:
        cycle_track(forward=True)
    elif key in ['left', 'backspace', 'a']:
        cycle_track(forward=False)
    elif key == ' ':
        toggle_pause()

    if hasattr(fig.canvas, 'get_tk_widget'):
        try:
            fig.canvas.get_tk_widget().focus_set()
        except Exception:
            pass

def on_mouse_click(event):
    """Enables seamless track switching via mouse (Right Click = Next Track)."""
    if hasattr(fig.canvas, 'get_tk_widget'):
        try:
            fig.canvas.get_tk_widget().focus_set()
        except Exception:
            pass
            
    if event.button == 3:  # Right Click
        cycle_track(forward=True)
    elif event.button == 1 and (event.inaxes == ax_track or event.inaxes is None):
        cycle_track(forward=True)
    elif event.button == 2:  # Middle Click
        toggle_pause()

def on_mouse_scroll(event):
    """Cycle tracks forward or backward with mouse scroll wheel."""
    if event.button == 'up':
        cycle_track(forward=False)
    elif event.button == 'down':
        cycle_track(forward=True)

# Register Matplotlib Canvas Events
fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('button_press_event', on_mouse_click)
fig.canvas.mpl_connect('scroll_event', on_mouse_scroll)

# Bind Tkinter level events to prevent OS from moving focus on Tab
if hasattr(fig.canvas, 'get_tk_widget'):
    try:
        _tk_c = fig.canvas.get_tk_widget()
        _tk_root = _tk_c.winfo_toplevel()
        
        def _on_tk_tab(e):
            cycle_track(forward=True)
            _tk_c.focus_set()
            return "break"

        def _on_tk_shift_tab(e):
            cycle_track(forward=False)
            _tk_c.focus_set()
            return "break"

        _tk_c.bind("<Tab>", _on_tk_tab)
        _tk_c.bind("<Shift-Tab>", _on_tk_shift_tab)
        _tk_c.bind("<ISO_Left_Tab>", _on_tk_shift_tab)
        _tk_root.bind("<Tab>", _on_tk_tab)
        _tk_root.bind("<Shift-Tab>", _on_tk_shift_tab)
        _tk_root.bind("<ISO_Left_Tab>", _on_tk_shift_tab)
    except Exception:
        pass

# Load initial circuit
load_and_render_circuit(initial_circuit_key)

# ==============================================================================
# 7. TELEMETRY PROCESSING ENGINE
# ==============================================================================
def process_telemetry_packet(frame):
    t = frame * 0.04
    circuit_pos = frame % len(fine_dist)
    dist_m = fine_dist[circuit_pos]
    dist_ratio = dist_m / target_length
    
    in_corner = False
    v_target = current_circuit.get("top_speed", 335.0)
    for lm in current_circuit.get("landmarks", []):
        if abs(dist_ratio - lm["dist_ratio"]) < 0.035:
            in_corner = True
            v_target = lm["speed_target"]
            break
            
    if in_corner:
        brake = max(0.0, 85.0 * np.abs(np.sin(t * 2.0)) - np.random.uniform(0, 3))
        throttle = max(0.0, 15.0 + np.random.normal(0, 1.5))
        speed = max(current_circuit.get("min_speed", 60.0), v_target + np.random.normal(0, 1.5))
    else:
        brake = 0.0
        throttle = min(100.0, 98.5 + np.random.normal(0, 0.4))
        speed = min(current_circuit.get("top_speed", 335.0), 220.0 + 8.5 * (frame % 20))
        
    return smooth_track_x[circuit_pos], smooth_track_y[circuit_pos], speed, throttle, brake

def engine_update(frame):
    cx, cy, speed, throttle, brake = process_telemetry_packet(frame)
    
    speed_buffer.append(speed)
    throttle_buffer.append(throttle)
    brake_buffer.append(brake)
    
    car_node.set_data([cx], [cy])
    ghost_node.set_data([cx + np.random.normal(0, 0.25)], [cy + np.random.normal(0, 0.25)])
    
    line_speed.set_ydata(list(speed_buffer))
    line_throttle.set_ydata(list(throttle_buffer))
    line_brake.set_ydata(list(brake_buffer))
    
    delta_time = frame * 0.04
    circuit_pos = frame % len(fine_dist)
    progress_pct = (fine_dist[circuit_pos] / target_length) * 100.0
    
    telemetry_matrix_text.set_text(
        f"CIRCUIT:      {current_circuit['name']} ({current_circuit['country']})\n"
        f"LAP PROGRESS: {progress_pct:05.1f}%  //  TIME: {delta_time:07.2f}s\n"
        f"LIVE VEL:     {speed:03.1f} KM/H  //  RPM: 11,840\n"
        f"THROTTLE:     {throttle:05.1f}%    //  BRAKE: {brake:05.1f}%\n"
        f"ERS STATUS:   REGEN DEPLOY  //  MGU-K: ACTIVE (120 kW)"
    )
    
    if brake > 70.0:
        status_header.set_text("SYS_STATUS: HEAVY DECELERATION ALPHA PHASE")
        status_header.set_color(HEX_BRAKE)
    else:
        status_header.set_text("SYS_STATUS: TELEMETRY STREAM NOMINAL // STRAT 2")
        status_header.set_color(HEX_THROTTLE)
        
    return line_speed, line_throttle, line_brake, car_node, ghost_node, status_header, telemetry_matrix_text

# ==============================================================================
# 8. SAFE ANIMATION & CLEAN EXIT HANDLER
# ==============================================================================
if __name__ == "__main__":
    ani = FuncAnimation(
        fig,
        engine_update,
        interval=20,
        blit=False,
        cache_frame_data=False
    )
    
    def on_close(event):
        if ani.event_source:
            ani.event_source.stop()
            
    fig.canvas.mpl_connect('close_event', on_close)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.94)

    if hasattr(fig.canvas, 'get_tk_widget'):
        try:
            _tk_c = fig.canvas.get_tk_widget()
            _tk_c.focus_force()
            _tk_root = _tk_c.winfo_toplevel()
            _tk_root.title("F1 Pit-Wall Live Telemetry & Strategy Suite")
        except Exception:
            pass

    plt.show()
