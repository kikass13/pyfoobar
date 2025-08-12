import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from enum import Enum

# =====================
# Config
# =====================
EPS = 0.1
LOOKAHEAD_POINTS = 15
WAIT_TIME = 2.0
DT = 0.1

# =====================
# States
# =====================
class MotionState(Enum):
    MOVING = 1
    WAITING = 2

class DriveState(Enum):
    FORWARD = 1
    REVERSE = 2
    STOPPED = 3

# =====================
# Reversal Controller (Online, forward-only search)
# =====================
class ReversalController:
    def __init__(self, wait_time=2.0, reversal_indices=None):
        self.state = MotionState.MOVING
        self.stop_start_time = None
        self.last_velocity_sign = 1
        self.wait_time = wait_time
        self.reversal_indices = reversal_indices or []
        self.resume_idx = None  # where to jump to after waiting

    def update(self, local_traj, current_pos, sim_time, global_idx):
        """
        local_traj: list of (x, y, v)
        current_pos: (x, y)
        sim_time: current time in seconds
        global_idx: current progress along global trajectory
        """
        if not local_traj:
            return 0.0

        # Map local indices to global indices
        search_indices = list(range(global_idx, global_idx + len(local_traj)))
        dists = [np.hypot(px - current_pos[0], py - current_pos[1]) for px, py, _ in local_traj]
        nearest_local_idx = int(np.argmin(dists))
        nearest_global_idx = search_indices[nearest_local_idx]

        # Find the next reversal index ahead of us
        next_reversal_idx = None
        for r_idx in self.reversal_indices:
            if r_idx > global_idx:
                next_reversal_idx = r_idx
                break

        if self.state == MotionState.MOVING:
            # Trigger stop just before reversal
            if next_reversal_idx is not None and nearest_global_idx >= next_reversal_idx - 1:
                self.state = MotionState.WAITING
                self.stop_start_time = sim_time
                self.resume_idx = next_reversal_idx
                return 0.0

            cmd_vel = local_traj[nearest_local_idx][2]
            if abs(cmd_vel) > EPS:
                self.last_velocity_sign = np.sign(cmd_vel)
            return cmd_vel

        elif self.state == MotionState.WAITING:
            # Stay stopped until wait time is over
            if sim_time - self.stop_start_time >= self.wait_time:
                self.state = MotionState.MOVING
                # Immediately resume from reversal index
                resume_local_idx = max(0, self.resume_idx - global_idx)
                resume_local_idx = min(resume_local_idx, len(local_traj)-1)
                cmd_vel = local_traj[resume_local_idx][2]
                if abs(cmd_vel) > EPS:
                    self.last_velocity_sign = np.sign(cmd_vel)
                return cmd_vel
            return 0.0

    def drive_state(self, cmd_vel):
        if abs(cmd_vel) < EPS:
            return DriveState.STOPPED
        elif cmd_vel > 0:
            return DriveState.FORWARD
        else:
            return DriveState.REVERSE


# =====================
# Simulated global trajectory
# =====================
forward_points = [(x, 0, 1.0) for x in np.linspace(0, 5, 25)]
reverse_points = [(x, 0, -1.0) for x in np.linspace(5, 0, 25)]
global_trajectory = forward_points + reverse_points + forward_points

# Precompute reversal indices
reversal_indices = [
    i for i in range(1, len(global_trajectory))
    if np.sign(global_trajectory[i][2]) != np.sign(global_trajectory[i-1][2])
]

# =====================
# Simulation variables
# =====================
robot_pos = [global_trajectory[0][0], global_trajectory[0][1]]
global_idx = 0
sim_time = 0.0
controller = ReversalController(wait_time=WAIT_TIME, reversal_indices=reversal_indices)
times, cmd_vels, drive_states = [], [], []

# Plot setup
fig, (ax_path, ax_vel) = plt.subplots(1, 2, figsize=(12, 5))

# Path plot
ax_path.plot([x for x, _, _ in global_trajectory], [y for _, y, _ in global_trajectory],
             '--', color='gray', label="Planned Path")
robot_dot, = ax_path.plot([], [], 'o', markersize=10, label="Robot")
ax_path.set_xlim(-1, 6)
ax_path.set_ylim(-2, 2)
ax_path.legend()
ax_path.set_title("Robot Path")
ax_path.grid(True)

# Velocity plot
total_time_est = len(global_trajectory) * DT + WAIT_TIME * 2
planned_times = [i*DT for i in range(len(global_trajectory))]
ax_vel.plot(planned_times, [v for _, _, v in global_trajectory], '--', color='gray', label="Planned Vel")
vel_line_cmd, = ax_vel.plot([], [], color='blue', label="Cmd Vel (live)")
ax_vel.set_ylim(-1.5, 1.5)
ax_vel.set_xlim(0, total_time_est)
ax_vel.set_title("Velocity Profile")
ax_vel.legend()
ax_vel.grid(True)

color_map = {
    DriveState.FORWARD: "green",
    DriveState.REVERSE: "red",
    DriveState.STOPPED: "yellow"
}
bg_regions = []
current_bg_color = None

# =====================
# Animation
# =====================
def animate(frame):
    global global_idx, sim_time, robot_pos, current_bg_color

    # Feed local lookahead trajectory to controller
    local_traj = global_trajectory[global_idx:global_idx + LOOKAHEAD_POINTS]
    cmd_vel = controller.update(local_traj, robot_pos, sim_time, global_idx)
    drive_state = controller.drive_state(cmd_vel)

    # Movement logic
    if abs(cmd_vel) > EPS and global_idx < len(global_trajectory)-1:
        global_idx += 1
        robot_pos = [global_trajectory[global_idx][0], global_trajectory[global_idx][1]]

    # Log data
    times.append(sim_time)
    cmd_vels.append(cmd_vel)
    drive_states.append(drive_state)

    # Path plot update
    robot_dot.set_data(robot_pos[0], robot_pos[1])
    robot_dot.set_color(color_map[drive_state])

    # Velocity plot background
    if current_bg_color != color_map[drive_state]:
        bg_regions.append(ax_vel.axvspan(sim_time, sim_time, color=color_map[drive_state], alpha=0.15))
        current_bg_color = color_map[drive_state]
    else:
        last_region = bg_regions[-1]
        last_region.set_xy([
            [last_region.xy[0][0], -1.5],
            [last_region.xy[1][0], 1.5],
            [sim_time, 1.5],
            [sim_time, -1.5]
        ])

    # Velocity line
    vel_line_cmd.set_data(times, cmd_vels)

    sim_time += DT
    return robot_dot, vel_line_cmd, *bg_regions

ani = animation.FuncAnimation(fig, animate, frames=300, interval=100, blit=True, repeat=False)
plt.tight_layout()
plt.show()
