import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from enum import Enum


class MotionState(Enum):
    MOVING = 1
    WAITING = 2

class DriveState(Enum):
    FORWARD = 1
    REVERSE = 2
    STOPPED = 3

class ReversalController:
    def __init__(self, trajectory, wait_time=2.0):
        self.trajectory = trajectory
        self.reversal_indices = self.find_reversal_points(trajectory)
        self.wait_time = wait_time
        self.state = MotionState.MOVING
        self.stop_start_time = None
        self.current_reversal_idx = None
        self.last_velocity_sign = 1  # assume forward at start
    def find_reversal_points(self, trajectory):
        reversal_indices = []
        prev_dir = 0
        for i, (_, _, v) in enumerate(trajectory):
            dir_now = 1 if v > 0 else (-1 if v < 0 else 0)
            if prev_dir != 0 and dir_now != 0 and dir_now != prev_dir:
                reversal_indices.append(i)
            if dir_now != 0:
                prev_dir = dir_now
        return reversal_indices
    def update(self, idx, sim_time):
        # Pick next reversal if needed
        if self.current_reversal_idx is None and self.reversal_indices:
            for rev_idx in self.reversal_indices:
                if rev_idx > idx:
                    self.current_reversal_idx = rev_idx
                    break

        if self.state == MotionState.MOVING:
            if self.current_reversal_idx is not None and idx >= self.current_reversal_idx:
                self.state = MotionState.WAITING
                self.stop_start_time = sim_time
                return 0.0
            vel = self.trajectory[idx][2]
            if vel != 0:
                self.last_velocity_sign = np.sign(vel)
            return vel

        elif self.state == MotionState.WAITING:
            if sim_time - self.stop_start_time >= self.wait_time:
                self.state = MotionState.MOVING
                if self.current_reversal_idx in self.reversal_indices:
                    self.reversal_indices.remove(self.current_reversal_idx)
                self.current_reversal_idx = None
                vel = self.trajectory[idx][2]
                if vel != 0:
                    self.last_velocity_sign = np.sign(vel)
                return vel
            return 0.0

    def drive_state(self, cmd_vel, eps=0.1):
        if abs(cmd_vel) < eps:
            return DriveState.STOPPED
        elif cmd_vel > 0:
            return DriveState.FORWARD
        else:
            return DriveState.REVERSE

# --------------------------
# Create fixed trajectory
# --------------------------
forward_points = [(x, 0, 1.0) for x in np.linspace(0, 5, 25)]
reverse_points = [(x, 0, -1.0) for x in np.linspace(5, 0, 25)]
trajectory = forward_points + reverse_points + forward_points + reverse_points

# --------------------------
# Simulation vars
# --------------------------
robot_pos = [trajectory[0][0], trajectory[0][1]]
idx = 0
sim_time = 0.0
dt = 0.1
controller = ReversalController(trajectory, wait_time=1.0)
times, cmd_vels, drive_states = [], [], []  # store drive state history

# --------------------------
# Figure setup
# --------------------------
fig, (ax_path, ax_vel) = plt.subplots(1, 2, figsize=(12, 5))

# Path plot
ax_path.plot([x for x, _, _ in trajectory], [y for _, y, _ in trajectory],
             '--', color='gray', label="Planned Path")
rev_points = controller.find_reversal_points(trajectory)
for rp in rev_points:
    ax_path.plot(trajectory[rp][0], trajectory[rp][1], 'ro', label="Reversal" if rp == rev_points[0] else "")
robot_dot, = ax_path.plot([], [], 'o', markersize=10, label="Robot")
ax_path.set_xlim(-1, 6)
ax_path.set_ylim(-2, 2)
ax_path.set_title("Robot Path")
ax_path.legend()
ax_path.grid(True)

# Velocity plot
total_time_est = len(trajectory) * dt + controller.wait_time * len(rev_points)
planned_times = [i*dt for i in range(len(trajectory))]
planned_vels_full = [v for _, _, v in trajectory]
ax_vel.plot(planned_times, planned_vels_full, '--', color='gray', label="Planned Vel")
vel_line_cmd, = ax_vel.plot([], [], color='blue', label="Cmd Vel (live)")
ax_vel.set_ylim(-1.5, 1.5)
ax_vel.set_xlim(0, total_time_est)
ax_vel.set_title("Velocity Profile")
ax_vel.legend()
ax_vel.grid(True)

# State-colored background regions
bg_regions = []
current_bg_color = None
color_map = {
    DriveState.FORWARD: "green",
    DriveState.REVERSE: "red",
    DriveState.STOPPED: "yellow"
}

# --------------------------
# Animation update
# --------------------------
def animate(frame):
    global idx, sim_time, robot_pos, current_bg_color

    cmd_vel = controller.update(idx, sim_time)
    drive_state = controller.drive_state(cmd_vel)

    # Move robot only if moving
    if cmd_vel != 0.0:
        idx = min(idx + 1, len(trajectory)-1)
        robot_pos = [trajectory[idx][0], trajectory[idx][1]]

    # Logging
    times.append(sim_time)
    cmd_vels.append(cmd_vel)
    drive_states.append(drive_state)

    # Robot dot color
    robot_dot.set_color(color_map[drive_state])

    # Background coloring for velocity plot
    if current_bg_color != color_map[drive_state]:
        # Start a new colored span
        bg_regions.append(ax_vel.axvspan(sim_time, sim_time, color=color_map[drive_state], alpha=0.15))
        current_bg_color = color_map[drive_state]
    else:
        # Extend last region
        last_region = bg_regions[-1]
        last_region.set_xy([
            [last_region.xy[0][0], -1.5],
            [last_region.xy[1][0], 1.5],
            [sim_time, 1.5],
            [sim_time, -1.5]
        ])

    # Update plots
    robot_dot.set_data(robot_pos[0], robot_pos[1])
    vel_line_cmd.set_data(times, cmd_vels)

    sim_time += dt
    return robot_dot, vel_line_cmd, *bg_regions

ani = animation.FuncAnimation(fig, animate, frames=300, interval=100, blit=True, repeat=False)
plt.tight_layout()
plt.show()
