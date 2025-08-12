import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from enum import Enum

EPS = 0.1
LOOKAHEAD_POINTS = 15
WAIT_TIME = 2.0
DT = 0.1
VEL_LAG = 0.3  # simulated PID lag
V_MAX = 1.5
REVERSAL_DISTANCE_THRESHOLD = 0.1  # m

class MotionState(Enum):
    MOVING = 1
    STOP_PENDING = 2
    WAITING = 3

class DriveState(Enum):
    FORWARD = 1
    REVERSE = 2
    STOPPED = 3


class ReversalController:
    def __init__(self, wait_time=2.0, reversal_indices=None, wait_points=10):
        self.state = MotionState.MOVING
        self.wait_time = wait_time
        self.wait_points = wait_points
        self.stop_start_time = None
        self.resume_idx = None
        self.reversal_indices = reversal_indices or []

    def update(self, global_traj, global_idx, current_pos, measured_vel, sim_time):
        # Find the next reversal index ahead of current position
        next_reversal_idx = None
        for r_idx in self.reversal_indices:
            if r_idx > global_idx:
                next_reversal_idx = r_idx
                break

        max_idx = len(global_traj) - 1

        def points_within_distance(global_traj, start_idx, dist):
            total_dist = 0.0
            idx = start_idx
            while idx < len(global_traj) - 1 and total_dist < dist:
                x0, y0, _ = global_traj[idx]
                x1, y1, _ = global_traj[idx + 1]
                total_dist += np.hypot(x1 - x0, y1 - y0)
                idx += 1
            return idx + 1  # include last point
        def close_enough_to_reversal():
            if next_reversal_idx is None:
                return False
            rev_x, rev_y, _ = global_traj[next_reversal_idx]
            dist = np.hypot(current_pos[0] - rev_x, current_pos[1] - rev_y)
            return dist < REVERSAL_DISTANCE_THRESHOLD

        local_traj = []

        if self.state == MotionState.MOVING:
            # Check if we are close enough to reversal and need to stop
            if next_reversal_idx is not None and close_enough_to_reversal():
                self.state = MotionState.STOP_PENDING
                self.resume_idx = next_reversal_idx
                segment_end = min(next_reversal_idx + 1, max_idx)
                local_traj = [(x, y, 0.0) for (x, y, v) in global_traj[global_idx:segment_end]]
                # Repeat points if too short
                if len(local_traj) < self.wait_points:
                    last_point = local_traj[-1]
                    repeats = self.wait_points - len(local_traj)
                    local_traj.extend([last_point] * repeats)
            else:
                end_idx = min(global_idx + LOOKAHEAD_POINTS, next_reversal_idx or max_idx, max_idx)
                local_traj = global_traj[global_idx:end_idx]

        elif self.state == MotionState.STOP_PENDING:
            if abs(measured_vel) < EPS and close_enough_to_reversal():
                self.state = MotionState.WAITING
                self.stop_start_time = sim_time
                local_traj = [(current_pos[0], current_pos[1], 0.0)] * self.wait_points
            else:
                segment_end = min(self.resume_idx + 1, max_idx)
                local_traj = [(x, y, 0.0) for (x, y, v) in global_traj[global_idx:segment_end]]
                # Repeat if too short
                if len(local_traj) < self.wait_points:
                    last_point = local_traj[-1]
                    repeats = self.wait_points - len(local_traj)
                    local_traj.extend([last_point] * repeats)

        elif self.state == MotionState.WAITING:
            if abs(measured_vel) < EPS and (sim_time - self.stop_start_time) >= self.wait_time:
                self.state = MotionState.MOVING
                resume_idx = self.resume_idx
                end_idx = min(resume_idx + LOOKAHEAD_POINTS, max_idx)
                local_traj = global_traj[resume_idx:end_idx]
            else:
                local_traj = [(current_pos[0], current_pos[1], 0.0)] * self.wait_points

        return local_traj

    def drive_state(self, cmd_vel):
        if abs(cmd_vel) < EPS:
            return DriveState.STOPPED
        elif cmd_vel > 0:
            return DriveState.FORWARD
        else:
            return DriveState.REVERSE


# ====== Setup ======
forward_points = [(x, 0, V_MAX) for x in np.linspace(0, 5, 25)]
reverse_points = [(x, 0, -V_MAX) for x in np.linspace(5, 0, 25)]
global_trajectory = forward_points + reverse_points + forward_points

reversal_indices = [
    i for i in range(1, len(global_trajectory))
    if np.sign(global_trajectory[i][2]) != np.sign(global_trajectory[i-1][2])
]

robot_pos = [global_trajectory[0][0], global_trajectory[0][1]]
global_idx = 0
sim_time = 0.0
measured_vel = 0.0
controller = ReversalController(wait_time=WAIT_TIME, reversal_indices=reversal_indices)

times, cmd_vels, measured_vels, drive_states = [], [], [], []

fig, (ax_path, ax_vel, ax_local) = plt.subplots(1, 3, figsize=(18, 5))

# Path plot
ax_path.plot([x for x, _, _ in global_trajectory], [y for _, y, _ in global_trajectory], '--', color='gray', label="Global Path")
robot_dot, = ax_path.plot([], [], 'o', markersize=10, label="Robot")
global_segment_line, = ax_path.plot([], [], 'b-', linewidth=4, alpha=0.5, label="Global Lookahead")
local_traj_line, = ax_path.plot([], [], 'r--', linewidth=3, alpha=0.8, label="Local Trajectory")
ax_path.set_xlim(-1, 6)
ax_path.set_ylim(-2, 2)
ax_path.legend()
ax_path.set_title("Robot Path")
ax_path.grid(True)

# Velocity plot
total_time_est = len(global_trajectory) * DT + WAIT_TIME * 5
planned_times = [i*DT for i in range(len(global_trajectory))]
ax_vel.plot(planned_times, [v for _, _, v in global_trajectory], '--', color='gray', label="Global Planned Vel")
vel_line_cmd, = ax_vel.plot([], [], 'b-', linewidth=4, label="Cmd Vel (first point)")
vel_line_meas, = ax_vel.plot([], [], 'r--', linewidth=3, label="Measured Vel")
ax_vel.set_ylim(-V_MAX*1.5, V_MAX*1.5)
ax_vel.set_xlim(0, total_time_est)
ax_vel.set_title("Velocity Profile")
ax_vel.legend()
ax_vel.grid(True)

# Local velocity profile plot (live only)
local_global_line, = ax_local.plot([], [], 'b-', linewidth=4, label="Global lookahead vel")
local_local_line, = ax_local.plot([], [], 'r--', linewidth=3, label="Local traj vel")
ax_local.set_ylim(-V_MAX*1.5, V_MAX*1.5)
ax_local.set_xlim(0, LOOKAHEAD_POINTS*DT)
ax_local.set_title("Current Step Velocity Profile")
ax_local.legend()
ax_local.grid(True)

color_map = {
    DriveState.FORWARD: "green",
    DriveState.REVERSE: "red",
    DriveState.STOPPED: "yellow"
}
state_colors = {
    MotionState.MOVING: '#d0f0c0',    # light green
    MotionState.STOP_PENDING: '#fce5cd',  # light orange
    MotionState.WAITING: '#f4cccc'     # light red
}


def animate(frame):
    global global_idx, sim_time, robot_pos, measured_vel

    local_traj = controller.update(global_trajectory, global_idx, robot_pos, measured_vel, sim_time)
    if not local_traj:
        return []

    cmd_vel = local_traj[0][2]
    drive_state = controller.drive_state(cmd_vel)

    measured_vel += (cmd_vel - measured_vel) * VEL_LAG

    # Move if in MOVING state
    if controller.state == MotionState.MOVING:
        if abs(cmd_vel) > EPS and global_idx < len(global_trajectory)-1:
            global_idx += 1
            robot_pos = [global_trajectory[global_idx][0], global_trajectory[global_idx][1]]
    elif controller.state == MotionState.WAITING:
        if abs(measured_vel) < EPS and (sim_time - controller.stop_start_time) >= WAIT_TIME:
            global_idx = controller.resume_idx
            robot_pos = [global_trajectory[global_idx][0], global_trajectory[global_idx][1]]

    times.append(sim_time)
    cmd_vels.append(cmd_vel)
    measured_vels.append(measured_vel)
    drive_states.append(drive_state)

    ### state color
    color = state_colors[controller.state]
    ax_path.set_facecolor(color)
    ax_vel.set_facecolor(color)
    ax_local.set_facecolor(color)

    # Path update
    robot_dot.set_data(robot_pos[0], robot_pos[1])
    robot_dot.set_color(color_map[drive_state])
    lookahead_end = min(global_idx + LOOKAHEAD_POINTS, len(global_trajectory)-1)
    gx = [pt[0] for pt in global_trajectory[global_idx:lookahead_end]]
    gy = [pt[1] for pt in global_trajectory[global_idx:lookahead_end]]
    global_segment_line.set_data(gx, gy)
    lx = [pt[0] for pt in local_traj]
    ly = [pt[1] for pt in local_traj]
    local_traj_line.set_data(lx, ly)

    # Velocity time history
    vel_line_cmd.set_data(times, cmd_vels)
    vel_line_meas.set_data(times, measured_vels)

    # Live local velocity profile vs global lookahead
    local_times = np.arange(len(local_traj)) * DT
    local_vels = [pt[2] for pt in local_traj]
    global_vels = [pt[2] for pt in global_trajectory[global_idx:lookahead_end]]

    local_global_line.set_data(np.arange(len(global_vels)) * DT, global_vels)
    local_local_line.set_data(local_times, local_vels)

    sim_time += DT
    return [robot_dot, global_segment_line, local_traj_line,
            vel_line_cmd, vel_line_meas,
            local_global_line, local_local_line]


ani = animation.FuncAnimation(fig, animate, frames=400, interval=DT*1000, blit=False, repeat=False)
plt.tight_layout()
plt.show()