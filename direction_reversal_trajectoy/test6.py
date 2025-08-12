import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from enum import Enum

EPS = 0.1
LOOKAHEAD_DIST = 5.0  # m
WAIT_TIME = 2.0
DT = 0.1
VEL_LAG = 0.3  # simulated PID lag
V_MAX = 1.5
REVERSAL_DISTANCE_THRESHOLD = 0.3  # m

class MotionState(Enum):
    MOVING = 1
    STOP_PENDING = 2
    WAITING = 3

class DriveState(Enum):
    FORWARD = 1
    REVERSE = 2
    STOPPED = 3

############################################################
class ReversalController:
    def __init__(self, wait_time=2.0, wait_points=25):
        self.state = MotionState.MOVING
        self.wait_time = wait_time
        self.wait_points = wait_points
        self.stop_start_time = None
        self.resume_idx = None

    def close_enough_to_reversal(self, current_pos, next_reversal_idx, global_traj):
        if next_reversal_idx is None:
            return False
        rev_x, rev_y, _ = global_traj[next_reversal_idx]
        dist = np.hypot(current_pos[0] - rev_x, current_pos[1] - rev_y)
        return dist < REVERSAL_DISTANCE_THRESHOLD

    def get_lookahead_end(self, traj, start_idx, lookahead_dist, next_rev_idx, max_index):
        dist_sum = 0.0
        prev_point = traj[start_idx]
        for i in range(start_idx + 1, len(traj)):
            p = traj[i]
            step_dist = np.hypot(p[0] - prev_point[0], p[1] - prev_point[1])
            dist_sum += step_dist
            if dist_sum > lookahead_dist:
                return i
            if next_rev_idx is not None and i >= next_rev_idx:
                return next_rev_idx
            prev_point = p
        return max_index

    def compute_distances(self, traj):
        dists = [0.0]
        for i in range(1, len(traj)):
            d = np.hypot(traj[i][0] - traj[i-1][0], traj[i][1] - traj[i-1][1])
            dists.append(dists[-1] + d)
        return dists

    def find_next_reversal_index(self, global_idx, global_traj):
        next_reversal_idx = None
        for i in range(global_idx + 1, len(global_traj)):
            if np.sign(global_traj[i][2]) != np.sign(global_traj[i-1][2]):
                next_reversal_idx = i
                break
        return next_reversal_idx
    
    def update(self, global_traj, global_idx, current_pos, measured_vel, sim_time):
        next_reversal_idx = self.find_next_reversal_index(global_idx, global_traj)

        max_idx = len(global_traj) - 1
        local_traj = []

        if self.state == MotionState.MOVING:
            if next_reversal_idx is not None and self.close_enough_to_reversal(current_pos, next_reversal_idx, global_traj):
                self.state = MotionState.STOP_PENDING
                self.resume_idx = next_reversal_idx
                segment_end = min(next_reversal_idx + 1, max_idx)
                local_traj = [(x, y, 0.0) for (x, y, v) in global_traj[global_idx:segment_end]]
                if len(local_traj) < self.wait_points:
                    last_point = local_traj[-1]
                    repeats = self.wait_points - len(local_traj)
                    local_traj.extend([last_point] * repeats)
            else:
                lookahead_end = self.get_lookahead_end(global_traj, global_idx, LOOKAHEAD_DIST, next_reversal_idx, max_idx)
                local_traj = global_traj[global_idx:lookahead_end]

        elif self.state == MotionState.STOP_PENDING:
            if abs(measured_vel) < EPS:
                self.state = MotionState.WAITING
                self.stop_start_time = sim_time
                local_traj = [(current_pos[0], current_pos[1], 0.0)] * self.wait_points
            else:
                segment_end = min(self.resume_idx + 1, max_idx)
                local_traj = [(x, y, 0.0) for (x, y, v) in global_traj[global_idx:segment_end]]
                if len(local_traj) < self.wait_points:
                    if len(local_traj) == 0:  # safeguard
                        local_traj = [(current_pos[0], current_pos[1], 0.0)]
                    last_point = local_traj[-1]
                    repeats = self.wait_points - len(local_traj)
                    local_traj.extend([last_point] * repeats)

        elif self.state == MotionState.WAITING:
            if abs(measured_vel) < EPS and (sim_time - self.stop_start_time) >= self.wait_time:
                self.state = MotionState.MOVING
                ##########
                # Skip the actual reversal point when resuming
                start_idx = self.resume_idx
                # Determine intended velocity sign after reversal
                if start_idx < max_idx:
                    intended_sign = np.sign(global_traj[start_idx + 1][2])
                else:
                    intended_sign = np.sign(global_traj[start_idx][2])
                # Move start_idx forward until velocity sign matches intended_sign
                while start_idx < max_idx and np.sign(global_traj[start_idx][2]) != intended_sign:
                    start_idx += 1
                # If for some reason we reach end, try moving backward
                while start_idx > 0 and np.sign(global_traj[start_idx][2]) != intended_sign:
                    start_idx -= 1
                ##########
                lookahead_end = self.get_lookahead_end(global_traj, start_idx, LOOKAHEAD_DIST, None, max_idx)
                local_traj = global_traj[start_idx:lookahead_end]
            else:
                local_traj = [(current_pos[0], current_pos[1], 0.0)] * self.wait_points

        return local_traj, next_reversal_idx

    def drive_state(self, cmd_vel):
        if abs(cmd_vel) < EPS:
            return DriveState.STOPPED
        elif cmd_vel > 0:
            return DriveState.FORWARD
        else:
            return DriveState.REVERSE

############################################################
forward_points = [(x, 0, V_MAX) for x in np.linspace(0, 5, 25)]
reverse_points = [(x, 0, -V_MAX) for x in np.linspace(5, 0, 25)]
global_trajectory = forward_points + reverse_points + forward_points
controller = ReversalController(wait_time=WAIT_TIME)
############################################################
robot_pos = list(global_trajectory[0][:2])
global_idx = 0
sim_time = 0.0
measured_vel = 0.0
cmd_vel = 0.0

times = []
cmd_vels = []
measured_vels = []
drive_states = []

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
ax_path, ax_vel, ax_local = axs

ax_path.plot([p[0] for p in global_trajectory], [p[1] for p in global_trajectory], 'k--', label='Global Trajectory')

robot_dot, = ax_path.plot([], [], 'bo', markersize=10, label='Robot')
global_lookahead_line, = ax_path.plot([], [], 'b-', linewidth=4, alpha=0.5, label='Global Lookahead')
local_traj_line, = ax_path.plot([], [], 'r--', linewidth=3, alpha=0.8, label='Local Trajectory')

ax_path.set_xlim(-1, 6)
ax_path.set_ylim(-2, 2)
ax_path.set_title('Robot Path')
ax_path.grid(True)
ax_path.legend()

ax_vel.set_title('Velocity Profile')
ax_vel.set_xlabel('Time (s)')
ax_vel.set_ylabel('Velocity (m/s)')
ax_vel.set_xlim(0, 60)
ax_vel.set_ylim(-V_MAX * 1.5, V_MAX * 1.5)
ax_vel.grid(True)
vel_cmd_line, = ax_vel.plot([], [], 'b-', linewidth=4, label='Cmd Vel (first local point)')
vel_meas_line, = ax_vel.plot([], [], 'r--', linewidth=3, label='Measured Vel')
ax_vel.legend()

ax_local.set_title('Local Velocity Profile')
ax_local.set_xlabel('Distance along local traj (m)')
ax_local.set_ylabel('Velocity (m/s)')
ax_local.set_xlim(0, LOOKAHEAD_DIST)
ax_local.set_ylim(-V_MAX * 1.5, V_MAX * 1.5)
ax_local.grid(True)
local_vel_line, = ax_local.plot([], [], 'r-', linewidth=3, label='Local Vel Profile')
global_vel_line, = ax_local.plot([], [], 'b--', linewidth=2, label='Global Vel Profile')
ax_local.legend()

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

def advance_along_path(robot_pos, global_idx, measured_vel, trajectory, dt):
    """Advance robot position along a trajectory by measured velocity,
    using current robot_pos as partial progress along the current segment."""

    dist_to_move = abs(measured_vel) * dt

    while global_idx < len(trajectory) - 1 and dist_to_move > 0:
        p0 = trajectory[global_idx]
        p1 = trajectory[global_idx + 1]

        # Calculate remaining distance from current robot_pos to p1
        # Instead of from p0 to p1, handle partial progress
        seg_total_dist = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if seg_total_dist < 1e-6:
            # Skip zero-length segments
            global_idx += 1
            continue

        # Distance from current robot_pos to p1
        dist_to_p1 = np.hypot(p1[0] - robot_pos[0], p1[1] - robot_pos[1])

        if dist_to_move >= dist_to_p1:
            # Move completely to p1 and advance index
            robot_pos = [p1[0], p1[1]]
            global_idx += 1
            dist_to_move -= dist_to_p1
        else:
            # Move partway towards p1 from current robot_pos
            ratio = dist_to_move / dist_to_p1
            robot_pos[0] += ratio * (p1[0] - robot_pos[0])
            robot_pos[1] += ratio * (p1[1] - robot_pos[1])
            dist_to_move = 0

    return robot_pos, global_idx

def update(frame):
    global robot_pos, global_idx, sim_time, measured_vel, cmd_vel

    local_traj, next_rev_idx = controller.update(global_trajectory, global_idx, robot_pos, measured_vel, sim_time)
    
    cmd_vel = local_traj[0][2] if len(local_traj) > 0 else 0.0
    measured_vel += (cmd_vel - measured_vel) * VEL_LAG
    robot_pos, global_idx = advance_along_path(robot_pos, global_idx, measured_vel, global_trajectory, DT)
    sim_time += DT

    max_idx = len(global_trajectory) - 1
    lookahead_end = controller.get_lookahead_end(global_trajectory, global_idx, LOOKAHEAD_DIST, next_rev_idx, max_idx)
    global_lookahead = global_trajectory[global_idx:lookahead_end]

    robot_dot.set_data(robot_pos[0], robot_pos[1])
    global_lookahead_line.set_data([p[0] for p in global_lookahead], [p[1] for p in global_lookahead])
    local_traj_line.set_data([p[0] for p in local_traj], [p[1] for p in local_traj])

    times.append(sim_time)
    cmd_vels.append(cmd_vel)
    measured_vels.append(measured_vel)
    drive_states.append(controller.drive_state(cmd_vel))

    vel_cmd_line.set_data(times, cmd_vels)
    vel_meas_line.set_data(times, measured_vels)
    ax_vel.set_xlim(max(0, sim_time - 10), sim_time + 0.1)

    local_vels = [p[2] for p in local_traj]
    global_vels = [p[2] for p in global_lookahead]
    local_indices = list(range(len(local_vels)))
    global_indices = list(range(len(global_vels)))
                                
    local_vel_line.set_data(local_indices, local_vels)
    global_vel_line.set_data(global_indices, global_vels)

    if len(global_indices) > 1 and len(local_indices) > 1:
        ax_local.set_xlim(0, max(global_indices[-1], local_indices[-1]))

    ### state color
    color = state_colors[controller.state]
    ax_path.set_facecolor(color)
    ax_vel.set_facecolor(color)
    ax_local.set_facecolor(color)

    return (robot_dot, global_lookahead_line, local_traj_line,
            vel_cmd_line, vel_meas_line, local_vel_line, global_vel_line)


############################################################

ani = animation.FuncAnimation(fig, update, interval=DT * 1000, blit=False)
plt.show()