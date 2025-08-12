import time
import matplotlib.pyplot as plt
from enum import Enum

class MotionState(Enum):
    MOVING = 1
    STOPPING_FOR_REVERSAL = 2
    WAITING_FOR_REVERSAL = 3

class ReversalController:
    def __init__(self, stop_wait_time=2.0):
        self.state = MotionState.MOVING
        self.current_direction = None  # +1 forward, -1 reverse
        self.stop_start_time = None
        self.stop_wait_time = stop_wait_time

    def update(self, current_speed, planned_trajectory, sim_time):
        # Determine current planned direction
        if not planned_trajectory:
            return 0.0

        planned_velocity_now = planned_trajectory[0][2]
        planned_dir = 1 if planned_velocity_now > 0 else (-1 if planned_velocity_now < 0 else 0)

        if self.current_direction is None:
            self.current_direction = planned_dir

        # Detect reversal ahead
        reversal_needed = self._detect_reversal_ahead(planned_trajectory)

        if self.state == MotionState.MOVING:
            if reversal_needed:
                self.state = MotionState.STOPPING_FOR_REVERSAL
                return 0.0
            else:
                return planned_velocity_now

        if self.state == MotionState.STOPPING_FOR_REVERSAL:
            if abs(current_speed) > 0.01:
                return 0.0  # keep stopping
            else:
                self.stop_start_time = sim_time
                self.state = MotionState.WAITING_FOR_REVERSAL
                return 0.0

        if self.state == MotionState.WAITING_FOR_REVERSAL:
            if sim_time - self.stop_start_time >= self.stop_wait_time:
                self.current_direction *= -1
                self.state = MotionState.MOVING
                return planned_velocity_now
            else:
                return 0.0

        return 0.0

    def _detect_reversal_ahead(self, traj):
        for _, _, v in traj[:5]:  # short lookahead
            if self.current_direction is None:
                continue
            dir_now = 1 if v > 0 else (-1 if v < 0 else 0)
            if dir_now != 0 and dir_now != self.current_direction:
                return True
        return False

# --- Simulation setup ---
controller = ReversalController(stop_wait_time=2.0)

# Create synthetic trajectory: forward 10s, reverse afterwards
planned_velocities = [1.0] * 10 + [-1.0] * 10
planned_trajectory_template = [(0, 0, v) for v in planned_velocities]

sim_dt = 0.5
sim_time = 0.0
actual_speed = 0.0

times, planned_v_list, cmd_v_list, states = [], [], [], []

for i in range(len(planned_velocities)):
    # Build current local trajectory (slice from i to end)
    traj = planned_trajectory_template[i:]
    planned_v = planned_velocities[i]

    cmd_v = controller.update(actual_speed, traj, sim_time)

    # Simulate perfect tracking for visualization
    actual_speed = cmd_v

    times.append(sim_time)
    planned_v_list.append(planned_v)
    cmd_v_list.append(cmd_v)
    states.append(controller.state.name)

    sim_time += sim_dt

# --- Visualization ---
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(times, planned_v_list, label="Planned Velocity", linestyle="--", color="gray")
ax1.plot(times, cmd_v_list, label="Commanded Velocity", color="blue")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Velocity (m/s)")
ax1.legend()
ax1.grid(True)

# Overlay states as colored background
unique_states = list(set(states))
colors = {
    "MOVING": "green",
    "STOPPING_FOR_REVERSAL": "orange",
    "WAITING_FOR_REVERSAL": "red"
}

for i in range(len(times) - 1):
    ax1.axvspan(times[i], times[i+1], color=colors[states[i]], alpha=0.1)

plt.title("Direction Reversal Handling Simulation")
plt.show()
