import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import tracker_simple as tracker

# --- Simulation Setup ---
np.random.seed(42)

# Simulate 2 moving objects
true_trajectories = [
    lambda t: np.array([100 + t * 2, 100 + t * 1.5]),
    lambda t: np.array([400 - t * 2, 300 - t * 1.0])
]

def generate_detections(t, noise_Factor=1.0):
    detections = []
    R = []
    for traj in true_trajectories:
        pos = traj(t)
        noise = np.random.randn(2) * noise_Factor
        R = np.diag([noise_Factor**2] * 2) ### [[noise_x², 0.0], [0.0, noise_y² ]]
        detection = tracker.Detection(pos + noise, R)
        detections.append(detection)
    return detections

# --- Visualization ---
fig, ax = plt.subplots()
ax.set_xlim(0, 600)
ax.set_ylim(0, 400)
ax.set_title("Multi-Hypothesis Kalman Tracker (Mahalanobis Distance)")

scatter_trks = ax.scatter([], [], color='blue', marker='s', label='Tracks')
scatter_dets = ax.scatter([], [], color='red', marker='x', label='Detections')
text_labels = []

mht = tracker.MultiHypothesisTracker(mahal_threshold=3.0)


def init():
    scatter_dets.set_offsets(np.empty((0, 2)))
    scatter_trks.set_offsets(np.empty((0, 2)))
    return scatter_dets, scatter_trks

def update(frame):
    global text_labels
    for label in text_labels:
        label.remove()
    text_labels = []

    detections = generate_detections(frame, noise_Factor=3.0)
    mht.update(detections)
    tracks = mht.get_tracks()

    detection_positions = [det.position for det in detections]
    scatter_dets.set_offsets(detection_positions)
    if tracks:
        track_positions = np.array([pos for _, pos in tracks])
        scatter_trks.set_offsets(track_positions)
    for tid, pos in tracks:
        txt = ax.text(pos[0]+5, pos[1]+5, f"ID:{tid}", color='blue', fontsize=8)
        text_labels.append(txt)

    return scatter_dets, scatter_trks, *text_labels

ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=False, interval=300)
plt.legend()
plt.show()
