import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation

import tracker_rect as tracker  # or adjust to your module path

np.random.seed(42)

dt = 0.250

# Simulated moving objects with size + yaw changing
true_objects = [
    lambda t: {
        'pos': np.array([100 + t * 2, 100 + t * 1.5]),
        'size': np.array([40.0 + 0.1 * np.sin(t / 5), 20.0 + 0.2 * np.cos(t / 3)]),
        'yaw': np.radians(45.0),
        'class': 0 if t < 20 else 2,
    },
    lambda t: {
        'pos': np.array([400 - t * 2, 300 - t * 1.0]),
        'size': np.array([24.0 + 0.1 * np.cos(t / 4), 38.0 + 0.1 * np.sin(t / 6)]),
        'yaw': np.radians(-45.0)* np.cos(t / 8),
        'class': 1,
    }
]

def generate_detections(t, noise_factor=3.0):
    detections = []
    for obj_fn in true_objects:
        obj = obj_fn(t)
        pos = obj['pos']
        size = obj['size']
        yaw = obj['yaw']
        class_id = obj['class']
        noisy_pos = pos + np.random.randn(2) * noise_factor
        noisy_size = size + np.random.randn(2) * 1.0
        noisy_velocity = np.zeros(2)
        noisy_yaw = yaw + np.random.randn() * 0.2  # add yaw noise
        R = np.diag([
            noise_factor**2, #x
            noise_factor**2, #y
            1000.0, # vx
            1000.0, # vy
            1.0**2, # width
            1.0**2, # length
            0.2**2, # yaw
        ])
        detection = tracker.Detection(noisy_pos, noisy_velocity, noisy_size, noisy_yaw, R, class_id=class_id)
        detections.append(detection)
    return detections

# --- Visualization ---
fig, ax = plt.subplots()
ax.set_xlim(0, 600)
ax.set_ylim(0, 400)
ax.set_title("Kalman Tracker with Yaw (Rotated Rectangles)")
ax.set_aspect('equal')

scatter_dets = ax.scatter([], [], color='red', marker='x', label='Detections')
text_labels = []
track_boxes = []

mht = tracker.MultiHypothesisTracker(mahal_threshold=5.0)

def init():
    scatter_dets.set_offsets(np.empty((0, 2)))
    return scatter_dets

def update(frame):
    global text_labels, track_boxes

    # Clear previous rectangles and text
    for rect in track_boxes:
        rect.remove()
    track_boxes = []

    for label in text_labels:
        label.remove()
    text_labels = []

    detections = generate_detections(frame, noise_factor=3.0)
    mht.update(detections, dt)
    tracks = mht.get_tracks()

    # Update detections
    detection_positions = [det.position for det in detections]
    scatter_dets.set_offsets(detection_positions)

    # Draw tracks as rotated rectangles
    for tid, pos, size, yaw in tracks:
        cx, cy = pos
        w, l = size

        # Draw rotated rectangle (yaw in radians)
        rect = Rectangle((-w/2, -l/2), w, l, linewidth=2, edgecolor='blue', facecolor='none')
        trans = Affine2D().rotate_around(0, 0, yaw).translate(cx, cy) + ax.transData
        rect.set_transform(trans)
        ax.add_patch(rect)
        track_boxes.append(rect)

        txt = ax.text(cx + 5, cy + 5, f"ID:{tid}", color='blue', fontsize=8)
        txtclass = ax.text(cx - 15, cy - 15, f"class:{tid}", color='red', fontsize=8)
        text_labels.append(txt)
        text_labels.append(txtclass)

    return scatter_dets, *track_boxes, *text_labels

ani = FuncAnimation(fig, update, frames=1000, init_func=init, blit=False, interval=int(1000*dt))
plt.legend()
plt.show()