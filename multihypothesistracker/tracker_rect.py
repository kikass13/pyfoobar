import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Detection:
    def __init__(self, pos, velocity, size, yaw, R, class_id=-1):
        self.class_id = class_id
        self.position = pos  # shape (2,)
        self.velocity = velocity
        self.size = size     # shape (2,) — width, length
        self.yaw = yaw
        self.R = R           # measurement noise (4x4): for pos and size
    def getZH(self):
        z = np.concatenate((self.position, self.velocity, self.size, [self.yaw])).reshape(7, 1)
        H = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0],  # y
            [0, 0, 0, 0, 0, 0, 0],  # vx
            [0, 0, 0, 0, 0, 0, 0],  # vy
            [0, 0, 0, 0, 1, 0, 0],  # width
            [0, 0, 0, 0, 0, 1, 0],  # length
            [0, 0, 0, 0, 0, 0, 1]   # yaw
        ])
        return z, H

class Track:
    def __init__(self, initial_position, inital_velocity, initial_size, initial_yaw, track_id, initial_noise_R=np.eye(7), class_id=-1):
        self.kf = self.create_kalman_filter(initial_position, inital_velocity, initial_size, initial_yaw, initial_noise_R)
        self.track_id = track_id
        self.class_id = class_id
        self.age = 1
        self.missed = 0
    def _state_transition_matrix(self, dt):
        return np.array([[1, 0, dt, 0, 0, 0, 0],    # x
                         [0, 1, 0, dt, 0, 0, 0],    # y
                         [0, 0, 1,  0, 0, 0, 0],    # vx
                         [0, 0, 0, 1, 0, 0, 0],     # vy
                         [0, 0, 0, 0, 1, 0, 0],     # w
                         [0, 0, 0, 0, 0, 1, 0],     # l
                         [0, 0, 0, 0, 0, 0, 1]])    # yaw
    def _process_noise_matrix(self, dt):
        Q = np.array([[1, 0, 0, 0, 0, 0, 0],    # x
                      [0, 1, 0, 0, 0, 0, 0],    # y
                      [0, 0, 1,  0, 0, 0, 0],    # vx
                      [0, 0, 0, 1, 0, 0, 0],     # vy
                      [0, 0, 0, 0, 1, 0, 0],     # w
                      [0, 0, 0, 0, 0, 1, 0],     # l
                      [0, 0, 0, 0, 0, 0, 1]])    # yaw
        return Q * dt ###scale with dt
    def create_kalman_filter(self, initial_position, inital_velocity, initial_size, initial_yaw, initial_noise_R):
        kf = KalmanFilter(dim_x=7, dim_z=7)
        dt = 1.0
        # initialize transition matrix
        kf.F = self._state_transition_matrix(dt)
        # Initial noise matrices
        kf.R = initial_noise_R
        kf.P *= 100.0
        kf.Q = self._process_noise_matrix(dt)
        ### initialize state
        kf.x[:2] = np.reshape(initial_position, (2, 1))  # x, y
        kf.x[2:4] = np.reshape(inital_velocity, (2, 1))  # vx, vy
        kf.x[4] = initial_size[0]
        kf.x[5] = initial_size[1]
        kf.x[6] = initial_yaw
        return kf
    def predict(self):
        self.kf.predict()
        self.age += 1
    def update(self, detection: Detection, dt):
        ### update state and noise matrices with dt
        self.kf.F = self._state_transition_matrix(dt)
        self.kf.Q = self._process_noise_matrix(dt)
        z,H = detection.getZH()
        self.kf.update(z, H=H)
        self.missed = 0

class MultiHypothesisTracker:
    def __init__(self, mahal_threshold=4.0, max_missed=5):
        self.tracks : list[Track] = []
        self.next_id = 0
        self.max_missed = max_missed
        self.mahal_threshold = mahal_threshold
    def newTrack(self, det):
        new_track = Track(det.position, det.velocity, det.size, det.yaw,
            self.next_id, class_id=det.class_id, initial_noise_R=det.R)
        # print(f"##### NEW TRACK {self.next_id}")
        self.tracks.append(new_track)
        self.next_id += 1
    def update(self, detections: list[Detection], dt):
        if not self.tracks:
            for det in detections:
                self.newTrack(det)
            return

        # Predict all tracks
        for track in self.tracks:
            track.predict()

        # Build cost matrix (Mahalanobis distances)
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                ### if detection class is track class, or track class is None (-1) 
                if track.class_id == -1 or det.class_id == track.class_id:
                    z, H = det.getZH()
                    x = track.kf.x
                    P = track.kf.P
                    R = det.R
                    y = z - H @ x
                    S = H @ P @ H.T + R
                    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
                    S += np.eye(S.shape[0]) * 1e-6
                    S_inv = np.linalg.pinv(S)
                    dist = float(np.sqrt((y.T @ S_inv @ y).item()))
                    cost_matrix[i, j] = dist
                else:
                    cost_matrix[i, j] = 9999.0

        # Assign detections to tracks
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        assigned_tracks = set()
        assigned_detections = set()
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] < self.mahal_threshold:
                self.tracks[r].update(detections[c], dt)
                assigned_tracks.add(r)
                assigned_detections.add(c)

        # Mark unassigned tracks
        for i, track in enumerate(self.tracks):
            if i not in assigned_tracks:
                track.missed += 1

        # Create new tracks
        for j, det in enumerate(detections):
            if j not in assigned_detections:
                self.newTrack(det)

        # Filter old tracks
        self.tracks = [t for t in self.tracks if t.missed < self.max_missed]

    def get_tracks(self):
        return [(t.track_id, t.kf.x[:2].flatten(), t.kf.x[4:6].flatten(), t.kf.x[6]) for t in self.tracks]
