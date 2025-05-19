import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Detection:
    def __init__(self, pos, R):
        self.position = pos
        self.R = R

class Track:
    def __init__(self, initial_position, track_id, initial_noise_R = np.eye(2)):
        self.kf = self.create_kalman_filter(initial_position, initial_noise_R)
        self.track_id = track_id
        self.age = 1
        self.missed = 0
    def create_kalman_filter(self, initial_position, initial_noise_R):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.R = initial_noise_R
        kf.P *= 100.0 ## initial process noise matrix
        kf.Q = np.eye(4) * 1.0  ### residual process noise 
        kf.x[:2] = np.reshape(initial_position, (2, 1))
        return kf
    def predict(self):
        self.kf.predict()
        self.age += 1
        return self.kf.x[:2].flatten()
    def update(self, measurement):
        self.kf.update(measurement)
        self.missed = 0

class MultiHypothesisTracker:
    def __init__(self, mahal_threshold=1.0, max_missed=5):
        self.tracks = []
        self.next_id = 0
        self.max_missed = max_missed
        self.mahal_threshold = mahal_threshold
    def update(self, detections : list[Detection]):
        if not self.tracks:
            for det in detections:
                self.tracks.append(Track(det.position, self.next_id, initial_noise_R=det.R))
                self.next_id += 1
            return
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        # predict each track
        for i, track in enumerate(self.tracks):
            predicted_position = track.predict()  # Call the predict() method
            # print(f"Predicted position for track {track.track_id}: {predicted_position}")
        # perform the Mahalanobis distance calculation
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                z = det.position.reshape(2, 1)
                # Kalman components
                H = track.kf.H
                x = track.kf.x
                P = track.kf.P
                R = track.kf.R
                # Innovation (residual)
                y = z - H @ x
                # Innovation covariance
                S = H @ P @ H.T + R
                S_regularized = S + np.eye(S.shape[0]) * 1e-6  # Adding a small value to the diagonal
                # S_inv = np.linalg.inv(S)
                S_inv = np.linalg.pinv(S_regularized) ##pseudo inverse
                # print("Sinv", S_inv)
                mahal_dist = float(np.sqrt((y.T @ S_inv @ y).item()))
                cost_matrix[i, j] = mahal_dist
        # perform assignment using the cost matrix
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        # update and track assigment
        assigned_tracks = set()
        assigned_detections = set()
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] < self.mahal_threshold:
                self.tracks[r].update(detections[c].position)
                assigned_tracks.add(r)
                assigned_detections.add(c)
        # handle misses
        for i, track in enumerate(self.tracks):
            if i not in assigned_tracks:
                track.missed += 1
        # handle new tracks
        for j, det in enumerate(detections):
            if j not in assigned_detections:
                print(f"new track {self.next_id}")
                self.tracks.append(Track(det.position, self.next_id, initial_noise_R=det.R))
                self.next_id += 1
        # filter bad tracks
        self.tracks = [t for t in self.tracks if t.missed < self.max_missed]
    def get_tracks(self):
        return [(track.track_id, track.kf.x[:2].flatten()) for track in self.tracks]
