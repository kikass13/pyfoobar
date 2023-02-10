import numpy as np
import math
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")
        self.n = F.shape[1]
        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)).flatten() if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = np.subtract(z, np.dot(self.H, self.x))
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = np.add(self.x, np.dot(K, y))
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
            (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():
    inputDt = 0.05
    outputDt = 0.01
    sq = outputDt ** 2
    ## state vector, constant acceleraton model 
    ### x, vx, ax
    # F = np.array([[1, outputDt, 0.5*sq], [0, 1, outputDt], [0, 0, 1]])
    ### do not apply acceleration onto position .. for reasons
    F = np.array([[1, outputDt, 0.0], [0, 1, outputDt], [0, 0, 1]])
    H = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    Q = np.array([[0.00001, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 10.0]])
    R = np.array([[10.0, 0.0, 0.0], [0.0, 10000.0, 0.0], [0.0, 0.0, 10000.0]])

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

    t = 0
    measurements = []
    measurementTimes = []
    lastMeasurement = None
    states = []
    stateTimes = []
    lastState = None
    stepSample = 0.01
    while t < 10.0:
        if lastMeasurement == None or t - lastMeasurement >= inputDt:
            lastMeasurement = t
            ### new input measurement
            z = math .pi * math.sin(t*5.0)
            noise = np.random.normal(0,0.5,1)[0]
            z = z + noise
            measurements.append(z)
            measurementTimes.append(t)
            zM = np.array([z, 0, 0]) ### x,vx, ax
            kf.update(zM)
        if lastState == None or t - lastState > outputDt:
            lastState = t
            s = kf.predict()
            print(s)
            print("===")
            states.append(s[0])
            stateTimes.append(t)
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        # for stamp, val in zip(stateTimes, states):
        #     if val >= 0.0:
        #         plt.axvline(x=stamp,  ymin = 0.0 , ymax = val)
        #     if val < 0.0:
        #         plt.axvline(x=stamp,  ymin = val , ymax = 0)
        plt.step(measurementTimes, measurements, "-r", label="course")
        plt.step(stateTimes, states, "-b", label="course")

        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

        t+=0.01


if __name__ == '__main__':
    example()