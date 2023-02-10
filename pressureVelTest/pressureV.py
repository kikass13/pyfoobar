import numpy as np
import math
import matplotlib.pyplot as plt

import Pid as pid

Kp = 7500.0
Ki = 100.0
Kd = 0.0
dt = 0.01

def emulatePressureForPosition(pos):
    return pos / 100.0

def main():
    T = 1.1
    positions = []
    times = []
    pressures = []
    targetpressures = []
    velocities = []
    positionState = 0
    time = 0.0
    target_pressure = 100.0

    pidParams = pid.Parameter(Kp=Kp, Ki=Ki, Kd=Kd, Ts=0.05, limitMin=-100000.0, limitMax=100000.0)
    controller = pid.Pid(pidParams)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
    ### maximise
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize()) 

    while T >= time:
        currentPressure = emulatePressureForPosition(positionState) 
        
        if time > 0.3 and time < 0.7:
            target_pressure = 100.0 + 20*math.sin(100*time)
        elif time > 0.8:
            target_pressure = 10.0

        targetpressures.append(target_pressure)
        times.append(time)
        positions.append(positionState)
        pressures.append(currentPressure)

        error = target_pressure - currentPressure
        v = controller(error, dt)
        
        velocities.append(v)
        positionState = positionState + v * dt
        time += dt

        # fig.clear()
        ax1.cla()
        ax2.cla()
        ax3.cla()
        # ax1.axhline(y = target_pressure, color = 'r', linestyle = '-')
        ax1.plot(times, targetpressures)
        ax1.plot(times, pressures)
        ax2.plot(times, velocities)
        ax3.plot(times, positions)
        # fig.canvas.draw()
        # plt.plot(times, pressures, "-b", label="position")
        # plt.plot(times, positions, "-b", label="position")
        # plt.grid(True)
        plt.pause(0.001)
    
if __name__ == '__main__':
    main()
