

import numpy as np
import math
import os
import time
import random
import matplotlib.pyplot as plt

import pid.Pid as pid

# Parameters
Lfc = 2.0  # [m] minimum look-ahead distance
Lfs = 1.0  # [s] look-ahead time
k = 1.0 # velocity lookahead distance gain 
### for speed control
# Kp = 1.0  # proportional gain
# Ki = 0.01  # integral gain
# Kd = 0.0001  # differential gain
### for distance control
Kp = 29.0  # proportional gain
Ki = 0.0  # integral gain
Kd = 9.0  # differential gain
dt = 0.1  # [s] time tick
vehicle_WB = 1.5  # [m] wheel base of vehicle
leader_WB = 0.5  # [m] wheel base of leader
vehicle_max_curvature = 1.0 ### [1/m] curvature of vehicle

show_animation = True


class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, wb=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.wb = wb
        self.rear_x = self.x - ((self.wb / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.wb / 2) * math.sin(self.yaw))
    def update(self, a, delta, dt):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        ### safetey for ~pi/2
        # self.yaw += self.v / WB * math.tan(delta) * dt
        r = 1.0
        if math.fabs(delta) < math.pi/2.0-0.15:
            r = math.tan(delta)
        self.yaw += self.v / self.wb * r * dt
        self.v += a * dt
        ### we cannot reverse
        if self.v < 0.0:
            self.v = 0.0
        self.rear_x = self.x - ((self.wb / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.wb / 2) * math.sin(self.yaw))
    def calc_distances(self, otherState):
        dx = self.rear_x - otherState.x
        dy = self.rear_y - otherState.y
        return dx, dy, math.hypot(dx, dy)
    def __repr__(self) -> str:
        return "State[%s, %s, yaw=%s, v=%s]" % (self.x, self.y, self.yaw, self.v)

class Obstacle :
    def __init__(self, center, size):
        self.center = center
        self.size = size
        self.rect = self.calcRect()
    def calcRect(self):
        self.l = self.center.x - self.size/2.0 #left
        self.r = self.center.x + self.size/2.0 #right
        self.u = self.center.y + self.size/2.0 #up
        self.b = self.center.y - self.size/2.0 #bottom
        return [
            Point(self.r, self.u),
            Point(self.r, self.b),
            Point(self.l, self.b),
            Point(self.l, self.u),
            ### extra point for drawing
            Point(self.r, self.u),
        ]
    def contains(self, p):
        for edge in self.rect:
            if p.x > self.l and p.x < self.r and p.y > self.b and p.y < self.u:
                return True
        return False
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def fromRPhi(r, phi):
        x = math.cos(phi) * r
        y = math.sin(phi) * r
        return Point(x,y)
    def np(self):
        return np.array((self.x, self.y, 0))
    def __add__(self, o):
        return Point(self.x + o.x, self.y + o.y)
    def __sub__(self, o):
        return Point(self.x - o.x, self.y - o.y)
    def __repr__(self) -> str:
        return "P[%s,%s]" % (self.x, self.y)

class Path:
    def __init__(self, start, goal):
        self.goal = goal
        self.start = start
        self.waypoints = []
        self.calculateWaypoints()
    def calculateWaypoints(self):
        pass
    def findNearest(self, target):
        distances = [np.linalg.norm(p.np()-target.np()) for p in self.waypoints]
        index = distances.index(min(distances))
        return self.waypoints[index]
    def pointAtDist(self, start, r):
        distances = [np.linalg.norm(p.np()-start.np()) for p in self.waypoints]
        nearestAtDist = self.waypoints[-1]
        for i,d in enumerate(distances):
            if d > r:
                nearestAtDist = self.waypoints[i]
                break
        return nearestAtDist
    def plot(self, plt):
        cx = [p.x for p in self.waypoints]
        cy = [p.y for p in self.waypoints]
        plt.plot(cx, cy, "-r", label="path")

class GoalOnlyPseudoPath(Path):
    def __init__(self, start, goal):
        super().__init__(start,goal)
    def calculateWaypoints(self):
        self.waypoints.append(Point(self.start.x, self.start.y))
        self.waypoints.append(Point(self.goal.x, self.goal.y))
class LinearPath(Path):
    def __init__(self, start, goal):
        super().__init__(start,goal)
    def calculateWaypoints(self):
        dx,dy,self.r = self.goal.calc_distances(self.start)
        point_sample_div = 10.0
        x_start = self.start.x
        x_inc = dx / point_sample_div
        x = [x_start + x_inc * i for i in range(0, int(point_sample_div)+1) ]
        y_start = self.start.y
        y_inc = dy / point_sample_div
        y = [y_start + y_inc * i for i in range(0, int(point_sample_div)+1) ]
        for px, py in zip(x,y):
            self.waypoints.append(Point(px, py)) 
class DubinsShortesPath(Path):
    def __init__(self, start, goal, yawStart, yawGoal):
        self.yawStart = yawStart
        self.yawGoal = yawGoal
        super().__init__(start,goal)
    def calculateWaypoints(self):
        import path_planning.dubins_path as dubins
        x, y, path_yaw, mode, lengths = dubins.plan_dubins_path(self.start.x,
                                                            self.start.y,
                                                            self.yawStart,
                                                            self.goal.x,
                                                            self.goal.y,
                                                            self.yawGoal,
                                                            vehicle_max_curvature)
        for px, py in zip(x,y):
            self.waypoints.append(Point(px, py)) 

def check_path_for_collision(path, obstacles):
    for p in path.waypoints:
        for o in obstacles:
            if o.contains(p):
                return True
    return False
def prepare_path_point(state, path, lookaheadDist):
    current = Point(state.x, state.y)
    ### project lookahead point by lookahead time and current yaw (max lookahead distance)
    # d = min(state.v * Lfs, Lfc)
    # projected = Point.fromRPhi(d, state.yaw)
    # # print("projected: %s" % projected)
    # lookahead = projected + current
    # print("lookahead: %s" % lookahead)
    ### project lookahead point relative to path
    nearestPoint = path.pointAtDist(current, lookaheadDist)
    ### find nearest path point to lookahead
    # nearestPoint = path.findNearest(lookahead)
    return nearestPoint
        
def pure_pursuit_steer_control(state, nextPoint, lookaheadDist):
    alpha = math.atan2(nextPoint.y - state.rear_y, nextPoint.x - state.rear_x) - state.yaw
    delta = math.atan2(2.0 * state.wb * math.sin(alpha) / lookaheadDist, 1.0)
    return delta, nextPoint

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def plot(state, leader, path, target, obstacles):
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    plot_arrow(state.x, state.y, state.yaw, fc='b')
    plot_arrow(leader.x, leader.y, leader.yaw, fc='r')
    path.plot(plt)
    # plt.plot(states.x, states.y, "-b", label="trajectory")
    plt.plot(target.x, target.y, "xg", label="target")
    obstacleY = [o.center.y for o in obstacles]
    oSize = [o.size for o in obstacles]
    for o in obstacles:
        x = [p.x for p in o.rect]
        y = [p.y for p in o.rect]
        plt.plot(x, y, "-")
    ax = plt.gca()
    ax.set_xlim([leader.x-10.0, leader.x+10.0])
    ax.set_ylim([leader.y-10.0, leader.y+10.0])
    # plt.axis("equal")
    plt.grid(True)
    plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
    plt.pause(0.001)

def main():
    ### course the leader will simulate
    #  target course
    cx = np.arange(0, 100, 0.05)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
    leaderIndex = 0
    course = []
    for px,py in zip(cx, cy):
        course.append(Point(px,py))
    ### obstacles on target course
    random.seed(1)
    obstacles = []
    # for i in range(0,50):
    #     randx = -5 + random.random() * (30- -5)
    #     randy = -5 + random.random() * (30- -5)
    #     randSize = 0.1 + random.random() * (4-0.1)
    #     obstacles.append(Obstacle(Point(randx, randy), randSize))
    
    ### initial state
    state = State(x=0.0, y=-2.0, yaw=1.5, v=0.0, wb=vehicle_WB)
    leader = State(x=2.0, y=0.0, yaw=0.0, v=4.0, wb=leader_WB)
    ### pid acceleration
    pidParams = pid.Parameter(Kp=Kp, Ki=Ki, Kd=Kd, Ts=0.05, limitMin=-10.0, limitMax=10.0)
    controller = pid.Pid(pidParams)
    dt = 0.02
    while leader:
        ### sanity check for distance
        distance = np.linalg.norm(Point(leader.x, leader.y).np()-Point(state.x, state.y).np())
        if distance > 5.0:
            print("leader too far away, doing nothing")
        else:
            ### calc path to target
            # path = GoalOnlyPseudoPath(state, leader)
            # path = LinearPath(state, leader)
            path = DubinsShortesPath(state, leader, state.yaw, leader.yaw)

            ### check if path is blocked
            collision = check_path_for_collision(path, obstacles)

            if not collision:
                ### calculate lookahead distance based on velocity
                ### lookahead dist is the given one, or if higher, our current speed based on lookadhead time 
                lookaheadDist = max(Lfc, k * state.v * Lfs)

                ### choose next point
                nextPoint = prepare_path_point(state, path, lookaheadDist)

                ## pure pursuit
                di, currentTargetPoint = pure_pursuit_steer_control(state, nextPoint, lookaheadDist)

                ### speed controlled
                # error = leader.v - state.v 
                ### dist controlled, stay 2m behind target
                error = distance - 2.0
                ### Calc control input
                ai = controller(error, dt)
            else:
                ai = -10.0

        ### update states
        ### leader update dynamically
        # leader.update(0.0, 0.1, dt)
        ### pre-planned course for leader update
        currentLeaderPos = Point(leader.x,leader.y) 
        nextCoursePoint = course[leaderIndex] 
        diffP = nextCoursePoint - currentLeaderPos
        nextYaw = math.atan2(diffP.y, diffP.x)
        leader.x = nextCoursePoint.x
        leader.y = nextCoursePoint.y
        leader.yaw = nextYaw
        leaderIndex += 1
        ####
        state.update(ai, di, dt) ### follower update
        # print("vehicle: %s" % state)
        # print("leader : %s" % leader)
        plot(state, leader, path, currentTargetPoint, obstacles)
        time.sleep(dt)
        
if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    main()
