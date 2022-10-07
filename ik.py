import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))



def make_kinematics(*args):
    bones = []
    bones_theta = []
    for length in args:
        bones.append(np.array(([length],[0])))
        bones_theta.append(0)
    
    return bones, bones_theta


def last_point(bones, args):
    rmat = np.identity(2)
    tvec = np.zeros((2,1))
    for bone, theta in zip(bones, args):
        rmat = rmat @ rot(theta)
        tvec = rmat @ rot(theta) @ bone + tvec
    return tvec

def get_kinematics(bones, thetas):
    rmat = np.identity(2)
    tvec = np.zeros((2,1))
    res = [tvec] 
    for bone, theta in zip(bones, thetas):
        rmat = rmat @ rot(theta)
        tvec = rmat @ rot(theta) @ bone + tvec
        res.append(tvec)
    return res

def solve(bones, bones_theta, target, eps = 0.0001):
    joc = np.zeros((2,2))
    from copy import deepcopy as dp
    for i, theta in enumerate(bones_theta):
        tmp = dp(bones_theta)
        tmp[i] += eps
        joc[:, [i]] = last_point(bones, tmp)/ eps

    print(joc)
    last_p = last_point(bones, bones_theta)
    res = np.linalg.solve(joc,-(last_p-target))
    print(res)
    print("h:",joc@res)
    print("l:",-last_p)
    for i in range(len(bones_theta)):
        bones_theta[i] += res[i,0]
    return bones, bones_theta

def draw(line, bones, bones_theta, target_p):
    colr = ['b', 'r', 'g']
    parent = np.array([[0],[0]])
    abs_bone =  get_kinematics(bones, bones_theta)
    x = []
    y = []
    c = []
    for i in range(len(bones)):
        x += [abs_bone[i][0,0], abs_bone[i+1][0,0]]
        y += [abs_bone[i][1,0], abs_bone[i+1][1,0]]
    line.set_data(x, y)


length1 = 3
length2 = 2


target_p = np.array([[10],[10]])

fig = plt.figure()
ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
line, = ax.plot([], [], lw=3)

bones, bones_theta = make_kinematics(length1, length2)

def animate(i):
    global bones, bones_theta,target_p
    bones, bones_theta = solve(bones, bones_theta, np.array([[5],[2]]))
    draw(line, bones, bones_theta, target_p)


anim = FuncAnimation(fig, animate, frames=200, interval=500)
plt.show()











