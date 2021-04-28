import numpy as np
import pandas as pd
from matplotlib import pylab as plt
import random 

random.seed(42)

# Create variable:
X = np.array([3, 2, 5, 4, 6, 7])

Y = np.squeeze([5 + 3*x + np.random.normal(-1,1,1) for x in X])

########################
### GRADIENT DESCENT ###
########################


# return the plot of data + linear hyperplane    
def plot_me(X, Y, B0, B1):
    plt.scatter(X, Y)
    plt.plot(X,  B0 + B1 * X)
    return plt.show()        

# compute cost function for knowing when to stop the gradient descent    
def cost_fct(B0, B1, X, Y):
    RSS = [] 
    m = len(X)
    for ind, x in enumerate(X):
        y_pred = B0 + B1 * x
        err = (y_pred - Y[ind])**2
        RSS.append(err)
    cost_function = np.sum(RSS)/(2*m)
    return cost_function
    
# return derivatives of each beta to update them
def update_weigths(B0, B1, X, Y):
    RSS_B0 = []
    RSS_B1 = [] 
    m = len(X)
    for ind, x in enumerate(X):
        y_pred = B0 + B1 * x
        err = (y_pred -Y[ind])
        err1 = x * (y_pred - Y[ind])
        RSS_B0.append(err)
        RSS_B1.append(err1) 
    der_B0 = np.sum(RSS_B0)/m
    der_B1 = np.sum(RSS_B1)/m
    return der_B0, der_B1

# compute gradient descent     
def fit_me(X, Y, alpha):
    # random value for B0 and B1
    list_B0 = []
    list_B1 = []
    cost_func_t = []
    m = len(X)
    B0 = B1 = 1
    # plot_me(X, Y, B0, B1)
    
    # first loop of the gradient descent
    cost = cost_fct(B0, B1, X, Y)
    der = update_weigths(B0, B1, X, Y)
    B0_updated = B0 - alpha*der[0]
    list_B0.append(B0_updated)
    B1_updated = B1 - alpha*der[1]
    list_B1.append(B1_updated)
    cost_updated = cost_fct(B0_updated, B1_updated, X, Y)
    cost_func_t.append(cost_updated)
    # plot_me(X, Y, B0_updated, B1_updated)
    
    # keep on decreasing till reach a minimun
    show = [10, 50, 100, 500, 1000, 1250, 1500, 1779]
    turn = 0
    while (cost - cost_updated) >= 0.0001:
        B0 = B0_updated
        B1 = B1_updated
        cost = cost_fct(B0, B1, X, Y)
        der = update_weigths(B0, B1, X, Y)
        B0_updated = B0 - alpha*der[0]
        B1_updated = B1 - alpha*der[1]
        cost_updated = cost_fct(B0_updated, B1_updated, X, Y)
        list_B0.append(B0_updated)
        list_B1.append(B1_updated)
        cost_func_t.append(cost_updated)
        # if turn in show:
        #     plot_me(X, Y, B0_updated, B1_updated)
        turn +=1
    print(turn)
        
    # return the final weight
    print("CONVERGED!")
    print("final plot B0 = {}, B1 = {}".format(B0_updated, B1_updated))
    # plot_me(X, Y, B0_updated, B1_updated)
    return B0_updated, B1_updated, list_B0, list_B1, cost_func_t, turn

B0_updated, B1_updated, list_B0, list_B1, cost_func_t, turn = fit_me(X, Y, 0.0001)


# # add fancy plot
# show = np.linspace(1, turn, 10, dtype=int)
# for index in show:
#     f, ([ax1, ax2]) = plt.subplots(1, 2, figsize=(12, 7))
#     ax1 = plt.subplot(1, 2, 1)
#     ax1.scatter(X, Y)
#     plt.plot(X,  list_B0[index] + list_B1[index] * X)
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     ax2 = plt.subplot(1, 2, 2)
#     plt.plot(list_B1, np.array(cost_func_t))
#     ax2.scatter(list_B1[index], np.array(cost_func_t)[index], marker='o', c="r")
#     plt.xlabel("Beta 1 values")
#     plt.ylabel("error")
#     plt.tight_layout()
#     plt.show()
    
    
    


# #########################
# ### Analytic solution ###
# #########################

# X = np.array([[3, 2], [2, 3]]) # matrices must be square to be invertible
# Y = 2*X[0] + 2*X[1]

# Solution = np.linalg.inv((X.T*X))*X.T*Y



#### ANIMATION WITH 2 SUBPLOT ####

### Animation 1 ##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter

betas = list(zip(list_B0, list_B1, cost_func_t))
index_list = [i for i in range(0, 1700, 10)]
batas = [betas[i] for i in index_list]
fig, ([ax1, ax2]) = plt.subplots(1, 2, figsize=(8, 6))

# fig, ax = plt.subplots()
xdata = []
ln, = ax1.plot([], [], c="r")
ln2, = ax2.plot([], [], marker='o', c="r")


def init():
    ax1.scatter(X, Y, c="black")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 25)
    ax1.set_title('Linear regression fitting process')
    ax1.set_ylabel('Scores examen')
    ax1.set_xlabel('Heure Harry Potter')
    ax2.plot(list_B1, np.array(cost_func_t), c="black")
    ax2.set_xlim(1, 3.5)
    ax2.set_ylim(0, 70)
    ax2.set_title('Optimisation using gradient descent')
    ax2.set_ylabel('Error')
    ax2.set_xlabel('Betas 1')

    return ln,

def update(frame):
    print(frame)
    X_ = X
    Y_ = frame[0] + frame[1] * X_
    ln.set_data(X_, Y_)
    X_1 = frame[1]
    Y_1 = frame[2]
    Eq = ax2.text(2.5, 50, "Y = {} + {} * X".format(np.round(frame[0], 1), np.round(frame[1], 1)))
    RSS = ax2.text(2.5, 45, "RSS = {}".format(np.round(frame[2], 2)))
    ln.set_data(X_, Y_)
    ln2.set_data(X_1, Y_1)
    return ln, ln2, Eq, RSS,

ani = FuncAnimation(fig, update, frames=batas,
                    init_func=init, interval=100, blit=True)
plt.show()

#### END ANIMATION 1 ######


#### ANIMATION WITH 3D SUBPLOT ON TOP
from mpl_toolkits.mplot3d import Axes3D
### Animation 2 ##

# needed for two first plot
betas = list(zip(list_B0, list_B1, cost_func_t))
index_list = [i for i in range(0, 1700, 10)]
batas = [betas[i] for i in index_list]

# info for 3D mesh
M = np.arange(-25, 25, 0.22)
B = np.arange(-25, 25, 0.22)
Z = [cost_fct(b, m, X, Y) for (b, m) in zip(M, B)]
M, B = np.meshgrid(M, B)
Z = [cost_fct(b, m, X, Y) for (b, m) in zip(np.ravel(B), np.ravel(M))]
Z = np.array(Z).reshape(M.shape)


fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

ax3.plot_wireframe(M, B, Z, rstride=1, cstride=1, color='lightblue', alpha=0.1)

Mplot = np.arange(-25, 3, 1)
bplot = np.arange(-22, 6, 1)
zplot = np.array([cost_fct(b, m, X, Y) for (b, m) in zip(Mplot, bplot)])
ax3.plot(Mplot, bplot, zplot, color='r', alpha=0.5)
ax3.scatter(Mplot, bplot, zplot, color='r')

# fig, ax = plt.subplots()
xdata = []
ln, = ax1.plot([], [], c="r")
ln2, = ax2.plot([], [], marker='o', c="r")


def init():
    ax1.scatter(X, Y, c="black")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 25)
    ax1.set_title('Linear regression fitting process')
    ax1.set_ylabel('Scores examen')
    ax1.set_xlabel('Heure Harry Potter')
    ax2.plot(list_B1, np.array(cost_func_t), c="black")
    ax2.set_xlim(1, 3.5)
    ax2.set_ylim(0, 70)
    ax2.set_title('Optimisation using gradient descent')
    ax2.set_ylabel('Error')
    ax2.set_xlabel('Betas 1')
    ax3.set_xlabel('Betas 1', labelpad=10)
    ax3.set_ylabel('Betas 0', labelpad=10)
    ax3.set_zlabel('RSS', labelpad=10, fontweight='bold')

    return ln,

def update(frame):
    print(frame)
    X_ = X
    Y_ = frame[0] + frame[1] * X_
    ln.set_data(X_, Y_)
    X_1 = frame[1]
    Y_1 = frame[2]
    Eq = ax2.text(2.5, 50, "Y = {} + {} * X".format(np.round(frame[0], 1), np.round(frame[1], 1)))
    RSS = ax2.text(2.5, 45, "RSS = {}".format(np.round(frame[2], 2)))
    ln.set_data(X_, Y_)
    ln2.set_data(X_1, Y_1)
    return ln, ln2, Eq, RSS,

ani = FuncAnimation(fig, update, frames=batas,
                    init_func=init, interval=100, blit=True)
plt.show()










