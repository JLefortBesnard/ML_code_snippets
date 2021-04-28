"""
# non_parametric_hypothesis_testing

Stats Jam: non_parametric_hypothesis_testing with permutation (permutation testing) 
=> super visualisation : https://www.jwilber.me/permutationtest/
"""

# Non paramteric hypothesis testing
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# toy example: grades negatively influenced by being a fan of harry potter

###############################################################################
# hypothesis: being a harry potter fan decrease the grade => one sided t test #
###############################################################################

# groupe A, harry potter fan
fan = [8, 7, 6, 7, 8, 4, 4, 5, 7, 8, 4, 9, 10, 10, 8, 7, 6, 7, 8, 10]

# groupe B, vodka drinker lol, controls
control = [6, 10, 7, 8, 10, 8, 10, 10, 6, 7, 8, 7, 5, 7, 9, 10, 9, 6, 5, 9]

# get whole set for permutation later
fancontrol = fan + control

# compute original means
fan_mean = np.mean(fan)
control_mean = np.mean(control)

# compute the metric of interest
def compute_metric(fan, control):
	metric = fan - control
	return metric

# compute original differences between group mean
original_metric = compute_metric(fan_mean, control_mean)
print(original_metric)

# launch the permutation
n_permutations = 1000
permutation_metrics = []
for i_iter in range(n_permutations):
	print(i_iter + 1)

	perm = np.random.permutation(fancontrol)
	fan_perm = perm[:20]
	control_perm = perm[20:]
	
	fan_perm_mean = np.mean(fan_perm)
	control_perm_mean = np.mean(control_perm)

	metric = compute_metric(fan_perm_mean, control_perm_mean)
	permutation_metrics.append(metric)

below = stats.scoreatpercentile(permutation_metrics, 5)





#### ANIMATION 1 ######

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter

perms = permutation_metrics

fig, ax = plt.subplots()
xdata = []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 10)
    return ln,

def update(frame):
	if len(xdata) == 50:
		ax.set_xlim(-2, 2)
		ax.set_ylim(-0.5, 20)
	if len(xdata) == 150:
		ax.set_xlim(-3, 3)
		ax.set_ylim(-0.5, 30)
	if len(xdata) == 300:
		ax.set_xlim(-4, 4)
		ax.set_ylim(-0.5, 70)
	if len(xdata) == 500:
		ax.set_xlim(-4, 4)
		ax.set_ylim(-0.5, 90)
	if len(xdata) == 700:
		ax.set_xlim(-5, 5)
		ax.set_ylim(-0.5, 110)
	xdata.append(frame)
	counts = Counter(xdata)
	X = []
	Y = []
	for key in counts.keys():
		for nb in range(0, counts[key]):
			X.append(key)
			Y.append(nb +1 )
	ln.set_data(X, Y)
	return ln,

ani = FuncAnimation(fig, update, frames=perms,
                    init_func=init, interval=2000, blit=True)
plt.show()

#### END ANIMATION 1 ######


#### ANIMATION 2 ######
fig, ax = plt.subplots()
xdata = []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 10)
    return ln,

def update(frame):
	if len(xdata) == 50:
		ax.set_xlim(-2, 2)
		ax.set_ylim(-0.5, 20)
	if len(xdata) == 150:
		ax.set_xlim(-3, 3)
		ax.set_ylim(-0.5, 30)
	if len(xdata) == 300:
		ax.set_xlim(-4, 4)
		ax.set_ylim(-0.5, 70)
	if len(xdata) == 500:
		ax.set_xlim(-4, 4)
		ax.set_ylim(-0.5, 90)
	if len(xdata) == 700:
		ax.set_xlim(-5, 5)
		ax.set_ylim(-0.5, 110)
	xdata.append(frame)
	counts = Counter(xdata)
	X = []
	Y = []
	for key in counts.keys():
		for nb in range(0, counts[key]):
			X.append(key)
			Y.append(nb +1 )
	ln.set_data(X, Y)
	return ln,

ani = FuncAnimation(fig, update, frames=perms,
                    init_func=init, interval=20, blit=True)
plt.show()
#### END ANIMATION 2 ######



# H0 distribution
plt.hist(permutation_metrics, bins=20, density=True, facecolor='black', alpha=0.75)
# plt.axvline(x = below, ymax=0.9, color = 'r', label = 'Significance',linestyle='--',  lw=1)
x = np.arange(min(permutation_metrics), below, 0.01)
y = [1] * len(x)
# plt.axvline(x = original_metric, y=1, color = 'blue', label = 'value',linestyle='-.',  lw=2)
plt.ylabel("frequence", fontsize=16, weight="bold")
plt.xlabel("H0 distribution", fontsize=16, weight="bold")
plt.title("Drawing H0")
plt.tight_layout()
plt.show()




# H0 distribution
plt.hist(permutation_metrics, bins=20, density=True, facecolor='black', alpha=0.75)
# plt.axvline(x = below, ymax=0.9, color = 'r', label = 'Significance',linestyle='--',  lw=1)
plt.text(-1.4, 0.4, "p<0.05", color="red")
x = np.arange(min(permutation_metrics), below, 0.01)
y = [1] * len(x)
plt.fill_between(x, y, color='red', alpha=0.5)
# plt.axvline(x = original_metric, y=1, color = 'blue', label = 'value',linestyle='-.',  lw=2)
plt.ylabel("frequence", fontsize=16, weight="bold")
plt.xlabel("H0 distribution", fontsize=16, weight="bold")
plt.title("Check for significance")
plt.tight_layout()
plt.show()





# check for significance
plt.hist(permutation_metrics, bins=20, density=True, facecolor='black', alpha=0.75)
# plt.axvline(x = below, ymax=0.9, color = 'r', label = 'Significance',linestyle='--',  lw=1)
plt.text(-1.4, 0.4, "p<0.05", color="red")
x = np.arange(min(permutation_metrics), below, 0.01)
y = [1] * len(x)
plt.fill_between(x, y, color='red', alpha=0.5)
plt.axvline(x = original_metric, color = 'blue', label = 'p value obtained',linestyle='-.',  lw=2)
plt.ylabel("frequence", fontsize=16, weight="bold")
plt.xlabel("H0 distribution", fontsize=16, weight="bold")
plt.legend()
plt.title("Check for significance")
plt.tight_layout()
plt.show()

if original_metric < below:
	print("group fan got significantly lower grades than control group (p>0.05)")
else:
	print("group fan did not get significantly lower grades than control group")






    



# toy example: watching harry potter movies negatively influence grades

####################################################################################
# hypothesis: watching harry potter movies decreases the grade => one sided t test #
####################################################################################

# Non parametric hypothesis testing for regression
import numpy as np
from scipy import stats
from sklearn import linear_model


# obtained grades
Y = [6, 1, 9, 2, 4, 9, 5, 2, 5, 3, 1, 8, 6, 5, 1, 9, 4, 7, 2, 1]

# number of hour spent watching harry potter movies
X = np.array([5, 9, 1, 8, 5, 1, 5, 8, 5, 7, 10, 2, 4, 5, 9, 1, 5, 5, 9, 8]).reshape((-1, 1))

# fit a linear model
reg = linear_model.LinearRegression()
reg.fit(X, Y)
original_metric = reg.coef_



# compute the metric of interest
def compute_metric(X, Y_perm):
	reg = linear_model.LinearRegression()
	reg.fit(X, Y_perm)
	metric = reg.coef_
	return metric

# launch the permutation
n_permutations = 1000
permutation_metrics = []
for i_iter in range(n_permutations):
	print(i_iter + 1)

	Y_perm = np.random.permutation(Y)
	metric = compute_metric(X, Y_perm)
	permutation_metrics.append(metric)
permutation_metrics = np.squeeze(permutation_metrics)
below = stats.scoreatpercentile(permutation_metrics, 5)

if original_metric < below:
	print("Watching more harry potter movies negatively influences grades (p>0.05)")
else:
	print("Watching more harry potter movies does not negatively influences grades")



# H0 distribution
plt.hist(permutation_metrics, bins=20, density=True, facecolor='black', alpha=0.75)
# plt.axvline(x = below, ymax=0.9, color = 'r', label = 'Significance',linestyle='--',  lw=1)
plt.text(-0.6, 0.4, "p<0.05", color="red")
x = np.arange(min(permutation_metrics), below, 0.01)
y = [1] * len(x)
plt.fill_between(x, y, color='red', alpha=0.5)
# plt.axvline(x = original_metric, y=1, color = 'blue', label = 'value',linestyle='-.',  lw=2)
plt.ylabel("frequence", fontsize=16, weight="bold")
plt.xlabel("H0 distribution", fontsize=16, weight="bold")
plt.title("Check for significance")
plt.tight_layout()
plt.show()

# check for significance
plt.hist(permutation_metrics, bins=20, density=True, facecolor='black', alpha=0.75)
# plt.axvline(x = below, ymax=0.9, color = 'r', label = 'Significance',linestyle='--',  lw=1)
plt.text(-0.6, 0.4, "p<0.05", color="red")
x = np.arange(min(permutation_metrics), below, 0.01)
y = [1] * len(x)
plt.fill_between(x, y, color='red', alpha=0.5)
plt.axvline(x = original_metric, color = 'blue', label = 'p value obtained',linestyle='-.',  lw=2)
plt.ylabel("frequence", fontsize=16, weight="bold")
plt.xlabel("H0 distribution", fontsize=16, weight="bold")
plt.legend()
plt.title("Check for significance")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
plt.scatter(X, Y)
original_metric = reg.coef_
plt.ylabel("Grades", fontsize=16, weight="bold")
plt.xlabel("Hours spent watching HP", fontsize=16, weight="bold")
plt.show()


import matplotlib.pyplot as plt
plt.scatter(X, Y)
x = np.arange(0, 10)
y = x*reg.coef_ + reg.intercept_
plt.plot(x, y, color='red')
original_metric = reg.coef_
plt.ylabel("Grades", fontsize=16, weight="bold")
plt.xlabel("Hours spent watching HP", fontsize=16, weight="bold")
plt.show()



