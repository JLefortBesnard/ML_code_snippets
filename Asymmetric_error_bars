import numpy as np
from matplotlib import pylab as plt
# Define variables
X = np.array([1, 2, 3, 4, 5, 6])
Y = np.squeeze([.5, 1.4, 3.6, -0.3, .34, -1])

# define error bars value lower and upper
err_l = [0.3, 0.4, 0.19, 0.52, 0.33, 0.9] # lower
err_u = [0.2, 0.24, 0.29, 0.23, 0.45, 0.13] # upper
err = [err_l, err_u]

# plot the graph
ax = plt.axes()
ax.errorbar(X, Y, yerr = np.array(err), fmt='o', ecolor="g", capthick=2)
# plt.bar(X, Y, yerr=err) # if bars instead of point
labels = ["", "first", "second", "third", "fourth", "fifth", "sixth"]
ax.set_xticklabels(labels)
plt.xticks(rotation=90, fontsize=7)
plt.tight_layout()
plt.show()
