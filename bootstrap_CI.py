"""
Calculate and plot Bootstrap intervals.
Example with the iris dataset from sklearn.
Get the 95% CI of the logistic regression weigths from 100 subsamples
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from scipy.stats import scoreatpercentile
import pandas as pd

# load and make dataset as pandas dataframe
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Target"] = iris.target  # setosa, versicolor, or virginica
df = df[df.Target < 2]  # keep only setosa and versicolor for this example
df.head(3)

# make sure you don't have any class imbalance problem
np.unique(df.Target, return_counts=True)

# define features and outcome
X = df.values[:, 0:4]
y = df.Target 

# run the logistic regression in our sample 
clf = LogisticRegression()
clf.fit(X, y)
acc_original = clf.score(X, y)
coef_original = clf.coef_[0]
print("score= %.2f") % (acc_original)

np.random.seed(0) # makes the random numbers predictable for replicable results

# Run the algorithm on 100 boostrapped subsamples 
# and keep the obtained coefs and accs for each
acc_bs = []
coef_bs = []
for i_subsample in range(100):
    # get the bootstrap subsample as a pandas dataframe
    sample_index = np.random.choice(df.index, len(df)) # resample with replacement
    df_bs = df.loc[sample_index]
                    
    # define features and outcome from the specific subsample       
    X = df_bs.values[:, 0:4]    
    y = df_bs.Target
            
    #run the logistic regression for the specific subsample 
    clf = LogisticRegression()
    clf.fit(X, y)
    acc = clf.score(X, y)
    
    # collect acc and coefs of the subsample
    acc_bs.append(acc)
    coef_bs.append(clf.coef_[0])
    print("acc: %.2f") % (acc)


# Extract the bootstrapped confidence intervals
df_bs = pd.DataFrame(np.array(coef_bs), columns = df.columns[0:4])
bs_err_dic = {}
for ind, item in enumerate(df_bs):
    bs_err = scoreatpercentile(df_bs[item], [2.5, 97.5], interpolation_method='fraction', axis=None)
    bs_err_dic[item] = bs_err

# calculating boostrap CI + and - values above and under the original data
err_l_ = []
err_u_ = []
for ind, name in enumerate(df.columns[0:4]):
    err_l = coef_original[ind] - bs_err_dic[name][0]
    err_u = bs_err_dic[name][1] - coef_original[ind]
    err_l_.append(err_l)
    err_u_.append(err_u)
err = np.array([err_l_, err_u_])


# plotting the results 

# function copy-pasted form stackoverflow to rotate and align labels with ticks
def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
    axes = []
    if which in ['x', 'both']:
        axes.append(ax.xaxis)
    elif which in ['y', 'both']:
        axes.append(ax.yaxis)
    for axis in axes:
        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)


X = df.columns[:4]
Y = coef_original # the weights obtained with the original sample
fig, ax = plt.subplots()
ax.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", ms=4)

# customize the figure
plt.text(2, 1.5, "score=%0.2f, n=%i" %(acc_original, len(df)), fontsize=8, ha='center', style="italic")
ax.set_xticklabels(X)
rotateTickLabels(ax, -55, 'x')
plt.xticks(fontsize=10)
plt.xlabel("")
plt.ylabel("Weight", fontsize=12, weight="bold")
plt.yticks(fontsize=10)
plt.axhline(0, color='grey')
plt.grid(True, axis="y")
plt.ylim([-1.7, 2.5])
plt.suptitle("Predict flower's type based on flower's features", y=1., fontsize=14, weight="bold")
plt.tight_layout()
plt.show()   

    
    