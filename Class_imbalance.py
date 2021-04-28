
"""
How to deal with class imbalance using Python
if only 5% of observations are women, then even a naive
model that predicts everybody as man will be 95% accurate. Clearly
this is not ideal. 
Using confusion matrices, precision, recall, F1 scores, and ROC curves is helpful
"""

# 4 types of solution:

#1. Collect more data.
#2. Change the metrics used to evaluate your model.
#3. Consider using a model’s built-in class weight parameters if possible; e.g. randomforest(class_weight="balanced")
#4. Downsampling, or upsampling.


# Features = array of inputs (X)
# Target = vector of dummies output (Y) (0 and 1)

######### DOWNSAMPLING ###############

# Downsample the majority class or upsample the minority class. 

# randomly sample without replacement from the majority class 
# to create a new set of observations equal to the minority class's size. 

# Indicies of each class' observations
ind_0 = np.where(target == 0)[0]
ind_1 = np.where(target == 1)[0]

# Number of observations in each class
n_0 = len(ind_0)
n_1 = len(ind_1)

# For every observation of class 0, randomly sample from majority class without replacement
ind_1_downsampled = np.random.choice(ind_1, size=n_0, replace=False)

# Join together the vectors of target 
np.hstack((target[ind_0], target[ind_1_downsampled]))

# Join together the arrays of features
np.vstack((features[ind_0,:], features[ind_1_downsampled,:]))


######### UPSAMPLING ###############

# randomly sample with replacement from the minority class to get
# the same number of observations from the minority and majority classes.


# For every observation in class 1, randomly sample from class 0 with replacement
ind_0_upsampled = np.random.choice(ind_0, size=n_1, replace=True)

# Join together the vectors of target
np.concatenate((target[ind_0_upsampled], target[ind_1]))

# Join together the arrays of features
np.vstack((features[ind_0_upsampled,:], features[ind_1,:]))
