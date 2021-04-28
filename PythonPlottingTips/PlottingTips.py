"""
A few tips to use plt.subplots()
Quick preview of what you can control and how to control it using matplotlib.pyplot.subplots()
You can run the script directly (python 2.7) but make sure you downloaded 
the two attached .jpg images (test.jpg and map.jpg) 
and that these images and you are in the same working directory.

"""


import seaborn as sns
from matplotlib import pylab as plt
import numpy as np
import pandas as pd


# create data
Y = [1, 2, -1, -2, 0]            
X = np.array([1, 2 , 3, 4, 5])
# confidence intervals
err = [[0.2, 0.75, 0.50, 0.80, 0.60], [0.80, 0.50, 0.60, 0.75, 1.2]]
# labels 
X_colnames = ["feature 1", "feature 2", "feature 3", "feature 4", "feature 5"]
labels = [""] + [X_colnames[i] for i in range(0, len(X_colnames))] # labels starts at index 1


###############################################################
###############################################################
###############################################################
# one plot
plt.scatter(X, Y)    
plt.show()


# one plot
plt.figure(figsize=(8, 8))
plt.scatter(X, Y)    
plt.show()



# two plots as 2 columns
f, ([ax1, ax2]) = plt.subplots(1, 2, figsize=(8, 8))    
plt.show()



# two plots as 2 rows
f, ([ax1, ax2]) = plt.subplots(2, 1, figsize=(8, 8))    
plt.show()



# four plots
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8))    
plt.show()



# six plots
f, ([ax1, ax2, ax5], [ax3, ax4, ax6]) = plt.subplots(2, 3, figsize=(8, 8))    
plt.show()

# Can be different sizes
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2)
gridsize = (3, 2)  
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid(gridsize, (2, 0))
ax3 = plt.subplot2grid(gridsize, (2, 1))
plt.show()

# Can even be completely different sizes
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2)    
ax1 = plt.subplot2grid(gridsize, (0, 0))
ax2 = plt.subplot2grid(gridsize, (2, 0))
ax3 = plt.subplot2grid(gridsize, (2, 1))
plt.show()


# can be an axe into an axe!
ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
plt.show()



# four plots
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8))    
plt.suptitle("useless axis intems")
plt.show()



# sharing the x-axis ticks
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col")    
plt.show()



# sharing the x- and y-axis ticks
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")    
plt.show()



###############################################################
###############################################################
###############################################################


# You have to define which plot in which, for example:    
# graph 1
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row") 
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# graph 2
ax2 = plt.subplot(2, 2, 2)
ax2.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# graph 3 and 4 are empty
plt.show()



###############################################################
###############################################################
###############################################################


# You can write something in the empty plot, for example:
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row") 
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# graph 2
ax2 = plt.subplot(2, 2, 2)
ax2.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# Write insite graph 3 and 4 that they are empty
ax3 = plt.subplot(2, 2, 3)
plt.text(0.5, 0.5, "I am empty", fontsize=8, ha='center', style="italic")
ax4 = plt.subplot(2, 2, 4)
plt.text(0.5, 0.5, "I am empty too", fontsize=8, ha='center', style="italic")

plt.show()


# you can also add an arrow!!
ax1 = plt.axes()
ax1.annotate("this is an insider so to say", xy=(0.6, 0.6), xytext=(0.1, 0.1),
                arrowprops=dict(facecolor="black", shrink=0.05))
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
plt.show()

###############################################################
###############################################################


# You can decide where to write stuff, for example:
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row") 
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# graph 2
ax2 = plt.subplot(2, 2, 2)
ax2.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# Write insite graph 3 and 4 that they are empty
ax3 = plt.subplot(2, 2, 3)
plt.text(0.2, 0.2, "I am empty", fontsize=8, ha='center', style="italic")
ax4 = plt.subplot(2, 2, 4)
plt.text(0.8, 0.8, "I am empty too", fontsize=8, ha='center', style="italic")

plt.show()

###############################################################
###############################################################


# You can colored writen stuff, for example:
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row") 
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# graph 2
ax2 = plt.subplot(2, 2, 2)
ax2.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# Write insite graph 3 and 4 that they are empty
ax3 = plt.subplot(2, 2, 3)
plt.text(0.2, 0.2, "I am empty", color="r", fontsize=8, ha='center', style="italic")
ax4 = plt.subplot(2, 2, 4)
plt.text(0.8, 0.8, "I am empty too", color="g", fontsize=8, ha='center', style="italic")

plt.show()



###############################################################
###############################################################
###############################################################


# all of them full 
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row") 
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# graph 2
ax2 = plt.subplot(2, 2, 2)
ax2.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# graph 3
ax3 = plt.subplot(2, 2, 3)
ax3.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# graph 4
ax4 = plt.subplot(2, 2, 4)
ax4.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)

plt.show()



###############################################################
###############################################################
###############################################################


# add a line in the middle to get the 0 axis
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row") 
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
plt.axhline(0, color='grey')
# graph 2
ax2 = plt.subplot(2, 2, 2)
ax2.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
plt.axhline(0, color='grey')
# Write insite graph 3 and 4 that they are empty
ax3 = plt.subplot(2, 2, 3)
ax3.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
plt.axhline(0, color='grey')
ax4 = plt.subplot(2, 2, 4)
ax4.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
plt.axhline(0, color='grey')

plt.show()



###############################################################
###############################################################
###############################################################



# reduce the size of your code with a for loop
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row") 
for i in range(0, 4):
    axe = plt.subplot(2, 2, i+1)
    axe.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
    plt.axhline(0, color='grey')
plt.suptitle("For loop: 5 lines of codes intead of almost 15", y=0.95, fontsize=14, weight="bold")
plt.show()
# The problem is that you can't really personnalize each graph



# It is still worth it to make a for loop because you will always have function that you
# want to apply to each graph such as plt.ylim() or plt.tight_layout()
# Still, we won't use it below just to make the function more clear in what they are doing


###############################################################
###############################################################
###############################################################

# Add some labels 

f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")    
# graph 1
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax1.set_xticklabels(labels)
plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
plt.ylabel("Label Y", fontsize=12, weight="bold")
plt.axhline(0, color='grey')

# graph 2
ax2 = plt.subplot(2, 2, 2)
ax2.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax2.set_xticklabels(labels)
plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
plt.ylabel("Label Y")
plt.axhline(0, color='grey')

# graph 3
ax3 = plt.subplot(2, 2, 3)
ax3.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax3.set_xticklabels(labels)
plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
plt.ylabel("Label Y", fontsize=12, weight="bold") 
plt.axhline(0, color='grey')
              
#graph 4
ax4 = plt.subplot(2, 2, 4)
ax4.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax4.set_xticklabels(labels)
plt.axhline(0, color='grey')
plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("X Label")
plt.ylabel("Y Label")

plt.show()


"""
Many problems here:

1. Too many labels
2. space in between labels and title
3. useless Title
4. miss main title
"""


###############################################################
###############################################################
###############################################################

# solve the above problems

f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")    
# graph 1
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax1.set_xticklabels(labels)
plt.title("Label X", fontsize=12, weight="bold")
# plt.xlabel("Label X")
plt.ylabel("Label Y", fontsize=12, weight="bold")
plt.axhline(0, color='grey')

# graph 2
ax2 = plt.subplot(2, 2, 2)
ax2.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax2.set_xticklabels(labels)
plt.title("Label X", fontsize=12, weight="bold")
# plt.xlabel("Label X")
# plt.ylabel("Label Y", fontsize=12, weight="bold")
plt.axhline(0, color='grey')

# graph 3
ax3 = plt.subplot(2, 2, 3)
ax3.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax3.set_xticklabels(labels)
# plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
plt.ylabel("Label Y", fontsize=12, weight="bold") 
plt.axhline(0, color='grey')
              
#graph 4
ax4 = plt.subplot(2, 2, 4)
ax4.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax4.set_xticklabels(labels)
plt.axhline(0, color='grey')
# plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("X label")
# plt.ylabel("Y label", fontsize=12, weight="bold")

plt.suptitle("Main Title", y=1, fontsize=14, weight="bold")
plt.tight_layout()
plt.show()


"""
We still can't really read the x axis (need to rotate them)
"""

###############################################################
###############################################################
###############################################################

# rotate the labels to 90 degres
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")    
# graph 1
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax1.set_xticklabels(labels, rotation = 90)
plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
plt.ylabel("Label Y", fontsize=12, weight="bold")
plt.axhline(0, color='grey')
plt.tight_layout()
plt.show()

# rotate the labels to 45 degres
f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")    
# graph 1
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax1.set_xticklabels(labels, rotation = 45)
plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
plt.ylabel("Label Y", fontsize=12, weight="bold")
plt.axhline(0, color='grey')
plt.tight_layout()
plt.show()

"""
then the problem we face is that the ticks are not aligned
"""


###############################################################
###############################################################
###############################################################

# To align x ticks, we can use this function (found in stackoverflow)
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

f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")    
# graph 1
ax1 = plt.subplot(2, 2, 1)
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax1.set_xticklabels(labels)
rotateTickLabels(ax1, -55, 'x')
plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
plt.ylabel("Label Y", fontsize=12, weight="bold")
plt.axhline(0, color='grey')

# graph 2
ax2 = plt.subplot(2, 2, 2)
ax2.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax2.set_xticklabels(labels)
rotateTickLabels(ax2, -55, 'x')
plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
# plt.ylabel("Label Y", fontsize=12, weight="bold")
plt.axhline(0, color='grey')

# graph 3
ax3 = plt.subplot(2, 2, 3)
ax3.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax3.set_xticklabels(labels)
rotateTickLabels(ax3, -55, 'x')
# plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
plt.ylabel("Label Y", fontsize=12, weight="bold") 
plt.axhline(0, color='grey')
              
#graph 4
ax4 = plt.subplot(2, 2, 4)
ax4.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax4.set_xticklabels(labels)
plt.axhline(0, color='grey')
rotateTickLabels(ax4, -55, 'x')
# plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("X label")
# plt.ylabel("Y label", fontsize=12, weight="bold")

plt.suptitle("Main Title", y=1, fontsize=14, weight="bold")
plt.tight_layout()
plt.show()



"""
Now, we can add more fancy stuff such as:

text inside
a grid
hide the x axis 
"""


###############################################################
###############################################################
###############################################################

# To align x ticks, we can use this function (found in stackoverflow)
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

f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="row")    
# graph 1
ax1 = plt.subplot(2, 2, 1)
plt.text(4, 1.5, "First plot", fontsize=8, ha='center', style="italic")
ax1.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# ax1.set_xticklabels(labels)
# rotateTickLabels(ax1, -55, 'x')
ax1.get_xaxis().set_visible(False)
plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
plt.ylabel("Label Y", fontsize=12, weight="bold")
plt.axhline(0, color='grey')
plt.grid(True, axis="y")

# graph 2
ax2 = plt.subplot(2, 2, 2)
plt.text(4, 1.5, "Second plot", fontsize=8, ha='center', style="italic")
ax2.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
# ax2.set_xticklabels(labels)
# rotateTickLabels(ax2, -55, 'x')
ax2.get_xaxis().set_visible(False)
plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Label X")
# plt.ylabel("Label Y", fontsize=12, weight="bold")
plt.axhline(0, color='grey')
plt.grid(True, axis="y")

# graph 3
ax3 = plt.subplot(2, 2, 3)
plt.text(4, 1.5, "Third plot", fontsize=8, ha='center', style="italic")
ax3.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax3.set_xticklabels(labels)
rotateTickLabels(ax3, -55, 'x')
# plt.title("Label X", fontsize=12, weight="bold")
plt.xlabel("Features label")
plt.ylabel("Label Y", fontsize=12, weight="bold") 
plt.axhline(0, color='grey')
plt.grid(True, axis="y")
              
#graph 4
ax4 = plt.subplot(2, 2, 4)
plt.text(4, 1.5, "Fouth plot", fontsize=8, ha='center', style="italic")
ax4.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", capthick=2)
ax4.set_xticklabels(labels)
ax4.legend(loc=(0.65, 0.8))
plt.axhline(0, color='grey')
rotateTickLabels(ax4, -55, 'x')
# plt.title("Label X", fontsize=12, weight="bold")
plt.grid(True, axis="y")
# plt.ylabel("Y label", fontsize=12, weight="bold")

plt.ylim([-3, 3])
sns.despine(bottom=True)
plt.suptitle("Main Title", y=1, fontsize=14, weight="bold")
plt.tight_layout()
plt.show()


###############################################################
###############################################################
###############################################################

# handle legend
# one plot
Yblue = [1, 2, -1, -2, 0]
Yred = [2, -2, 0, 1, 2]  
plt.scatter(X, Yblue, color = "blue")
plt.scatter(X, Yred, color= "red")
plt.show()

# play with markers
plt.scatter(X, Yblue, color = "blue", marker= "x")
plt.scatter(X, Yred, color= "red", marker= ",")
plt.show()

# even more markers (script from Jake Vanderplas)
rng = np.random.RandomState(0)
for marker in ["o", ".", ",", "x", "+", "v", "<", ">", "s", "d"]:
    plt.plot(rng.rand(5), rng.rand(5), marker, label="marker='{}'".format(marker))
plt.legend()
plt.show()

# plt.legend is not enough 
plt.scatter(X, Yblue, color = "blue")
plt.scatter(X, Yred, color= "red")
plt.legend()
plt.show()

# add legend
plt.scatter(X, Yblue, color = "blue", label="blue")
plt.scatter(X, Yred, color= "red", label="red")
plt.legend()
plt.show()

# legend location
plt.scatter(X, Yblue, color = "blue", label="blue")
plt.scatter(X, Yred, color= "red", label="red")
plt.legend(loc=(0.65, 0.8))
plt.show()

# legend location
plt.scatter(X, Yblue, color = "blue", label="blue")
plt.scatter(X, Yred, color= "red", label="red")
plt.legend(loc=(0.30, 0.30))
plt.grid(True)
plt.show()


# play with the marker size 
Yblue = [1, 2, -1, -2, 0]
Yred = [2, -2, 0, 1, 2] 
sizes = [500, 120, 50, 2000, 50] 
plt.scatter(X, Yblue, s=sizes, color = "blue")
plt.scatter(X, Yred, s= sizes, color= "red")
# plt.colorbar()
plt.show()

# & intensity
plt.scatter(X, Yblue, s=sizes, alpha = 0.6, color = "blue")
plt.scatter(X, Yred, s= sizes, alpha= 0.2, color= "red")
# plt.colorbar()
plt.show()

# create a legend for the size
plt.scatter(X, Yblue, s=sizes, alpha = 0.6, color = "blue")
plt.scatter(X, Yred, s= sizes, alpha= 0.2, color= "red")
# plt.colorbar()
for size in [100, 500, 1000]:
    plt.scatter([], [], alpha=0.3, c="blue", s=size, label=str(size) +' km$^2$')
plt.legend()
plt.show()
    
##############################################################################
# for the following part, you need two images in your working directory
# one named test.jpg and map.jpg
##############################################################################

# insert image and plot over it
im = plt.imread('test.jpg') # make sure the location of the image and yours are the same
fig, ax = plt.subplots()
ax.imshow(im)
X = [350, 400, 100, 750, 500]
Yblue = [200, 500, 100, 430, 300]
Yred = [300, 250, 150, 550, 450] 
sizes = [500, 120, 50, 2000, 50] 
plt.scatter(X, Yblue, s=sizes, alpha = 0.6, color = "blue")
plt.scatter(X, Yred, s= sizes, alpha= 0.2, color= "red")
# plt.colorbar()
for ind, size in enumerate([100, 500, 1000]):
    plt.scatter([], [], alpha=0.3, c="blue", s=size, label=str((ind + 1) *2) +' m$^2$')
plt.legend()
plt.show()


# insert image and plot over it
im = plt.imread('map.jpg') # make sure the location of the image and yours are the same
fig, ax = plt.subplots()
ax.imshow(im)
X = [470, 670, 630, 420, 480]
Yblue = [400, 450, 240, 320, 800]
sizes = [500, 250, 400, 400, 100] 
plt.scatter(X, Yblue, s=sizes, alpha = 0.6, color = "blue")
# plt.colorbar()
for size in [50, 200, 400]:
    plt.scatter([], [], alpha=0.3, c="blue", s=size, label=str(size) +' m$^2$')
plt.legend()
plt.show()



# insert image and plot over it
im = plt.imread('map.jpg') # make sure the location of the image and yours are the same
fig, ax = plt.subplots()
ax.imshow(im)
X = [470, 670, 630, 420, 480]
Yblue = [400, 450, 240, 320, 800]
sizes = [500, 250, 400, 400, 100] 
plt.scatter(X, Yblue, s=sizes, alpha = 0.6, color = "blue")
plt.text(470, 450, "Bienvenue", fontsize=12, ha='center', weight="bold", color="b")
plt.text(670, 500, "Wilkommen", fontsize=12, ha='center', weight="bold", color="b")
plt.text(630, 290, "Gruess Gott", fontsize=12, ha='center', weight="bold", color="b")
plt.text(420, 370, "Hello", fontsize=12, ha='center', weight="bold", color="b")
plt.text(480, 850, "As-salam alaykom", fontsize=12, ha='center', weight="bold", color="b")

# plt.colorbar()
for size in [50, 200, 400]:
    plt.scatter([], [], alpha=0.3, c="blue", s=size, label=str(size) +' m$^2$')
plt.legend()
plt.show()


# script from sklearn webpage
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


# Compute confusion matrix (nothing inside)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix only color")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()



# Compute confusion matrix (values written inside)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix values inside")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

fmt = '.2f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()



# Compute confusion matrix (values written inside, nothing when 0)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("sparse confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

fmt = '.2f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if int(format(cm[i, j])) > 0:
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()




