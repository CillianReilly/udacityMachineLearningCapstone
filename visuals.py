###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import numpy as np
import matplotlib.patches as mpatches
import sklearn.learning_curve as curves
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import ShuffleSplit, train_test_split
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score

def ModelLearning(X, y):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = pl.figure(figsize=(10,7))

    # Create three different models based on max_depth
    for k, depth in enumerate([3,5,8,10]):
        
        # Create a Decision tree regressor at max_depth = depth
        regressor = DecisionTreeClassifier(max_depth = depth)

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(regressor, X, y, \
            cv = cv, train_sizes = train_sizes, scoring = 'r2')
        
        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)

        # Subplot the learning curve 
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')
        
        # Labels
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])
    
    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Decision Tree Classifier Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()


def ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, test_scores = curves.validation_curve(DecisionTreeClassifier(), X, y, \
        param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'r2')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Classifier Complexity Performance')
    pl.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
    pl.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
    pl.fill_between(max_depth, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    pl.fill_between(max_depth, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')
    
    # Visual aesthetics
    pl.legend(loc = 'lower right')
    pl.xlabel('Maximum Depth')
    pl.ylabel('Score')
    pl.ylim([-0.05,1.05])
    pl.show()


def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate(['driverId','constructorId']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 842, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0,100))
        ax.set_yticks([0,25,50,75,100])
        ax.set_yticklabels([0,25,50,75,100])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Discrete Race Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Discrete Race Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1, accuracy2, f12):
	"""
	Visualization code to display results of various learners.

	inputs:
	  - results:a dictionary of results of supervised learners
	  - accuracy: The score for the naive predictor
	  - f1: The score for the naive predictor
	"""

	# Create figure
	fig, ax = pl.subplots(2, 3, figsize = (100,70))

	# Constants
	bar_width = 0.15
	colors = ['#A00000','#00A0A0','#00A000','#FB8B24','#D90368']

	# Super loop to plot four panels of data
	for k, learner in enumerate(results.keys()):
		for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
			for i in np.arange(3):
			
				# Creative plot code
				ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
				ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
				ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
				ax[j//3, j%3].set_xlabel("Training Set Size")
				ax[j//3, j%3].set_xlim((-0.1, 3.0))

	# Add unique y-labels
	ax[0, 0].set_ylabel("Time (in seconds)")
	ax[0, 1].set_ylabel("Accuracy Score")
	ax[0, 2].set_ylabel("F-score")
	ax[1, 0].set_ylabel("Time (in seconds)")
	ax[1, 1].set_ylabel("Accuracy Score")
	ax[1, 2].set_ylabel("F-score")

	# Add titles
	ax[0, 0].set_title("Model Training")
	ax[0, 1].set_title("Accuracy Score on Training Subset")
	ax[0, 2].set_title("F-score on Training Subset")
	ax[1, 0].set_title("Model Predicting")
	ax[1, 1].set_title("Accuracy Score on Testing Set")
	ax[1, 2].set_title("F-score on Testing Set")

	# Add horizontal lines for naive predictors
	ax[0, 1].axhline(y = accuracy, xmin = 0, xmax = 1, linewidth = 5, color = 'k', linestyle = 'dashed')
	ax[1, 1].axhline(y = accuracy, xmin = 0, xmax = 1, linewidth = 5, color = 'k', linestyle = 'dashed')
	ax[0, 2].axhline(y = f1, xmin = 0, xmax = 1, linewidth = 5, color = 'k', linestyle = 'dashed')
	ax[1, 2].axhline(y = f1, xmin = 0, xmax = 1, linewidth = 5, color = 'k', linestyle = 'dashed')
	
	ax[0, 1].axhline(y = accuracy2, xmin = 0, xmax = 1, linewidth = 5, color = 'k', linestyle = 'dashed')
	ax[1, 1].axhline(y = accuracy2, xmin = 0, xmax = 1, linewidth = 5, color = 'k', linestyle = 'dashed')
	ax[0, 2].axhline(y = f12, xmin = 0, xmax = 1, linewidth = 5, color = 'k', linestyle = 'dashed')
	ax[1, 2].axhline(y = f12, xmin = 0, xmax = 1, linewidth = 5, color = 'k', linestyle = 'dashed')

	# Set y-limits for score panels
	ax[0, 1].set_ylim((0.90, 1))
	ax[0, 2].set_ylim((0.90, 1))
	ax[1, 1].set_ylim((0.90, 1))
	ax[1, 2].set_ylim((0.90, 1))

	# Create patches for the legend
	patches = []
	for i, learner in enumerate(results.keys()):
		patches.append(mpatches.Patch(color = colors[i], label = learner))
	pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
			loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 32)

	#Aesthetics
	pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 32, y = 1.10)
	#pl.tight_layout()
	pl.show()

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Create the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()  
