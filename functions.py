import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score

#Taken from fomlads library
def import_for_classification(
        ifname, input_cols=None, target_col=None, classes=None):
    """
    parameters
    ----------
    ifname -- filename/path of data file.
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    classes -- list of the classes to plot

    returns
    -------
    inputs -- the data as a numpy.array object  
    targets -- the targets as a 1d numpy array of class ids
    input_cols -- ordered list of input column names
    classes -- ordered list of classes
    """
    # if no file name is provided then use synthetic data
    dataframe = pd.read_csv(ifname)
    print("dataframe.columns = %r" % (dataframe.columns,) )
    N = dataframe.shape[0]
    # if no target name is supplied we assume it is the last colunmn in the 
    # data file
    if target_col is None:
        target_col = dataframe.columns[-1]
        potential_inputs = dataframe.columns[:-1]
    else:
        potential_inputs = list(dataframe.columns)
        # target data should not be part of the inputs
        potential_inputs.remove(target_col)
    # if no input names are supplied then use them all
    if input_cols is None:
        input_cols = potential_inputs
    print("input_cols = %r" % (input_cols,))
    # if no classes are specified use all in the dataset
    if classes is None:
        # get the class values as a pandas Series object
        class_values = dataframe[target_col]
        classes = class_values.unique()
    else:
        # construct a 1d array of the rows to keep
        to_keep = np.zeros(N,dtype=bool)
        for class_name in classes:
            to_keep |= (dataframe[target_col] == class_name)
        # now keep only these rows
        dataframe = dataframe[to_keep]
        # there are a different number of dat items now
        N = dataframe.shape[0]
        # get the class values as a pandas Series object
        class_values = dataframe[target_col]
    print("classes = %r" % (classes,))
    # We now want to translate classes to targets, but this depends on our 
    # encoding. For now we will perform a simple encoding from class to integer.
    targets = np.empty(N)
    for class_id, class_name in enumerate(classes):
        is_class = (class_values == class_name)
        targets[is_class] = class_id
    #print("targets = %r" % (targets,))

    # We're going to assume that all our inputs are real numbers (or can be
    # represented as such), so we'll convert all these columns to a 2d numpy
    # array object
    inputs = dataframe[input_cols].values
    return inputs, targets, input_cols, classes

#Taken from fomlads library
def train_and_test_filter(N, test_fraction=None):
    """
    Randomly generates filters for a train/test split for data of size N.

    parameters
    ----------
    N - the dataset size
    test_fraction - a fraction (between 0 and 1) specifying the proportion of
        the data to use as test data.

    returns
    -------
    train_filter - a boolean vector of length N, where if ith element is
        True if the ith data-point belongs to the training set, and False if
        otherwise
    test_filter - a boolean vector of length N, where if ith element is
        True if the ith data-point belongs to the testing set, and False if
        otherwise
    """
    if test_fraction is None:
        test_fraction = 0.5
    p = [test_fraction,(1-test_fraction)]
    train_filter = np.random.choice([False,True],size=N, p=p)
    test_filter = np.invert(train_filter)
    return train_filter, test_filter

#Taken from fomlads library
def train_and_test_partition(inputs, targets, train_filter, test_filter, use_fixed_filter=True):
    """
    Splits a data matrix (or design matrix) and associated targets into train
    and test parts.

    parameters
    ----------
    inputs - a 2d numpy array whose rows are the datapoints, or can be a design
        matric, where rows are the feature vectors for data points.
    targets - a 1d numpy array whose elements are the targets.
    train_filter - A list (or 1d array) of N booleans, where N is the number of
        data points. If the ith element is true then the ith data point will be
        added to the training data.
    test_filter - (like train_filter) but specifying the test points.
    use_fixed_filter - a Boolean, specifying whether to use the fixed filters provided as csv files in this folder.

    returns
    -------     
    train_inputs - the training input matrix
    train_targets - the training targets
    test_inputs - the test input matrix
    test_targets - the test targtets
    """
    if use_fixed_filter == True:
        train_filter = pd.read_csv('train_filter_fixed.csv')['0'].to_numpy()
        test_filter = pd.read_csv('test_filter_fixed.csv')['0'].to_numpy()
    # get the indices of the train and test portion
    univariate = (len(inputs.shape) == 1)
    if univariate:
        # if inputs is a sequence of scalars we should reshape into a matrix
        inputs = inputs.reshape((inputs.size,1))
    train_inputs = inputs[train_filter,:]
    test_inputs = inputs[test_filter,:]
    train_targets = targets[train_filter]
    test_targets = targets[test_filter]
    if univariate:
        train_inputs = train_inputs.flatten()
        test_inputs = test_inputs.flatten()
    return train_inputs, train_targets, test_inputs, test_targets

#Taken from exercise notebook
def fold_train_test_filter(inputs, num_folds=10):
    """
    Randomly generates filters for a train/test split for data of size N.
    
    parameters
    ----------
    inputs - a 2d numpy array, a training data set, whose rows are the datapoints, where rows are the feature vectors for data
    points.
    num_folds - the number of folds for cross validation
    
    returns
    ----------
    folds - a list of tuples, each corresponding to a 'fold', which are pairs of training and validation filters. The validation 
    filter for a given fold is a Boolean vector with length equal to the number of rows of the inputs array, where if the ith 
    element is equal to the fold number, the ith data-point belongs to the validation set, and to the training set otherwise. The 
    training filter is the same as the validation filter with the sets reversed.
    
    """
    partitions = np.random.randint(0,num_folds,inputs.shape[0])
    folds = []
    for f in range(num_folds):
        # data points with partition equal to fold id are 
        # validation points in fold f
        fold_valid_filter = (partitions == f)
        # all other points are training
        fold_train_filter = ~fold_valid_filter
        # store the fold
        folds.append((fold_train_filter, fold_valid_filter))
    return folds

#Adapted from exercise notebook
def grid_search_cross_val_svm(folds, inputs, targets, C_range, gamma_range):
    """
    Performs grid search with cross validation with given fold filters, inputs, targets and C and gamma hyperparameter ranges
    for SVM model with (default) rbf kernel.
    
    parameters
    ----------
    folds - a list of tuples, each corresponding to a 'fold', which are pairs of training and validation filters. The validation 
    filter for a given fold is a Boolean vector with length equal to the number of rows of the inputs array, where if the ith 
    element is equal to the fold number, the ith data-point belongs to the validation set, and to the training set otherwise. The 
    training filter is the 'reverse' of this.
    inputs - a 2d numpy array, a training data set, whose rows are the datapoints, where rows are the feature vectors for data
    points.
    targets - a 1d numpy array whose elements are the targets.
    C_range - a list of C hyperparameter values to include in grid search
    gamma_range - a list of gamma hyperparameter values to include in grid search
    
    returns
    ----------
    scores - a list of cross validation results in which each element is itself a list consisting of [c, g, f, f1] where:
    c - the C hyperparameter evaluated
    g - the gamma hyperparameter evaluated
    f - the fold number
    f1 - the macro f1 score for the model trained with these hyperparameters for one permutation of this fold.
    """
    # define regularisation parameters to evaluate
    # lambdas = np.logspace(-17,-8,11)
    # need to store a result for every fold for every lambda
    scores = []

    for c in C_range:
        for g in gamma_range:
            for f, (fold_train_filter, fold_valid_filter) in enumerate(folds):
                inputs_train, targets_train, inputs_valid, targets_valid = train_and_test_partition(
                inputs, targets, fold_train_filter, fold_valid_filter, use_fixed_filter=False)
                svc = SVC(C=c, verbose=1, gamma=g)
                svc.fit(inputs_train, targets_train)
                score = svc.score(inputs_train, targets_train)
                targets_pred = svc.predict(inputs_valid)
                f1 = f1_score(targets_valid, targets_pred, average='macro')
                #av_fold_score = sum(fold_scores)/len(fold_scores)
                scores.append([c, g, f, f1])
        
    return scores