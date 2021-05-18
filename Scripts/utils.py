# coding: utf-8
import sys, os, re
import numpy as np

# -------------------------------------------------------------------------------------------------

def parse_args():
    """
    parse and validate arguments for train_xxx.py runs
    """
    if len(sys.argv) != 2:
        raise Exception("Expecting a classifier root folder as first arg")
    if not os.path.exists(sys.argv[1]):
        raise Exception("Passed classifier root folder does not exist: '%s'" % sys.argv[1])

    root_path = sys.argv[1]
    if root_path.endswith("/") or root_path.endswith("\\"):
        root_path = root_path[0:-1]
    name = os.path.basename(root_path)

    number_of_runs = 5 # TODO

    random_seed = 300 # TODO
    # apply RANDOM_SEED
    np.random.seed(random_seed)
    
    return { 
        'classifier_root_path': root_path,
        'classifier_name': name,
        'number_of_runs': number_of_runs,
        'random_seed': random_seed
    }


# -------------------------------------------------------------------------------------------------

def load_dataset(dataset_path: str):
    """
    load a sononym classifier csv dataset
    """
    import pandas as pd

    print('Loading data...')
    df = pd.read_csv(dataset_path)

    # split label column and get feature names 
    print('Preparing data...')
    assert(df.columns[0] == "label" and df.columns[1] == "id")
    df_labels = df[df.columns[0]]
    df_ids = df[df.columns[1]]
    df_data = df[df.columns[2:]]
    feature_names = list(df.columns[2:])
    num_classes = np.max(df_labels) + 1

    print('  Dataset has %s samples with %s features and %s classes.' % 
        (len(df_data), len(df.columns) - 1, num_classes))

    return df_labels, df_ids, df_data, num_classes, feature_names


# -------------------------------------------------------------------------------------------------

def split_data_sets(df_labels, df_ids, df_data, num_classes, n_splits=1):
    """
    n_splits > 1: create n_splits statified folds
    n_splits = 1: create a single statified split with a test_size factor of 1/5

    Expects a pandas dataset as input. returns keras np.array classification sets 
    """
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
    from keras.utils import np_utils

    print("Splitting train/test data sets...")

    sss = None
    if (n_splits == 1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1/5)
    else:
        sss = StratifiedKFold(n_splits, shuffle=True)

    X_train_sets, X_test_sets, y_test_sets, y_train_sets = [], [], [], []

    for train_index, test_index in sss.split(df_data, df_labels):
        # deal with augmented samples
        train_index, test_index = split_augmented_samples(train_index, test_index, df_ids)
        
        # fetch pandas rows for current split
        X_train, X_test = df_data.iloc[train_index], df_data.iloc[test_index]
        y_train, y_test = df_labels.iloc[train_index], df_labels.iloc[test_index]
        
        # convert to np.arrays
        X_train_sets.append(np.array(X_train, dtype=np.float32).reshape((-1, X_train.shape[1])))
        X_test_sets.append(np.array(X_test, dtype=np.float32).reshape((-1, X_test.shape[1])))
        y_train_sets.append(np_utils.to_categorical(y_train, num_classes))
        y_test_sets.append(np_utils.to_categorical(y_test, num_classes))

    return X_train_sets, X_test_sets, y_test_sets, y_train_sets

# -------------------------------------------------------------------------------------------------

def split_augmented_samples(train_index: list, test_index: list, df_ids):
    """
    modify train_index and test_index splits, so that the test set contains no augmented
    samples of the train set and vice versa
    """
    is_augmented_sample_re = re.compile(r".*_aug\d+\.[a-z]+$")

    # remove all augmented test samples
    na_test_index = list(test_index)
    for ti in test_index:
        if is_augmented_sample_re.match(df_ids[ti]):
            na_test_index.remove(ti)
    
    # remove all augmented train samples
    na_train_index = list(train_index)
    for ti in train_index:
        if is_augmented_sample_re.match(df_ids[ti]):
            na_train_index.remove(ti)

    """
    # remove all test set's augmented samples from the train set 
    na_train_index = list(train_index)
    for ti in na_test_index:
        test_sample_filename = os.path.splitext(df_ids[ti])[0]
        for tj in list(na_train_index):
            if is_augmented_sample_re.match(df_ids[tj]) and \
               df_ids[tj].startswith(test_sample_filename):
                # print("Removing '%s' from train set" % df_ids[tj])
                na_train_index.remove(tj)
    """
    
    # shuffle contents
    np.random.shuffle(na_test_index)
    np.random.shuffle(na_train_index)

    return na_train_index, na_test_index

# -------------------------------------------------------------------------------------------------

def split_descriptors(X): 
    """
    Reshape given (samples, features) descriptor data into 
      a 2d spectrum time series data with shape (samples, time_series, band_values) 
      and the remaining 1d statistical features into shape (samples, statistics)
    """

    # expected descriptor data layout (time_series x feature_count)
    time_series_length = 48
    feature_count = 35
    if (len(X[0]) != time_series_length*feature_count):
        raise Exception("Unexpected dataset feature count")
        
    # time series / statistics layout
    spectral_feature_count = 14
    statistics_feature_count = feature_count - spectral_feature_count

    # convert to 2d input
    X = X.reshape((-1, feature_count, time_series_length))

    # input data shape for CNNs/LSTMs is: (batch_size, time_steps, data_dim)
    X = np.swapaxes(X, 1, 2)

    # splice away statistics from data
    def split_time_series(X):
        time_series = np.zeros((X.shape[0], time_series_length, spectral_feature_count))
        for b in range(len(X)):
            for t in range(len(X[b])):
                time_series[b][t] = X[b][t][0:spectral_feature_count]
        # (samples, time_series, band_features)
        return time_series

    # splice away the spectrum from data
    def split_statistics(X):
        statistics = np.zeros((X.shape[0], time_series_length, statistics_feature_count))
        for b in range(len(X)):
            for t in range(len(X[b])):
                statistics[b][t] = X[b][t][spectral_feature_count:feature_count]
        shape = statistics.shape
        # flatten from 3D to 2D (samples, statistics)
        return statistics.reshape(shape[0], shape[1] * shape[2])

    return split_time_series(X), split_statistics(X)

# -------------------------------------------------------------------------------------------------

def plot_losses(train_loss: list, val_loss: list):
    """
    plot train and validation loss values
    """
    import matplotlib.pyplot as plt

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


# -------------------------------------------------------------------------------------------------

def plot_confusion_matrix(cm,
                          target_names=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import itertools
    import matplotlib.pyplot as plt

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
