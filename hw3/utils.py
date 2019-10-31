import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import keras

def wine_preprocess():
    df = pd.read_csv('winequality-white.csv',sep=';')

    #categorize wine quality
    bins = (2,6.5,9)
    group_names = [0,1]
    categories = pd.cut(df['quality'], bins, labels = group_names)
    df['quality'] = categories

    ## Divide into features and output
    X = df.loc[:,'fixed acidity':'alcohol']
    y = np.array(df.loc[:,'quality'])

    ## splitting into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    ## Standardize the data before training the model
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def forest_preprocess(keras_training=False):
    ##===========================##
    ##   Checking Missing Value  ##
    ##===========================##

    df = pd.read_csv('forest_train.csv')
    print(df.columns[:-1])

    ## Divide into features and output
    X= df.loc[:, 'Elevation':'Soil_Type40']
    y= np.array(df.loc[:, 'Cover_Type'])
    y= y- 1


    ## splitting into training and testing
    size = 10
    X_train, X_test, y_train, y_test= train_test_split(X, y, stratify=y, test_size=0.2)
    X_train_num = X_train.iloc[:, :size]
    X_train_cat = X_train.iloc[:, size:]

    X_test_num = X_test.iloc[:, :size]
    X_test_cat = X_test.iloc[:, size:]

    scaler = StandardScaler()
    scaler.fit(X_train_num)
    X_train_num = scaler.transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)

    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    if keras_training:
        # convert class vectors to binary class matrices
        num_classes = 7
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    print('finished preparing the data')

    return X_train,X_test,y_train,y_test

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring='recall'):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    if scoring != 'recall':
        print('Using accuracy score as scoring metrics')
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring='accuracy')
    else:
        print('Using recall score as scoring metrics')
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring='recall')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


