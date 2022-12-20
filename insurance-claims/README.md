# Car Insurance Claims

Dataset source: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification

## Dataset

The dataset contains a total of 44 columns. One of those columns is a unique identifier, one
is a boolean value which specifies whether the customer filed a claim. After dropping these two
columns, we recieve a dataset of 42 attributes. The dataset contains approximately 59'000 rows.
The dataset is highly imbalanced, of the roughly 59'000 records in the dataset, only just below
4'000 records resulted in a claim being filed. This is unsurprising, as the insurance companies
would have gone bankrupt long ago, if the data were balanced, but also undesirable in terms of
machine learning and data processing.

The dataset contains numerical and categorical attributes, as well as binary flags.

## Preprocessing

Before we start creating models, the dataset has to be preprocessed. During preprocessing,
categorical attributes are one-hot encoded, binary values are converted from strings to
`0`/`1`, and some columns are dropped. We drop the unique identifier, car and engine model,
as those attributes contain a lot of different distinct values, and we do not care as much
about the name of the model, as much as we care about the characteristics and technical
specifications of the machines. Lastly, we also drop the `steering_type` column, which
contains information about what kind of power-steering, if any, is used in the car.
The column is somewhat redundant, as we already have the `is_power_steering` boolean flag.
Dropping the categorical column for a yes/no flag might result in a loss of information,
but the dataset already contains a lot of attributes and we do not want to add more.

The preprocessed dataset contains a total of 73 attributes.

## Training And Model Evaluation

We will perform K-fold crossvalidation for all classifiers, with `K=3` due to performance
reasons.

The metrics used to compare the models will be F1-score, accuracy, and we will even construct
a `collections.Counter` from the predictions to have a closer look at what predictions the models
are making (if the use of the `Counter` doesn't make sense now, it will make sense soon).

Since we're splitting the dataset and training the model multiple times on the same dataset, we will
compute the average F1-score and the average accuracy for all three runs and compare
the classifier configurations by their average performance. `Counter`s will be summed up.

## Classifiers

Five classifiers were used and compared in the project. These classifiers include

* Decision Tree
* Random Forest
* XGBoost decision trees/forests
* Support Vector Machine
* Multi-layer Perceptron

## Scaled Data

First, we will try to trained the classifiers on scaled data. The data is scaled using an
`sklearn.preprocessing.StandardScaler`. There are no attempts to classify unscaled data,
all training and classification is performed on rescaled datasets.

### Decision Tree

We construct multiple decision trees with their depths equal to multiples of five.
From the table, we can see that whilst the F-score increases for deeper trees,
the accuracy actually decreases. Due to the imbalance present in the data, a classifier
which classifies everything as a non-claim will always perform very well, and there's a lot
to lose when training our own classifiers.

| max_depth | f1 score | accuracy | counter |
| --- | --- | --- | --- |
| 5 | 0.003 | 0.936 | {0: 58573, 1: 19} |
| 10 | 0.014 | 0.933 | {0: 58342, 1: 250} |
| 15 | 0.041 | 0.919 | {0: 57410, 1: 1182} |
| 20 | 0.063 | 0.899 | {0: 56014, 1: 2578} |
| 25 | 0.077 | 0.884 | {0: 54965, 1: 3627} |
| 30 | 0.083 | 0.874 | {0: 54310, 1: 4282} |
| 35 | 0.086 | 0.872 | {0: 54123, 1: 4469} |
| 40 | 0.087 | 0.872 | {0: 54110, 1: 4482} |
| 45 | 0.086 | 0.872 | {0: 54132, 1: 4460} |
| 50 | 0.089 | 0.872 | {0: 54075, 1: 4517} |
| 55 | 0.089 | 0.872 | {0: 54096, 1: 4496} |
| 60 | 0.092 | 0.873 | {0: 54117, 1: 4475} |
| 65 | 0.088 | 0.872 | {0: 54134, 1: 4458} |
| 70 | 0.091 | 0.872 | {0: 54082, 1: 4510} |

Model performance doesn't seem to change much beyond `max_depth=30`.

### Random Forest

We set the tree depth to levels that proved reasonable for singular trees,
and experiment with number of trees in the forest. Yet again,
the classifiers which classify (almost) everthing as a non-claim are more accurate,
although they have a lower F-score.

In terms of F-score, a decision forest does not outperform a single tree.

Tree depth seems to have less of an effect than the number of estimators. Increasing
the number of trees in the forest causes the classifier to classify more and more
records as non-claims, decreasing F-score and increasing accuracy.

| max_depth | n_estimators | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- |
| 45 | 3 | 0.055 | 0.913 | {0: 56924, 1: 1668} |
| 45 | 5 | 0.041 | 0.920 | {0: 57430, 1: 1162} |
| 45 | 10 | 0.016 | 0.929 | {0: 58136, 1: 456} |
| 45 | 100 | 0.017 | 0.931 | {0: 58241, 1: 351} |
| 50 | 3 | 0.048 | 0.911 | {0: 56884, 1: 1708} |
| 50 | 5 | 0.038 | 0.920 | {0: 57446, 1: 1146} |
| 50 | 10 | 0.014 | 0.930 | {0: 58154, 1: 438} |
| 50 | 100 | 0.018 | 0.932 | {0: 58257, 1: 335} |
| 55 | 3 | 0.055 | 0.911 | {0: 56801, 1: 1791} |
| 55 | 5 | 0.042 | 0.920 | {0: 57463, 1: 1129} |
| 55 | 10 | 0.019 | 0.929 | {0: 58111, 1: 481} |
| 55 | 100 | 0.016 | 0.931 | {0: 58257, 1: 335} |

### XGBoost

We set the objective to `binary:hinge` to obtain binary results.

We experiment with the number of trees in the forest, `alpha`, `lambda`
and `gamma` training parameters. Maximum depth is constant across all sets and is
set to `55`.

| alpha | objective | lambda | eta | booster | num_parallel_tree | max_depth | gamma | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 55 | - | 0.077 | 0.892 | {0: 55481, 1: 3111} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 55 | 0.500 | 0.069 | 0.891 | {0: 55448, 1: 3144} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 55 | 1.000 | 0.069 | 0.891 | {0: 55448, 1: 3144} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 55 | 1.500 | 0.075 | 0.890 | {0: 55393, 1: 3199} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 1 | 55 | - | 0.065 | 0.902 | {0: 56168, 1: 2424} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 55 | - | 0.082 | 0.894 | {0: 55602, 1: 2990} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 55 | 0.500 | 0.071 | 0.890 | {0: 55429, 1: 3163} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 55 | 1.000 | 0.071 | 0.890 | {0: 55429, 1: 3163} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 55 | 1.500 | 0.076 | 0.895 | {0: 55706, 1: 2886} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 3 | 55 | - | 0.062 | 0.901 | {0: 56125, 1: 2467} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 55 | - | 0.076 | 0.893 | {0: 55573, 1: 3019} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 55 | 0.500 | 0.068 | 0.892 | {0: 55529, 1: 3063} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 55 | 1.000 | 0.068 | 0.892 | {0: 55529, 1: 3063} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 55 | 1.500 | 0.067 | 0.889 | {0: 55397, 1: 3195} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 10 | 55 | - | 0.065 | 0.902 | {0: 56168, 1: 2424} |

XGBoost outperforms Random Forests in terms of F-score, but it falls short of Decision Trees. It is also
worse in terms of accuracy.

Increasing `alpha` and `lambda` beyond their default values causes the model to classify more records
as non-claims. Increasing the number of trees in the forest results in a subtle drop in F-score.

### SVC

The radial basis function and the linear function kernel live in a utopian universe where cyclists
actually abide by the traffic laws (or better yet, do not exist at all), and people use their turn signals.
But alas, the classifier which only predicts zeroes is the most accurate one we've had so far.

Since the sigmoid kernel finally predicts a couple claims, we can also try increasing the `C` parameter.
Increasing `C` leads to a higher F-score and more claim predictions.

| degree | kernel | gamma | C | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- | --- | --- |
| - | rbf | scale | 1.000 | 0.000 | 0.936 | {0: 58592} |
| - | linear | scale | 1.000 | 0.000 | 0.936 | {0: 58592} |
| 3 | poly | scale | 1.000 | 0.001 | 0.936 | {0: 58572, 1: 20} |
| - | sigmoid | scale | 1.000 | 0.055 | 0.898 | {0: 55990, 1: 2602} |
| - | sigmoid | scale | 10.000 | 0.069 | 0.884 | {0: 55042, 1: 3550} |

### Multi-layer Perceptron

The neural network appears to classify almost everything as a non-claim. A rather surprising result
is that we get a better classifier by increasing the number of hidden layers. Or, at least the F-score
improves and the classifier starts classifying some records as claims.

RELU appears to perform better than sigmoid, at least in terms of F-score.

| solver | activation | hidden_layer_sizes | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- | --- |
| adam | relu | (146,) | 0.003 | 0.935 | {0: 58537, 1: 55} |
| adam | logistic | (146,) | 0.002 | 0.936 | {0: 58579, 1: 13} |
| lbfgs | relu | (146,) | 0.006 | 0.935 | {0: 58504, 1: 88} |
| lbfgs | logistic | (146,) | 0.000 | 0.936 | {0: 58592} |
| adam | relu | (146, 146) | 0.033 | 0.925 | {0: 57771, 1: 821} |
| adam | logistic | (146, 146) | 0.010 | 0.933 | {0: 58394, 1: 198} |
| lbfgs | relu | (146, 146) | 0.009 | 0.933 | {0: 58367, 1: 225} |
| lbfgs | logistic | (146, 146) | 0.000 | 0.936 | {0: 58592} |
| adam | relu | (219,) | 0.003 | 0.935 | {0: 58519, 1: 73} |
| adam | logistic | (219,) | 0.001 | 0.936 | {0: 58581, 1: 11} |
| lbfgs | relu | (219,) | 0.005 | 0.934 | {0: 58481, 1: 111} |
| lbfgs | logistic | (219,) | 0.000 | 0.936 | {0: 58592} |

## Data enhancement
