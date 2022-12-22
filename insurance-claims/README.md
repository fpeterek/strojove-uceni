# Car Insurance Claims

Dataset source: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification

## Running the script

```sh
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

python classification
```

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

* DecisionTree
* RandomForest
* XGBoost decision trees/forests
* Support Vector Machine
* Multi-layer Perceptron

# Experiments

## Scaled Data

First, we will try to train the classifiers on scaled data. The data is scaled using an
`sklearn.preprocessing.StandardScaler`. There are no attempts to classify unscaled data,
all training and classification is performed on rescaled datasets.

### DecisionTree

We construct multiple decision trees with their depths equal to multiples of five.
From the table, we can see that whilst the F-score increases for deeper trees,
the accuracy actually decreases. Due to the imbalance present in the data, a classifier
which classifies everything as a non-claim will always perform very well, and there's plenty
of opportunities to lose accuracy when training custom classifiers compared to just classifying
everything as a non-claim.

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

### RandomForest

We set the tree depth to levels that proved reasonable for singular trees,
and experiment with number of trees in the forest. Yet again,
the classifiers which classify (almost) everthing as a non-claim are more accurate,
although they have a lower F-score.

In terms of F-score, a decision forest does not outperform a single tree.

Tree depth seems to have less of an effect than the number of estimators, at least in our
set with a limited number of configurations. Increasing the number of trees in the forest
causes the classifier to classify more and more records as non-claims, decreasing F-score
and increasing accuracy.

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

We experiment with the number of trees in the forest and `alpha`, `lambda`
and `gamma` training parameters. Maximum depth is constant across all configurations and is
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

XGBoost outperforms RandomForest in terms of F-score, but it falls short of DecisionTrees. It is also
worse in terms of accuracy.

Increasing `alpha` and `lambda` beyond their default values causes the model to classify more records
as non-claims. Increasing the number of trees in the forest results in a subtle drop in F-score.

### SVC

The radial basis function and the linear function kernel live in a utopian universe where cyclists
actually abide by the traffic laws (or better yet, do not exist at all), and drivers use their turn signals.
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
is that we get a better classifier by increasing the number of hidden layers. Or at least the F-score
improves and the classifier starts classifying some records as claims.

Choosing ReLU as the activation function results in higher F1-score but equivalent accuracy compared
to the Sigmoid function.

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

As our next experiment, we try to enhance the data using the SMOTE algorithm to obtain a balanced training
set. The enhanced dataset is then scaled, as in the previous experiment.

### DecisionTree

The first four trees have a relatively large F-score, but their accuracy is low. They also
classify way too many records as claims. Performance stops to differ beyond maximum depth
of 25 and more. The classifier still doesn't perform well.

| max_depth | f1 score | accuracy | counter |
| --- | --- | --- | --- |
| 5 | 0.148 | 0.589 | {1: 24493, 0: 34099} |
| 10 | 0.144 | 0.657 | {1: 19768, 0: 38824} |
| 15 | 0.135 | 0.735 | {0: 44408, 1: 14184} |
| 20 | 0.122 | 0.776 | {0: 47371, 1: 11221} |
| 25 | 0.096 | 0.864 | {0: 53527, 1: 5065} |
| 30 | 0.093 | 0.869 | {0: 53894, 1: 4698} |
| 35 | 0.091 | 0.864 | {0: 53587, 1: 5005} |
| 40 | 0.093 | 0.865 | {0: 53630, 1: 4962} |
| 45 | 0.091 | 0.864 | {0: 53555, 1: 5037} |
| 50 | 0.091 | 0.864 | {0: 53592, 1: 5000} |
| 55 | 0.093 | 0.865 | {0: 53639, 1: 4953} |
| 60 | 0.090 | 0.865 | {0: 53619, 1: 4973} |
| 65 | 0.090 | 0.865 | {0: 53670, 1: 4922} |
| 70 | 0.088 | 0.865 | {0: 53691, 1: 4901} |

### RandomForest

Unlike in the first experiment on the original dataset, RandomForest manages to outperform
a single DecisionTree in terms of F-score (if we disregard the first two very inaccurate trees).

Increasing the number of estimators appears to decrease the amount of records classified as claims.

| max_depth | n_estimators | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- |
| 45 | 3 | 0.113 | 0.818 | {0: 50301, 1: 8291} |
| 45 | 5 | 0.111 | 0.832 | {0: 51278, 1: 7314} |
| 45 | 10 | 0.103 | 0.858 | {0: 53067, 1: 5525} |
| 45 | 100 | 0.103 | 0.858 | {0: 53061, 1: 5531} |
| 50 | 3 | 0.111 | 0.815 | {0: 50142, 1: 8450} |
| 50 | 5 | 0.106 | 0.833 | {0: 51401, 1: 7191} |
| 50 | 10 | 0.103 | 0.861 | {0: 53287, 1: 5305} |
| 50 | 100 | 0.102 | 0.861 | {0: 53265, 1: 5327} |
| 55 | 3 | 0.113 | 0.823 | {0: 50668, 1: 7924} |
| 55 | 5 | 0.106 | 0.836 | {1: 7019, 0: 51573} |
| 55 | 10 | 0.094 | 0.864 | {0: 53519, 1: 5073} |
| 55 | 100 | 0.102 | 0.862 | {0: 53325, 1: 5267} |

### XGBoost

XGBoost seems to perform worse in terms of accuracy and F-score than RandomForest, however, the number
of records classified as claims is closer to the actual number of claims, than in the case of RandomForest.

Changing the `gamma` parameter seems to have no effect. Increasing `alpha` and `lambda` results in fewer
records being classified as claims, slightly increasing accuracy. Increasing the number of trees in the forest
appears to have little to no effect.

| alpha | objective | lambda | eta | booster | num_parallel_tree | max_depth | gamma | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 55 | - | 0.092 | 0.861 | {0: 53389, 1: 5203} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 55 | 0.500 | 0.086 | 0.863 | {0: 53530, 1: 5062} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 55 | 1.000 | 0.088 | 0.861 | {0: 53411, 1: 5181} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 55 | 1.500 | 0.085 | 0.865 | {0: 53683, 1: 4909} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 1 | 55 | - | 0.091 | 0.876 | {0: 54353, 1: 4239} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 55 | - | 0.093 | 0.858 | {0: 53141, 1: 5451} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 55 | 0.500 | 0.087 | 0.861 | {0: 53421, 1: 5171} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 55 | 1.000 | 0.086 | 0.861 | {0: 53424, 1: 5168} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 55 | 1.500 | 0.096 | 0.865 | {0: 53618, 1: 4974} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 3 | 55 | - | 0.085 | 0.879 | {0: 54570, 1: 4022} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 55 | - | 0.091 | 0.860 | {0: 53336, 1: 5256} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 55 | 0.500 | 0.087 | 0.860 | {0: 53358, 1: 5234} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 55 | 1.000 | 0.086 | 0.860 | {0: 53350, 1: 5242} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 55 | 1.500 | 0.089 | 0.864 | {0: 53576, 1: 5016} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 10 | 55 | - | 0.092 | 0.876 | {0: 54346, 1: 4246} |

### SVC

The Support Vector Machine performs horribly regardless of choice of kernel or training parameters.
It's only slightly more accurate than marking every second record as a claim.

| degree | kernel | gamma | C | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- | --- | --- |
| - | rbf | scale | 1.000 | 0.148 | 0.534 | {0: 30332, 1: 28260} |
| - | linear | scale | 1.000 | 0.147 | 0.555 | {0: 31733, 1: 26859} |
| 3 | poly | scale | 1.000 | 0.150 | 0.550 | {0: 31324, 1: 27268} |
| - | sigmoid | scale | 1.000 | 0.119 | 0.514 | {0: 30050, 1: 28542} |
| - | sigmoid | scale | 10.000 | 0.122 | 0.517 | {0: 30142, 1: 28450} |

### Multi-layer Perceptron

A Neural Network based classifier doesn't perform well, either.

The adam optimizer reaches a higher accuracy but a lower F-score than the lbfgs optimizer.
Yet again, increasing the number of hidden layers improves the network.

As for the activation function, ReLU seems to work slightly better than Sigmoid.

| solver | activation | hidden_layer_sizes | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- | --- |
| adam | relu | (146,) | 0.144 | 0.693 | {1: 17230, 0: 41362} |
| adam | logistic | (146,) | 0.143 | 0.673 | {1: 18634, 0: 39958} |
| lbfgs | relu | (146,) | 0.147 | 0.627 | {0: 36691, 1: 21901} |
| lbfgs | logistic | (146,) | 0.149 | 0.608 | {0: 35334, 1: 23258} |
| adam | relu | (146, 146) | 0.126 | 0.764 | {0: 46496, 1: 12096} |
| adam | logistic | (146, 146) | 0.140 | 0.695 | {1: 17035, 0: 41557} |
| lbfgs | relu | (146, 146) | 0.149 | 0.647 | {0: 38055, 1: 20537} |
| lbfgs | logistic | (146, 146) | 0.153 | 0.582 | {0: 33415, 1: 25177} |
| adam | relu | (219,) | 0.140 | 0.662 | {1: 19288, 0: 39304} |
| adam | logistic | (219,) | 0.142 | 0.676 | {1: 18401, 0: 40191} |
| lbfgs | relu | (219,) | 0.146 | 0.630 | {0: 36944, 1: 21648} |
| lbfgs | logistic | (219,) | 0.152 | 0.600 | {0: 34712, 1: 23880} |

## Dimension Reduction

In our last experiment, we try to perfrom dimension reduction using Principle Component Analysis.
Dimension reduction is performed after data enhancement, but before scaling.

The dataset is reduced to 20 dimensions. Due to time/performance constraints, no attempts were made
to reduce the dataset to other sizes.

### DecisionTree

Accuracy increases and F1-score decreases when we try to increase tree depth. However, the classifier
doesn't perform very well.

| max_depth | f1 score | accuracy | counter |
| --- | --- | --- | --- |
| 5 | 0.139 | 0.504 | {1: 30025, 0: 28567} |
| 10 | 0.136 | 0.518 | {1: 28944, 0: 29648} |
| 15 | 0.127 | 0.583 | {1: 24255, 0: 34337} |
| 20 | 0.124 | 0.622 | {0: 37046, 1: 21546} |

### RandomForest

RandomForest, once again, performs better than DecisionTree. Unfortunately, that doesn't mean much,
as the bar was already set very low.

Increasing depth makes the model more accurate. Increasing the number of trees results in fewer records
being classified as claims.

| max_depth | n_estimators | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- |
| 10 | 3 | 0.140 | 0.536 | {1: 27887, 0: 30705} |
| 10 | 5 | 0.140 | 0.553 | {0: 31909, 1: 26683} |
| 10 | 10 | 0.140 | 0.564 | {0: 32622, 1: 25970} |
| 10 | 100 | 0.141 | 0.570 | {0: 32978, 1: 25614} |
| 15 | 3 | 0.128 | 0.615 | {0: 36473, 1: 22119} |
| 15 | 5 | 0.133 | 0.618 | {0: 36551, 1: 22041} |
| 15 | 10 | 0.134 | 0.633 | {0: 37529, 1: 21063} |
| 15 | 100 | 0.132 | 0.639 | {0: 37968, 1: 20624} |
| 20 | 3 | 0.126 | 0.657 | {0: 39366, 1: 19226} |
| 20 | 5 | 0.122 | 0.685 | {1: 17253, 0: 41339} |
| 20 | 10 | 0.125 | 0.683 | {0: 41119, 1: 17473} |
| 20 | 100 | 0.126 | 0.695 | {0: 41881, 1: 16711} |

### XGBoost

Finally, XGBoost manages to outperform a RandomForest. Whilst the F1-score is slightly lower, the accuracy
is higher than that of the RandomForest. Still, the models aren't very accurate.

Deeper trees perform better in terms of accuracy and only slightly worse in terms of F1-score, however,
the number of trees in the forest seems to have little to no noticeable effect. Increasing `alpha` and
`lambda` results in a slightly better model. Increasing `gamma` has little to no effect.

XGB based models perform the best of all tested classifiers in terms of accuracy on the dataset
with a reduced number of dimensions.

| alpha | objective | lambda | eta | booster | num_parallel_tree | max_depth | gamma | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 10 | - | 0.125 | 0.658 | {0: 39418, 1: 19174} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 10 | 0.500 | 0.123 | 0.663 | {0: 39794, 1: 18798} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 10 | 1.000 | 0.128 | 0.651 | {1: 19702, 0: 38890} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 10 | 1.500 | 0.124 | 0.663 | {0: 39800, 1: 18792} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 1 | 10 | - | 0.125 | 0.670 | {1: 18342, 0: 40250} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 10 | - | 0.125 | 0.662 | {1: 18872, 0: 39720} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 10 | 0.500 | 0.123 | 0.666 | {0: 40053, 1: 18539} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 10 | 1.000 | 0.122 | 0.671 | {0: 40386, 1: 18206} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 10 | 1.500 | 0.126 | 0.650 | {1: 19684, 0: 38908} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 3 | 10 | - | 0.122 | 0.674 | {0: 40569, 1: 18023} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 10 | - | 0.123 | 0.660 | {1: 18948, 0: 39644} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 10 | 0.500 | 0.122 | 0.662 | {0: 39744, 1: 18848} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 10 | 1.000 | 0.126 | 0.643 | {0: 38396, 1: 20196} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 10 | 1.500 | 0.126 | 0.657 | {1: 19236, 0: 39356} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 10 | 10 | - | 0.123 | 0.665 | {1: 18652, 0: 39940} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 15 | - | 0.120 | 0.704 | {0: 42651, 1: 15941} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 15 | 0.500 | 0.122 | 0.694 | {0: 41937, 1: 16655} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 15 | 1.000 | 0.120 | 0.701 | {0: 42454, 1: 16138} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 15 | 1.500 | 0.118 | 0.705 | {0: 42746, 1: 15846} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 1 | 15 | - | 0.117 | 0.717 | {0: 43557, 1: 15035} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 15 | - | 0.118 | 0.696 | {0: 42105, 1: 16487} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 15 | 0.500 | 0.119 | 0.709 | {0: 43015, 1: 15577} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 15 | 1.000 | 0.122 | 0.686 | {1: 17222, 0: 41370} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 15 | 1.500 | 0.120 | 0.707 | {0: 42828, 1: 15764} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 3 | 15 | - | 0.118 | 0.708 | {0: 42954, 1: 15638} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 15 | - | 0.121 | 0.699 | {0: 42287, 1: 16305} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 15 | 0.500 | 0.120 | 0.688 | {0: 41548, 1: 17044} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 15 | 1.000 | 0.122 | 0.685 | {0: 41348, 1: 17244} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 15 | 1.500 | 0.120 | 0.693 | {0: 41915, 1: 16677} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 10 | 15 | - | 0.124 | 0.703 | {0: 42442, 1: 16150} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 20 | - | 0.118 | 0.730 | {0: 44425, 1: 14167} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 20 | 0.500 | 0.111 | 0.733 | {0: 44747, 1: 13845} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 20 | 1.000 | 0.119 | 0.734 | {0: 44675, 1: 13917} |
| - | binary:hinge | - | 0.600 | gbtree | 1 | 20 | 1.500 | 0.114 | 0.732 | {0: 44649, 1: 13943} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 1 | 20 | - | 0.117 | 0.738 | {0: 44933, 1: 13659} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 20 | - | 0.113 | 0.728 | {0: 44393, 1: 14199} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 20 | 0.500 | 0.113 | 0.733 | {0: 44708, 1: 13884} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 20 | 1.000 | 0.115 | 0.729 | {0: 44384, 1: 14208} |
| - | binary:hinge | - | 0.600 | gbtree | 3 | 20 | 1.500 | 0.113 | 0.733 | {0: 44723, 1: 13869} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 3 | 20 | - | 0.113 | 0.737 | {0: 44973, 1: 13619} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 20 | - | 0.115 | 0.732 | {0: 44626, 1: 13966} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 20 | 0.500 | 0.118 | 0.731 | {0: 44490, 1: 14102} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 20 | 1.000 | 0.117 | 0.732 | {0: 44581, 1: 14011} |
| - | binary:hinge | - | 0.600 | gbtree | 10 | 20 | 1.500 | 0.117 | 0.732 | {0: 44584, 1: 14008} |
| 0.500 | binary:hinge | 1.500 | 0.600 | gbtree | 10 | 20 | - | 0.116 | 0.737 | {0: 44948, 1: 13644} |

### SVC

The Support Vector Machine performs badly. The sigmoid kernel appears to perform the worst.

| degree | kernel | gamma | C | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- | --- | --- |
| - | rbf | scale | 1.000 | 0.140 | 0.535 | {1: 27930, 0: 30662} |
| - | linear | scale | 1.000 | 0.146 | 0.541 | {0: 30832, 1: 27760} |
| 3 | poly | scale | 1.000 | 0.142 | 0.532 | {1: 28190, 0: 30402} |
| - | sigmoid | scale | 1.000 | 0.117 | 0.499 | {0: 29104, 1: 29488} |
| - | sigmoid | scale | 10.000 | 0.119 | 0.507 | {0: 29541, 1: 29051} |

### Multi-layer Perceptron

For the first time, a neural network with more than one hidden layer doesn't perform all
that much better. ReLU appears to fit the task slightly better than Sigmoid. Overall, the MLP
didn't perform very well.

| solver | activation | hidden_layer_sizes | f1 score | accuracy | counter |
| --- | --- | --- | --- | --- | --- |
| adam | relu | (40,) | 0.140 | 0.571 | {0: 33077, 1: 25515} |
| adam | logistic | (40,) | 0.139 | 0.558 | {1: 26339, 0: 32253} |
| lbfgs | relu | (40,) | 0.139 | 0.556 | {1: 26484, 0: 32108} |
| lbfgs | logistic | (40,) | 0.139 | 0.556 | {1: 26469, 0: 32123} |
| adam | relu | (40, 40) | 0.138 | 0.572 | {0: 33263, 1: 25329} |
| adam | logistic | (40, 40) | 0.139 | 0.534 | {1: 27958, 0: 30634} |
| lbfgs | relu | (40, 40) | 0.139 | 0.565 | {1: 25865, 0: 32727} |
| lbfgs | logistic | (40, 40) | 0.136 | 0.556 | {0: 32206, 1: 26386} |
| adam | relu | (60,) | 0.139 | 0.550 | {1: 26860, 0: 31732} |
| adam | logistic | (60,) | 0.141 | 0.549 | {0: 31592, 1: 27000} |
| lbfgs | relu | (60,) | 0.139 | 0.562 | {0: 32558, 1: 26034} |
| lbfgs | logistic | (60,) | 0.139 | 0.546 | {1: 27148, 0: 31444} |

## Conclusion

Due to the strong imbalance present in the data, the most accurate classifier predicts `0` (a non-claim)
for every input. Tree-based methods stopped improving after a certain depth was reached - thus, we have
reason to believe that about two thirds of the attribute are actually unimportant in predicting whether
the customer is going to file a claim. However, dimension reduction down to 20 attributes seemed to worsen
the predictors by a large margin. If the goal was to create a classifier of the highest possible accuracy,
it would be better to try a less aggressive approach dimension reduction, or to just avoid PCA and instead
select the most important attributes and drop the rest. A larger neural network with multiple hidden layers
or XGBoost would probably perform the best after some parameter tuning.

### Addendum

Out of curiosity, I decided to test how well a classifier which marked every other record as a claim would
perform.

| f1 score | accuracy | counter |
| --- | --- | --- |
| 0.115 | 0.501 | {1: 29297, 0: 29295} |
