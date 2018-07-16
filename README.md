# RuleFit
Implementation of a rule based prediction algorithm based on [the rulefit algorithm from Friedman and Popescu (PDF)](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf)

The algorithm can be used for predicting an output vector y given an input matrix X. In the first
step a tree ensemble is generated with gradient boosting. The trees are then used to form
rules, where the paths to each node in each tree form one rule. A rule is a binary decision if
an observation is in a given node, which is dependent on the input features that were used
in the splits. The ensemble of rules together with the original input features are then being input
 in a L1-regularized linear model, also called Lasso, which estimates the effects of each rule on
the output target but at the same time estimating many of those effects to zero.

You can use rulefit for predicting a numeric response (categorial not yet implemented).
The input has to be a numpy matrix with only numeric values.

## Installation

The latest version can be installed from the master branch using pip:

```
pip install git+git://github.com/christophM/rulefit.git
```

Another option is to clone the repository and install using `python setup.py install` or `python setup.py develop`.

## Usage

### Train your model:
```python
import numpy as np
import pandas as pd

from rulefit import RuleFit

boston_data = pd.read_csv("boston.csv", index_col=0)

y = boston_data.medv.values
X = boston_data.drop("medv", axis=1)
features = X.columns
X = X.as_matrix()

rf = RuleFit()
rf.fit(X, y, feature_names=features)

```

If you want to have influence on the tree generator you can pass the generator as argument:

```python
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=500, max_depth=10, learning_rate=0.01)
rf = RuleFit(gb)

rf.fit(X, y, feature_names=features)

```

### Predict

```python

rf.predict(X)

```

### Inspect rules:

```python
rules = rf.get_rules()

rules = rules[rules.coef != 0].sort_values("support", ascending=False)

print(rules)
```

## Notes
- In contrast to the original paper, the generated trees are always fitted with the same maximum depth.
  In the original implementation the maximum depth of the tree are drawn from a distribution each time
- This implementation is in progress. If you find a bug, don't hesitate to contact me.

## Changelog
All notable changes to this project will be documented here.

### [v0.3] - IN PROGRESS
- set default of exclude_zero_coef to False in get_rules(): 
- syntax fix (Issue 21)

### [v0.2] - 2017-11-24
- Introduces classification for RuleFit
- Adds scaling of variables (Friedscale)
- Allows random size trees for creating rules

### [v0.1] - 2016-06-18
- Start changelog and versions
