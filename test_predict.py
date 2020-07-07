import pandas as pd
import numpy as np

from rulefit import RuleFit

#load data
boston_data = pd.read_csv("boston.csv", index_col=0)

y = boston_data.medv.values
X = boston_data.drop("medv", axis=1)
features = X.columns
X = X.as_matrix()

y_class = y.copy()
y_class[y_class < 21] = 0
y_class[y_class >= 21] = +1
N = X.shape[0]

#fit
rf = RuleFit(tree_size=4, sample_fract='default', max_rules=2000,
             memory_par=0.01,
             tree_generator=None,
             rfmode='classify', lin_trim_quantile=0.025,
             lin_standardise=True, exp_rand_tree_size=True, random_state=1)
rf.fit(X, y_class, feature_names=features)

#predict
y_pred = rf.predict(X)
y_proba = rf.predict_proba(X)

#basic checks for probabilities
assert np.min(y_proba) >= 0
assert np.max(y_proba) <= 1

#test that probabilities match actual predictions
np.testing.assert_array_equal(np.rint(np.array(y_proba[:,1])), y_pred)

