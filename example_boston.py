import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from rulefit import RuleFit


boston_data = pd.read_csv("boston.csv", index_col=0)

y = boston_data.medv.values
X = boston_data.drop("medv", axis=1)
features = X.columns
X = X.as_matrix()

gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.01)
rf = RuleFit(gb)

rf.fit(X, y, feature_names=features)

rules = rf.get_rules()

rules = rules[rules.coef != 0].sort("support")
