import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from rulefit import RuleFit



boston_data = pd.read_csv("boston.csv")

y = boston_data.medv.values
X = boston_data.drop("medv", axis=1)

gb = GradientBoostingRegressor(n_estimators=5000, max_depth=3)
rf = RuleFit(gb)

rf.fit(X.as_matrix(), y, feature_names=X.columns)

rules = rf.get_rules()

rules = rules[rules.coef != 0].sort("support")
