import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from rulefit import RuleFit


boston_data = pd.read_csv("boston.csv", index_col=0)

y = boston_data.medv.values
X = boston_data.drop("medv", axis=1)
features = X.columns
X = X.values

typ = 'regressor' #regressor or classifier

if typ == 'regressor':
    rf = RuleFit(
        rfmode='regress',
        tree_generator=RandomForestRegressor()
    )
    rf.fit(X, y, feature_names=features)
    y_pred = rf.predict(X)
    insample_rmse = np.sqrt(np.sum((y_pred - y)**2)/len(y))
elif typ == 'classifier':
    y_class = y.copy()
    y_class[y_class < 21] = -1
    y_class[y_class >= 21] = +1
    N = X.shape[0]
    rf = RuleFit(   rfmode='classify',
                    tree_generator=RandomForestClassifier()
                )
    rf.fit(X, y_class, feature_names=features)
    y_pred = rf.predict(X)
    y_proba = rf.predict_proba(X)
    insample_acc = sum(y_pred == y_class) / len(y_class)
rules = rf.get_rules()

rules = rules[rules.coef != 0].sort_values(by="support")
num_rules_rule = len(rules[rules.type == 'rule'])
num_rules_linear = len(rules[rules.type == 'linear'])
print(rules.sort_values('importance', ascending=False))
