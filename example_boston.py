import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from rulefit import RuleFit


boston_data = pd.read_csv("boston.csv", index_col=0)

y = boston_data.medv.values
X = boston_data.drop("medv", axis=1)
features = X.columns
X = X.as_matrix()

typ='classifier' #regressor or classifier

if typ=='regressor':
    rf = RuleFit(tree_size=4,sample_fract='default',max_rules=2000,
             memory_par=0.01,
             tree_generator=None,
            rfmode='regress',lin_trim_quantile=0.025,
            lin_standardise=True, exp_rand_tree_size=True,random_state=1) 
    rf.fit(X, y, feature_names=features)
    y_pred=rf.predict(X)
    insample_rmse=np.sqrt(np.sum((y_pred-y)**2)/len(y))
elif typ=='classifier':
    y_class=y.copy()
    y_class[y_class<21]=-1
    y_class[y_class>=21]=+1
    N=X.shape[0]
    rf = RuleFit(tree_size=4,sample_fract='default',max_rules=2000,
                 memory_par=0.01,
                 tree_generator=None,
                rfmode='classify',lin_trim_quantile=0.025,
                lin_standardise=True, exp_rand_tree_size=True,random_state=1) 
    rf.fit(X, y_class, feature_names=features)
    y_pred=rf.predict(X)
    insample_acc=len(y_pred==y_class)/len(y_class)
rules = rf.get_rules()

rules = rules[rules.coef != 0].sort_values(by="support")
num_rules_rule=len(rules[rules.type=='rule'])
num_rules_linear=len(rules[rules.type=='linear'])

