import numpy as np
import pandas as pd
from rulefit import RuleFit



## Create artificial data set with
n = 10000
x1 = np.random.normal(scale=1, size=n)
x2 = np.random.normal(loc=0, scale=1, size=n)
x3 = np.random.normal(size=n)
x4 = np.random.normal(size=n)

eps = np.random.normal(loc=0, scale=0.1, size=n)

y = 5 * ((x1 > 1).astype(int) * (x2 <  -1).astype(int)) + 0.3 * x4 + eps


X = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})



rf = RuleFit()
rf.fit(X.values, y, X.columns)
rf.fit(X.values, y)

rules = rf.get_rules(exclude_zero_coef=True)

print(rules)
