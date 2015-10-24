"""Linear model of tree-based decision rules

This method implement the RuleFit algorithm

The module structure is the following:

- ``RuleCondition`` implements a binary feature transformation
- ``Rule`` implements a Rule composed of ``RuleConditions``
- ``RuleEnsemble`` implements an ensemble of ``Rules``
- ``RuleFit`` implements the RuleFit algorithm

"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV

class RuleCondition():
    """Class for binary rule condition

    Warning: this class should not be used directly.
    """

    def __init__(self,
                 feature,
                 threshold,
                 operator,
                 feature_name = None):
        self.feature = feature
        self.threshold = threshold
        self.operator = operator
        self.feature_name = feature_name

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix, shape=(n_samples, n_features)

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        if self.operator == "<=":
            res =  1 * (X[:,self.feature] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:,self.feature] > self.threshold)
        return res




class Rule():
    """Class for binary Rules from list of conditions

    Warning: this class should not be used directly.
    """
    def __init__(self,
                 rule_conditions):
        self.conditions = rule_conditions

    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x,y: x * y, rule_applies)

    def __str__(self):
        return  " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()








class RuleEnsemble():
    """Ensemble of binary decision rules

    This class implements an ensemble of decision rules that extracts rules from
    an ensemble of decision trees.

    Parameters
    ----------
    tree_list: List or array of DecisionTreeClassifier or DecisionTreeRegressor
        Trees from which the rules are created

    feature_names: List of strings, optional (default=None)
        Names of the features

    Attributes
    ----------
    rules: List of Rule
        The ensemble of rules extracted from the trees
    """
    def __init__(self,
                 tree_list,
                 feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.rules = []
        ## TODO: Move this out of __init__
        self._extract_rules()

    def _extract_rules(self):
        """Recursively extract rules from each tree in the ensemble

        """
        for tree in self.tree_list:
            self._traverse_nodes(tree[0].tree_)


    def _traverse_nodes(self, tree, node_id=0, operator=None, threshold=None, feature=None, conditions=[]):
        """Extract rule in node

        """
        if node_id != 0:
            if self.feature_names is None:
                feature_name = None
            else:
                feature_name = self.feature_names[feature]

            rule_condition = RuleCondition(feature=feature, threshold=threshold, operator=operator, feature_name=feature_name)
            ## Create new Rule from old rule + new condition
            new_conditions = conditions  + [rule_condition]
            new_rule = Rule(new_conditions)
            self.rules.append(new_rule)


        else:
            new_conditions = []

        ## if not terminal node
        if not tree.feature[node_id] == -2:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            left_node_id = tree.children_left[node_id]
            self._traverse_nodes(tree, left_node_id, "<=", threshold, feature, new_conditions)

            right_node_id = tree.children_right[node_id]
            self._traverse_nodes(tree, right_node_id, ">", threshold, feature, new_conditions)
        else:
            return None

    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix, shape=(n_samples, n_features)

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, n_out)
            Transformed dataset. Each column represents one rule.
        """
        return np.array([rule.transform(X) for rule in self.rules]).T

    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()




class RuleFit(BaseEstimator, TransformerMixin):
    """Rulefit class


    Parameters
    ----------
    n_estimators: integer, optional (default=10)
        The number of trees used for rule extraction

    feature_names: list of strings, optional (default=None)
        The names of the features (columns)

    Attributes
    ----------
    rule_ensemble: RuleEnsemble
        The rule ensemble
    """
    def __init__(self,
                 n_estimators=10,
                 feature_names=None):
        ##TODO: Move feature_names to fit method
        ##TODO: Turn RuleFit into MetaEstimator and allow different tree-generating algorithms
        self.tree_generator = "GradientBoosting"
        self.n_estimators = n_estimators
        self.feature_names = feature_names


    def fit(self, X, y=None):

        ## initialise tree generator
        self.tree_generator = GradientBoostingRegressor(n_estimators=self.n_estimators)

        ## fit tree generator
        self.tree_generator.fit(X, y)

        ## extract rules
        self.rule_ensemble = RuleEnsemble(self.tree_generator.estimators_, feature_names=self.feature_names)

        ## concatenate original features and rules
        X_rules = self.rule_ensemble.transform(X)
        X_concat = np.concatenate((X, X_rules), axis=1)

        ## initialise Lasso
        self.lscv = LassoCV()

        ## fit Lasso
        self.lscv.fit(X_concat, y)
        return self

    def predict(self, X):
        """Predict outcome for X

        """

        X_rules = self.rule_ensemble.transform(X)
        X_concat = np.concatenate((X, X_rules), axis=1)

        return self.lscv.predict(X_concat)

    def transform(self, X=None, y=None):
        """Transform dataset.

        Parameters
        ----------
        X : array-like matrix, shape=(n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency.

        Returns
        -------
        X_transformed: matrix, shape=(n_samples, n_out)
            Transformed data set
        """
        ##TODO
        ## Apply rules to X
        ## concatenate X and rules_x
        ## remove columns with beta == 0
        ## return concatenated data set
        pass

    def fit_transform(self, X, y=None):
        ## TODO
        pass

