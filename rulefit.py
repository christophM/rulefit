import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class RuleCondition():

    def __init__(self, feature, threshold, operator, feature_name = None):
        self.feature = feature
        self.threshold = threshold
        self.operator = operator
        self.feature_name = feature_name
        if operator == "<=":
            self.func = lambda x: int(x <= self.threshold)
        elif operator == ">":
            self.func = lambda x: int(x > self.threshold)


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X):
        if self.operator == "<=":
            res =  1 * (X[:,self.feature] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:,self.feature] > self.threshold)
        return res





class Rule():

    def __init__(self, rule_conditions):
        self.conditions = rule_conditions

    def transform(self, features):
        rule_applies = [x.transform(features) for x in self.conditions]
        return np.prod(rule_applies)

    def transform(self, X):
        applied_conditions = map(lambda x: x.transform(X), self.conditions)
        return reduce(lambda x,y: x * y, applied_conditions)


    def emit_rules(self):
        ## build short rules
        return [self] + [Rule(self.conditions[:-x]) for x in range(1, len(self.conditions))]

    def __str__(self):
        return  " & ".join([x.__str__() for x in self.conditions])

    def __repre__(self):
        return self.__str__()




## Brainstorm if you really need a class for that or if you do a function
class RuleEnsemble():

    def __init__(self, tree, feature_names=None):
        self.tree = tree.tree_
        self.feature_names = feature_names
        self.rules = []
        self.extract_rules()

    def extract_rules(self):
        self.traverse_nodes()
        return self.rules


    def traverse_nodes(self, node_id=0, operator=None, threshold=None, feature=None, conditions=[]):
        ## add to self.rules. only for non-root nodes
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
        if not self.tree.feature[node_id] == -2:
            feature = self.tree.feature[node_id]
            threshold = self.tree.threshold[node_id]

            left_node_id = self.tree.children_left[node_id]
            self.traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)

            right_node_id = self.tree.children_right[node_id]
            self.traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
        else:
            return None



class RulesSet():

    def __init__(self, trees_list, feature_names):
        self.trees = trees_list
        self.feature_names = feature_names
        self.rules = []
        self.traverse_trees()

    def traverse_trees(self):
        for tree in self.trees:
            self.rules += RuleEnsemble(tree[0], feature_names = self.feature_names).extract_rules()
        return None


    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X):
        return np.array([rule.transform(X) for rule in self.rules]).T

    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()




class Rulefit(BaseEstimator, TransformerMixin):

    trees = []
    rules = []

    def __init__(self):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X=None, y=None):
        pass

    def fit_transform(self, X, y=None):
        pass
