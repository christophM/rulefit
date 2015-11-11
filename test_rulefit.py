from rulefit import RuleCondition, Rule, RuleEnsemble, RuleFit
import numpy as np



rule_condition_smaller = RuleCondition(1, 5, "<=", 0.4)
rule_condition_greater = RuleCondition(0, 1, ">", 0.1)

X = np.array([[1,2,3], [4,5,6], [7,8,9]])

## Testing RuleCondition
def test_rule_condition_hashing_equal1():
    assert (RuleCondition(1, 5, "<=", 0.4) == RuleCondition(1, 5, "<=", 0.4))

def test_rule_condition_hashing_equal2():
    assert (RuleCondition(1, 5, "<=", 0.5) == RuleCondition(1, 5, "<=", 0.4))

def test_rule_condition_hashing_different1():
    assert (RuleCondition(1, 4, "<=", 0.4) != RuleCondition(1, 5, "<=", 0.4))

def test_rule_condition_hashing_different2():
    assert (RuleCondition(1, 5, ">", 0.4) != RuleCondition(1, 5, "<=", 0.4))

def test_rule_condition_hashing_different2():
    assert (RuleCondition(2, 5, ">", 0.4) != RuleCondition(1, 5, ">", 0.4))

def test_rule_condition_smaller():
    np.testing.assert_array_equal(rule_condition_smaller.transform(X),
                                  np.array([1,1,0]))
def test_rule_condition_greater():
    np.testing.assert_array_equal(rule_condition_greater.transform(X),
                                  np.array([0,1,1]))



## Testing rule
rule = Rule([rule_condition_smaller, rule_condition_greater])

def test_rule_transform():
    np.testing.assert_array_equal(rule.transform(X),
                                  np.array([0,1,0]))

def test_rule_equality():
    rule2 = Rule([rule_condition_greater, rule_condition_smaller])
    assert rule == rule2


## Test rule extractions function
## TODO

## RuleEnsemble
## - Construct ensemble with 2 short trees and test results
## - Test filter rules with only rules that only have the "<=" operator
## - Test filter short rules
## - Test transform function

