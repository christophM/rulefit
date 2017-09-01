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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
from functools import reduce
import  scipy 
import cvglmnet
import cvglmnetCoef
import cvglmnetPlot
import cvglmnetPredict

class GLMCV():
    def __init__(self,family='gaussian',incr_feats=None,decr_feats=None):
        self.cvfit=None
        self.family=family
        self.incr_feats=np.asarray([]) if incr_feats is None else np.asarray(incr_feats)
        self.decr_feats=np.asarray([]) if decr_feats is None else np.asarray(decr_feats)

    def fit(self,x,y):        
        cv_loss='class' if self.family=='binomial' else 'deviance'
        coef_limits=scipy.array([[scipy.float64(-scipy.inf)], [scipy.float64(scipy.inf)]]) # default, no limits on coefs
        # set up constraints
        if  len( self.incr_feats)>0 or  len(self.decr_feats)>0 :
            coef_limits=np.zeros([2,x.shape[1]])
            for i_feat in np.arange(x.shape[1]):
                coef_limits[0,i_feat]=-np.inf if i_feat not in self.incr_feats-1 else 0.
                coef_limits[1,i_feat]=np.inf if i_feat not in self.decr_feats-1 else 0.
            coef_limits=scipy.array(coef_limits)
        self.cvfit = cvglmnet.cvglmnet(x = x.copy(), y = y.copy(), nfolds=5,family = self.family, ptype = cv_loss, nlambda = 20,intr=True,cl=coef_limits)
        coef=cvglmnetCoef.cvglmnetCoef(self.cvfit, s = 'lambda_min')
        self.coef_=coef[1:,0]
        self.intercept_=coef[0,0]
    def predict(self,x):
        return cvglmnetPredict.cvglmnetPredict(self.cvfit, newx = x, s = 'lambda_min', ptype = 'class')
        
   

class RuleCondition():
    """Class for binary rule condition

    Warning: this class should not be used directly.
    """

    def __init__(self,
                 feature_index,
                 threshold,
                 operator,
                 support,
                 feature_name = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
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
            res =  1 * (X[:,self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:,self.feature_index] > self.threshold)
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))


class FriedScale():
    """Performs scaling of linear variables according to Friedman et al. 2005 Sec 5

    Each variable is firsst Winsorized l->l*, then standardised as 0.4 x l* / std(l*)
    Warning: this class should not be used directly.
    """    
    def __init__(self,trim_quantile=0.0):
        self.trim_quantile=trim_quantile
        self.scale_multipliers=None
        self.winsor_lims=None
        
    def train(self,X):
        # get winsor limits
        self.winsor_lims=np.ones([2,X.shape[1]])*np.inf
        self.winsor_lims[0,:]=-np.inf
        if self.trim_quantile>0:
            for i_col in np.arange(X.shape[1]):
                lower=np.percentile(X[:,i_col],self.trim_quantile*100)
                upper=np.percentile(X[:,i_col],100-self.trim_quantile*100)
                self.winsor_lims[:,i_col]=[lower,upper]
        # get multipliers
        scale_multipliers=np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals=len(np.unique(X[:,i_col]))
            if num_uniq_vals>2: # don't scale binary variables which are effectively already rules
                X_col_winsorised=X[:,i_col].copy()
                X_col_winsorised[X_col_winsorised<self.winsor_lims[0,i_col]]=self.winsor_lims[0,i_col]
                X_col_winsorised[X_col_winsorised>self.winsor_lims[1,i_col]]=self.winsor_lims[1,i_col]
                scale_multipliers[i_col]=0.4/np.std(X_col_winsorised)
        self.scale_multipliers=scale_multipliers
        
    def scale(self,X):
        return X*self.scale_multipliers

class Rule():
    """Class for binary Rules from list of conditions

    Warning: this class should not be used directly.
    """
    def __init__(self,
                 rule_conditions,value):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        self.value=value
        self.rule_direction=None
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

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


def extract_rules_from_tree(tree, feature_names=None):
    """Helper to turn a tree into as set of rules
    """
    rules = set()

    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       conditions=[]):
        if node_id != 0:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(feature_index=feature,
                                           threshold=threshold,
                                           operator=operator,
                                           support = tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                                           feature_name=feature_name)
            new_conditions = conditions + [rule_condition]
            #new_rule = Rule(new_conditions,tree.value[node_id][0][0])
            #rules.update([new_rule])
        else:
            new_conditions = []
                ## if not terminal node
        if tree.children_left[node_id] != tree.children_right[node_id]: #not tree.feature[node_id] == -2:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)
            
            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
        else: # a leaf node
            if len(new_conditions)>0:
                new_rule = Rule(new_conditions,tree.value[node_id][0][0])
                rules.update([new_rule])
            else:
                print('********** WHAT THE??? ****** ' + str(tree.node_count))
            return None

    traverse_nodes()
    
    return rules



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
        self.rules = set()
        ## TODO: Move this out of __init__
        self._extract_rules()
        self.rules=list(self.rules)

    def _extract_rules(self):
        """Recursively extract rules from each tree in the ensemble

        """
        for tree in self.tree_list:
            rules = extract_rules_from_tree(tree[0].tree_,feature_names=self.feature_names)
            self.rules.update(rules)

    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X,coefs=None):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix, shape=(n_samples, n_features)

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, n_out)
            Transformed dataset. Each column represents one rule.
        """
        rule_list=list(self.rules) 
        if   coefs is None:
            return np.array([rule.transform(X) for rule in rule_list]).T
        else: # else use the coefs to filter the rules we bother to interpret
            res= np.array([rule_list[i_rule].transform(X) for i_rule in np.arange(len(rule_list)) if coefs[i_rule]!=0]).T
            res_=np.zeros([X.shape[0],len(rule_list)])
            res_[:,coefs!=0]=res
            return res_
    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()




class RuleFit(BaseEstimator, TransformerMixin):
    """Rulefit class


    Parameters
    ----------
        tree_generator: object GradientBoostingRegressor or GradientBoostingClassifier, optional (default=None)

    Attributes
    ----------
    rule_ensemble: RuleEnsemble
        The rule ensemble

    feature_names: list of strings, optional (default=None)
        The names of the features (columns)

    """
    def __init__(self,tree_size=4,sample_fract='default',max_rules=2000,
                 memory_par=0.01,
                 tree_generator=None,n_feats=None,incr_feats=[],decr_feats=[],
                rfmode='regress',lin_trim_quantile=0.025,
                lin_standardise=True, exp_rand_tree_size=True,
                model_type='rl',random_state=None):
        self.tree_generator = tree_generator
        self.n_feats=n_feats
        self.incr_feats=np.asarray([] if incr_feats is None else incr_feats)
        self.decr_feats=np.asarray([] if decr_feats is None else decr_feats)
        self.mt_feats=np.asarray(list(self.incr_feats)+list(self.decr_feats))
        self.nmt_feats=np.asarray([j for j in np.arange(n_feats)+1 if j not in self.mt_feats])
        self.rfmode=rfmode
        self.lin_trim_quantile=lin_trim_quantile
        self.lin_standardise=lin_standardise
        self.friedscale=FriedScale(trim_quantile=lin_trim_quantile)
        self.exp_rand_tree_size=exp_rand_tree_size
        self.max_rules=max_rules
        self.sample_fract=sample_fract 
        self.max_rules=max_rules
        self.memory_par=memory_par
        self.tree_size=tree_size
        self.random_state=random_state
        self.model_type=model_type
        
    def fit(self, X, y=None, feature_names=None):
        """Fit and estimate linear combination of rule ensemble

        """
        ## Enumerate features if feature names not provided
        N=X.shape[0]
        if feature_names is None:
            self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
        else:
            self.feature_names=feature_names
        if 'r' in self.model_type:
            ## initialise tree generator
            if self.tree_generator is None:
                n_estimators_default=int(np.ceil(self.max_rules/self.tree_size))
                self.sample_fract_=min(0.5,(100+6*np.sqrt(N))/N)
                if   self.rfmode=='regress':
                    self.tree_generator = GradientBoostingRegressor(n_estimators=n_estimators_default, max_leaf_nodes=self.tree_size, learning_rate=self.memory_par,subsample=self.sample_fract_,random_state=self.random_state,max_depth=100)
                else:
                    self.tree_generator =GradientBoostingClassifier(n_estimators=n_estimators_default, max_leaf_nodes=self.tree_size, learning_rate=self.memory_par,subsample=self.sample_fract_,random_state=self.random_state,max_depth=100)
    
            if   self.rfmode=='regress':
                if type(self.tree_generator) not in [GradientBoostingRegressor,RandomForestRegressor]:
                    raise ValueError("RuleFit only works with RandomForest and BoostingRegressor")
            else:
                if type(self.tree_generator) not in [GradientBoostingClassifier,RandomForestClassifier]:
                    raise ValueError("RuleFit only works with RandomForest and BoostingClassifier")
    
            ## fit tree generator
            if not self.exp_rand_tree_size: # simply fit with constant tree size
                self.tree_generator.fit(X, y)
            else: # randomise tree size as per Friedman 2005 Sec 3.3
                np.random.seed(self.random_state)
                tree_sizes=np.random.exponential(scale=self.tree_size-2,size=int(np.ceil(self.max_rules*2/self.tree_size)))
                tree_sizes=np.asarray([2+np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))],dtype=int)
                i=int(len(tree_sizes)/4)
                while np.sum(tree_sizes[0:i])<self.max_rules:
                    i=i+1
                tree_sizes=tree_sizes[0:i]
                self.tree_generator.set_params(warm_start=True) 
    #            num_rules=0
                for i_size in np.arange(len(tree_sizes)):
                    size=tree_sizes[i_size]
                    self.tree_generator.set_params(n_estimators=len(self.tree_generator.estimators_)+1)
                    self.tree_generator.set_params(max_leaf_nodes=size)
                    self.tree_generator.set_params(random_state=i_size+self.random_state) # warm_state=True seems to reset random_state, such that the trees are highly correlated, unless we manually change the random_sate here.
                    self.tree_generator.get_params()['n_estimators']
                    self.tree_generator.fit(np.copy(X, order='C'), np.copy(y, order='C'))
                    # count leaves (a check)
    #                tree_=self.tree_generator.estimators_[j_][0].tree_
    #                test_=tree_.children_left+ tree_.children_left
    #                num_leaves=len(test_[test_==-2])
    #                num_rules=num_rules+mean_leaves
    #            print('num rules: ' + str(num_rules))
    #                print('tree_size: ' + str(size) + ' mean actual number of leaf nodes: ' + str(mean_leaves) + ' for ' + str(cnt)+ ' trees')
                self.tree_generator.set_params(warm_start=False) 
            tree_list = self.tree_generator.estimators_
            if isinstance(self.tree_generator, RandomForestRegressor) or isinstance(self.tree_generator, RandomForestClassifier):
                 tree_list = [[x] for x in self.tree_generator.estimators_]
                 
            ## extract rules
            self.rule_ensemble = RuleEnsemble(tree_list = tree_list,
                                              feature_names=self.feature_names)
            ## filter for upper and lower rules only (if needed)
            self.num_rules_peak_=len(self.rule_ensemble.rules)
            if len(self.mt_feats)>0: 
                filtered_rules=set()
                for rule in self.rule_ensemble.rules:
                    conditions=list(rule.conditions)
                    all_incr=np.all([conditions[c].operator[0]==('>' if conditions[c].feature_index in self.incr_feats-1 else '<') for c in [cc for cc in np.arange(len(conditions)) if conditions[cc].feature_index in self.mt_feats-1]])
                    all_decr=np.all([conditions[c].operator[0]==('<' if conditions[c].feature_index in self.incr_feats-1 else '>') for c in [cc for cc in np.arange(len(conditions)) if conditions[cc].feature_index in self.mt_feats-1]])
                    if (all_incr and rule.value>0) or (all_decr and rule.value<0):
                        rule.rule_direction=+1 if all_incr else -1
                        filtered_rules.add(rule)
                #print('started with ' + str(len(self.rule_ensemble.rules)) + ' rules, now have ' + str(len(filtered_rules)))
                self.rule_ensemble.rules=list(filtered_rules)
            ## concatenate original features and rules
            X_rules = self.rule_ensemble.transform(X)
        
        ## standardise linear variables if requested (for regression model only)
        if 'l' in self.model_type: 
            if self.lin_standardise:
                self.friedscale.train(X)
                X_regn=self.friedscale.scale(X)
            else:
                X_regn=X.copy()            
        
        ## Compile Training data
        X_concat=np.zeros([X.shape[0],0])
        if 'l' in self.model_type:
            X_concat = np.concatenate((X_concat,X_regn), axis=1)
        if 'r' in self.model_type:
            if X_rules.shape[0] >0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)

        ## initialise Lasso
        if len(self.mt_feats)==0:
            if self.rfmode=='regress':
                self.lscv = LassoCV()
            else:
                self.lscv=GLMCV(family='binomial')
        else:
            rule_dirns=np.asarray([r.rule_direction for r in self.rule_ensemble.rules])
            incr_feats_with_rules=np.asarray(np.hstack([self.incr_feats,self.n_feats+np.asarray([i_rule+1 for i_rule in np.arange(len(rule_dirns)) if rule_dirns[i_rule]>0])]))
            decr_feats_with_rules=np.asarray(np.hstack([self.decr_feats,self.n_feats+np.asarray([i_rule+1 for i_rule in np.arange(len(rule_dirns)) if rule_dirns[i_rule]<0])]))
            if self.rfmode=='regress':
                self.lscv=GLMCV(family='gaussian',incr_feats=incr_feats_with_rules,decr_feats=decr_feats_with_rules)
            else:
                self.lscv=GLMCV(family='binomial',incr_feats=incr_feats_with_rules,decr_feats=decr_feats_with_rules)

        ## fit Lasso
        self.lscv.fit(X_concat, y)
        
        return self

    def predict(self, X):
        """Predict outcome for X

        """
        X_concat=np.zeros([X.shape[0],0])
        if 'l' in self.model_type:
            if self.lin_standardise:
                X_concat = np.concatenate((X_concat,self.friedscale.scale(X)), axis=1)
            else:
                X_concat = np.concatenate((X_concat,X), axis=1)
        if 'r' in self.model_type:
            rule_coefs=self.lscv.coef_[self.n_feats:]
            X_rules = self.rule_ensemble.transform(X,coefs=rule_coefs)
            if X_rules.shape[0] >0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)
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
        return self.rule_ensemble.transform(X)

    def get_rules(self, exclude_zero_coef=True):
        """Return the estimated rules

        Parameters
        ----------
        exclude_zero_coef: If True (default), returns only the rules with an estimated
                           coefficient not equalt to  zero.

        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """

        n_features= len(self.lscv.coef_) - len(self.rule_ensemble.rules)
        rule_ensemble = list(self.rule_ensemble.rules)
        output_rules = []
        ## Add coefficients for linear effects
        for i in range(0, n_features):
            if self.lin_standardise:
                coef=self.lscv.coef_[i ]*self.friedscale.scale_multipliers[i]
            else:
                coef=self.lscv.coef_[i ]
            output_rules += [(self.feature_names[i], 'linear',coef, 1)]
        ## Add rules
        for i in range(0, len(self.rule_ensemble.rules)):
            rule = rule_ensemble[i]
            coef=self.lscv.coef_[i + n_features]
            output_rules += [(rule.__str__(), 'rule', coef,  rule.support,rule.rule_direction)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type","coef", "support","dirn"])
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules
