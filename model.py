''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from math import log2
from typing import Protocol
import ipykernel 
import jupyter
import matplotlib
import numpy as np
import plotly 
import seaborn


import pandas as pd


# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the MajorityBaseline and DecisionTree classes further down.
class Model(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        ...

    def predict(self, x: pd.DataFrame) -> list:
        ...



class MajorityBaseline(Model):
    __most_common = None
    def __init__(self):
        pass


    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
        '''

        # YOUR CODE HERE
        self.__most_common = max(set(y), key=y.count)
        return 
    

    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''
        # YOUR CODE HERE
        y_hat = [self.__most_common] * len(x)
        return y_hat



class DecisionTree(Model):
    __tree = {}
    most_common = None
    def __init__(self, depth_limit: int = None, ig_criterion: str = 'entropy'):
        '''
        Initialize a new DecisionTree

        Args:
            depth_limit (int): the maximum depth of the learned decision tree. Should be ignored if set to None.
            ig_criterion (str): the information gain criterion to use. Should be one of "entropy" or "collision".
        '''
        
        self.depth_limit = depth_limit
        self.ig_criterion = ig_criterion

    # Helper method that calcualtes entropy for a given feature
    def entropy(self, feature: list):
        #Calculate for lables
        _, counts = np.unique(feature, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities)) 
    
    # Helper method that calcaulates info gain
    def info_gain(self, x: pd.DataFrame, y: pd.Series):
        entropy_s = self.entropy(y)
        info_gains = {}  
        
        for feature_name in x.columns:  
            feature = x[feature_name]
            unique_values, counts = np.unique(feature, return_counts=True)
            weighted_entropy = 0
            
            for value, count in zip(unique_values, counts):
                subset_labels = y[feature == value]
                weighted_entropy += (count / len(feature)) * self.entropy(subset_labels)
            
            info_gains[feature_name] = entropy_s - weighted_entropy
        return info_gains
    
    def id3(self, x: pd.DataFrame, y: list, features: list, depth: int):
        #BASE CASES
        if self.entropy(y) == 0: # if all labels are the same
            return y[0]
        if not features:    # If there are no more features
            return max(set(y), key=y.count)
        if depth >= self.depth_limit:  #If max depth reached
            return max(set(y), key=y.count)
        else:
            info_gain = self.info_gain(x[features], pd.Series(y)) # Calculate info gain with all remaining features
            if 'label' in info_gain:
                del info_gain['label']
            best_feature = max(info_gain, key=info_gain.get) # Get the remaining feature with the highest info gain
            features_copy = features.copy()
            features_copy.remove(best_feature)
            tree = {best_feature: {}}
            
            unique_values = np.unique(x[best_feature]) # Get each unique value of that feautre and the count of each feature
            for value in unique_values:
                subset_x = x[x[best_feature] == value].drop(columns=[best_feature]).reset_index(drop=True)
                subset_y = [y[i] for i in range(len(y)) if x[best_feature].iloc[i] == value]
                subtree = self.id3(subset_x, subset_y, features_copy, depth + 1)
                tree[best_feature][value] = subtree
            return tree

    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a decision tree from a dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
            - Ignore self.depth_limit if it's set to None
            - Use the variable self.ig_criterion to decide whether to calulate information gain 
              with entropy or collision entropy
        '''

        # YOUR CODE HERE
        # If all labels are the same 
        self.most_common = max(set(y), key=y.count)
        if self.depth_limit == None:
            self.depth_limit = np.inf
        self.__tree = self.id3(x, y, x.columns.tolist(), 0)
        return 
    
    # Helper method for calculating one instance
    def predict_instance(self, instance: dict):
        tree = self.__tree  

        while isinstance(tree, dict):  # While the tree is not a leaf node
            feature = next(iter(tree))  # Get the root feature of the current subtree
            value = instance.get(feature)  # Get the feature value for this instance
        
            tree = tree.get(feature, {}).get(value, None)
        
            # If the value doesn't exist in the tree, return None (or a default label)
            if tree is None:
                return self.most_common

        return tree  # Return the leaf node (label)


    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''
        # YOUR CODE HERE
        y_hat=[]
        for _, row in x.iterrows():
            instance = row.to_dict()
            y_hat.append(self.predict_instance(instance))
        return y_hat

