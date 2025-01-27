
import argparse
from typing import Tuple

import pandas as pd

from data import load_data
from model import DecisionTree
from train import train, evaluate



def cross_validation(cv_folds: list, depth_limit_values: list, ig_criterion: str = 'entropy') -> Tuple[dict, float]:
    '''
    Runs cross-validation to determine the best hyperparameters.

    Args:
        cv_folds (list): a list of dataframes, each corresponding to a fold of the data
        depth_limit_values (list): a list of depth_limit hyperparameter values to try
        ig_criterion (str): the information gain variant to use. Should be one of "entropy" or "collision".

    Returns:
        dict: a dictionary with the best hyperparameters discovered during cross-validation
        float: the average cross-validation accuracy corresponding to the best hyperparameters

    '''

    best_hyperparams = {'depth_limit': None}
    best_avg_accuracy = 0
    
    for depth in depth_limit_values: # Loop through each depth that needs to be tested
        accuracies = [] # Create a list of all accuracies each k_fold iteration
        for i in range(len(cv_folds)): # Loop k times
            copy = cv_folds.copy() 
            validation = copy.pop(i) # Get validation fold
            
            x_train = pd.concat(copy, axis=0).reset_index(drop=True)
            y_train = x_train.iloc[:,-1].tolist() # Get training labels
            validation = validation.reset_index(drop=True)
            x_test = validation.iloc[:, :-1]  # Extract testing features
            y_test = validation.iloc[:, -1].tolist()  # Extract testing labels as a list

            dt = DecisionTree(depth, 'entropy') # Create new tree
            dt.train(x_train, y_train) # Train tree
            accuracy = evaluate(dt, x_test, y_test) # Calculate accuracy
            accuracies.append(accuracy)

        average = sum(accuracies) / len(accuracies) #get average accuracy at this depth
        if average > best_avg_accuracy: #Check if better than previous depth
            best_avg_accuracy = average
            best_hyperparams = {'depth_limit': depth}
    
    return best_hyperparams, best_avg_accuracy


# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--depth_limit_values', '-d', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6], 
        help='The list (comma separated) of maximum depths to try.')
    parser.add_argument('--ig_criterion', '-i', type=str, choices=['entropy', 'collision'], default='entropy',
        help='Which information gain variant to use.')
    args = parser.parse_args()


    # load data
    data_dict = load_data()
    cv_folds = data_dict['cv_folds']

    # run cross_validation
    best_hyperparams, best_accuracy = cross_validation(
        cv_folds=cv_folds, 
        depth_limit_values=args.depth_limit_values, 
        ig_criterion=args.ig_criterion)
    
    # print best hyperparameters and accuracy
    print('\nBest hyperparameters from cross-validation:\n')
    for name, value in best_hyperparams.items():
        print(f'{name:>15}: {value}')
    print(f'       accuracy: {best_accuracy:.3f}\n')
