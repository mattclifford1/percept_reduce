'''
dataset util for get a random subsample/proportions of a dataset
'''
import random
import numpy as np

def get_indicies(proportions=[0.7, 0.3], total_instances=60000, seed=42):
    '''
    get random inidcies of each proportion. e.g. for train, val, test sets
    '''
    if sum(proportions) != 1:
        raise ValueError(f'proportions need to sum to 1, got: {proportions}')
    # get randomly ordered inds of the dataset
    inds = list(range(total_instances))
    random.seed(seed)
    random.shuffle(inds)
    # get each proportion
    ind_list = []
    prev_ind = 0
    for prop in proportions:
        next_ind = prev_ind + int(total_instances*prop)
        ind_list.append(inds[prev_ind:next_ind])
        prev_ind = next_ind
    return ind_list


if __name__ =='__main__':
    print(get_indicies([0.2, 0.4, 0.4], 10))