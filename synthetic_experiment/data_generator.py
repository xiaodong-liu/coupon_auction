import numpy as np
from scipy.stats import lognorm
import os
import random
import pickle

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def uniform_distribution(n_bidder, n_auction):
    ctr_initial = np.random.random((n_auction, n_bidder)) * 0.495 + 0.005
    valuation = np.random.random((n_auction, n_bidder)) * 45 + 5
    return ctr_initial, valuation

def lognorm_distribution(n_bidder, n_auction):
    ctr_initial = np.random.random((n_auction, n_bidder)) * 0.495 + 0.005
    valuation = lognorm.rvs(s=1, scale = np.exp(3), size=(n_auction, n_bidder))
    return ctr_initial, valuation


if __name__ == '__main__':
    set_all_seeds(2024)
    # dir = 'data/uniform.npy'
    # ctr_initial, valuation = uniform_distribution(8, 100000)
    # data = {
    #     'ctr': ctr_initial,
    #     'value': valuation
    # }
    # with open(dir, 'wb') as f:
    #     pickle.dump(data, f)

    dir = 'data/lognorm.npy'
    ctr, value = lognorm_distribution(8, 100000)
    data = {
        'ctr': ctr,
        'value': value
    }
    with open(dir, 'wb') as f:
        pickle.dump(data, f)