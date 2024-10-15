import numpy as np
from scipy.stats import lognorm
import pickle
from tqdm import tqdm

def read_data(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    ctr, value = data['ctr'], data['value']
    return ctr, value

def virtual_value_function(valuations):
    return valuations - ((1 - lognorm.cdf(valuations, s = 1, scale = np.exp(3)))/ lognorm.pdf(valuations, s = 1, scale=np.exp(3)))

def inverse_virtual_value_function(x, upper_bound, eps):
    l = x
    r = upper_bound
    mid = (l + r) / 2.0

    tmp = virtual_value_function(mid) - x
    while (np.abs(tmp) > eps):
        l = mid if tmp < 0 else l
        r = mid if tmp >= 0 else r
        mid = (l + r) / 2.0
        tmp = virtual_value_function(mid) - x
    return mid

def myerson_auction(ctr, valuation):
    n_auction, n_bidder = valuation.shape

    virtual_values = virtual_value_function(valuation)
    rank_score = ctr * virtual_values
    rank_index = np.argsort(rank_score, axis=-1)
    winner_index = rank_index[:, -1]
    second_index = rank_index[:, -2]

    inverse_input = rank_score[np.arange(n_auction), second_index] / ctr[np.arange(n_auction), winner_index]

    inverse_input = np.where(inverse_input < 0, 0, inverse_input)
    payment = np.zeros(n_auction)
    for i in tqdm(range(n_auction)):
        if rank_score[i, winner_index[i]] > 0:
            payment[i] = inverse_virtual_value_function(inverse_input[i], valuation[i, winner_index[i]], eps=0.01)
    # payment = inverse_virtual_value_function(inverse_input, valuation[np.arange(n_auction), winner_index], eps = 0.01)
    welfare = ctr * valuation
    revenue = ctr[np.arange(n_auction), winner_index] * payment
    return np.mean(welfare[np.arange(n_auction), winner_index]), np.mean(revenue)

def main():
    dir = '../data/lognorm.npy'
    ctr, value = read_data(dir)
    social, revenue = myerson_auction(ctr, value)
    print(f"Social Welfare: {social-revenue}, Revenue: {revenue}")

if __name__ == '__main__':
    main()
