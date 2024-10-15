import numpy as np
from scipy.stats import uniform
import pickle

def read_data(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    ctr, value = data['ctr'], data['value']
    return ctr, value
# 在Myerson Auction中，payment

def virtual_value_function(valuations):
    a = valuations - ((1 - uniform.cdf(valuations, loc=5, scale=45)) / uniform.pdf(valuations, loc=5,scale=45))
    return a

def inverse_virtual_value_function(x):
    return (x + 50.0) / 2.0

def myerson_auction(ctr, valuation):
    n_auction, n_bidder = valuation.shape
    virtual_values = virtual_value_function(valuation)
    rank_score = ctr * virtual_values
    rank_index = np.argsort(rank_score, axis = -1)
    winner_index = rank_index[:, -1]
    second_index = rank_index[:, -2]
    inverse_input = rank_score[np.arange(n_auction), second_index] / ctr[np.arange(n_auction), winner_index]
    payment = inverse_virtual_value_function(inverse_input)

    welfare = ctr * valuation
    revenue = ctr[np.arange(n_auction), winner_index] * payment
    return np.mean(welfare[np.arange(n_auction), winner_index]), np.mean(revenue)


def main():
    dir = '../data/uniform.npy'
    ctr, value = read_data(dir)
    social, revenue = myerson_auction(ctr, value)
    print(f"Advertisers Utility: {social-revenue}, Revenue: {revenue}")

if __name__ == '__main__':
    main()