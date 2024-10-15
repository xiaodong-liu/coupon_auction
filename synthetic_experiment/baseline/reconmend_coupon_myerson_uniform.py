import numpy as np
from scipy.stats import uniform
import pickle

def read_data(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    ctr, value = data['ctr'], data['value']
    return ctr, value

def virtual_value_function(value):
    return value - ((1 - uniform.cdf(value, loc=5,scale=45))/uniform.pdf(value, loc=5, scale=45))

def inverse_value_function(x):
    return (x + 50.0) / 2.0


def coupon_function(ctr, value):
    n_auction, n_bidder = value.shape

    coupons = np.array([0, 2, 4, 8])
    ctr_boost = np.array([1, 1.1, 1.2, 1.3])

    coupons_extend = np.broadcast_to(coupons, (n_auction, n_bidder, 4))
    ctr_boost_extend = np.broadcast_to(ctr_boost, (n_auction, n_bidder, 4))

    ctr_extend = ctr.repeat(4).reshape((n_auction, n_bidder, 4))
    valuation_extend = value.repeat(4).reshape((n_auction, n_bidder, 4))

    virtual_value_extend = virtual_value_function(valuation_extend)
    rank_scores = (ctr_extend * ctr_boost_extend) * (virtual_value_extend - coupons_extend)
    best_coupon_index = np.argsort(rank_scores, axis = -1)

    best_index = best_coupon_index[:, :, -1]
    best_coupon = np.take_along_axis(coupons_extend, best_index[..., np.newaxis], axis=-1).squeeze(-1)
    best_ctrs = np.take_along_axis(ctr_extend * ctr_boost_extend, best_index[...,np.newaxis], axis=-1).squeeze(-1)

    return best_ctrs, best_coupon

def myerson_auction(ctr, valuation):
    n_auction, n_bidder = valuation.shape

    virtual_value = virtual_value_function(valuation)
    
    best_ctrs, best_coupon = coupon_function(ctr, valuation)
    rank_score = best_ctrs * (virtual_value - best_coupon)
    rank_index = np.argsort(rank_score, axis = -1)
    winner_index = rank_index[:, -1]
    second_index = rank_index[:, -2]


    inverse_input = rank_score[np.arange(n_auction), second_index] / best_ctrs[np.arange(n_auction), winner_index] + best_coupon[np.arange(n_auction), winner_index]
    inverse_input = np.where(inverse_input > 0, inverse_input, 0)
    payment = np.zeros(n_auction)
    is_allocate = rank_score[np.arange(n_auction), winner_index] > 0
    for i in range(n_auction):
        if rank_score[i, winner_index[i]] > 0:
            payment[i] = inverse_value_function(inverse_input[i])

    welfare = best_ctrs * (valuation-best_coupon)
    revenue = best_ctrs[np.arange(n_auction), winner_index] * (payment - best_coupon[np.arange(n_auction), winner_index]) * is_allocate
    return np.mean(welfare[np.arange(n_auction), winner_index] * is_allocate), np.mean(revenue)

def main():
    dir = "../data/uniform.npy"
    ctr, value = read_data(dir)
    social, revenue = myerson_auction(ctr, value)
    print(f"Advertisers Utility: {social-revenue}, Revenue: {revenue}")

if __name__ == '__main__':
    main()