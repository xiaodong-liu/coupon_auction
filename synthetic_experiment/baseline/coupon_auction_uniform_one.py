import numpy as np
from scipy.stats import uniform
import pickle
from scipy import integrate
def read_data(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    ctr, value = data['ctr'], data['value']
    return ctr, value

def virtual_value_function(valuations):
    return valuations - ((1 - uniform.cdf(valuations, loc=5, scale=45)) / uniform.pdf(valuations, loc = 5, scale=45))

def coupon_function(ctr, valuation):
    n_auction, n_bidder = valuation.shape

    coupons = np.array([0, 2, 4, 8])
    ctr_boost = np.array([1, 1.1, 1.2, 1.3])
    coupon_extend = np.broadcast_to(coupons, (n_auction, n_bidder, 4))
    ctr_boost_extend = np.broadcast_to(ctr_boost, (n_auction, n_bidder, 4))

    ctr_extend = ctr.repeat(4).reshape((n_auction, n_bidder, 4))
    valuation_extend = valuation.repeat(4).reshape((n_auction, n_bidder, 4))

    virtual_values = virtual_value_function(valuation_extend)

    rank_score = (ctr_extend * ctr_boost_extend) * (virtual_values - coupon_extend)
    best_coupon_index = np.argsort(rank_score, axis = -1)
    best_index = best_coupon_index[:, :, -1]
    best_coupon = np.take_along_axis(coupon_extend, best_index[...,np.newaxis], axis = -1).squeeze(-1)
    best_ctrs = np.take_along_axis(ctr_extend * ctr_boost_extend, best_index[..., np.newaxis], axis=-1).squeeze(-1)
    return best_ctrs, best_coupon

def helper(ctr_initial, value):
    n_auction, = ctr_initial.shape
    coupons = np.array([0, 2, 4, 8])
    ctr_boost = np.array([1, 1.1, 1.2, 1.3])
    
    coupon_extend = np.broadcast_to(coupons, (n_auction, 4))
    ctr_boost_extend = np.broadcast_to(ctr_boost, (n_auction, 4))

    ctr_initial_extend = ctr_initial.repeat(4).reshape((n_auction, 4))
    value_extend = value.repeat(4).reshape((n_auction, 4))
    rank_score = (ctr_initial_extend * ctr_boost_extend) * (virtual_value_function(value_extend) - coupon_extend)
    best_coupon_index = np.argsort(rank_score, axis = -1)
    best_index = best_coupon_index[:, -1]

    return rank_score[np.arange(n_auction), best_index]



def inverse_virtual_value_function(ctr_initial, x, upper_bound, eps):
    l = np.zeros_like(x) + 5
    r = np.copy(upper_bound)
    mid = (l + r) / 2.0

    tmp = helper(ctr_initial, mid) - x
    while(np.all(np.abs(tmp)) > eps):
        l = np.where(tmp < 0, mid, l)
        r = np.where(tmp >= 0, mid, r)
        mid = (l + r) / 2.0
        tmp = helper(ctr_initial, mid) - x

    return mid

def vectorized_linspace(start, stop, num):
    t = np.linspace(0, 1, num)
    return start[:, np.newaxis] + t[np.newaxis, :] * (stop - start)[:, np.newaxis]

def payment_rule(ctr_initial, mid, value):
    n_auction, = ctr_initial.shape
    value_enumerate = vectorized_linspace(mid, value, 1000)
    value_enumerate_extend = value_enumerate.repeat(4).reshape((n_auction, 1000, 4))
    ctr_initial_extend = ctr_initial.repeat(1000 * 4).reshape((n_auction, 1000, 4))
    
    coupons = np.broadcast_to(np.array([0, 2, 4, 8]), (n_auction, 1000, 4))
    ctr_boost = np.broadcast_to(np.array([1, 1.1, 1.2, 1.3]), (n_auction, 1000, 4))

    rank_score = (ctr_initial_extend * ctr_boost) * (virtual_value_function(value_enumerate_extend) - coupons)

    best_coupon_index = np.argsort(rank_score, axis = -1)
    best_index = best_coupon_index[:, :, -1]
    best_ctrs = np.take_along_axis(ctr_initial_extend * ctr_boost, best_index[..., np.newaxis], axis = -1).squeeze(-1)
    payments = np.zeros(n_auction)
    for i in range(n_auction):
        payments[i] = integrate.simpson(y = best_ctrs[i], x = value_enumerate[i])
    return payments


def myerson_auction(ctr, value, coupons, ctr_initial):
    n_auction, n_bidder = value.shape

    virtual_values = virtual_value_function(value)
    rank_score = ctr * (virtual_values - coupons)
    rank_index = np.argsort(rank_score, axis = -1)

    winner_index = rank_index[:, -1]
    second_index = rank_index[:, -2]

    inverse_input = inverse_virtual_value_function(
        ctr_initial[np.arange(n_auction), winner_index], 
        rank_score[np.arange(n_auction), second_index], 
        value[np.arange(n_auction), winner_index],
        0.001
    )

    payment = payment_rule(
        ctr_initial[np.arange(n_auction), winner_index],
        inverse_input, 
        value[np.arange(n_auction), winner_index]
    )

    print(np.mean(rank_score[np.arange(n_auction), winner_index]))
    welfare = np.mean(ctr[np.arange(n_auction), winner_index] * value[np.arange(n_auction), winner_index])

    revenue = np.mean(ctr[np.arange(n_auction), winner_index] * value[np.arange(n_auction), winner_index]-payment)

    return welfare, revenue




if __name__ == '__main__':

    dir = '../data/uniform.npy'
    ctr, value = read_data(dir)

    ctr_new, coupons = coupon_function(ctr, value)

    welfare, revenue = myerson_auction(ctr_new, value, coupons, ctr)

    print(f"Welfare: {welfare}, Revenue: {revenue}")
