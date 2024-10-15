import numpy as np
from scipy.stats import lognorm
import pickle 
from tqdm import tqdm

def read_data(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    ctr, value = data['ctr'], data['value']
    return ctr, value

def coupon_function(ctr, valuation):
    n_auction, n_bidder= valuation.shape
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
    best_coupon = np.take_along_axis(coupon_extend, best_index[..., np.newaxis], axis=-1).squeeze(-1)
    best_ctrs = np.take_along_axis(ctr_extend * ctr_boost_extend, best_index[..., np.newaxis], axis=-1).squeeze(-1)
    return best_ctrs, best_coupon

def virtual_value_function(valuations):
    return valuations - ((1 - lognorm.cdf(valuations, s = 1, scale = np.exp(3))) / lognorm.pdf(valuations, s = 1, scale=np.exp(3)))

def inverse_virtual_value_function(x, upper_bound, eps):
    l = x
    r = upper_bound
    mid = (l + r) / 2.0
    
    tmp = virtual_value_function(mid) - x
    while (np.abs(tmp)> eps):
        l = mid if tmp < 0 else l
        r = mid if tmp >=0 else r
        mid = (l + r) / 2.0
        tmp = virtual_value_function(mid) - x

    return mid

def myerson_auction(ctr, valuation, coupons):
    n_auction, n_bidder = valuation.shape

    virtual_values = virtual_value_function(valuation)
    rank_score = ctr * (virtual_values - coupons)
    rank_index = np.argsort(rank_score, axis=-1)
    winner_index = rank_index[:, -1]
    second_index = rank_index[:, -2]

    inverse_input = rank_score[np.arange(n_auction), second_index] / ctr[np.arange(n_auction), winner_index] + coupons[np.arange(n_auction), winner_index]
    
    inverse_input = np.where(inverse_input < 0, 0, inverse_input)
    is_allocate = rank_score[np.arange(n_auction), winner_index] > 0
    print(np.mean(rank_score[np.arange(n_auction), winner_index] * is_allocate))
    cost = ctr * coupons
    payment = np.zeros(n_auction)
    for i in tqdm(range(n_auction)):
        if is_allocate[i]:
            payment[i] = inverse_virtual_value_function(inverse_input[i], valuation[i, winner_index[i]], eps = 0.01)
    welfare = ctr * (valuation - coupons)
    revenue = ctr[np.arange(n_auction), winner_index] * (payment-coupons[np.arange(n_auction), winner_index]) * is_allocate
    return np.mean(welfare[np.arange(n_auction), winner_index] * is_allocate), np.mean(revenue)

def main():
    dir = '../data/lognorm.npy'
    ctr, value = read_data(dir)
    ctr_new, coupons = coupon_function(ctr, value)
    social, revenue = myerson_auction(ctr_new, value, coupons)
    print(f"Social Welfare: {social-revenue}, Revenue: {revenue}")

if __name__ == '__main__':
    main()
