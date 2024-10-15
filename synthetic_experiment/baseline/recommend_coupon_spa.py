import numpy as np
import pickle
import random

def read_data(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    ctr, valuation = data['ctr'], data['value']
    return ctr, valuation

def coupon_function(ctr, valuation):
    n_auction, n_bidder = valuation.shape
    coupons = np.array([0, 2, 4, 8])
    ctr_boost = np.array([1, 1.1, 1.2, 1.3])
    # Enumerate all coupons to determine the best coupon for each bidder in each auction
    coupons_extend = np.broadcast_to(coupons, (n_auction, n_bidder, 4))
    ctr_boost_extend = np.broadcast_to(ctr_boost, (n_auction, n_bidder, 4))
    ctr_extend = ctr.repeat(4).reshape((n_auction, n_bidder, 4))
    valuation_extend = valuation.repeat(4).reshape((n_auction, n_bidder, 4))
    rank_scores = (ctr_extend * ctr_boost_extend) * (valuation_extend - coupons_extend)
    best_coupon_index = np.argsort(rank_scores, axis = -1)
    best_index = best_coupon_index[:, :, -1]
    best_coupon = np.take_along_axis(coupons_extend, best_index[...,np.newaxis], axis=-1).squeeze(-1)
    best_ctrs = np.take_along_axis(ctr_extend * ctr_boost_extend, best_index[..., np.newaxis], axis=-1).squeeze(-1)
    return best_ctrs, best_coupon


def second_price_auction(ctr, valuation):
    n_aucton, n_bidder = valuation.shape
    rank_score = ctr * valuation
    rank_index = np.argsort(rank_score, axis = -1)
    social_welfare = rank_score[np.arange(n_aucton), rank_index[:, -1]]
    payment = rank_score[np.arange(n_aucton), rank_index[:, -2]]
    return np.mean(social_welfare), np.mean(payment)

def main():
    dir = "../data/lognorm.npy"
    ctr, valuation = read_data(dir)
    ctr_new, coupons = coupon_function(ctr, valuation)
    social, revenue = second_price_auction(ctr_new, valuation-coupons)
    print(f"Advertiser Utility: {social-revenue}, Revenue: {revenue}")

if __name__ == '__main__':
    main()

