import numpy as np
from scipy.stats import uniform
from scipy import integrate
import pickle

def read_data(dir):
    with open(dir, 'rb') as f:
        data =pickle.load(f)
    ctr, value = data['ctr'], data['value']
    return ctr, value


def virtual_value_function(valuations):
    return valuations - ((1 - uniform.cdf(valuations, loc = 5, scale = 45)) / uniform.pdf(valuations, loc = 5, scale = 45))

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



def vectorized_find_index(value_matrix, k_vector):
    def helper(value_list, k):
        l, r = 0, len(value_list) - 1
        while(l < r):
            mid = (l + r + 1) // 2
            if value_list[mid] > k:
                r = mid - 1
            else:
                l = mid
        return l 
    n_auction, num_points = value_matrix.shape
    indices = np.zeros(n_auction, dtype=int)
    for i in range(n_auction):
        indices[i] = helper(value_matrix[i], k_vector[i])
    return indices

def vectorized_allocation_rule(ctr):
    n_auction, = ctr.shape
    value_enumerate = np.linspace(uniform.ppf(0.01, loc=5, scale=45), uniform.ppf(0.99, loc = 5, scale = 45), 1000)
    value_enumerate_extend = np.broadcast_to(value_enumerate, (n_auction, 1000))
    value_enumerate_extend = value_enumerate_extend.repeat(4).reshape((n_auction, 1000, 4))
    coupons = np.array([0, 2, 4, 8])
    ctr_boost = np.array([1, 1.1, 1.2, 1.3])
    ctr_extend = ctr.repeat(1000 * 4).reshape((n_auction, 1000, 4))
    coupons_extend = np.broadcast_to(coupons, (n_auction, 1000, 4))
    ctr_boost_extend = np.broadcast_to(ctr_boost, (n_auction, 1000, 4))

    rank_score = (ctr_extend * ctr_boost_extend) * (virtual_value_function(value_enumerate_extend) - coupons_extend)

    best_coupon_index = np.argsort(rank_score, axis = -1)
    best_index = best_coupon_index[:, :, -1]
    best_coupon = np.take_along_axis(coupons_extend, best_index[..., np.newaxis], axis=-1).squeeze(-1)
    best_ctrs = np.take_along_axis(ctr_extend * ctr_boost_extend, best_index[..., np.newaxis], axis=-1).squeeze(-1)
    best_virtual_values = np.take_along_axis(rank_score, best_index[..., np.newaxis], axis = -1).squeeze(-1)
    allocations = np.where(best_ctrs > 0, best_ctrs, 0)

    return allocations, value_enumerate, best_virtual_values




def main():
    dir = '../data/uniform.npy'
    ctr, value = read_data(dir)
    n_auction, n_bidder = value.shape
    ctr_new, coupons = coupon_function(ctr, value)

    virtual_values = virtual_value_function(value)

    rank_score = ctr_new * (virtual_values - coupons)
    rank_index = np.argsort(rank_score, axis = -1)
    first_index = rank_index[:, -1]
    second_index = rank_index[:, -2]

    social_welfare = ctr_new * value

    payment_input = rank_score[np.arange(n_auction), second_index]

    allocations, value_enumerate, best_virtual_values = vectorized_allocation_rule(ctr[np.arange(n_auction), first_index])

    start = vectorized_find_index(best_virtual_values, payment_input)
    end = vectorized_find_index(best_virtual_values, rank_score[np.arange(n_auction), first_index])

    payment = []
    for i, (s, e) in enumerate(zip(start, end)):
        if rank_score[i, first_index[i]] > 0:
            payment.append(allocations[i, e] * value[i, first_index[i]] - integrate.trapezoid(allocations[i, s:e], value_enumerate[s:e]))
        else:
            payment.append(0)
    cost = ctr_new * coupons
    is_allocate = rank_score[np.arange(n_auction), first_index] > 0
    coupon_cost = np.mean(cost[np.arange(n_auction), first_index] * is_allocate)
    print(f'cost: {np.mean(cost[np.arange(n_auction), first_index])}')
    return np.mean(social_welfare[np.arange(n_auction), first_index] * is_allocate), np.mean(payment), coupon_cost

    # revenue = payment_rule(ctr[np.arange(n_auction), first_index], value[np.arange(n_auction), first_index], payment_input)



if __name__ == '__main__':
    welfare, revenue, coupon_cost = main()
    print(f"Social Welfare: {welfare-revenue}, Revenue: {revenue-coupon_cost}")

