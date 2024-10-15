import numpy as np
import pickle


def read_data(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    ctrs, valuations = data['ctr'], data['value']
    return ctrs, valuations

def second_price_auction(ctrs, valuations):
    num_auction, n_bidder = ctrs.shape

    rank_scores = ctrs * valuations

    rank_index = np.argsort(rank_scores, axis=-1)
    social_welfares = rank_scores[np.arange(num_auction), rank_index[:, -1]]
    payments = rank_scores[np.arange(num_auction), rank_index[:, -2]]

    return np.mean(social_welfares), np.mean(payments)


if __name__ == '__main__':
    dir = '../data/uniform.npy'
    ctrs, valuations = read_data(dir)
    sw, revenue = second_price_auction(ctrs, valuations)
    print(f"Advertisers Utility: {sw-revenue}, Revenue: {revenue}")