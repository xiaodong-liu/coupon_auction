import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
from scipy.stats import lognorm
from tqdm import tqdm
import os
import random
from scipy import integrate
def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def vectorized_linspace(start, stop, num):
    t = torch.linspace(0, 1, num)
    return start.unsqueeze(1) + t.unsqueeze(0) * (stop - start).unsqueeze(1)


class CVR_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_embedding = nn.Embedding(981, 8)
        self.first_industry_name = nn.Embedding(15, 4)
        self.second_industry_name = nn.Embedding(58, 8)
        self.seller_embedding = nn.Embedding(685, 8)
        self.category_level_1_name = nn.Embedding(47, 8)
        self.category_level_2_name = nn.Embedding(146, 8)
        self.detail_industry_id = nn.Embedding(59, 8)
        self.industry_id = nn.Embedding(16, 4)
        self.big_industry_id = nn.Embedding(16, 4)
        self.age_segment_ser = nn.Embedding(7, 4)
        self.user_active_degree = nn.Embedding(9, 4)
        self.device_brand = nn.Embedding(65, 8)
        self.country = nn.Embedding(66, 8)
        self.province = nn.Embedding(228, 8)
        self.city = nn.Embedding(520, 8)
        self.linear = nn.Linear(105, 1)

    def forward(self, x, numerical_feature):
        items = self.item_embedding(x[:, 0])
        first_industry = self.first_industry_name(x[:, 1])
        second_industry = self.second_industry_name(x[:, 2])
        seller_embedding = self.seller_embedding(x[:, 3])
        category_level_1_embedding = self.category_level_1_name(x[:, 4])
        category_level_2_embedding = self.category_level_2_name(x[:, 5])
        detail_industry_id = self.detail_industry_id(x[:, 6])
        industry_id = self.industry_id(x[:, 7])
        big_industry_id = self.big_industry_id(x[:, 8])
        age_segment_ser = self.age_segment_ser(x[:, 9])
        user_active_degree = self.user_active_degree(x[:, 10])
        device_brand = self.device_brand(x[:, 11])
        country = self.country(x[:, 12])
        province = self.province(x[:, 13])
        city = self.city(x[:, 14])
        feature_concat = torch.cat(
            (items, first_industry, second_industry, seller_embedding,
             category_level_1_embedding, category_level_2_embedding, detail_industry_id, industry_id, big_industry_id,
             age_segment_ser, user_active_degree, device_brand, country, province, city, numerical_feature), 1)
        logit = self.linear(feature_concat)
        return logit

    def enumerate_coupon_values(self, x, numerical_feature, enumerate_size, coupon_upper_bound):
        n_auction, _ = x.shape
        items = self.item_embedding(x[:, 0])
        first_industry = self.first_industry_name(x[:, 1])
        second_industry = self.second_industry_name(x[:, 2])
        seller_embedding = self.seller_embedding(x[:, 3])
        category_level_1_embedding = self.category_level_1_name(x[:, 4])
        category_level_2_embedding = self.category_level_2_name(x[:, 5])
        detail_industry_id = self.detail_industry_id(x[:, 6])
        industry_id = self.industry_id(x[:, 7])
        big_industry_id = self.big_industry_id(x[:, 8])
        age_segment_ser = self.age_segment_ser(x[:, 9])
        user_active_degree = self.user_active_degree(x[:, 10])
        device_brand = self.device_brand(x[:, 11])
        country = self.country(x[:, 12])
        province = self.province(x[:, 13])
        city = self.city(x[:, 14])

        category_features = torch.cat(
            (items, first_industry, second_industry, seller_embedding,
             category_level_1_embedding, category_level_2_embedding, detail_industry_id,
             industry_id, big_industry_id, age_segment_ser, user_active_degree, device_brand,
             country, province, city), 1
        )
        n_auction, category_feature_dim = category_features.shape
        n_auction, numerical_feature_dim = numerical_feature.shape
        # 然后每个复制n份，最后改一个维度的值
        category_features_extend = category_features.unsqueeze(1).repeat(1, enumerate_size, 1)
        numerical_features_extend = numerical_feature.unsqueeze(1).repeat(1, enumerate_size, 1)

        coupons_enumeration_extend = vectorized_linspace(torch.zeros(n_auction), coupon_upper_bound, enumerate_size)
        numerical_features_extend[:, :, -1] = coupons_enumeration_extend
        feature_concat = torch.cat(
            (category_features_extend, numerical_features_extend), dim=-1
        )
        logit = self.linear(feature_concat)
        return logit



def virtual_value_function(x, mu, sigma):
    return x - ((1 - lognorm.cdf(x, s=sigma, scale=np.exp(mu))) / lognorm.pdf(x, s=sigma, scale=np.exp(mu)))

def find_index(values, k):
    l, r = 0, len(values) -1
    while(l < r):
        mid = (l + r + 1) // 2
        if values[mid] > k:
            r = mid - 1
        else:
            l = mid
    return l


def payment_function(cvr_model, payment_input, value, mu, sigma, category_feature, numerical_feature):
    sigmoid = nn.Sigmoid()
    value_enumerate = torch.linspace(0.01, value, 2000)
    category_feature = category_feature.unsqueeze(0)
    numerical_feature = numerical_feature.unsqueeze(0)
    cvrs_tmp = sigmoid(cvr_model.enumerate_coupon_values(category_feature, numerical_feature, 100, value)).detach().squeeze()
    virtual_values = virtual_value_function(value_enumerate, mu, sigma)
    cvrs_tmp_extend = torch.broadcast_to(cvrs_tmp, (2000, 100))
    virtual_values_extend = virtual_values.reshape(-1, 1).repeat(1, 100)
    coupons_enumerate = torch.linspace(0, value, 100)
    coupons_enumerate_extend = torch.broadcast_to(coupons_enumerate, (2000, 100))
    rank_scores = cvrs_tmp_extend * (virtual_values_extend - coupons_enumerate_extend)

    rank_scores_sort = torch.argsort(rank_scores, dim = -1)
    best_index = rank_scores_sort[:, -1]
    cvrs = cvrs_tmp_extend[torch.arange(2000), best_index]
    # 计算start
    # 然后根据这两个位置，计算积分，并求解
    best_rank_scores = rank_scores[torch.arange(2000), best_index]
    s = find_index(best_rank_scores, payment_input)
    t = cvrs[-1] * value
    v = integrate.trapezoid(cvrs[s:], value_enumerate[s:])
    return t, v
# tensor(138.5154)
# cvr: tensor(1)

if __name__ == '__main__':
    set_all_seeds(1024)
    cvr_model = CVR_Model()
    cvr_model.load_state_dict(torch.load('dataset/cvr_model_11.pt'))
    cvr_model.eval()

    user_info_select = pd.read_csv("dataset/user_info_selected_processed.csv")
    item_info_select = pd.read_csv("dataset/item_info_processed.csv")

    x_item = np.array(item_info_select)
    user_info = np.array(user_info_select)
    sigmoid = nn.Sigmoid()
    values = torch.tensor(x_item[:, -1])
    mus = torch.zeros(100)
    sigmas = torch.ones(100)
    for i in range(100):
        mus[i] = torch.log(values[i]) - 0.5 - 1

    values = torch.zeros((100000, 100))
    for i in range(100):
        values[:, i] = torch.tensor(lognorm.rvs(size=(100000,), s = sigmas[i], scale=np.exp(mus[i])))

    cvrs = torch.zeros([100000, 100])
    best_coupons = torch.zeros([100000, 100])

    virtual_value = torch.zeros_like(values)
    for i in range(100):
        virtual_value[:, i] = virtual_value_function(values[:, i], mus[i], sigmas[i])

    for i in tqdm(range(100)):
        user_feature = user_info[:, 1:10]
        item_feature = np.broadcast_to(x_item[i, :10], (100000, 10))
        x_feature = np.concatenate([item_feature[:, 0:9], user_feature, item_feature[:, 9:10]], axis=-1)
        category = torch.LongTensor(x_feature[:, 0:15])
        numerical = torch.FloatTensor(x_feature[:, 15:])
        coupons = torch.zeros((100000, 1))
        cvrs_tmp = sigmoid(cvr_model.enumerate_coupon_values(category, torch.cat([numerical, coupons], dim=-1), 100,
                                                             values[:, i])).detach().squeeze(-1)

        coupons_enumerate = vectorized_linspace(torch.zeros(100000), values[:, i], 100)
        virtual_values_extend = virtual_value[:, i].unsqueeze(1).repeat(1, 100)
        rank_scores = cvrs_tmp * (virtual_values_extend - coupons_enumerate)
        rank_scores_sort = torch.argsort(rank_scores, dim=-1)
        best_index = rank_scores_sort[:, -1]
        cvrs[:, i] = cvrs_tmp[torch.arange(100000), best_index]
        best_coupons[:, i] = coupons_enumerate[torch.arange(100000), best_index]

    rank_scores = cvrs * (virtual_value - best_coupons)
    rank_scores_rerank =torch.argsort(rank_scores, dim = -1)
    winner_index = rank_scores_rerank[:, -1]
    second_index = rank_scores_rerank[:, -2]
    welfare_all = cvrs * (values-best_coupons)
    allocation = rank_scores[torch.arange(100000), winner_index] > 0
    payment_input = (rank_scores[torch.arange(100000), second_index])
    # payment_input = torch.where(payment_input > 0, payment_input, 0)
    cost = cvrs * best_coupons
    cost = cost[torch.arange(100000), winner_index] * allocation
    payments = torch.zeros(100000,)
    welfares = torch.zeros(100000,)
    print(rank_scores[torch.arange(100000), winner_index])
    for i in tqdm(range(100000)):
        if allocation[i] is False:
            continue
        ad_index = winner_index[i]
        user_feature = user_info[i, 1:10]
        item_feature = x_item[ad_index, :10]
        x_feature = np.concatenate([item_feature[0:9], user_feature, item_feature[9:10]], axis=-1)
        category = torch.LongTensor(x_feature[0:15])
        numerical = torch.FloatTensor(x_feature[15:])
        coupons = torch.zeros((1,))
        numerical = torch.cat([numerical, coupons], dim = -1)
        t, v = payment_function(cvr_model, payment_input[i], values[i, winner_index[i]], mus[winner_index[i]], sigmas[winner_index[i]], category, numerical)

        payments[i] = t-v
        welfares[i] = t
    print(f"Welfares: {torch.mean(welfares)}")
    welfare = torch.mean(welfares)
    print(f"Payment: {torch.mean(payments)}")
    print(f"Cost: {torch.mean(cvrs[torch.arange(100000),winner_index] * best_coupons[torch.arange(100000), winner_index] * allocation)}")
    revenue = torch.mean((pa                           yments - cvrs[torch.arange(100000),winner_index] * best_coupons[torch.arange(100000), winner_index]) * allocation)
    print(f"Welfare: {welfare-torch.mean(payments)}, Revenue: {revenue}")