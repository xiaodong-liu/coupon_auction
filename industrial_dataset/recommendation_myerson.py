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
def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_all_seeds(1024)

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
    return  x - ((1 - lognorm.cdf(x, s = sigma, scale = np.exp(mu))) / lognorm.pdf(x, s=sigma, scale=np.exp(mu)))

def inverse_value_function(x, upper_bound, eps, mu, sigma):
    l = 0
    r = upper_bound
    mid = (l + r) / 2.0
    max_iteration = 100
    cnt = 1
    tmp = virtual_value_function(mid, mu, sigma) - x
    while(np.abs(tmp) > eps):
        if (cnt > max_iteration):
            return -1
        if tmp < 0:
            l = mid
        else:
            r = mid
        mid = (l + r) / 2.0
        tmp = virtual_value_function(mid, mu, sigma) - x
        cnt += 1
    return mid

def inverse_value_function_debug(x, upper_bound, eps, mu, sigma):
    cnt = 0
    max_iterations = 1000
    l = 0
    r = upper_bound
    mid = (l + r) / 2.0
    if(virtual_value_function(r, mu, sigma) - x < 0):
        print(x, r) 
    tmp = virtual_value_function(mid, mu, sigma) - x
    while(np.abs(tmp) > eps):
        if tmp < 0:
            l = mid
        else:
            r = mid
        mid = (l + r) / 2.0
        tmp = virtual_value_function(mid, mu, sigma) - x
        cnt += 1
    return mid

def payment_rule(value, payment_input, mu, sigma):
    if(virtual_value_function(value, mu, sigma) < 0):
        return 0.0
    eps = 0.1
    p = inverse_value_function(payment_input, value, eps, mu, sigma)
    return p

def payment_rule_debug(value, payment_input, mu, sigma):
    eps = 0.01
    p = inverse_value_function_debug(payment_input, value, eps, mu, sigma)
    return p
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
    values[:, i] = torch.tensor(lognorm.rvs(size = (100000,), s = sigmas[i], scale = np.exp(mus[i])))

cvrs = torch.zeros([100000, 100])
best_coupons = torch.zeros([100000, 100])

virtual_value = torch.zeros_like(values)
for i in range(100):
    virtual_value[:, i] = virtual_value_function(values[:, i], mus[i], sigmas[i])

for i in tqdm(range(100)):
    user_feature = user_info[:, 1:10]
    item_feature = np.broadcast_to(x_item[i, :10], (100000, 10))
    x_feature = np.concatenate([item_feature[:, 0:9], user_feature, item_feature[:, 9:10]], axis = -1)
    category = torch.LongTensor(x_feature[:, 0:15])
    numerical = torch.FloatTensor(x_feature[:, 15:])
    coupons = torch.zeros((100000, 1))
    cvrs_tmp = sigmoid(cvr_model.enumerate_coupon_values(category, torch.cat([numerical,coupons], dim=-1), 100, values[:, i])).detach().squeeze(-1)
    
    coupons_enumerate = vectorized_linspace(torch.zeros(100000), values[:, i], 100)
    virtual_values_extend = virtual_value[:, i].unsqueeze(1).repeat(1, 100)
    rank_scores = cvrs_tmp * (virtual_values_extend - coupons_enumerate)
    rank_scores_sort = torch.argsort(rank_scores, dim = -1)
    best_index = rank_scores_sort[:, -1]
    cvrs[:, i] = cvrs_tmp[torch.arange(100000), best_index]
    best_coupons[:, i] = coupons_enumerate[torch.arange(100000), best_index]



print(best_coupons)
rank_scores = cvrs * (virtual_value - best_coupons)

rank_scores_rerank = torch.argsort(rank_scores, dim = -1)

winner_index = rank_scores_rerank[:, -1]
second_index = rank_scores_rerank[:, -2]

welfare_all = cvrs * (values - best_coupons)

allocation = rank_scores[torch.arange(100000), winner_index] > 0


payment_input = (rank_scores[torch.arange(100000), second_index] / cvrs[torch.arange(100000), winner_index]) + best_coupons[torch.arange(100000), winner_index]
payment_input = torch.where(payment_input >= 0, payment_input, 0)
payments = torch.zeros(100000,)
print(rank_scores[torch.arange(100000), winner_index])
for i in tqdm(range(100000)):
        # payments[i] = payment_rule_debug(values[i, winner_index[i]], payment_input[i], mus[winner_index[i]], sigmas[winner_index[i]])
    if(allocation[i] is False):
        continue
    t = payment_rule(values[i, winner_index[i]], payment_input[i], mus[winner_index[i]], sigmas[winner_index[i]])
    payments[i] = t


welfare = torch.mean(welfare_all[torch.arange(100000), winner_index] * allocation)
revenue = torch.mean(cvrs[torch.arange(100000),winner_index] * (payments - best_coupons[torch.arange(100000), winner_index]) * allocation)
print(f"Coupon Cost: {torch.mean(cvrs[torch.arange(100000),winner_index] * best_coupons[torch.arange(100000), winner_index] * allocation)}")
print(f"Welfare: {welfare-revenue}, Revenue: {revenue}")
