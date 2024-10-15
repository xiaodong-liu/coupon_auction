import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
from scipy.stats import lognorm
import random
import os
from tqdm import tqdm


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
        items = self.item_embedding(x[..., 0])
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

def main(values, x_item, user_info, ad_idx):
    cvr_model = CVR_Model()
    cvr_model.load_state_dict(torch.load('dataset/cvr_model_11.pt'))
    cvr_model.eval()

    # user_info_select = pd.read_csv("dataset/user_info_selected_processed.csv")
    # item_info_select = pd.read_csv("dataset/item_info_processed.csv")
    #
    # x_item = np.array(item_info_select)
    # user_info = np.array(user_info_select)
    sigmoid = nn.Sigmoid()
    cvrs = torch.zeros([100000, 100])
    best_coupons = torch.zeros([100000, 100])
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
        values_extend = values[:, i].unsqueeze(1).repeat(1, 100)
        rank_scores = cvrs_tmp * (values_extend - coupons_enumerate)
        rank_scores_sort = torch.argsort(rank_scores, dim=-1)
        best_index = rank_scores_sort[:, -1]
        cvrs[:, i] = cvrs_tmp[torch.arange(100000), best_index]
        best_coupons[:, i] = coupons_enumerate[torch.arange(100000), best_index]
        if i == ad_idx:
            cvrs[:, i] = cvrs_tmp[torch.arange(100000), 0]
            best_coupons[:, i] = 0

    rank_scores = cvrs * (values - best_coupons)

    rank_scores_rerank = torch.argsort(rank_scores, dim=-1)
    winner_index = rank_scores_rerank[:, -1]
    second_index = rank_scores_rerank[:, -2]
    # for i in range(100):
    #     print(f"{i}: {torch.sum(winner_index == i)}")
    is_ad = (winner_index == ad_idx)
    welfare = torch.mean(rank_scores[torch.arange(100000), winner_index] * is_ad)
    revenue = torch.mean(rank_scores[torch.arange(100000), second_index] * is_ad)

    print(f"{ad_idx}: Welfare: {welfare - revenue}, Revenue: {revenue}")

    cvrs = torch.zeros([100000, 100])
    best_coupons = torch.zeros([100000, 100])
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
        values_extend = values[:, i].unsqueeze(1).repeat(1, 100)
        rank_scores = cvrs_tmp * (values_extend - coupons_enumerate)
        rank_scores_sort = torch.argsort(rank_scores, dim=-1)
        best_index = rank_scores_sort[:, -1]
        cvrs[:, i] = cvrs_tmp[torch.arange(100000), best_index]
        best_coupons[:, i] = coupons_enumerate[torch.arange(100000), best_index]
        # if i == ad_idx:
        #     cvrs[:, i] = cvrs_tmp[torch.arange(100000), 0]
        #     best_coupons[:, i] = 0

    rank_scores = cvrs * (values - best_coupons)

    rank_scores_rerank = torch.argsort(rank_scores, dim=-1)
    winner_index = rank_scores_rerank[:, -1]
    second_index = rank_scores_rerank[:, -2]
    # for i in range(100):
    #     print(f"{i}: {torch.sum(winner_index == i)}")
    is_ad = (winner_index == ad_idx)
    welfare = torch.mean(rank_scores[torch.arange(100000), winner_index] * is_ad)
    revenue = torch.mean(rank_scores[torch.arange(100000), second_index] * is_ad)

    print(f"{ad_idx}: Welfare: {welfare - revenue}, Revenue: {revenue}")

if __name__ == '__main__':
    set_all_seeds(1024)
    user_info_select = pd.read_csv("dataset/user_info_selected_processed.csv")
    item_info_select = pd.read_csv("dataset/item_info_processed.csv")

    x_item = np.array(item_info_select)
    user_info = np.array(user_info_select)
    values = torch.tensor(x_item[:, -1])
    mus = torch.zeros(100)
    sigmas = torch.ones(100)
    for i in range(100):
        mus[i] = torch.log(values[i]) - 0.5 - 1

    values = torch.zeros((100000, 100))
    for i in range(100):
        values[:, i] = torch.tensor(lognorm.rvs(size=(100000,), s=sigmas[i], scale=np.exp(mus[i])))

    for i in range(100):
        main(values,x_item, user_info, i)