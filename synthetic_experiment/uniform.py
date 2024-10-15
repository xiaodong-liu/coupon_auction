
import numpy as np
from scipy.stats import lognorm, uniform
import matplotlib.pyplot as plt
from scipy import integrate

class Coupon_Auction(object):
    def __init__(self, n, m):
        # m auction, n bidders
        
        self.num_bidder = n 
        self.num_auction = m
        self.valuations = self.random_generate(self.num_auction, self.num_bidder)
    
    def coupon_function(self):
        raise NotImplemented

    def random_generate(self, m, n):
        raise NotImplemented
    
    def allocation_payment(self):
        raise NotImplemented
    

class Args(object):
    def __init__(self):
        self.num_bidder =
        self.num_auction = 
        self.valuefunction = 


if __name__ == '__main__':
    
    coupon_auction = Coupon_Auction()

