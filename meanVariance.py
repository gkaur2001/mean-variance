# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
"""
Spyder Editor

This is a temporary script file.
"""
#Reads in the excel file as a data frame and adjust the data to get excess returns for the 11 assets
data=pd.read_excel('assetclass_data_monthly_2009.xlsx')
data.iloc[:, 1:12] = data.iloc[:, 1:12].sub(data['Cash'], axis = 0)
adj_data = data.iloc[:, 1:12]

# Calculating the mean excess returns, volatility, and Sharpe ratio for each of the 11 assets
mu = adj_data.mean()
sigma = adj_data.std()
summary = pd.DataFrame({'Mean': mu, 'Volatility': sigma, 'Sharpe Ratio': mu/sigma})
print(summary)
print()

# Calculates the tangency portfolio of given data and return the weights and covariance matrix
def tangency_weights (data) :
    cov_matrix = data.cov()
    inverse_cov = np.linalg.inv(cov_matrix)
    mu_tilde = data.mean()
    tan_ratios = inverse_cov @ mu_tilde
    adj_ratios = tan_ratios / tan_ratios.sum()
    weights = pd.Series(adj_ratios, index = mu_tilde.index)
    return weights, cov_matrix
    
# Calculates tangency weights of the given data
tan_weights, cov_matrix = tangency_weights(adj_data)
print(tan_weights)
print()

# Calculates the mean of portfolio weights
def portfolio_mean (mu, weights):
    return mu @ weights

# Calculates the volatitility (standard deviation) of portfolio weights
def portfolio_vol (weights, cov_matrix):
    return np.sqrt(weights.transpose() @ cov_matrix @ weights)

# Mean, volatility, and sharpe ratio of the tangency portfolio
tan_mean = portfolio_mean(mu, tan_weights)
print(tan_mean)
tan_vol = portfolio_vol(tan_weights, cov_matrix)
print(tan_vol)
tan_sharpe = np.sqrt(mu.transpose() @ np.linalg.inv(cov_matrix) @ mu)
print(tan_sharpe)
print()

# Adjusts portfolio weights for target mean return
def target_adj (target_mean, actual_mean, tan_weights):
    return (target_mean / actual_mean) * tan_weights

# Calculates the weights, volatility, and Sharpe ratio for the target mean of 0.0067
target_mean = 0.0067
tar_weights = target_adj(target_mean,tan_mean, tan_weights)
print(tar_weights)
print()
tar_vol = portfolio_vol(tar_weights, cov_matrix)
print(tar_vol)
tar_sharpe = target_mean / tar_vol
print(tar_sharpe)
print()

# Calculates portfolio weights for target mean of 0.0067 with only domestic and foreign equity assets
equity_data = data.iloc[:, 1:3]
eq_tan_weights, eq_cov = tangency_weights(equity_data)
eq_mean = portfolio_mean(equity_data.mean(), eq_tan_weights)
tar_eq_weights = target_adj(target_mean , eq_mean, eq_tan_weights)
print(tar_eq_weights)
print()

# Calculates portfolio weights fro the equity data with the addition of .001 to excess returns monthly
adj_eq_data = equity_data
adj_eq_data['Foreign Equity'] = equity_data['Foreign Equity'].add(.001)
adj_eq_weights, adj_eq_cov = tangency_weights(equity_data)
adj_eq_mean = portfolio_mean(equity_data.mean(), eq_tan_weights)
tar_adj_weights = target_adj(target_mean, adj_eq_mean, adj_eq_weights)
print(tar_adj_weights)
print()

# Calculates portfolio weights for a diagonalized covariance matrix
def modified_weights (data):
    cov_matrix = np.diag(np.diag(data.cov()))
    inverse_cov = np.linalg.inv(cov_matrix)
    mu_tilde = data.mean()
    tan_ratios = inverse_cov @ mu_tilde
    adj_ratios = tan_ratios / tan_ratios.sum()
    weights = pd.Series(adj_ratios, index = mu_tilde.index)
    return weights, cov_matrix
    
# Calculates portfolio weights fro entire data set using modified covariance matrix
mod_tan_weights, mod_cov = modified_weights(adj_data)
print(mod_tan_weights)
print()

# Calculates the portfolio weights up until 2016
sixteen_data = adj_data.iloc[1:96, :]
sixteen_weights, sixteen_cov = tangency_weights(sixteen_data)
sixteen_mean = portfolio_mean(sixteen_data.mean(), sixteen_weights)
tar_sixteen_weights = target_adj(target_mean, sixteen_mean, sixteen_weights)
print(tar_sixteen_weights)
print()

# Calculates Sharpe ratio for data until 2016
tar_sixteen_vol = portfolio_vol(tar_sixteen_weights, sixteen_cov)
tar_sixteen_sharpe = target_mean / tar_sixteen_vol
print(tar_sixteen_sharpe)
print()

# Calucualtes portfolio weights for 2017 to 2019
recent_data = adj_data.iloc[96:, :]
rec_weights, rec_cov = tangency_weights(recent_data)
rec_mean = portfolio_mean(recent_data.mean(), rec_weights)
tar_rec_weights = target_adj(target_mean, rec_mean, rec_weights)

# Calucualtes Sharpe ratio for 2017 to 2019
tar_rec_vol = portfolio_vol(tar_rec_weights, rec_cov)
tar_rec_sharpe = target_mean / tar_rec_vol
print(tar_rec_sharpe)
print()

#Calculates Sharpe ratio using diagonalized matrix with data up until 2016
six_mod_weights, six_mod_cov = modified_weights(sixteen_data)
six_mod_mean = portfolio_mean(sixteen_data.mean(), six_mod_weights)
tar_six_weights = target_adj(target_mean, six_mod_mean, six_mod_weights)
tar_six_vol = portfolio_vol(tar_six_weights, six_mod_cov)
tar_six_sharpe = target_mean / tar_six_vol
print(tar_six_sharpe)
print()

#Calculates Sharpe ratio using diagonalized matrix with data up from 2017 to 2019
rec_mod_weights, rec_mod_cov = modified_weights(recent_data)
rec_mod_mean = portfolio_mean(recent_data.mean(), rec_mod_weights)
tar_rec_weights = target_adj(target_mean, rec_mod_mean, rec_mod_weights)
tar_rec_vol = portfolio_vol(tar_rec_weights, rec_mod_cov)
tar_rec_sharpe = target_mean / tar_rec_vol
print(tar_rec_sharpe)
