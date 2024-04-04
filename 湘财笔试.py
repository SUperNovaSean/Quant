#%%
############ 第一题
import numpy as np
import pandas as pd
df = pd.read_csv("./daily.csv")
df = df.set_index(["TICKER", "DATE"])
# %%
vol = df[["volume"]].unstack(0)
shr = df[["float_a_shr"]].unstack(0)
vol.columns = vol.columns.get_level_values(1)
shr.columns = shr.columns.get_level_values(1)
T = vol/shr
ret = df[["adj_close"]].unstack(0).pct_change()
ret.columns = ret.columns.get_level_values(1)

rolling_T = np.lib.stride_tricks.sliding_window_view(T, axis=0, window_shape=20)
rolling_ret = np.lib.stride_tricks.sliding_window_view(ret, axis=0, window_shape=20)
group_max = (np.argsort(-rolling_T) < 4).astype(int)
group_min = (np.argsort(rolling_T) < 4).astype(int)
R1 = (group_max * rolling_ret).sum(axis=-1)/group_max.sum(axis=-1)
R2 = (group_min * rolling_ret).sum(axis=-1)/group_min.sum(axis=-1)
R = R1-R2
res = pd.DataFrame(R, index=ret.index[19:], columns=ret.columns)

# %%
############ 第二题
import numpy as np
ret = df[["adj_close"]].unstack(0).pct_change()
ret.columns = ret.columns.get_level_values(1)
ret_d = np.sign(np.nan_to_num(ret)).astype(np.int8)
cs_ret_d = ret_d[:, None, :] * ret_d[:, :, None]
cs_ret_d = (cs_ret_d > 0).astype(np.int8)
rolling_ret = np.lib.stride_tricks.sliding_window_view(cs_ret_d, axis=0, window_shape=50)
n_same = rolling_ret.sum(-1)
n_res = []
from joblib import Parallel, delayed
from tqdm import tqdm
def compute_partition(i):
    res = np.argpartition(n_same[i, ...], -30)[-30:]
    return res
with Parallel(n_jobs=16) as parallel:
    n_res = parallel(delayed(compute_partition)(i) for i in tqdm(range(n_same.shape[0])))

stock_index = np.array(n_res).transpose(2, 0 ,1)
average_yield_rates = np.zeros((2813, 194))

ret_data = ret.values[49:]
for stock_idx in range(stock_index.shape[0]):
    for time_idx in range(stock_index.shape[1]):
        related_stocks_indices = stock_index[stock_idx, time_idx, :]
        related_stocks_yield_rates =ret_data[time_idx, related_stocks_indices]
        average_yield = np.mean(related_stocks_yield_rates)
pd.DataFrame(average_yield_rates.T, index=ret.index[49:], columns=ret.columns)


# %%
import numpy as np

def d2(X, Y):
    n = len(X)
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    std_X = np.std(X, ddof=1)
    std_Y = np.std(Y, ddof=1)

    corr_coef = np.sum((X - mean_X) * (Y - mean_Y)) / ((n - 1) * std_X * std_Y)
    d_corr_d_x1 = (-2 * (X[0] - mean_X) * np.sum((Y - mean_Y) * (X - mean_X)) - 
                  (n - 1) * std_Y * np.sum((Y - mean_Y)) + 
                  corr_coef * (n - 1) * std_Y * (n * (X[0] - mean_X))) / ((n - 1)**2 * std_X**3 * std_Y)
    
    d_corr_d_x1 = (-2 * (X[0] - mean_X) * np.sum((Y - mean_Y) * (X - mean_X)) - 
                  (n - 1) * std_Y * np.sum((Y - mean_Y)) + 
                  corr_coef * (n - 1) * std_Y * (n * (X[0] - mean_X))) / ((n - 1)**2 * std_X**3 * std_Y)
    
    d_corr_d_x1_plus_delta = (-2 * (X_plus_delta[0] - mean_X_plus_delta) * np.sum((Y - mean_Y) * (X_plus_delta - mean_X_plus_delta)) - 
                             (n - 1) * std_Y * np.sum((Y - mean_Y)) + 
                             corr_coef_plus_delta * (n - 1) * std_Y * (n * (X_plus_delta[0] - mean_X_plus_delta))) / ((n - 1)**2 * std_X_plus_delta**3 * std_Y)
    
    d2_corr_d_x1_2 = (d_corr_d_x1_plus_delta - d_corr_d_x1) / delta
    
    return d2_corr_d_x1_2

# Example usage:
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 2.5, 3, 3.5, 4])

second_derivative = d2(X, Y)
# %%
