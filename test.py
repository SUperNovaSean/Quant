#%%
import pandas as pd
import numpy as np

#%%
# 处理市值
market_value = pd.read_csv('/home/puzh/task/hw2/Shareholder Value Market Value.csv', index_col=0)
market_value = market_value.set_index(["S_INFO_WINDCODE", "TRADE_DT"]).unstack(0)
market_value.columns = market_value.columns.droplevel(0)

mask_big_mv =  market_value.gt(market_value.median(1), axis=0)
mask_small_mv = market_value.lt(market_value.median(1), axis=0)

# %%
asset_tt = pd.read_csv('/home/puzh/task/hw2/TOT_ASSETS.csv', index_col=0)
asset_tt = (asset_tt.groupby(["S_INFO_WINDCODE", "TRADE_DT"]).sum()).unstack(0).ffill()
asset_tt.columns = asset_tt.columns.droplevel(0)

asset_lb = pd.read_csv('/home/puzh/task/hw2/TOT_LIAB.csv', index_col=0)
asset_lb = (asset_lb.groupby(["S_INFO_WINDCODE", "TRADE_DT"]).sum()).unstack(0).ffill()
asset_lb.columns = asset_lb.columns.droplevel(0)

bm = (asset_tt - asset_lb)/market_value

mask_high_bm = bm.gt(bm.median(1), axis=0)
mask_low_bm = bm.lt(bm.median(1), axis=0)


# %%
close = pd.read_csv("/home/puzh/task/hw2/CLOSE.csv", index_col=0)
close = close.set_index(["S_INFO_WINDCODE", "TRADE_DT"]).unstack(0)
close.columns = close.columns.droplevel(0)

#%%
ret = close.pct_change().shift(-1)

m_ret = ret.mean(axis=1)
mv_ret = ret[mask_big_mv].mean(1) - ret[mask_small_mv].mean(1)
bm_ret = ret[mask_high_bm].mean(1) - ret[mask_low_bm].mean(1)


#%%
import numpy as np
from sklearn.linear_model import LinearRegression

def linear_regression_residuals(x, y):
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(x)

    residuals = y - y_pred

    return residuals

#%%
from tqdm import tqdm
factors = pd.concat([m_ret, mv_ret, bm_ret], axis=1)
factors.columns = ["RMRF", "SMB", "HML"]
factors.index = pd.to_datetime(factors.index.astype(str))
ret.index = pd.to_datetime(ret.index.astype(str))
t_res = []
for (mn, mn_df), (mn, mn_ret) in tqdm(zip(factors.groupby(factors.index.month), ret.groupby(ret.index.month))):
    code_res = []
    for ret_code in ret.columns:
        y = mn_ret[ret_code].values
        res = linear_regression_residuals(mn_df.values, y)
        code_res.append(res)
    code_res = np.vstack(code_res)
    t_res.append(code_res)
final_res = np.hstack(t_res)
resudial = pd.DataFrame(final_res.T, index=factors.index, columns=ret.columns)

resudial_m = resudial.resample("1M").mean()
rolling_mean = resudial_m.shift(1).rolling(6).mean()
ffm_moment = resudial_m.shift(1).rolling(6).sum() / np.sqrt((resudial_m - rolling_mean).rolling(6).sum()**2)
fft_moment_d = ffm_moment.resample("D").ffill()
fft_moment_d = fft_moment_d["20210901":"20230901"].dropna(how="all")
ret_ = ret.reindex(index=fft_moment_d.index, columns=fft_moment_d.columns)
#%%
from bk_test import BKInterface
bkm = BKInterface(ret_)
bkm.run_backtest(fft_moment_d)
                    
    
#%%

tsmom = (ret - ret.ewm(span=12*30).mean()).rolling(12*30).mean().apply(lambda x: np.sign(x))*(ret - ret.ewm(span=12*30).mean())/(ret - ret.ewm(span=12*30).std())
tsmom = tsmom["20210901":"20230901"].dropna(how="all")
ret_ = ret.reindex(index=tsmom.index, columns=tsmom.columns)
from bk_test import BKInterface
bkm = BKInterface(ret_)
bkm.run_backtest(tsmom)
                    
    



# %%
