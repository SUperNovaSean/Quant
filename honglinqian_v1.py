import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from bk_test import BKInterface

# Load data and preprocess
def load_data(file_path, index_cols):
    data = pd.read_csv(file_path, index_col=0)
    try:
        data = data.set_index(index_cols).unstack(0)
    except:
        data = data.groupby(index_cols).sum().unstack(0)
    data.columns = data.columns.droplevel(0)
    return data

def calculate_bm(asset_tt, asset_lb, market_value):
    bm = (asset_tt - asset_lb) / market_value
    return bm

def calculate_residuals(x, y):
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    residuals = y - y_pred
    return residuals

def calculate_factors(ret, market_value, bm):
    mask_big_mv = market_value.gt(market_value.median(1), axis=0)
    mask_small_mv = market_value.lt(market_value.median(1), axis=0)
    mask_high_bm = bm.gt(bm.median(1), axis=0)
    mask_low_bm = bm.lt(bm.median(1), axis=0)

    m_ret = ret.mean(axis=1)
    mv_ret = ret[mask_big_mv].mean(1) - ret[mask_small_mv].mean(1)
    bm_ret = ret[mask_high_bm].mean(1) - ret[mask_low_bm].mean(1)
    return m_ret, mv_ret, bm_ret

def calculate_rolling_moments(resudial):
    resudial_m = resudial.resample("1M").mean()
    rolling_mean = resudial_m.shift(1).rolling(6).mean()
    ffm_moment = resudial_m.shift(1).rolling(6).sum() / np.sqrt((resudial_m - rolling_mean).rolling(6).sum() ** 2)
    return ffm_moment

def main():
    # Load data
    market_value = load_data('/home/puzh/task/hw2/Shareholder Value Market Value.csv', ["S_INFO_WINDCODE", "TRADE_DT"])
    asset_tt = load_data('/home/puzh/task/hw2/TOT_ASSETS.csv', ["S_INFO_WINDCODE", "TRADE_DT"])
    asset_lb = load_data('/home/puzh/task/hw2/TOT_LIAB.csv', ["S_INFO_WINDCODE", "TRADE_DT"])
    close = load_data("/home/puzh/task/hw2/CLOSE.csv", ["S_INFO_WINDCODE", "TRADE_DT"])

    # Calculate book-to-market ratio
    bm = calculate_bm(asset_tt, asset_lb, market_value)

    # Calculate factors
    ret = close.pct_change().shift(-1)
    m_ret, mv_ret, bm_ret = calculate_factors(ret, market_value, bm)
    factors = pd.concat([m_ret, mv_ret, bm_ret], axis=1)
    # Calculate residuals
    residuals = []
    for (mn, mn_df), (mn, mn_ret) in tqdm(zip(factors.groupby(factors.index.month), ret.groupby(ret.index.month))):
        code_res = []
        for ret_code in ret.columns:
            y = mn_ret[ret_code].values
            res = calculate_residuals(mn_df.values, y)
            code_res.append(res)
        code_res = np.vstack(code_res)
        residuals.append(code_res)
    final_res = np.hstack(residuals)
    resudial = pd.DataFrame(final_res.T, index=factors.index, columns=ret.columns)

    # Calculate rolling moments
    ffm_moment = calculate_rolling_moments(resudial)

    # Backtest
    fft_moment_d = ffm_moment.resample("D").ffill()
    fft_moment_d = fft_moment_d["20210901":"20230901"].dropna(how="all")
    ret_ = ret.reindex(index=fft_moment_d.index, columns=fft_moment_d.columns)
    bkm = BKInterface(ret_)
    bkm.run_backtest(fft_moment_d)

if __name__ == "__main__":
    main()