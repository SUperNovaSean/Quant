from loguru import logger
import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator

#%%
N_GROUPS = 10
class BKInterface:
    def __init__(self, cross_return) -> None:
        self.cross_return = cross_return

    def prepare_data(self, signals, cross_return):
        signals = signals.reindex(columns=cross_return.columns)
        return signals, cross_return


    
    def run_backtest(self, signals, percentage=None, univ=None):
        signals = signals.sort_index()
        dates = signals.index
        cross_return = self.cross_return
        signals, cross_return = self.prepare_data(signals, cross_return)
        logger.info("Calculate long-short return")
        metrics_returns, metrics_cum_returns,  cum_returns, cum_excess_returns = self.cumulative_return(signals, cross_return, percentage, univ)
        logger.info("Calculate grouped return")
        grouped_cum_returns = self.tiered_return(signals, cross_return, n_groups=N_GROUPS, univ=univ)
        logger.info("Calculate IC")
        ic_summary = self.cumulative_ic(signals, cross_return, percentage)
        logger.info("Calculate metrics")
        metrics = self.metrics(signals,metrics_cum_returns, metrics_returns)
        logger.info("Plot long-short return")
        self.plot_return(dates, cum_returns, cum_excess_returns, metrics)
        logger.info("Plot IC")
        self.plot_ic(dates, ic_summary)
        logger.info("Plot grouped return")
        self.plot_group_return(dates, grouped_cum_returns)

    def tiered_return(self, signals, cross_return,  n_groups, univ):
        quantiles = np.linspace(0, 1, n_groups+1)
        lb = np.nanquantile(signals, quantiles[:n_groups], 1)
        ub = np.nanquantile(signals, quantiles[-n_groups:], 1)
        lb = lb[..., np.newaxis].repeat(signals.shape[1], axis=-1).transpose(1, 2, 0)
        ub = ub[..., np.newaxis].repeat(signals.shape[1], axis=-1).transpose(1, 2, 0)
        mask = (signals.values[..., None] < ub) & (signals.values[..., None] > lb)
        group_return = np.nansum(cross_return.values[..., None] * mask, axis=1)/ (np.nansum(mask, axis=1) +1e-5)
        group_cum_return = np.cumprod(1 +np.nan_to_num(group_return), axis=0)
        bench_return = np.nanmean(cross_return, 1) if univ is None else np.nanmean(cross_return[univ], 1)
        bench_return =  np.cumprod(1 + np.nan_to_num(bench_return), axis=0)
        gc_return = {"groups": group_cum_return, "bench": bench_return}
        return gc_return


    
    def cumulative_return(self, signals, cross_return,  percentage, univ):
        percentage = 0.5 if percentage is None else percentage
        quantiles = np.nanquantile(signals, [1 - percentage, percentage], axis=1)
        f = lambda x: quantiles[x][:, None].repeat(signals.shape[1], axis=-1)
        long_quantile, short_quantile = f(0), f(1)
        long_return = (cross_return * (signals > long_quantile)).sum(axis=1)/((signals > long_quantile).sum(axis=1) +1e-5)
        short_return = (cross_return * (signals < short_quantile)).sum(axis=1)/((signals < long_quantile).sum(axis=1) +1e-5)
        bench_return = cross_return.mean(axis=1) if univ is None else cross_return[univ].mean(axis=1)
        long_short_return = np.cumprod(1 + long_return.fillna(0) - short_return.fillna(0), axis=0)
        long_bench_return =  np.cumprod(1 + long_return.fillna(0) - bench_return.fillna(0), axis=0)
        bench_short_return =  np.cumprod(1 + bench_return.fillna(0) - short_return.fillna(0), axis=0)
        cum_long_return =  np.cumprod(1 + long_return.fillna(0), axis=0)
        cum_short_return =  np.cumprod(1 + short_return.fillna(0), axis=0)
        cum_bench_return =  np.cumprod(1 + bench_return.fillna(0), axis=0)
        metrics_returns = {"long-short": long_return - short_return, "long": long_return - bench_return, "short": bench_return - short_return}
        metrics_cum_returns = {"long-short": long_short_return, "long":long_bench_return, "short": bench_short_return}
        cum_returns = {"long": cum_long_return, "short": cum_short_return, "bench": cum_bench_return}
        cum_excess_returns = {"long-short": long_short_return, "long-bench": long_bench_return, "bench-short": bench_short_return}
        return metrics_returns, metrics_cum_returns,  cum_returns, cum_excess_returns
        
    def cumulative_ic(self, signals, cross_return, percentage=None):
        percentage = 0.5 if percentage is None else percentage
        quantiles = np.nanquantile(signals, [1 - percentage, percentage], axis=1)
        f = lambda x: quantiles[x][:, None].repeat(signals.shape[1], axis=-1)
        long_quantile, short_quantile = f(0), f(1)
        ic = cross_return.corrwith(signals, axis=1)
        cum_ic = cross_return.corrwith(signals, axis=1).cumsum()

        long_short_ic = cross_return.corrwith(signals, axis=1).mean()
        long_ic = cross_return[signals > long_quantile].corrwith(signals[signals >long_quantile], axis=1).mean()
        short_ic = cross_return[signals < short_quantile].corrwith(signals[signals < short_quantile], axis=1).mean()

        long_short_rankic = cross_return.corrwith(signals, method="spearman", axis=1).mean()
        long_rankic = cross_return[signals > long_quantile].corrwith(signals[signals >long_quantile], method="spearman", axis=1).mean()
        short_rankic = cross_return[signals < short_quantile].corrwith(signals[signals < short_quantile], method="spearman", axis=1).mean()
        ic_summary = {
                        "long_short_ic": ic, 
                        "cum_long_short_ic": cum_ic,  
                        "ic": [long_short_ic, long_ic, short_ic], 
                        "rank_ic": [long_short_rankic, long_rankic, short_rankic]
                        
                    }
        return ic_summary
    
    def get_weight(self, signals,  percentage=None):
        percentage = 0.5 if percentage is None else percentage
        quantiles = np.nanquantile(signals, [percentage], axis=1)
        quantiles = quantiles[0][:, None].repeat(signals.shape[1], axis=-1)
        weight = (signals.values > quantiles)/(signals.values > quantiles).sum(axis=1, keepdims=True)
        return weight


   
    def metrics(self, signals, cum_excess_returns, returns):
        net_value = {key:val[-1] for key, val in cum_excess_returns.items()}
        ann_return = dict(zip(returns.keys(), list(map(lambda x: np.nanmean(x)*252, returns.values()))))
        ann_volatility = dict(zip(returns.keys(), list(map(lambda x: np.nanstd(x)*np.sqrt(252), returns.values()))))
        sharpe = dict(zip(ann_return.keys(), list(map(lambda x: abs(ann_return[x]/ann_volatility[x]), ann_return.keys()))))
        max_drawdown = dict(zip(cum_excess_returns.keys(), list(map(lambda x: np.max((np.maximum.accumulate(x) - x)/np.maximum.accumulate(x)), cum_excess_returns.values()))))
        weight = self.get_weight(signals)
        turnover = np.nanmean(np.nansum(np.abs(np.diff(weight, axis=0)), axis=1))
        metrics = {
                    'Total_Net_Value': net_value, 
                    'Annualized_Return': ann_return, 
                    'Annualized_Volatility': ann_volatility, 
                    'Sharpe': sharpe,
                    'MaxDrawdown': max_drawdown, 
                    "Turnover": turnover
                    }
        return metrics
    
    def plot_group_return(self, dates, group_cum_return):
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, N_GROUPS))
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        dates = [dt.strftime('%Y-%m-%d') for dt in dates]
        ax.set_facecolor('#f5f5f5')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        gp_return = group_cum_return["groups"]
        mean_return = group_cum_return["bench"]

        ax.plot(dates, mean_return, color='gray', linewidth=2.5, label='bench', alpha=0.5)
        for i in range(gp_return.shape[1]):
            ax.plot(range(len(dates)),gp_return[:, i], label=f'Group {i+1}', color=colors[i])
        ax.set_xticks(range(len(dates)), dates, rotation=45)
        ax.xaxis.set_major_locator(MaxNLocator(20))
        ax.legend(loc='lower left')
        plt.show()

    def plot_return(self, dates, cum_returns, cum_excess_returns, metrics):
        
        colors = {
            'long': 'indianred',
            'short': 'goldenrod',
            'bench': 'gray',
            "long-short":  'indianred',
            "long-bench":  'goldenrod',
            "bench-short":  'gray',
        }
        fig, axs = plt.subplots(3, 1, figsize=(20, 20), height_ratios=[4, 4, 2])
        dates = [dt.strftime('%Y-%m-%d') for dt in dates]
        # fig.subplots_adjust(bottom=0.7)
        for key in cum_returns.keys():
            axs[0].plot(range(len(dates)), cum_returns[key], label=key, color=colors[key])
        axs[0].legend()
        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[0].set_xticks(range(len(dates)), dates, rotation=45)
        axs[0].xaxis.set_major_locator(MaxNLocator(20))
        axs[0].set_facecolor('#f5f5f5')

        for key in cum_excess_returns.keys():
            axs[1].plot(range(len(dates)), cum_excess_returns[key], label=key, color=colors[key])
        axs[1].legend()
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[1].set_xticks(range(len(dates)), dates, rotation=45)
        axs[1].xaxis.set_major_locator(MaxNLocator(20))
        axs[1].set_facecolor('#f5f5f5')

        columns = ["long-short", "long", "short"]
        rows = list(metrics.keys())
        data = []
        for val in metrics.values():
            if isinstance(val, dict):
                data.append([f"{v:.3f}" for v in val.values()])
            else:
                data.append([f"{val:.3f}", " ", " "])
        axs[2].axis('off')
        BBOX = axs[2].get_position()
        BBOX = BBOX.from_bounds(BBOX.x0, BBOX.y0, BBOX.width, BBOX.height*5)
        table = axs[2].table(cellText=data,
                  rowLabels=rows,
                  colLabels=columns,
                  colWidths = [0.2]*3,
                  loc='center left',
                  colLoc='left',  
                  cellLoc='left',
                  bbox=BBOX)  
        
        
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('none')
        
        plt.show()


    def plot_ic(self, dates, ic_summary):
        color = {
            "ic": 'powderblue',    
            "cumulative_ic": 'teal'
        }

        fig, axs = plt.subplots(2, 1, figsize=(20, 10), height_ratios=[4, 1])
        dates = [dt.strftime('%Y-%m-%d') for dt in dates] 
        axs[0].set_facecolor('#f5f5f5')
        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs_ = axs[0].twinx()
        axs[0].plot(range(len(dates)), ic_summary["long_short_ic"], color=color["ic"], label='IC', alpha=0.8)
        axs_.plot(range(len(dates)), ic_summary["cum_long_short_ic"], color=color["cumulative_ic"], linestyle='--', label='Cumulative IC')
        axs[0].set_xticks(range(len(dates)), dates, rotation=45)
        axs[0].xaxis.set_major_locator(MaxNLocator(20))
        axs[0].legend()
        axs_.legend()
        columns = ["long-short", "long", "short"]
        rows = ["IC", "Rank_IC"]
        data = [[f"{v:.3f}" for v in ic_summary['ic']], [f"{v:.3f}" for v in ic_summary['rank_ic']]]
        
        axs[1].axis('off')
        BBOX = axs[1].get_position()
        BBOX = BBOX.from_bounds(BBOX.x0, BBOX.y0, BBOX.width, BBOX.height*5)
        table = axs[1].table(cellText=data,
                  rowLabels=rows,
                  colLabels=columns,
                  colWidths = [0.2]*3,
                  loc='center left',
                  colLoc='left',  
                  cellLoc='left',
                  bbox=BBOX 
        )

        
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('none')
        
        plt.show()
   

    

    