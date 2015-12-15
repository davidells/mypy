up_down = consecutive_up_down(GLD['Adj Close'])
adj_close = GLD['Adj Close']

df = pd.DataFrame({
    'up_streak': up_down['up_streak'],
    'down_streak': up_down['down_streak'],
    'atr_5': average_true_range(GLD, 5),
    #'atr_20': average_true_range(GLD, 20),
    #'atr_50': average_true_range(GLD, 50),
    'over_ma_5': adj_close > util.ma(adj_close, 5),
    'over_ma_10': adj_close > util.ma(adj_close, 10),
    'over_ma_20': adj_close > util.ma(adj_close, 20),
    'over_ma_20': adj_close > util.ma(adj_close, 20),
    #'hist_ret_5': adj_close.pct_change().shift(5).fillna(0),
    #'hist_ret_4': adj_close.pct_change().shift(4).fillna(0),
    'hist_ret_3': adj_close.pct_change().shift(3).fillna(0),
    'hist_ret_2': adj_close.pct_change().shift(2).fillna(0),
    'hist_ret_1': adj_close.pct_change().shift(1).fillna(0)    
})