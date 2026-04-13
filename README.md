backtest_template.py combines directional trading and market making strategies depending on the signal.
The signal is composed of:
* imbalance_l1: the difference between number of buyers and sellers at the best bid and ask prices, calculated by (bid_volume1 - ask_volume1)/(bid_volume1 + ask_volume1)
* imbalance_l3: the difference between number of buyers and sellers at the top 3 bid and ask prices, calculated by (bid_volume1,2,3 - ask_volume1,2,3)/(bid_volume1,2,3 + ask_volume1,2,3)
* return: the log-normalized difference between midprice_{t} and midprice_{t-1}. Midprice is defined as the average of the current highest buyer bid price and lowest seller ask price.

The function will loop through all rows of the dataset and calculate the Profit and Losses (PnL) and the Sharpe score.
