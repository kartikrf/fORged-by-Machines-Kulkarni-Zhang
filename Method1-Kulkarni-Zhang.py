import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
# from plotly.plotly import plot_mpl
#import statsmodels.api as sm
#from sklearn import datasets, linear_model
#from sklearn.model_selection import train_test_split
#from statsmodels.tsa.stattools import adfuller
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from statsmodels.tsa.arima_model import ARIMA
#import statistics
#import matplotlib.pyplot as plt


initial_inventory = 73
demand_history = pd.read_csv('Ten-Year-Demand.csv', index_col=0, parse_dates=True, squeeze=True)
result = seasonal_decompose(demand_history, model="additive")
actual_demand = pd.read_csv('Two-Year-Demand-New.csv', index_col=0, parse_dates=True, squeeze=True)
actual_demand_new1 = actual_demand[0:12]
actual_demand_new2 = actual_demand[12:24]
optimal_order_quantity = np.zeros(24)
optimal_ending_inventory = np.zeros(24)
optimal_backorder = np.zeros(24)
#print(actual_demand_new1)
#print(actual_demand_new2)
print("PLEASE WAIT..THE CODE IS RUNNING..")
for j in range(2):
    # result.plot()
    # pyplot.show()
    # print(demand_history.head())

    # Plot the data to check if stationary (constant mean and variance), as many time series models require the data to be stationary

    # pyplot.plot(demand_history)
    # pyplot.show()

    # Difference the data to make it more stationary and plot to check if the data looks more stationary
    # Differencing subtracts the next value by the current value Best not to over-difference the data,
    # as this could lead to inaccurate estimates # Make sure to leave no missing values, as this could cause
    # problems when modeling later

    #demand_history_diff1 = demand_history.diff().fillna(demand_history)
    # pyplot.plot(demand_history_diff1)
    # pyplot.show()

    #demand_history_diff2 = demand_history_diff1.diff().fillna(demand_history_diff1)
    # pyplot.plot(demand_history_diff2)
    # pyplot.show()

    # plot_acf(demand_history_diff1)
    # pyplot.show()

    # plot_pacf(demand_history_diff2)
    # pyplot.show()

    #demand_history_train, demand_history_test = train_test_split(demand_history, test_size = 0.0, shuffle = False)
    demand_history_train = demand_history
    # print(demand_history_train)
    #print(demand_history_test)

    #model_demand_history = sm.tsa.statespace.SARIMAX(demand_history_train, order=(0,1,0), seasonal_order=(1,1,1,12))
    #results = model_demand_history.fit()

    demand_forecast_model = pm.auto_arima(demand_history_train ,start_p=1, start_q=1,
                                 max_p=3, max_q=3, m=12,
                                 start_P=0, seasonal=True,
                                 d=1, D=1, trace=False,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True)

    #results = demand_forecast_model.fit()

    #print(demand_forecast_model.summary())
    #print("*******************C.I**************")
    #forecast, stderr, conf = demand_forecast_model.forecast()
    #print(conf)
    #print(stderr)
    #print(demand_forecast_model.cov_params_approx())
    # demand_history_diff2 = demand_history_diff1.diff().fillna(demand_history_diff1)
    # pyplot.plot(demand_history_diff2)
    # pyplot.show()
    # ARMA1model_demand_history.fit(demand_history)
    #arma_res = demand_forecast_model.fit(demand_history)

    future_demand = demand_forecast_model.predict(n_periods = 13)
    #std_error = demand_forecast_model.bse()
    #print("STDERROR")

    #print(std_error)
    #fig, ax = plt.subplots(figsize=(10, 8))
    #fig = arma_res.plot_predict(start='1996-01-01', end='2006-01-01', ax=ax)
    #legend = ax.legend(loc='upper left')
    #dem_plot = np.zeros(120)
    #for i in range(120):
    #    dem_plot[i] = demand_history_train[i]
    #print(dem_plot)
    #print(future_demand[0])
    #pyplot.plot(dem_plot)
    #pyplot.plot(future_demand, color = 'red')
    #pyplot.show()
    if j==0:
        actual_demand_new = actual_demand_new1
    else:
        actual_demand_new = actual_demand_new2

    #print(actual_demand_new[0])

    order_quantity_ideal = np. zeros(12)
    inventory = np.zeros(12)
    backorder = np.zeros(12)
    order_quantity_ideal[0] = future_demand[0] + future_demand[1] - initial_inventory
    #order_quantity_ideal[11] = future_demand[11]
    inventory[0] = max(0, initial_inventory - actual_demand_new[0])
    backorder[0] = max(0, actual_demand_new[0] - initial_inventory)

    for i in range(1, 12):
        order_quantity_ideal[i] = future_demand[i+1]
        inventory[i] = max(0, inventory[i-1] + order_quantity_ideal[i - 1] - actual_demand_new[i] - backorder[i-1])
        backorder[i] = max(0, actual_demand_new[i] + backorder[i - 1] - inventory[i-1] - order_quantity_ideal[i-1])
        if backorder[i] > 5:
            order_quantity_ideal[i] = order_quantity_ideal[i] + backorder[i]
        if inventory[i] > 20:
            order_quantity_ideal[i] = order_quantity_ideal[i] - inventory[i]
    #print("Demand forecast", future_demand)
    #print("Order Quantity:", order_quantity_ideal)
    #print(inventory, backorder)
    initial_inventory = order_quantity_ideal[11]
    #print("Intitial inv: ", initial_inventory)
    demand_history = np.concatenate((demand_history_train, actual_demand_new))

    #("*****************New Demand*************", demand_history)

    for x in range(12*j, 12*j + 12):
        optimal_order_quantity[x] = order_quantity_ideal[x - 12*j]
        optimal_ending_inventory[x] = inventory[x - 12*j]
        optimal_backorder[x] = backorder[x - 12*j]

print("-------------------OPTIMAL SOLUTION--------------------")
print("Order Quantity :")
print(optimal_order_quantity)
print("Ending Inventory :")
print(optimal_ending_inventory)
print("Backorder :")
print(optimal_backorder)
inv_cost = optimal_ending_inventory
for y in range(24):
    if optimal_ending_inventory[y] > 90:
        inv_cost[y] = 2 * inv_cost[y]
    else:
        continue
back_cost = 3 * optimal_backorder

cum_inv= np.cumsum(inv_cost)
cum_back = np.cumsum(back_cost)
total_inventory_cost = cum_inv[23]
total_backorder_cost = cum_back[23]

print("------------------------COSTS-------------------------")
print("Inventory Costs:")
print(inv_cost)
print("Backordering Costs:")
print(back_cost)

print("OVERALL COSTS:", total_inventory_cost+total_backorder_cost)