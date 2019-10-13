from gurobipy import *
import sys
import math
import numpy as np
import pandas as pd
import csv

actual_demand = pd.read_csv('Two-Year-Demand-New.csv', index_col=0, parse_dates=True, squeeze=True)
initial_inventory = 73
future_demand=[95.785167, 95.386705, 109.039909, 94.991456, 96.02944, 109.988194, 101.448577, 85.736381, 111.617761, 106.539466, 108.080133, 115.764111, 98.25996528, 98.55706555, 111.565617, 97.09340264, 98.91681081, 111.4736759, 103.0156426, 87.49652351, 112.8125065, 107.9559793, 109.1617965, 117.3376]
order_quantity_ideal = np. zeros(24)
inventory = np.zeros(25)
inventory[0]=initial_inventory
backorder = np.zeros(25)
holdingcostary=np.zeros(24)
backordercostary=np.zeros(24)
backorder[1]=future_demand[0]-inventory[0]
m = Model("INVENTORY PLANNING")
m.Params.OutputFlag = 0
order_quantity_ideal = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="last_order")

scenarios = 1000
incidence = np.zeros(scenarios)
mu = np.zeros(24)
sigma = np.zeros(24)
demand = np.zeros(scenarios)
for i in range(scenarios):
    incidence[i] = i
inventory_cost = m.addVars(incidence, lb=0, vtype=GRB.CONTINUOUS, name="inventory_cost")
backorder_cur = m.addVars(incidence, lb=0, vtype=GRB.CONTINUOUS, name="backorder_cur")
inventory_cur = m.addVars(incidence, lb=0, vtype=GRB.CONTINUOUS, name="inventory_cur")
#result="result.csv"
#csv = open(result, "wb")
#columnTitleRow = "month, beginning inventory, order quality, ending inventory, holding cost, backorder cost\n"
#csv.write(columnTitleRow)


for i in range(0,24):
    mu[i] = future_demand[i]
    sigma[i] = 0.05*future_demand[i]
    demand = np.random.normal(mu[i], sigma[i], scenarios)
    m.setObjective(0.001*inventory_cost.sum('*')+0.001*3*backorder_cur.sum('*'),GRB.MINIMIZE)
    for j in range(scenarios):
        m.addConstr(inventory_cost[j]>=inventory_cur[j])
        m.addConstr(inventory_cost[j]>=2*inventory_cur[j]-90)
        m.addConstr(inventory[i]-backorder[i]+order_quantity_ideal-demand[j]==inventory_cur[j]-backorder_cur[j])
        if i==0:
            m.addConstr(order_quantity_ideal==0)
    m.optimize()
    #print('Obj: %g' % m.objVal)
    x=m.getVarByName("last_order")
    inventory[i+1] = max(0, inventory[i] + x.x - actual_demand[i] - backorder[i])
    backorder[i+1] = max(0, actual_demand[i] + backorder[i] - inventory[i] - x.x)
    holdingcost=min(90, inventory[i])+max(0,2*(inventory[i]-90))
    holdingcostary[i-1]=holdingcost
    backordercostary[i-1]=3*backorder[i]
    print('--------------------month %g-----------------' % i)
    print('beginning inventoty of month %g: %g' % (i, inventory[i-1]) )
    print('order number of month %g is %g'% (i, x.x))
    print('ending inventoty of month %g: %g' % (i, inventory[i]) )
    print('holding cost of month %g: %g' % (i, holdingcost))
    print('backorder cost of month %g: %g' % (i, 3*backorder[i]))
    m.reset(0)
    m.remove(m.getConstrs())

print('--------------------Summary-----------------')
print('total holding cost: %g' % np.sum(holdingcostary))
print('average holding cost: %g' % np.mean(holdingcostary))
print('total backorder cost: %g' % np.sum(backordercostary))
print('average backorder cost: %g' % np.mean(backordercostary))
total=np.sum(holdingcostary)+np.sum(backordercostary)
print('total cost: %g' % total)
