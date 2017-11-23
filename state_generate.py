#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("jpy24h.csv")
df = df.reset_index(drop=True)
dfd = df.iloc[:1500]

# macd 相關state
dfd['dif'] = (dfd['Close'].ewm(span = 12).mean()-dfd['Close'].ewm(span = 26).mean()) #訊號線
dfd['dem'] = dfd['dif'].ewm(span = 9).mean() #macd
dfd['osc'] = dfd['dif'] - dfd['dem']
dfd['osc_dummy'] = dfd['osc'].apply(lambda x:0 if x>=0 else 1) # osc state -> neg =1,pos = 0
dfd['dif_dummy'] = dfd['dif'].apply(lambda x:0 if x>=0 else 1) # 訊號線 state -> neg =1,pos = 0
dfd['dem_dummy'] = dfd['dem'].apply(lambda x:0 if x>=0 else 1) # macd state -> neg =1,pos = 0

#定義交叉與糾結
dfd['cross'] = 0
dfd['cross'][dfd['osc_dummy'] != dfd['osc_dummy'].shift()] = 1
dfd['cross_dummy'] = dfd['cross']*np.sign(dfd['osc'])+0  # 黃金交叉 ＝ 1 同邊 = 0
dfd['cross_dummy'] = dfd['cross_dummy'].astype(int)
dfd['cross_dummy'] = dfd['cross_dummy'].replace(-1,2) # 死亡交叉 = 2

dfd['rol_cross'] = dfd['cross'].rolling(6,min_periods = 1).sum() #糾結（前6跟bar內出現兩次以上交叉)
dfd['rol_cross'] = dfd['rol_cross'].apply(lambda x: 1 if x > 1 else 0) #糾結 = 1 , 其他 = 0 

def rsi_d(x):
    if 0 <= x <= 30:
        return 0
    if 30 < x <= 50:
        return 1
    if 50 < x <= 70:
        return 2
    if 70 < x <= 100:
        return 3
    
dfd['RSI_dummy'] = dfd['RSI'].apply(rsi_d)

# 定義背離
dfd['bali'] = 0

temp = dfd[dfd['cross_dummy'] == 2].index
for idx in temp:
    try:
        idx_next = temp[idx+1]
        if dfd.loc[idx,'dif'] > dfd.loc[idx_next,'dif']: # 當兩次死叉出現時 後面的訊號線位置比前面低 且後面的價格比前面高
            if dfd.loc[idx,'Close'] <= dfd.loc[idx_next,'Close']:
                dfd.loc[idx_next:idx_next+3,'bali'] = 1  # 頂背離 = 1
    except:
        continue

temp = dfd[dfd['cross_dummy'] == 1].index
for idx in temp:
    try:
        idx_next = temp[idx+1]
        if dfd.loc[idx,'dif'] < dfd.loc[idx_next,'dif']: # 當兩次金叉出現時 後面的訊號線位置比前面高 且後面的價格比前面低
            if dfd.loc[idx,'Close'] >= dfd.loc[idx_next,'Close']:
                dfd.loc[idx_next:idx_next+3,'bali'] = 1  # 底背離 = 1
    except:
        continue          

# 背離二版 (正在寫)
''' 
dfd['osc_h/l'] = 0
dfd['cluster_h/l'] = 0

for idx in range(3,len(dfd)):
    if abs(dfd.loc[idx,'osc']) < abs(dfd.loc[idx-1,'osc']):
        if abs(dfd.loc[idx,'osc']) < abs(dfd.loc[idx-2,'osc']):
            if abs(dfd.loc[idx-2,'osc']) > abs(dfd.loc[idx-3,'osc']):
                if (dfd.loc[idx,'osc_dummy'] == dfd.loc[idx-1,'osc_dummy']) &(dfd.loc[idx,'osc_dummy'] == dfd.loc[idx-2,'osc_dummy']) &((dfd.loc[idx-2,'osc_dummy'] == dfd.loc[idx-3,'osc_dummy'])):
                    dfd.loc[idx-2,'cluster_h/l_p'] =1
                    dfd.loc[idx-2,'osc_h/l'] = dfd.loc[idx-2,'osc']
'''

state_list = ['RSI_dummy','osc_dummy','dif_dummy','dem_dummy','cross_dummy'] #'bali','rol_cross'
#糾結與背離目前太不平均 導致許多state幾乎不會出現

dfd.to_csv('macd_st.csv',index=None,encoding = 'utf-8')

# plot macd
fig, ax = plt.subplots(nrows=4, ncols=1)
plt.subplot(4, 1, 1)
plt.plot(dfd['Close'])
plt.subplot(4, 1, 2)
plt.plot(dfd['RSI'])
plt.subplot(4, 1, 3)
plt.plot(dfd['dif'])
plt.plot(dfd['dem'])
plt.subplot(4, 1, 4)
colors = np.array([(1,0,0)]*len(dfd['osc']))
colors[dfd['osc'] >= 0] = (0,0,1)
plt.bar(dfd.index,dfd['osc'],color = colors)
plt.show()
