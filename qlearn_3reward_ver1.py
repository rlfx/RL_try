# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def state_generate(df,state_list):
    # generate state
    for idx in range(len(df)):
        aaa = ''
        for each in state_list:
            s = str(df.loc[idx,each])
            aaa += s
        df.loc[idx,'state'] = aaa

    df['state'] = df['state'].astype(str)
    return df

class Qlearn(object):
    def __init__(self,data,state_list):
        self.data        = data
        self.episode     = 500
        self.epsilon     = 0.4  
        self.lr          = 0.2
        self.discount    = 0.9
        self.decay_rate  = 0.9
        self.window_size = 1
        self.actions     = len(STATE)
        self.state       = self.initState(data,state_list)
        self.Q           = self.initQ(data,state_list)
        self.entry_price = 0
        self.exit_price  = 0
        self.position    = 0 
        self.cash        = 10000
        self.day_net     = 10000

    def initState(self,data,state_list):
        state_num = 1
        for each in state_list:
            state_num *= len(data[each].unique())
        return state_num  
    
    def initQ(self,data,state_list):
        num_list = []
        for each in state_list:
            num_list.append(len(data[each].unique()))
        return np.zeros(np.append(num_list,self.actions),dtype=np.float32)
    
    def StateInQ(self,state_t):
        s_tuple = ()
        for s in state_t:
            s_tuple += (int(s),)
        return s_tuple   

    def getQVector(self,state_t):
        Q_selection = self.Q
        for s in state_t:
            Q_selection = Q_selection[int(s)]
        return Q_selection

    def take_greedy_action(self,n):
        if np.random.random(1) >= self.epsilon/(1+n*self.decay_rate):
            return True
        else:
            return False

    def take_optimal_action(self,state_t):
        return  np.argmax(self.getQVector(state_t))

    def take_random_action(self):
        return np.random.randint(0, self.actions)

    def get_action(self, state_t, n):
        if all(q == self.getQVector(state_t)[0] for q in self.getQVector(state_t)): # all q is equal
            return self.take_random_action()
        else:
            if self.take_greedy_action(n):
                return self.take_optimal_action(state_t)
            else:
                return self.take_random_action()

    def get_unrealized_PnL(self,action_t,open_t,close_t):
        actions        = np.array(STATE, dtype =np.int32)
        action         = actions[action_t]
        self.position += action
        self.cash     -= action*open_t
        day_net        = self.cash + self.position*close_t
        reward         = day_net - self.day_net 
        self.day_net   = day_net        
        return reward
    
    def get_realized_PnL(self,action_t,close_t):
        '''
        if position = 0        : action_space = {buy,hold,sell}
        if position = 1 (long) : action_space = {hold,sell}
        if position =-1 (short): action_space = {buy,hold}
        
        當一進一出後,計算instant_pnl當作此次交易的reward
        '''
        actions     = np.array(STATE, dtype =np.int32)
        action      = actions[action_t]
        instant_pnl = 0
        if action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = close_t
            elif self.position == -1:
                self.position = 0
                self.exit_price = close_t
                instant_pnl = self.entry_price - self.exit_price
                self.entry_price = 0          
        if action == -1:
            if self.position == 0:
                self.position = -1
                self.entry_price = close_t
            elif self.position == 1:
                self.position = 0
                self.exit_price = close_t
                instant_pnl = self.exit_price - self.entry_price
                self.entry_price = 0
        return instant_pnl
    
    def get_1day_operation_PnL(self,action_t,close_t,close_t1):
        actions = np.array(STATE, dtype =np.int32)
        action  = actions[action_t]
        reward  = action*(close_t1-close_t)
        return reward    
    
    def updateQ(self,state_t,state_t1,action,reward):
        s_tuple = self.StateInQ(state_t)
        next_s_tuple = self.StateInQ(state_t1)
        self.Q[s_tuple][action] = (1-self.lr)*self.Q[s_tuple][action] +self.lr*(reward+self.discount*max(self.Q[next_s_tuple]))
        Q_range = max(self.Q[s_tuple]) - min(self.Q[s_tuple])
        return Q_range
        
    def main(self,mode):
        n = 0
        acc_list = []
        Q_range_list = []
        for each_episode in range(self.episode):
            reward = 0
            if mode == 1:
                self.position = 0  
                self.cash = 10000
                self.day_net = 10000
                
            for sample in range(len(self.data)-1):
                state = self.data.loc[sample,'state']
                action = self.get_action(state,n)
                next_state = self.data.loc[sample+1,'state']
                
                if mode == 0: # realized PnL
                    instant_pnl = self.get_realized_PnL(action,self.data.loc[sample,'Close'])
                    Q_range = self.updateQ(state,next_state,action,instant_pnl)
                    Q_range_list.append(Q_range)
                    reward += instant_pnl
                    if n > 498:
                        Q_range_list.append(Q_range)
                
                if mode == 1: # unrealized PnL
                    unreal_pnl = self.get_unrealized_PnL(action,self.data.loc[sample,'Open'],self.data.loc[sample,'Close'])
                    Q_range = self.updateQ(state,next_state,action,unreal_pnl)
                    reward += unreal_pnl
                    if n > 498:
                        Q_range_list.append(Q_range)
                
                if mode == 2: # 1 day operation PnL
                    oneday_pnl = self.get_1day_operation_PnL(action,self.data.loc[sample,'Close'],self.data.loc[sample+1,'Close'])
                    Q_range = self.updateQ(state,next_state,action,oneday_pnl)
                    reward += oneday_pnl
                    if n > 498:
                        Q_range_list.append(Q_range)
                                  
            print('epoch:',n,'acc_reward:',reward)
            acc_list.append(reward) 
            n+=1
        return (acc_list,Q_range_list,self.Q)

if __name__ == '__main__':
    df = pd.read_csv("macd_st.csv",encoding='utf-8')
    state_list = ['RSI_dummy','osc_dummy','dem_dummy'] #'dif_dummy','cross_dummy'
    STATE = [ -1, 0, 1]

    data = state_generate(df,state_list)

    RLmodel = Qlearn(data,state_list)
    (acc_list,Q_range_list,Q) = RLmodel.main(1)

    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.plot(acc_list)

    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.scatter(x = range(len(Q_range_list)),y = Q_range_list)