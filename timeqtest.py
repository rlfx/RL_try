# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Qlearn(object):
    def __init__(self,data):
        self.data = data
        self.episode = 500
        self.epsilon = 0.4
        self.lr = 0.2
        self.discount = 0.9
        self.decay_rate = 0.9           
        self.states = len(data)
        self.actions = 5  #買1 2 不動 賣1 2
        self.window_size = 1
        self.Q = np.zeros((len(data),5), dtype=np.float32)

    def get_action(self, currentstate,n):
        if np.sum(self.Q[currentstate,:]) == 0:
            return np.random.randint(0, self.actions)
        else:
            if np.random.random(1) < self.epsilon/(1+n*self.decay_rate): #(1+m/3*self.decay_rate) old_version
                return np.random.randint(0, self.actions)
            else:
                return np.argmax(self.Q[currentstate,:])

    def get_reward(self, currentaction,tClose,t1Close):
        actions = np.array([-2,-1,0,1,2], dtype=np.int32)
        reward = actions[currentaction]*(t1Close-tClose)/tClose
        return reward

    def main(self):
        n = 0
        m = 0
        acc_list = []
        states_list = []
        for each_episode in range(self.episode):
            reward_list = []
            acc_reward = 0
            for sample in range(len(self.data)-1):
                if sample >= self.window_size-1:
                    state = self.data.loc[sample,'state']
                    action = self.get_action(state,n)
                    
                    try:
                        reward = self.get_reward(action,self.data.loc[sample,'Close'],self.data.loc[sample+1,'Close'])
                    except:
                        continue
                    
                    next_state = self.data.loc[sample+1,'state']
                    self.Q[state,action] = (1-self.lr)*self.Q[state,action] + self.lr*(reward+self.discount*np.argmax(self.Q[next_state,:]))
                    reward_list.append(reward)
                    states_list.append(state)
                    acc_reward += reward
                    acc_y_reward = acc_reward / len(self.data)*252
                    m += 1

            print('epoch:',n,acc_y_reward,np.min(reward_list),np.max(reward_list))
            acc_list.append(acc_reward)
            n+=1
        return (acc_list,states_list,self.Q)


if __name__ == '__main__':
    df = pd.read_csv("macd_st.csv",encoding='utf-8')
    #state_list = ['RSI_dummy','osc_dummy','dif_dummy','dem_dummy','cross_dummy']
    
    # generate state
    '''
    for idx in range(len(df)):
        aaa = ''
        for each in state_list:
            s = str(df.loc[idx,each])
            aaa += s
        df.loc[idx,'state'] = aaa
    '''
    df['state'] = df.index
    
    RLmodel = Qlearn(df)
    (acc_list,states_list,Q) = RLmodel.main()
    #test_reward = RLmodel.test_q(test_df)
    #test_year_r = test_reward / len(test_df)*252
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.plot(acc_list)

    print('Q表中非零的個數:',np.count_nonzero(Q),'非零比例:',np.count_nonzero(Q) /(1499*5))
    print('出現的state個數:',len(list(set(states_list))),'比例:',len(list(set(states_list)))/1499)