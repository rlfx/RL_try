import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#test_df = dfd.iloc[1000:]
#test_df.reset_index(drop=True,inplace=True)

class Qlearn(object):
    def __init__(self,data,state_list):
        self.data = data
        self.episode = 50
        self.epsilon = 0.4
        self.lr = 0.2
        self.discount = 0.9
        self.decay_rate = 0.9
        
        num_list = []
        state_num = 1
        for each in state_list:
            num = len(data[each].unique())
            num_list.append(num)
            state_num *= num
            
        self.states = state_num
        self.actions = 5  #買1 2 不動 賣1 2
        self.window_size = 1
        self.Q = np.zeros((num_list[0],num_list[1],num_list[2],num_list[3],num_list[4],5), dtype=np.float32)

    def get_action(self, currentstate,n):
        if np.sum(self.Q[int(currentstate[0]),int(currentstate[1]),int(currentstate[2]),int(currentstate[3]),int(currentstate[4]),:]) == 0:
            return np.random.randint(0, self.actions)
        else:
            if np.random.random(1) < self.epsilon/(1+n*self.decay_rate): #(1+m/3*self.decay_rate) old_version
                return np.random.randint(0, self.actions)
            else:
                return np.argmax(self.Q[int(currentstate[0]),int(currentstate[1]),int(currentstate[2]),int(currentstate[3]),int(currentstate[4]):])
    '''
    def get_reward(self, currentaction,tClose,t1Close):
        actions = np.array([-2,-1,0,1,2], dtype=np.int32)
        reward = actions[currentaction]*(t1Close-tClose)/tClose
        return reward
    '''

    def get_reward(self,currentaction,tClose,position,cash):
        actions = np.array([-2,-1,0,1,2], dtype=np.int32)
        action = actions[currentaction]
        position+= action
        cash -= action*tClose
        net_reward = cash + position*tClose

        return (position,cash,net_reward)
    
    def main(self):
        n = 0
        m = 0
        acc_list = []
        states_list = []
        for each_episode in range(self.episode):
            reward_list = []
            #acc_reward = 0
            position = 0
            cash = 1000000
            for sample in range(len(self.data)-1):
                if sample >= self.window_size-1:
                    state = self.data.loc[sample,'state']
                    action = self.get_action(state,n)
                    
                    try:
                        (position,cash,reward) = self.get_reward(action,self.data.loc[sample,'Close'],position,cash)
                    except:
                        continue
                    
                    next_state = self.data.loc[sample+1,'state']
                    self.Q[int(state[0]),int(state[1]),int(state[2]),int(state[3]),int(state[4]),action] = (1-self.lr)*self.Q[int(state[0]),int(state[1]),int(state[2]),int(state[3]),int(state[4]),action] + self.lr*(reward+self.discount*np.argmax(self.Q[int(next_state[0]),int(next_state[1]),int(next_state[2]),int(next_state[3]),int(next_state[4]),:]))
                    reward_list.append(reward)
                    states_list.append(state)
                    #acc_reward += reward
                    #acc_y_reward = acc_reward / len(self.data)*252
                    m += 1

            print('epoch:',n,reward,np.min(reward_list),np.max(reward_list))#acc_y_reward
            acc_list.append(reward) #acc_reward
            n+=1
        return (acc_list,states_list,self.Q)

    def test_q(self,data):
        reward_list = []
        state_list = []
        m = 1000000
        acc_reward = 0
        for sample in range(len(data)-1):
            if sample >= self.window_size-1:
                state = self.get_state(data.loc[sample-(self.window_size-1):sample])
                action = self.get_action(state,m)
                try:
                    reward = self.get_real_action(action,data.loc[sample,'Close'],data.loc[sample+1,'Close'])
                except:
                    continue
                
                state_list.append(state)
                reward_list.append(reward)
                acc_reward += reward 
        return acc_reward


if __name__ == '__main__':
    df = pd.read_csv("macd_st.csv",encoding='utf-8')
    state_list = ['RSI_dummy','osc_dummy','dif_dummy','dem_dummy','cross_dummy']
    # generate state
    for idx in range(len(df)):
        aaa = ''
        for each in state_list:
            s = str(df.loc[idx,each])
            aaa += s
        df.loc[idx,'state'] = aaa
    
    df['state'] = df['state'].astype(str)
    
    RLmodel = Qlearn(df,state_list)
    (acc_list,states_list,Q) = RLmodel.main()
    #test_reward = RLmodel.test_q(test_df)
    #test_year_r = test_reward / len(test_df)*252
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.plot(acc_list)

    print('Q表中非零的個數:',np.count_nonzero(Q),'非零比例:',np.count_nonzero(Q) / 480)
    print('出現的state個數:',len(list(set(states_list))),'比例:',len(list(set(states_list)))/96)