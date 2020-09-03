# ディープ・ニューラルネットワークの構築
from .config_learn import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
import random
import time
import sys

from collections import namedtuple
import warnings
warnings.simplefilter('ignore')

Transition = namedtuple('Transition', ('episode', 'step' ,'state', 'action', 'next_state', 'reward', 'reward_p'))

# device = 'cpu'
print(device)

''' for Action Branched DQN '''


# nn_option : 7 (LSTM version)
class LSTMBranchedNet(nn.Module):

    def __init__(self, num_actions, num_products, n_fc1, n_fc2, LSTM_MEMORY, n_fc3, n_branch, num_states):
        self.n_branch = n_branch
        self.num_states = num_states
        self.num_products = num_products
        self.num_actions = num_actions
        self.LSTM_MEMORY = LSTM_MEMORY

        super(LSTMBranchedNet, self).__init__()
        self.fc1 = nn.Linear(self.num_states, n_fc1)
        # self.fc2 = nn.Linear(n_fc1, n_fc2)  # state representation
        self.lstm = nn.LSTM(n_fc1, LSTM_MEMORY, 1)  # (Input, Hidden, Num Layers)

        fc_branch = [nn.Linear(LSTM_MEMORY, n_fc3) for i in range(self.n_branch)]
        self.fc_branch_layer = nn.ModuleList(fc_branch)
        fc_branch2 = [nn.Linear(n_fc3, self.num_actions) for i in range(self.n_branch)]
        self.fc_branch_layer2 = nn.ModuleList(fc_branch2)

    def forward(self, x, hidden_state, cell_state):
        h1 = F.relu(self.fc1(x))
        # h2 = F.relu(self.fc2(h1))
        # print(h2.shape)
        # print(h2.view(len(x),  1, -1).shape)

        # LSTM
        h3, (next_hidden_state, next_cell_state) = self.lstm(h1.view(len(x),  1, -1), (hidden_state, cell_state))

        # branch
        h4 = [F.relu(fc_branch_layer(h3)) for fc_branch_layer in self.fc_branch_layer]
        output = [fc_branch_layer(h4) for h4, fc_branch_layer in zip(h4, self.fc_branch_layer2)]
        return output

    def init_states(self) -> [Variable, Variable]:
        hidden_state = Variable(torch.zeros(1, 1, self.LSTM_MEMORY)).to(device)
        cell_state = Variable(torch.zeros(1, 1, self.LSTM_MEMORY)).to(device)
        return hidden_state, cell_state

    def reset_states(self, hidden_state, cell_state):
        hidden_state[:, :] = 0
        cell_state[:, :] = 0
        return hidden_state.detach(), cell_state.detach()


# nn_option : 3
class BranchedNet(nn.Module):

    def __init__(self, num_actions, num_products, n_fc1, n_fc2, n_fc3, n_branch, num_states):
        self.n_branch = n_branch
        self.num_states = num_states
        self.num_products = num_products
        self.num_actions = num_actions

        super(BranchedNet, self).__init__()
        self.fc1 = nn.Linear(self.num_states, n_fc1)
        self.fc2 = nn.Linear(n_fc1, n_fc2)  # state representation
        # 1) State Value
        self.s = nn.Linear(n_fc2, 1)
        # 2) Branching Architecture         #A_d(s, a_d)
        fc_branch = [nn.Linear(n_fc2, n_fc3) for i in range(self.n_branch)]
        self.fc_branch_layer = nn.ModuleList(fc_branch)
        fc_branch2 = [nn.Linear(n_fc3, self.num_actions) for i in range(self.n_branch)]
        self.fc_branch_layer2 = nn.ModuleList(fc_branch2)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = [F.relu(fc_branch_layer(h2)) for fc_branch_layer in self.fc_branch_layer]

        adv = [fc_branch_layer(h3) for h3, fc_branch_layer in zip(h3, self.fc_branch_layer2)]
        val = [self.s(h2).expand(-1, adv[0].size(1)) for i in range(self.n_branch)]

        # productごとのバッチをfc_branch_layerに入れる
        # output = [fc_branch_layer2(h3) for fc_branch_layer2, h3 in zip(self.fc_branch_layer2, h3)]
        output = [val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1)) for val, adv in zip(val,adv)]

        return output


# nn_option : 4
class BranchedNetDuelState(nn.Module):

    def __init__(self, num_actions, num_products, n_fc1, n_fc2, n_fc3,n_branch, num_states):
        self.n_branch = n_branch
        self.num_states = num_states
        self.num_products = num_products
        self.num_actions = num_actions

        super(BranchedNetDuelState, self).__init__()
        self.fc1 = nn.Linear(self.num_states, n_fc1)
        self.fc2 = nn.Linear(n_fc1, n_fc2)  # state representation

        fc_branch = [nn.Linear(n_fc2, n_fc3) for i in range(self.n_branch)]
        self.fc_branch_layer = nn.ModuleList(fc_branch)

        # 1) State Value
        s_branch = [nn.Linear(n_fc3, 1) for i in range(self.n_branch)]  # state value
        self.s_branch_layer = nn.ModuleList(s_branch)

        # 2) Branching Architecture         #A_d(s, a_d)
        fc_branch2 = [nn.Linear(n_fc3, self.num_actions) for i in range(self.n_branch)]
        self.fc_branch_layer2 = nn.ModuleList(fc_branch2)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = [F.relu(fc_branch_layer(h2)) for fc_branch_layer in self.fc_branch_layer]

        adv = [fc_branch_layer(h3) for h3, fc_branch_layer in zip(h3, self.fc_branch_layer2)]
        val = [s_branch_layer(h3).expand(-1, adv[0].size(1)) for h3, s_branch_layer in zip(h3, self.s_branch_layer)]

        # Qの算出 (optionいくつかある）
        # 1)  naive approach
        # output = [val + adv for adv in adv]
        # for i, h in enumerate(h_branch):
        #     h_branch[i] = state_value + h

        # 2)  average reduction approach
        output = [val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1)) for val, adv in zip(val,adv)]
        # 3)  maximum reduction approach

        # print('adv:', val)
        # print('val:', val)
        # print('output:', output)

        return output


# nn_option : 6
class BranchedNetDuel(nn.Module):

    def __init__(self, num_actions, num_products, n_fc1, n_fc2, n_fc3,n_branch, num_states):
        self.n_branch = n_branch
        self.num_states = num_states
        self.num_products = num_products
        self.num_actions = num_actions

        super(BranchedNetDuel, self).__init__()
        self.fc1 = nn.Linear(self.num_states, n_fc1)
        self.fc2 = nn.Linear(n_fc1, n_fc2)  # state representation
        # 1) State Value
        self.s = nn.Linear(n_fc2, 1) #state value

        # 2) Branching Architecture
        fc_branch = [nn.Linear(n_fc2, n_fc3) for i in range(self.n_branch)]
        self.fc_branch_layer = nn.ModuleList(fc_branch)

        fc_branch2 = [nn.Linear(n_fc3, self.num_actions) for i in range(self.n_branch)]
        self.fc_branch_layer2 = nn.ModuleList(fc_branch2)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = [F.relu(fc_branch_layer(h2)) for fc_branch_layer in self.fc_branch_layer]

        # productごとのバッチをfc_branch_layerに入れる
        # adv = [self.fc_branch_layer[i](h2) for i in range(self.n_branch)]
        adv = [fc_branch_layer(h3) for h3, fc_branch_layer in zip(h3, self.fc_branch_layer2)]
        val = [self.s(h2).expand(-1, adv[0].size(1)) for i in range(self.n_branch)]

        # Qの算出 (optionいくつかある）
        # 1)  naive approach
        # output = [val + adv for adv in adv]
        # for i, h in enumerate(h_branch):
        #     h_branch[i] = state_value + h

        # 2)  average reduction approach
        output = [val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1)) for val, adv in zip(val,adv)]
        # output = [val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1)) for adv in adv]
        # 3)  maximum reduction approach

        # print('adv:', val)
        # print('val:', val)
        # print('output:', output)

        return output


class BranchedBrain:
    def __init__(self, num_states_per_product, num_products, num_ret, num_actions, option, num_states):
        self.num_products = num_products
        self.num_ret = num_ret

        self.n_branch = int(num_products * num_ret)
        self.num_states = num_states
        # self.num_states = int(num_states_per_product * self.n_branch)
        self.num_actions = num_actions
        self.clip = CLIP
        self.RewardAllocation = True
        self.option = option

        if option == 6:
            self.RewardAllocation = False

        self.memory = ReplayMemory(CAPACITY)  # 経験を記憶するメモリオブジェクトを生成
        self.multi_step_transitions = []  # multi step learning用に追加 / sglab

        if option ==7:
            self.BATCH_SIZE = 179
        else:
            self.BATCH_SIZE = BATCH_SIZE

        # hidden layer
        n_fc1 = 512
        n_fc2 = 256

        if self.n_branch > 10:
            n_fc1 = 2048
            n_fc2 = 1024

        n_fc3 = 128
        LSTM_MEMORY = 128

        # ニューラルネットワークを構築
        if option==3:#Not Duel
            self.main_q_network = BranchedNet(num_actions, num_products, n_fc1, n_fc2,n_fc3,
                                                  self.n_branch, self.num_states).to(device)
            self.target_q_network = BranchedNet(num_actions, num_products, n_fc1,n_fc2,n_fc3,
                                                self.n_branch, self.num_states).to(device)

        elif option ==4:#Duel State
            self.main_q_network = BranchedNetDuelState(num_actions, num_products, n_fc1, n_fc2,n_fc3,
                                              self.n_branch, self.num_states).to(device)
            self.target_q_network = BranchedNetDuelState(num_actions, num_products, n_fc1, n_fc2,n_fc3,
                                                self.n_branch, self.num_states).to(device)

        elif option == 5:#Duel (shared state value)
            self.main_q_network = BranchedNetDuel(num_actions, num_products, n_fc1, n_fc2,n_fc3,
                                              self.n_branch, self.num_states).to(device)
            self.target_q_network = BranchedNetDuel(num_actions, num_products, n_fc1, n_fc2,n_fc3,
                                                self.n_branch, self.num_states).to(device)

        elif option == 6:  # Duel (No Reward Allocation, shared state value)
            self.main_q_network = BranchedNetDuel(num_actions, num_products, n_fc1, n_fc2,n_fc3,
                                                  self.n_branch, self.num_states).to(device)
            self.target_q_network = BranchedNetDuel(num_actions, num_products, n_fc1, n_fc2,n_fc3,
                                                    self.n_branch, self.num_states).to(device)

        elif option == 7:  #  branching with LSTM
            self.main_q_network = LSTMBranchedNet(num_actions, num_products, n_fc1, n_fc2, LSTM_MEMORY, n_fc3,
                                                  self.n_branch, self.num_states).to(device)
            self.target_q_network = LSTMBranchedNet(num_actions, num_products, n_fc1, n_fc2, LSTM_MEMORY, n_fc3,
                                                    self.n_branch, self.num_states).to(device)

        # print(self.main_q_network)  # ネットワークの形を出力
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=LR_DECAY)

        # TD誤差のメモリオブジェクトを生成
        self.td_error_memory = TDerrorMemory(CAPACITY)

    def estimate_palettes(self, state):
        action = self.decide_action_with_estimate(state)
        return self.calc_palettes(action[0])

    def calc_palettes(self, action):
        res = 0
        for i in range(self.n_branch):
            res += action[i] * self.lots[i]
        return res

    def decide_action_with_estimate(self, state):
        ''' 他エージェントの行動を予測した上で自分の行動を決める'''
        self.main_q_network.eval()  # ネットワークを推論モードに切り替える
        with torch.no_grad():
            for _ in range(5):
                Q_s = self.main_q_network(state.to(device))
                action = [Q.max(1)[1].view(1, 1) for Q in Q_s]
                action = torch.unsqueeze(torch.Tensor(action), 0).type(torch.LongTensor)
                forecasted_palettes = self.calc_palettes(action[0])
                state[0, self.num_states-1] = forecasted_palettes
        return action

    def decide_action(self, state, episode, greedy, sampling, has_wh_const, wh_inv_lot):
        '''現在の状態に応じて、行動を決定する'''
        # ε-greedy法で徐々に最適行動のみを採用する
        # WH在庫を制約とするかどうかをパラメータとする
        epsilon = START_EPSILON * (1 / (episode//100 + 1)) + FINAL_EPSILON

        self.main_q_network.eval()  # ネットワークを推論モードに切り替える
        with torch.no_grad():
            # POMDPに対応してstateのリストに対して期待値をとり，それに対してargmaxする
            if sampling:
                state = torch.cat(state)
                Q_s = self.main_q_network(state.to(device))
                # Q_s = self.main_q_network(state.to("cpu"))
                Q_s = [Q.mean(dim=0) for Q in Q_s]
                action = [Q.argmax().view(1, 1) for Q in Q_s]
                action = torch.unsqueeze(torch.Tensor(action), 0).type(torch.LongTensor)
            else:
                if self.option == 7:
                    # 現在のstateだけでなく、現在のepisodeにおけるこれまでのstateを全て渡す
                    if len(self.memory.memory) == 0:
                        state_batch = state
                    elif episode == np.unique(np.array(Transition(*zip(*self.memory.memory)).episode))[-1]:
                        batch = Transition(*zip(*list(filter(lambda x: x.episode == episode, self.memory.memory))))
                        state_batch = torch.cat(batch.state)
                        state_batch = torch.cat((state_batch, state), 0)
                    else:
                        state_batch = state
                    train_hidden_state, train_cell_state = self.main_q_network.init_states()
                    Q_s = self.main_q_network(state_batch.view(len(state_batch), 1, -1), train_hidden_state, train_cell_state)
                    Q_s = [Q_s.view(-1, self.num_actions)[-1].view(1, -1) for Q_s in Q_s]
                else:
                    Q_s = self.main_q_network(state.to(device))

                # if USE_INDEPENDENT_LEARN:
                #     action = self.decide_action_with_estimate(state)
                action = [Q.max(1)[1].view(1, 1) for Q in Q_s]
                action = torch.unsqueeze(torch.Tensor(action), 0).type(torch.LongTensor)

                # print(f'in decide_action: state: {state}, action: {action}')
                # print(f'-- amount : {self.calc_palettes(action[0])}')
                # print(f'-- {state[0][0]}, {action[0][0]}')

                ''' WH在庫制約の考慮'''
                if has_wh_const:
                    # 在庫がある中で全拠点の合計のQが最大になるactionをそれぞれの拠点にて選択する
                    action = np.array(action).reshape(self.num_products, self.num_ret)
                    Q_s_reshaped = np.array([Q.detach().numpy() for Q in Q_s]).reshape(self.num_products, -1, self.num_actions)
                    action = self.get_opt_action_under_const(Q_s_reshaped, action, wh_inv_lot)
                    action = torch.unsqueeze(torch.Tensor(action), 0).type(torch.LongTensor)
                # print('BEST UNDER CONST', action)

        if greedy:
            # print('greedily chosen action:', action)
            return action

        if epsilon > np.random.uniform(0, 1):
            # random action
            if has_wh_const:
                action = self.get_random_action_under_const(wh_inv_lot)
                action = torch.unsqueeze(torch.Tensor(action), 0).type(torch.LongTensor)
                # print('RANDOM', action)
            else:
                # randomに行動する場合はある単一の枝のみとする
                # random対象のbranch index
                # randomに行動するのは，学習対象の枝とする
                idx = (episode // BRANCH_SWITCHING_FREQ) % self.n_branch
                # idx = random.randrange(self.n_branch)
                action[0][idx] = random.randrange(self.num_actions)
                # print('random', idx, action)
                # action = torch.LongTensor(
                #     [[random.randrange(self.num_actions) for i in range(self.n_branch)]])

        return action

    def get_random_action_under_const(self, wh_inv_lot):
        ''' randomに取得する場合 '''
        # WH在庫を上限とした制約を満たす組み合わせからランダムに行動を決める
        # 全ての拠点に対して同時に決定する
        selected_action = []
        for const in wh_inv_lot:
            random_action = []
            cnt = 0
            for q0 in range(self.num_actions):
                for q1 in range(self.num_actions):
                    for q2 in range(self.num_actions):
                        if q0 + q1 + q2 <= const:
                            random_action.append([cnt, [q0, q1, q2]])
                            cnt = cnt + 1
            # 在庫制約を満たす組み合わせからランダムに一つ選ぶ
            s = random.randrange(len(random_action))
            selected_action.extend(random_action[s][1])

        return selected_action

    def get_opt_action_under_const(self, Q_s_reshaped, action, wh_inv_lot):
        opt_action = []
        # inv(WH)も取得する
        for Q, action_, const in zip(Q_s_reshaped, action, wh_inv_lot):
            if action_.sum() > const:
                ''' 最適解が在庫の関係で選べない場合'''
                # 全商品でのループ
                Q_opt = - np.inf
                # print(action_[0], action_[1], action_[2])
                # print('constraint:', const)
                # 拠点数固定版
                for q0 in range(action_[0] + 1):
                    for q1 in range(action_[1] + 1):
                        for q2 in range(action_[2] + 1):
                            if q0 + q1 + q2 <= const:
                                Q_sum = Q[0][q0] + Q[1][q1] + Q[2][q2]
                                if Q_sum > Q_opt:
                                    Q_opt = Q_sum
                                    opt_action_ = np.array([q0, q1, q2])
            else:
                ''' 在庫が制約ではない場合'''
                opt_action_ = action_
            # print('OPT', opt_action_)
            opt_action.extend(opt_action_)

        return opt_action

    def add_memory(self, transition):
        ''' Multi Stepの場合は，　multi-step分の割引報酬およびNstep後のstateをtransitionとしてReplayMemoryに送る　'''
        if 1 < NSTEP:
            transition = self._get_multi_step_transition(transition)

        if transition == None:
            # Nstep分溜まっていない時はReplayMemoryに入れない
            return

        self.memory.push(transition)
        self.td_error_memory.push(0)  # tentative // for multi step learning

    def _get_multi_step_transition(self, transition):
        self.multi_step_transitions.append(transition)
        if len(self.multi_step_transitions) < NSTEP:
            return None

        next_state = transition.next_state
        nstep_reward = 0
        nstep_reward_p = torch.zeros(self.n_branch, device=device)

        for i in range(NSTEP):
            r = self.multi_step_transitions[i].reward
            r_p = self.multi_step_transitions[i].reward_p
            nstep_reward += r * GAMMA ** i
            nstep_reward_p += torch.Tensor([r * GAMMA ** 1 for r in r_p]).to(device)

            if self.multi_step_transitions[i].next_state is None:
                next_state = None
                break

        episode, step, state, action, _, _, _ = self.multi_step_transitions.pop(0)

        return Transition(episode, step, state, action, next_state, nstep_reward, torch.unsqueeze(nstep_reward_p, 0))

    def replay(self, episode):
        '''Experience Replayでネットワークの結合パラメータを学習'''
        st = time.time()
        # 1. メモリサイズの確認
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. ミニバッチの作成
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.reward_p_batch, self.non_final_next_states = self.make_minibatch(
            episode)

        # 3. 教師信号となるQ(s_t, a_t)値を求める
        temp = time.time()
        self.expected_state_action_values = self.get_expected_state_action_values()
        time_teacher = time.time() - temp

        # 4. 結合パラメータの更新
        st_learn = time.time()
        self.update_main_q_network(episode)

        ed = time.time()
        # print(f'learning time: {(ed-st_learn)/(ed-st):.0%}, make teacher: {(time_teacher)/(ed-st):.0%}')

    def make_minibatch(self, episode):
        '''2. ミニバッチの作成'''

        # 2.1 メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(BATCH_SIZE)
        '''
        if episode < 30:
        else:
            # TD誤差に応じてミニバッチを取り出すに変更
            indexes = self.td_error_memory.get_prioritized_indexes(BATCH_SIZE)
            transitions = [self.memory.memory[n] for n in indexes]
        '''

        # 2.2 各変数をミニバッチに対応する形に変形
        if self.option == 7:
            # 現在投入途中のepisodeは含めない
            episodes = np.unique(np.array(Transition(*zip(*self.memory.memory)).episode))#[0:-1]
            episode_num = random.sample(list(episodes), 1)[0]
            batch = Transition(*zip(*list(filter(lambda x: x.episode == episode_num, self.memory.memory))))
            # max_episode = episodes.max()
        else:
            batch = Transition(*zip(*transitions))


        # 2.3 各変数の要素をミニバッチに対応する形に変形し、ネットワークで扱えるようVariableにする
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        reward_p_batch = torch.cat(batch.reward_p)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        if self.option ==7:
            state_batch = state_batch.view(len(state_batch), 1, -1)
            non_final_next_states = non_final_next_states.view(len(non_final_next_states), 1, -1)

        return batch, state_batch, action_batch, reward_batch, reward_p_batch, non_final_next_states

    def get_expected_state_action_values(self):
        '''3. 教師信号となるQ(s_t, a_t)値を求める'''

        # 3.1 ネットワークを推論モードに切り替える
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 ネットワークが出力したQ_d(s_t, a_t)を求める
        # action dimensionごとのlistとして受け取る
        if self.option ==7:
            self.BATCH_SIZE = self.state_batch.shape[0] # steps数
            train_hidden_state, train_cell_state = self.main_q_network.init_states()
            Qs = self.main_q_network(self.state_batch, train_hidden_state, train_cell_state)
            Qs = [Qs.view(-1, self.num_actions) for Qs in Qs]
        else:
            Qs = self.main_q_network(self.state_batch)

        self.state_action_values = [Qs[i].gather(1, torch.unsqueeze(self.action_batch[:, i], 1)) for i in
                                    range(self.n_branch)]

        # 3.3 max{Q_d(s_t+1, a)}値を求める。ただし次の状態があるかに注意。
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        next_state_values = [torch.zeros(self.BATCH_SIZE, device=device) for i in range(self.n_branch)]
        a_m = [torch.zeros(self.BATCH_SIZE).type(torch.LongTensor).to(device) for i in range(self.n_branch)]

        # 次の状態でのQ_d(s,a)値が最大であるaction(a_m)をMain Q-Networkから求める
        if self.option ==7:
            next_Qs = self.main_q_network(self.non_final_next_states, train_hidden_state, train_cell_state)
            next_Qs = [next_Qs.view(-1, self.num_actions) for next_Qs in next_Qs]
        else:
            next_Qs = self.main_q_network(self.non_final_next_states)

        for i in range(self.n_branch):
            a_m[i][non_final_mask] = next_Qs[i].detach().max(1)[1]

        # 次の状態があるものだけにフィルター
        a_m_non_final_next_states = [a[non_final_mask] for a in a_m]

        # 次の状態があるindexの、行動a_mのQ値をtarget Q-Networkから求める
        if self.option ==7:
            next_Qs_target = self.target_q_network(self.non_final_next_states, train_hidden_state, train_cell_state)
            next_Qs_target = [next_Qs_target.view(-1, self.num_actions) for next_Qs_target in next_Qs_target]
        else:
            next_Qs_target = self.target_q_network(self.non_final_next_states)

        for i in range(self.n_branch):
            next_state_values[i][non_final_mask] = next_Qs_target[i].gather(1, torch.unsqueeze(
                a_m_non_final_next_states[i].type(torch.LongTensor).to(device), 1)).detach().squeeze()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        # 1) naive
        # expected_state_action_values = [self.reward_batch + (GAMMA ** NSTEP) * next_state_values for next_state_values in next_state_values]
        # 1) naive ( indepent Q_d)
        if self.RewardAllocation:
            expected_state_action_values = [self.reward_p_batch + (GAMMA ** NSTEP) * next_state_values for
                                            self.reward_p_batch, next_state_values in
                                            zip(torch.t(self.reward_p_batch), next_state_values)]
        else:
            expected_state_action_values = [self.reward_batch + (GAMMA ** NSTEP) * next_state_values for next_state_values in next_state_values]
            '''
            expected_state_action_values = [reward_batch + (GAMMA ** NSTEP) * next_state_values for
                                            reward_batch, next_state_values in
                                            zip(torch.t(self.reward_batch), next_state_values)]
            '''


        # 2) max operation
        # target_Q_max = torch.Tensor([np.array([next_state_values[j][i] for j in range(self.n_branch)]).max() for i in range(next_state_values[0].size()[0])])
        # expected_state_action_values = self.reward_batch + (GAMMA ** NSTEP) * target_Q_max
        # 3) mean operation
        # target_Q_mean = torch.Tensor([np.array([next_state_values[j][i] for j in range(self.n_branch)]).mean() for i in range(next_state_values[0].size()[0])])

        # branchごとのリストとして返す
        return expected_state_action_values

    def update_main_q_network(self, episode):
        '''4. 結合パラメータの更新'''
        def show_params(stop=False):
            for idx, p in enumerate(self.main_q_network.parameters()):
                print(idx, p.shape)
                print(p.grad)
                print(p)
                if p.grad is None:
                    print(f'({episode}): {idx} - none grad')
            if stop:
                sys.exit()

        # 4.1 ネットワークを訓練モードに切り替える
        self.main_q_network.train()
        self.optimizer.zero_grad()

        # 4.2 損失関数を計算する
        # 単一の枝ごとにパラメータを更新する
        loss = torch.zeros(1).type(torch.FloatTensor).to(device)

        if USE_INDEPENDENT_LEARN and ((episode // (self.n_branch*10)) % 2 == 1):
            # 枝ごとに更新する
            # かつ共通部分のパラメータは固定する
            # これにより対象の枝以外の方策は固定され，非定常環境となることを防ぐ
            idx = (episode // BRANCH_SWITCHING_FREQ) % self.n_branch
            loss += F.smooth_l1_loss(self.state_action_values[idx].view(-1),
                                     self.expected_state_action_values[idx])
            loss.backward()
            # 共通部分を止める
            self.main_q_network.fc1.weight.grad /= 100000
            self.main_q_network.fc1.bias.grad /= 100000
            self.main_q_network.fc2.weight.grad /= 100000
            self.main_q_network.fc2.bias.grad /= 100000
        else:
            # 共通部分も含め，すべての枝を同時に更新していく
            for i in range(self.n_branch):
                loss += F.smooth_l1_loss(self.state_action_values[i].view(-1),
                                         self.expected_state_action_values[i])
            loss.backward()
            #  gradient scaling
            self.main_q_network.fc1.weight.grad /= self.n_branch
            self.main_q_network.fc1.bias.grad /= self.n_branch
            self.main_q_network.fc2.weight.grad /= self.n_branch
            self.main_q_network.fc2.bias.grad /= self.n_branch

        # 自動的にブランチごとのバックプロパゲーションになるはず
        if self.option == 7:
            for p in self.main_q_network.parameters():
                p.grad = p.grad / (self.n_branch)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.main_q_network.parameters(), self.clip)
        self.optimizer.step()

    def update_target_q_network(self):  # DDQNで追加
        '''Target Q-NetworkをMainと同じにする'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def update_td_error_memory(self):  # PrioritizedExperienceReplayで追加
        '''TD誤差メモリに格納されているTD誤差を更新する'''

        # ネットワークを推論モードに切り替える
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 全メモリでミニバッチを作成
        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        reward_p_batch = torch.cat(batch.reward_p)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        # ネットワークが出力したQ(s_t, a_t)を求める
        if self.option ==7:
            self.BATCH_SIZE = self.state_batch.shape[0] # steps数
            train_hidden_state, train_cell_state = self.main_q_network.init_states()
            Qs = self.main_q_network(self.state_batch, train_hidden_state, train_cell_state)
            Qs = [Qs.view(-1, self.num_actions) for Qs in Qs]
        else:
            Qs = self.main_q_network(state_batch)
        state_action_values = [Qs[i].gather(1, torch.unsqueeze(action_batch[:, i], 1)) for i in range(self.n_branch)]

        # next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))

        # まずは全部0にしておく、サイズはメモリの長さである
        next_state_values = [torch.zeros(len(self.memory), device=device) for i in range(self.n_branch)]
        a_m = [torch.zeros(len(self.memory)).type(torch.LongTensor).to(device) for i in range(self.n_branch)]

        # 次の状態でのQ_d(s,a)値が最大であるaction(a_m)をMain Q-Networkから求める
        if self.option ==7:
            next_Qs = self.main_q_network(non_final_next_states, train_hidden_state, train_cell_state)
            next_Qs = [next_Qs.view(-1, self.num_actions) for next_Qs in next_Qs]
        else:
            next_Qs = self.main_q_network(non_final_next_states)
        for i in range(self.n_branch):
            a_m[i][non_final_mask] = next_Qs[i].detach().max(1)[1]

        # 次の状態があるものだけにフィルター
        a_m_non_final_next_states = [a[non_final_mask] for a in a_m]

        # 次の状態があるindexの、行動a_mのQ値をtarget Q-Networkから求める
        if self.option ==7:
            next_Qs_target = self.target_q_network(non_final_next_states, train_hidden_state, train_cell_state)
            next_Qs_target = [next_Qs_target.view(-1, self.num_actions) for next_Qs_target in next_Qs_target]
        else:
            next_Qs_target = self.target_q_network(non_final_next_states)
        for i in range(self.n_branch):
            next_state_values[i][non_final_mask] = next_Qs_target[i].gather(
                1,
                torch.unsqueeze(a_m_non_final_next_states[i].type(torch.LongTensor).to(device), 1)).detach().squeeze()

        # expected_state_action_values = [reward_batch + (GAMMA ** NSTEP) * next_state_values for next_state_values in next_state_values]
        if self.RewardAllocation:
            expected_state_action_values = [reward_p_batch + (GAMMA ** NSTEP) * next_state_values for
                                            reward_p_batch, next_state_values in
                                            zip(torch.t(reward_p_batch), next_state_values)]
        else:
            expected_state_action_values = [reward_batch + (GAMMA ** NSTEP) * next_state_values for next_state_values in next_state_values]

        # max operation ver (保留)
        # target_Q_max = torch.Tensor([np.array([next_state_values[j][i] for j in range(self.n_branch)]).max() for i in range(next_state_values[0].size()[0])])
        # expected_state_action_values = reward_batch + (GAMMA ** NSTEP) * target_Q_max

        # TD誤差を求める
        print('Replay Memory size is...:',len(self.memory), len(self.td_error_memory.memory))

        td_errors = torch.zeros(len(self.memory), device=device)
        for l in [abs(expected_state_action_values[i] - state_action_values[i].view(-1)) for i in range(self.n_branch)]:
            td_errors += l

        # TD誤差メモリを更新、Tensorをdetach()で取り出し、NumPyにしてから、Pythonのリストまで変換
        # self.td_error_memory.memory = td_errors.detach().to("cpu").numpy().tolist()
        self.td_error_memory.memory = td_errors.detach().cpu().numpy().tolist()


# TD誤差を格納するメモリクラスを定義します
class TDerrorMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def clear_memory(self):
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, td_error):
        '''TD誤差をメモリに保存します'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)

    def get_prioritized_indexes(self, batch_size):
        '''TD誤差に応じた確率でindexを取得'''

        # TD誤差の和を計算
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)  # 微小値を足す

        # batch_size分の乱数を生成して、昇順に並べる
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        # 作成した乱数で串刺しにして、インデックスを求める
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (
                        abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1

            # 微小値を計算に使用した関係でindexがメモリの長さを超えた場合の補正
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)

        return indexes

    def update_td_error(self, updated_td_errors):
        '''TD誤差の更新'''
        self.memory = updated_td_errors


# 経験を保存するメモリクラスを定義します
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def clear_memory(self):
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, transition):
        # multi step learningのため変更 / sglab
        '''transition = (state, action, state_next, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        # self.memory[self.index] = Transition(state, action, state_next, reward)
        self.memory[self.index] = transition

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)
