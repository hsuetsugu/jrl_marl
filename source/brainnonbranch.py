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

Transition = namedtuple('Transition', ('episode', 'step' ,'state', 'action', 'next_state', 'reward'))
print(device)


class IndependentNet(nn.Module):
    def __init__(self, num_actions, num_products, n_fc1, n_fc2, n_fc3, n_branch, num_states):
        self.n_branch = n_branch
        self.num_states = num_states
        self.num_actions = num_actions

        super(IndependentNet, self).__init__()
        self.fc1 = nn.Linear(self.num_states, n_fc1)
        self.fc2 = nn.Linear(n_fc1, n_fc2)
        self.fc3 = nn.Linear(n_fc2, n_fc3)
        self.fc4 = nn.Linear(n_fc3, num_actions)

        '''
        if FUNCTION_APPROX_LINEAR:
            self.fc1 = nn.Linear(self.num_states, num_actions)
        else:
            assert NN_LAYER > 1, 'LAYER should be more than 2'
            if NN_LAYER == 2:
                self.fc1 = nn.Linear(self.num_states, n_fc1)
                self.fc2 = nn.Linear(n_fc1, num_actions)
            elif NN_LAYER == 3:
                self.fc1 = nn.Linear(self.num_states, n_fc1)
                self.fc2 = nn.Linear(n_fc1, n_fc2)
                self.fc3 = nn.Linear(n_fc2, num_actions)
            else:
                self.fc1 = nn.Linear(self.num_states, n_fc1)
                self.fc2 = nn.Linear(n_fc1, n_fc2)
                self.fc3 = nn.Linear(n_fc2, n_fc3)
                self.fc4 = nn.Linear(n_fc3, num_actions)
        '''

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        output = self.fc4(h3)
        return output
        '''
        if FUNCTION_APPROX_LINEAR:
            return self.fc1(x)
        else:
            h1 = F.relu(self.fc1(x))
            if NN_LAYER == 2:
                return self.fc2(h1)
            h2 = F.relu(self.fc2(h1))
            if NN_LAYER == 3:
                return self.fc3(h2)
            h3 = F.relu(self.fc3(h2))
            output = self.fc4(h3)
            return output
        '''

class IndependentBrain:
    def __init__(self, num_states_per_product, num_products, num_ret, num_actions, option, num_states):
        self.num_products = num_products
        self.num_ret = num_ret
        self.n_branch = int(num_products * num_ret)
        self.num_states = num_states
        self.num_actions = num_actions
        self.clip = CLIP
        self.RewardAllocation = True
        self.option = option
        self.BATCH_SIZE = BATCH_SIZE

        self.memory = ReplayMemory(CAPACITY)  # 経験を記憶するメモリオブジェクトを生成
        self.td_error_memory = TDerrorMemory(CAPACITY)

        self.multi_step_transitions = []  # multi step learning用に追加

        self.main_q_network = IndependentNet(num_actions, num_products, n_fc1, n_fc2,n_fc3,
                                             self.n_branch, self.num_states).to(device)
        self.target_q_network = IndependentNet(num_actions, num_products, n_fc1,n_fc2,n_fc3,
                                               self.n_branch, self.num_states).to(device)

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=LR_DECAY)

    @staticmethod
    def _deide_action_boltzman(Qs, temperature_parameter):
        # ボルツマン選択（未使用）
        def boltzmann_distribution(array):
            if temperature_parameter == 0:
                result = np.zeros([len(array)])
                result[random.choice(len(array))] = 1
                return result
            return array ** (1 / temperature_parameter) / np.sum(array ** (1 / temperature_parameter))

        boltzmann_array = boltzmann_distribution(Qs)
        random_value = random.uniform(0, 1)
        threshold = 0
        for i, value in enumerate(boltzmann_array):
            threshold += value
            if random_value < threshold:
                return i
        return i

    def decide_action(self, state, episode, greedy=False, target=False, joint_random=0):
        def boltzman_decision(Qs):
            # Qに比例して選択させる（ただし0は除く）
            # 温度パラメータはepisodeに従って無限大から始まり0に近付いて行くものとする
            temperature = 10000 * (BOLTZMAN_TEMP_DECAY ** episode)
            temperature = max(temperature, BOLTZMAN_TEMP_MIN)

            q = Qs[0].detach().cpu().numpy()
            exp_q = np.exp(q / temperature)
            prob = exp_q / np.sum(exp_q)

            if random.uniform(0, 1) < 0.002:
                print(f'({episode}), {temperature:.2f}, {q}, {exp_q}, prob: {prob}')

            if np.isnan(prob[0]):
                return random.randrange(self.num_actions)

            random_value = random.uniform(0, 1)
            threshold = 0
            for i, value in enumerate(prob):
                threshold += value
                if random_value < threshold:
                    return i
            return i

        # ε-greedy法で徐々に最適行動のみを採用する
        if EPSILON_CONSTANT:
            epsilon = START_EPSILON
        else:
            epsilon = START_EPSILON * (1 / (episode//DECAY_FREQ + 1)) + FINAL_EPSILON

        with torch.no_grad():
            if target:
                self.target_q_network.eval()
                Qs = self.target_q_network(state.to(device))
            else:
                self.main_q_network.eval()
                Qs = self.main_q_network(state.to(device))
            action = Qs.max(1)[1].view(1, 1)
        if greedy:
            return action, Qs.max(1)[1]

        # joint-randomでrandomにjoint-actionをとる場合
        if E_GREEDY_JOINT and joint_random == 1:
            # ランダムに選択する：決定空間が大きい場合には収束に時間がかかる
            ret = boltzman_decision(Qs)
            action[0, 0] = ret
            return action, None

        # joint-randomで一斉に0をとる場合
        if E_GREEDY_JOINT and joint_random == 2:
            # 全ての商品が0
            action[0, 0] = 0
            return action, None

        # independent-randomでrandomにとる場合
        if epsilon/self.n_branch > np.random.uniform(0, 1):
            ret = boltzman_decision(Qs)
            action[0, 0] = ret
            return action, None

        return action, Qs.max(1)[1]

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

        for i in range(NSTEP):
            r = self.multi_step_transitions[i].reward
            nstep_reward += r * GAMMA ** i

            if self.multi_step_transitions[i].next_state is None:
                next_state = None
                break

        episode, step, state, action, _, _ = self.multi_step_transitions.pop(0)

        return Transition(episode, step, state, action, next_state, nstep_reward)

    def replay(self, episode):
        '''Experience Replayでネットワークの結合パラメータを学習'''
        st = time.time()
        # 1. メモリサイズの確認
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. ミニバッチの作成
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = \
            self.make_minibatch(episode)

        # 3. 教師信号となるQ(s_t, a_t)値を求める
        temp = time.time()
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4. 結合パラメータの更新
        st_learn = time.time()
        self.update_main_q_network(episode)
        ed = time.time()

        # 教師信号作成とlearnで同じくらいかかっている
        # if DEBUG_PRINT:
        #     print(f'learning time: {(ed-st_learn):.4f}sec, make teacher: {st_learn-temp:.4f}sec')

    def make_minibatch(self, episode):
        '''2. ミニバッチの作成'''

        # 2.1 メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 各変数をミニバッチに対応する形に変形
        batch = Transition(*zip(*transitions))
        # print(batch.state)
        # print(batch.action)
        # print(batch.reward)

        # 2.3 各変数の要素をミニバッチに対応する形に変形し、ネットワークで扱えるようVariableにする
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        '''3. 教師信号となるQ(s_t, a_t)値を求める'''
        # 3.1 ネットワークを推論モードに切り替える
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 ネットワークが出力したQ_d(s_t, a_t)を求める
        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        # 3.3 max{Q_d(s_t+1, a)}値を求める。ただし次の状態があるかに注意。
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        a_m = torch.zeros(self.BATCH_SIZE).type(torch.LongTensor).to(device)

        # 次の状態でのQ_d(s,a)値が最大であるaction(a_m)をMain Q-Networkから求める
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 次の状態があるものだけにフィルター
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 次の状態があるindexの、行動a_mのQ値をtarget Q-Networkから求める
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = self.reward_batch + (GAMMA ** NSTEP) * next_state_values

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
        st = time.time()
        loss = torch.zeros(1).type(torch.FloatTensor).to(device)
        loss += F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        st_backward = time.time()
        loss.backward()

        # gradient clipping
        if ENABLE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(self.main_q_network.parameters(), self.clip)

        # lossの正負に応じて学習率を変える＝gradientを小さくする
        if loss < 0:
            for p in self.main_q_network.parameters():
                p.grad /= HYSTERETIC_BETA

        st_step = time.time()
        self.optimizer.step()
        # if DEBUG_PRINT:
        #     print(f'calc_loss:{st_backward - st:.4f}sec, '
        #           f'backward:{st_step - st_backward:.4f}sec, '
        #           f'optimizer.step : {time.time() - st_step:.4f}sec')

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
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        Qs = self.main_q_network(state_batch)
        state_action_values = Qs.gather(1, action_batch)

        # next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))

        # まずは全部0にしておく、サイズはメモリの長さである
        next_state_values = torch.zeros(len(self.memory), device=device)
        a_m = torch.zeros(len(self.memory)).type(torch.LongTensor).to(device)

        # 次の状態でのQ_d(s,a)値が最大であるaction(a_m)をMain Q-Networkから求める
        next_Qs = self.main_q_network(non_final_next_states)
        a_m[non_final_mask] = next_Qs.detach().max(1)[1]

        # 次の状態があるものだけにフィルター
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 次の状態があるindexの、行動a_mのQ値をtarget Q-Networkから求める
        next_state_values[non_final_mask] = self.target_q_network(
            non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        td_errors = (reward_batch + (GAMMA ** NSTEP) * next_state_values) - \
            state_action_values.squeeze()

        # TD誤差メモリを更新、Tensorをdetach()で取り出し、NumPyにしてから、Pythonのリストまで変換
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
