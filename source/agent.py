from config_learn import *
from scm import Node, ResultMemory
from scm import IGNORE_STEP, BUFFER, Cinv_ret, Cpel_ret, Ctrans, Cinv_wh, Cpel_wh
from collections import namedtuple
from brainnonbranch import IndependentBrain, Transition
from brainbranched import BranchedBrain
Transition_branched = namedtuple('Transition', ('episode', 'step' ,'state', 'action', 'next_state', 'reward', 'reward_p'))

import logging
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import matplotlib.pyplot as plt
import copy
import os
import datetime
from datetime import datetime
import csv
import pickle
import gc
import time
import psutil
import itertools
import numpy as np
import math
# import multiprocessing as mp
# import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method
from multiprocessing import cpu_count

from joblib import Parallel, delayed

import random
from deap import base
from deap import creator
from deap import tools
from itertools import chain
import sys

def unwrap_self_f(brain, episode):
    # メソッドfをクラスメソッドとして呼び出す関数
    brain.replay(episode)

def _unwrap_forward(net, state):
    net.eval()
    return net(state)

def unwrap_forward(args):
    return _unwrap_forward(*args)

class Agent():
    def __init__(self,scenario, env,num_actions_per_product, state_real:bool, obs_only:bool,sample_val:bool,
                 op_trans_alloc):
        self.scenario = scenario
        self.Ctrans = Ctrans
        self.env = env
        self.season = env.season
        self.error_rate = env.error_rate
        self.num_ret = env.num_ret
        self.num_products = env.num_products
        self.obs_only = obs_only
        self.sample_val = sample_val
        self.op_trans_alloc = op_trans_alloc
        self.agent_initialized_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # state space
        # print(state_real)
        self.state_real = state_real

        if USE_FORECAST:
            self.num_states_per_product = 5
        else:
            self.num_states_per_product = 2
        print('num_states:', self.num_states_per_product)

        self.num_states = self.num_states_per_product * self.num_products * self.num_ret

        # action space
        self.num_actions_per_product = num_actions_per_product
        self.num_actions = self.num_actions_per_product ** (self.num_products * self.num_ret)#商品数の数だけactionスペースは増える

        self.num_time = env.num_time
        self.version = ""
        self.comp_at =  datetime.now().strftime("%Y%m%d%H%M%S")
        self.model_path = 'model/weight.pth'
        self.scm_best_path = 'scm/best_scm.pth'
        self.scm_val_path = 'scm/val_scm.pth'

        self.op_trans_alloc = env.op_trans_alloc

    def write_to_csv(self, out_list):
        self.file = 'result/{}_{}.csv'.format(self.comp_at, self.version)
        f = open(self.file, 'a')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(out_list)
        f.close()

    def write_to_csv_val(self, out_list):
        self.file_val = 'result_val/{}_{}_val.csv'.format(self.comp_at, self.version)
        f = open(self.file_val, 'a')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(out_list)
        f.close()

    def _set_logger(self):
        # Logger Setting
        self.model_path = 'model/{}_{}_weight.pth'.format(self.comp_at, self.version)
        self.scm_best_path = 'scm/{}_{}_best_scm.pth'.format(self.comp_at, self.version)
        self.scm_val_path = 'scm/{}_{}_val_scm.pth'.format(self.comp_at, self.version)
        self.LOG =  'log/{}_{}.log'.format(self.comp_at, self.version)

        self.logger = getLogger(__name__)
        log_fmt = Formatter('%(asctime)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
        handler = StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(log_fmt)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        handler = FileHandler(self.LOG, 'a')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(log_fmt)
        self.logger.addHandler(handler)
        
        self.logger.info(self.LOG)
        
    ''' Q-Leavrning Agent Functions '''
    def update_q_function(self, episode):
        '''Q関数を更新する'''
        self.brain.replay(episode)

    def memorize_multi(self, idx, episode, step, state, action, state_next, reward):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        ''' multi-step learning用に変更'''
        ''' RNN対応するためシーケンシャルに経験を貯めるためのepisode, stepを記憶'''
        # self.brain.memory.push(state, action, state_next, reward)
        self.brains[idx].add_memory(Transition(episode, step, state, action, state_next, reward))

    def memorize(self, episode, step, state, action, state_next, reward, reward_tuple):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        ''' multi-step learning用に変更'''
        ''' RNN対応するためシーケンシャルに経験を貯めるためのepisode, stepを記憶'''
        # self.brain.memory.push(state, action, state_next, reward)
        self.brain.add_memory(Transition_branched(episode, step, state, action, state_next, reward, reward_tuple))

    def update_target_q_function(self):
        '''Target Q-NetworkをMain Q-Networkと同じに更新'''
        self.brain.update_target_q_network()
        
    def memorize_td_error(self, td_error):  # PrioritizedExperienceReplayで追加
        '''TD誤差メモリにTD誤差を格納'''
        self.brain.td_error_memory.push(td_error)
        
    def update_td_error_memory(self):  # PrioritizedExperienceReplayで追加
        '''TD誤差メモリに格納されているTD誤差を更新する'''
        if self.option_nn == 10:
            for brain in self.brains:
                brain.update_td_error_memory()
        else:
            self.brain.update_td_error_memory()
    ''' Q-Learning Agent Functions <END> '''
        
    def load_weight(self, brain):
        param = torch.load(PATH)
        brain.main_q_network.load_state_dict(param)
        brain.target_q_network.load_state_dict(param)

    def sim_s_q_policy_memorize(self, with_forcst, pred_fresh, memorize, episode,trans_cost_mode):
        # self.logger.info('Simulate static (s,Q)policy...  with Forecast : {}'.format(with_forcst))
        self.env_sq = copy.deepcopy(self.env)
        if pred_fresh:
            self.env_sq.reflesh_pred()  # 予測値のリフレッシュ
        self.env_sq.create_result_memory()  # 結果格納用のResultMemoryを作る
        episode_reward = 0

        observation = self.reset(self.num_states)
        state = observation  # 観測をそのまま状態sとして使用
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0).to(device)

        for step in range(self.num_time):
            action = []
            # actionの決定
            for i, node in enumerate(self.env_sq.nodes):
                # action.extend(node.simulate_s_q_policy(step, with_forcst))
                action.extend(node.simulate_s_s_policy(step, with_forcst,trans_cost_mode))

            if memorize:
                action = np.clip(np.array(action), a_min=0,
                                 a_max=self.num_actions_per_product - 1)  # num_action以上にはならないように制御する
                action_origin = self.get_action_idx_from_list(action)
                action_origin = torch.LongTensor([[action_origin]])
                # print(action, action_origin)

            action_converted = np.array(action).reshape(self.num_products, self.num_ret)

            done, reward, _ = self.env_sq.step(action_converted, step)
            if done:  # 最後
                state_next = None
            else:
                state_next, e_state_next = self.get_next_state(step, self.env_sq.nodes, False)

            if step > IGNORE_STEP:
                episode_reward += reward
                if memorize:
                    self.memorize(
                        episode,
                        step,
                        state,
                        action_origin.to(device),
                        state_next,
                        - reward.to(device)
                    )
                    if step % 90 == 0:
                        # print(episode, step)
                        self.update_q_function(episode)

            state = state_next
            # 終了時の処理
            if (done) and (memorize):
                # TD誤差メモリの中身を更新する
                self.update_td_error_memory()
                # DDQNで追加、2試行に1度、Target Q-NetworkをMainと同じにコピーする
                if (episode % 2 == 0):
                    self.update_target_q_function()
                break

        if not memorize:
            self.logger.info('EpisodeReward: %d' % (episode_reward))
            self.env_sq.calc_episode_performance()
            self.env_sq.episode_performance_print(self.logger)
        if (memorize) and (episode % 50) == 0:
            self.logger.info('(%d)EpisodeReward: %d  Loss: %d' % (
            episode, episode_reward, (np.array(self.brain.td_error_memory.memory) ** 2).mean()))

    def sim_s_q_policy(self, with_forcst,pred_fresh,memorize,episode,trans_cost_mode):
        if memorize:
            self.sim_s_q_policy_memorize(with_forcst,pred_fresh,memorize,episode,trans_cost_mode)

        # self.logger.info('Simulate static (s,Q)policy...  with Forecast : {}'.format(with_forcst))
        self.env_sq = copy.deepcopy(self.env)
        if pred_fresh:
            self.env_sq.reflesh_pred() #予測値のリフレッシュ
        self.env_sq.create_result_memory() #結果格納用のResultMemoryを作る
        episode_reward = 0

        for step in range(self.num_time):
            action = []
            # actionの決定
            for i, node in enumerate(self.env_sq.nodes):
                # action.extend(node.simulate_s_q_policy(step, with_forcst))
                action.extend(node.simulate_s_s_policy(step, with_forcst,trans_cost_mode))

            action_converted = np.array(action).reshape(self.num_products,self.num_ret)
            
            done, reward ,_ = self.env_sq.step(action_converted, step)
            if step > IGNORE_STEP:
                episode_reward += reward

        if not memorize:
            self.logger.info(f'EpisodeReward: {episode_reward}')
            self.env_sq.calc_episode_performance()
            self.env_sq.episode_performance_print(self.logger)

    def optimize_ga(self, mode='scs', capacitated=False):
        # s:trigger, S :reorder
        if 'step' in self.scenario:
            s_min, s_max = 0, 80
            S_min, S_max = 5, 150
        else:
            s_min, s_max = 0, 20
            S_min, S_max = 5, 30
        # t: periodic SS
        t_min, t_max = 0, 10
        # s in QS
        tot_s_min, tot_s_max = 0, 60

        assert mode in ['scs','ss', 'scds', 'qs'], 'mode should be either scs or ss or qs'

        print_setting_ga(self.logger)

        def uniform():
            if mode == 'scs':
                s = list(np.random.randint(s_min, s_max, self.num_products))
                c = list(np.random.randint(s_min, s_max, self.num_products))
                S = list(np.random.randint(S_min, S_max, self.num_products))
                ret = list(chain.from_iterable([[s, c, S] for s, c, S in zip(s, c, S)]))
            if mode == 'scds':
                s = list(np.random.randint(s_min, s_max, self.num_products))
                c = list(np.random.randint(s_min, s_max, self.num_products))
                d = list(np.random.randint(s_min, s_max, self.num_products))
                S = list(np.random.randint(S_min, S_max, self.num_products))
                ret = list(chain.from_iterable([[s, c, d, S] for s, c, d, S in zip(s, c, d, S)]))
            if mode == 'ss':
                t = list(np.random.randint(t_min, t_max, 1))
                s = list(np.random.randint(s_min, s_max, self.num_products))
                S = list(np.random.randint(S_min, S_max, self.num_products))
                ret = t + list(chain.from_iterable([[s, S] for s, S in zip(s, S)]))
            if mode == 'qs':
                tot_s = list(np.random.randint(tot_s_min, tot_s_max, 1))
                S = list(np.random.randint(S_min, S_max, self.num_products))
                ret = tot_s + S
            return ret

        def get_lower():
            if mode == 'scs':
                ret = [s_min, s_min, S_min] * self.num_products
            if mode == 'scds':
                ret = [s_min, s_min, s_min, S_min] * self.num_products
            if mode == 'ss':
                ret = [t_min] + [s_min, S_min] * self.num_products
            if mode == 'qs':
                ret = [tot_s_min] + [S_min] * self.num_products
            return ret

        def get_upper():
            if mode == 'scs':
                ret = [s_max, s_max, S_max] * self.num_products
            if mode == 'scds':
                ret = [s_max, s_max, s_max, S_max] * self.num_products
            if mode == 'ss':
                ret = [t_max] + [s_max, S_max] * self.num_products
            if mode == 'qs':
                ret = [tot_s_max] + [S_max] * self.num_products
            return ret

        def evaluateInd(individual):
            params = np.array(individual)
            if mode == 'scs':
                val = [self.sim_scs_policy(params, capacitated=capacitated).item() for _ in range(NUM_SAMPLE_GA)]
            elif mode == 'scds':
                val = [self.sim_scs_policy(params, add_d=True, capacitated=capacitated).item() for _ in range(NUM_SAMPLE_GA)]
            elif mode == 'qs':
                val = [self.sim_qs_policy(params, capacitated=capacitated).item() for _ in range(NUM_SAMPLE_GA)]
            else:
                val = [self.sim_ss_periodic_policy(params, capacitated=capacitated).item() for _ in range(NUM_SAMPLE_GA)]
            return [np.array(val).mean()]

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        toolbox.register("individual", tools.initIterate, creator.Individual, uniform)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)

        # toolbox.register("mutate", tools.mutUniformInt, low=0, up=30, indpb=0.2)
        toolbox.register('mutate', tools.mutPolynomialBounded, eta=0.5, low=get_lower(), up=get_upper(), indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluateInd)

        # multi-processing
        # pool = Pool()
        # toolbox.register("map", pool.map)

        # 初期集団を生成する
        pop = toolbox.population(n=self.num_products * NUM_POPULATION)
        # pop = toolbox.population(n=NUM_POPULATION)

        # 初期集団の個体を評価する
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):  # zipは複数変数の同時ループ
            ind.fitness.values = fit
        # self.logger.info("  %i の個体を評価" % len(pop))

        # 進化計算開始
        for g in range(NGEN):
            # 選択
            # 次世代の個体群を選択
            offspring = toolbox.select(pop, len(pop))
            # 個体群のクローンを生成
            offspring = list(map(toolbox.clone, offspring))

            # 交叉
            # 偶数番目と奇数番目の個体を取り出して交差
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    # 交叉された個体の適合度を削除する
                    del child1.fitness.values
                    del child2.fitness.values

            # 変異
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 適合度が計算されていない個体を集めて適合度を計算
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # self.logger.info("  %i の個体を評価" % len(invalid_ind))

            # 次世代群をoffspringにする
            pop[:] = offspring

            # すべての個体の適合度を配列にする
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            self.logger.info(f"({g})世代, len: {len(invalid_ind)}, min: {min(fits):.1f}, max: {max(fits):.1f}, mean: {mean:.1f}")

        best_ind = tools.selBest(pop, 1)[0]
        self.logger.info("最も優れていた個体: %s, %s" % (best_ind, best_ind.fitness.values))

        return best_ind

    def optimize_scs_params(self, all_cond):
        res = []
        best_reward = float('inf')

        all_params = []
        for param in all_cond:
            all_params.append(range(param['s']['lb'], param['s']['ub']))
            all_params.append(range(param['c']['lb'], param['c']['ub']))
            all_params.append(range(param['S']['lb'], param['S']['ub']))

        tot = len(list(itertools.product(*all_params)))
        print(f'total # of combinations : {tot}')

        for idx, param in enumerate(itertools.product(*all_params)):
            self.set_scs_params(np.array(param).reshape(2,3))
            val = self.sim_scs_policy()
            res.append(val)
            if val < best_reward:
                best_reward = val
                best_param = param
                print(f'({idx}/{tot}): current best: {best_reward}, param={best_param}')

        return best_reward, best_param, res

    def sim_ss_periodic_policy(self, params, capacitated=False, draw_result=False, write=False):
        ''' P(S, s)方策をシミュレートする'''
        self.bench_params = params
        self.env_ss_periodic = copy.deepcopy(self.env)

        t = max(1, int(params[0]))
        params = params[1:].reshape(self.num_products, 2)
        self.env_ss_periodic.param_ss_periodic_t = t
        for i, node in enumerate(self.env_ss_periodic.nodes):
            node.set_ss_periodic_params(params[i])

        self.env_ss_periodic.update_demand()
        self.env_ss_periodic.create_result_memory()  # 結果格納用のResultMemoryを作る
        episode_reward = 0

        for step in range(self.num_time):
            action = []

            # tごとに発注機会が訪れる
            if step % self.env_ss_periodic.param_ss_periodic_t == 0:
                # actionの決定
                for i, node in enumerate(self.env_ss_periodic.nodes):
                    for ret in node.iRet:
                        action.extend(node.simulate_ss_periodic_policy(step))
                if capacitated:
                    action = self.modify_heuristic_capacitated(self.env_ss_periodic, action, step)
                action_converted = np.array(action).reshape(self.num_products, self.num_ret)
            else:
                action_converted = np.zeros(self.num_products * self.num_ret).reshape(self.num_products, self.num_ret)

            done, reward, _ = self.env_ss_periodic.step(action_converted, step)
            if step > IGNORE_STEP:
                episode_reward += reward

        if draw_result:
            self.logger.info(f'EpisodeReward: {episode_reward}')
            self.env_ss_periodic.calc_episode_performance()
            self.env_ss_periodic.episode_performance_print(self.logger)

        if write:
            self.output_result('P(s, S)', episode_reward.item(), capacitated)

        return episode_reward

    def sim_qs_policy(self, params, capacitated=False, draw_result=False, write=False):
        ''' QS方策をシミュレートする'''
        self.bench_params = params
        self.env_qs = copy.deepcopy(self.env)

        s = params[0]
        params = params[1:].reshape(self.num_products)
        self.env_qs.param_qs_s = s
        for i, node in enumerate(self.env_qs.nodes):
            node.set_qs_params(params[i])

        self.env_qs.update_demand()
        self.env_qs.create_result_memory()
        episode_reward = 0

        for step in range(self.num_time):
            action = []

            tot_inv = 0
            for i, node in enumerate(self.env_qs.nodes):
                for ret in node.iRet:
                    tot_inv += ret.inv_position[step]

            # 合計在庫量がs以下なら商品ごとにSiまで発注する
            if tot_inv <= self.env_qs.param_qs_s:
                # actionの決定
                for i, node in enumerate(self.env_qs.nodes):
                    for ret in node.iRet:
                        action.extend(node.simulate_qs_policy(step))
                if capacitated:
                    action = self.modify_heuristic_capacitated(self.env_qs, action, step)
                action_converted = np.array(action).reshape(self.num_products, self.num_ret)
            else:
                action_converted = np.zeros(self.num_products * self.num_ret).reshape(self.num_products, self.num_ret)

            done, reward, _ = self.env_qs.step(action_converted, step)
            if step > IGNORE_STEP:
                episode_reward += reward

        if draw_result:
            self.logger.info(f'EpisodeReward: {episode_reward}')
            self.env_qs.calc_episode_performance()
            self.env_qs.episode_performance_print(self.logger)

        if write:
            self.output_result('QS', episode_reward.item(), capacitated)

        return episode_reward

    def modify_heuristic_capacitated(self, env, action, step):
        cnt = 0
        flgs = [0] * self.num_products
        orders = []
        inv_p = []
        vals = []
        vals_inc = []
        reorder = []
        mustorder = []
        trigger = []
        lots = []
        for idx, node in enumerate(env.nodes):
            for Ret in node.iRet:
                orders.append(action[idx] * Ret.lot)
                inv_p.append(Ret.inv_position[step])
                lots.append(Ret.lot)
                reorder.append(Ret.param_reorder_point)
                mustorder.append(Ret.param_mustorder_point)
                trigger.append(Ret.param_trigger_point)

                # 1ロット減らした後の発注量
                vals.append(orders[idx] - Ret.lot)
                # 1ロット増やした場合の発注量
                vals_inc.append(orders[idx] + Ret.lot)

        inv_p = np.array(inv_p)
        vals = np.array(vals)
        vals_inc = np.array(vals_inc)
        reorder = np.array(reorder)
        mustorder = np.array(mustorder)
        trigger = np.array(trigger)

        # すべて0なら何もしない
        if sum(orders) == 0:
            return action

        # capacitatedの場合
        if self.scenario in ['ex1', 'ex5', 'ex6']:
            # 必ず減らす
            while sum(orders) > env.trans_cap:
                # 調整後の在庫量が最もSに近い商品を選択
                idx = abs(reorder - (inv_p + vals)).argmin()
                # 発注量が0なら対象から外す
                if orders[idx] == 0:
                    flgs[idx] = -1
                    continue
                # 調整可能な商品がなくなれば終了
                if flgs[idx] == -1:
                    break
                orders[idx] = vals[idx]
                vals[idx] -= lots[idx]
                action[idx] -= 1
            return action

        # step-wiseの場合
        elif 'step' in self.scenario:
            # return action

            # 最後一つの積載率が50％を切っている場合は1つ分減らす，そうでない場合は満歳にするまで増やす
            temp_order = sum(orders) - (sum(orders) // env.trans_cap) * env.trans_cap
            if temp_order == 0:
                return action
            cond_dec = sum(orders) > env.trans_cap and temp_order < env.trans_cap * 0.5
            cond_inc = temp_order >= env.trans_cap * 0.5
            # print('before', sum(orders), cond_dec, cond_inc)

            if cond_dec:
                # 最後の1つ分を減らす
                num = sum(orders) // env.trans_cap
                while sum(orders) > num * env.trans_cap:
                    # print('dec', flgs)
                    # 調整後の在庫量が最もSに近い商品を選択
                    idx = abs(reorder - (inv_p + vals)).argmin()
                    # 調整可能な商品がなくなれば終了
                    if flgs[idx] == -1:
                        break
                    # 発注量が既に0 もしくは 調整後inv_pがmust_orderを下回っている場合は対象から外す
                    if orders[idx] == 0 or (inv_p[idx] + vals[idx]) < mustorder[idx]:
                        flgs[idx] = -1
                        continue
                    orders[idx] = vals[idx]
                    vals[idx] -= lots[idx]
                    action[idx] -= 1

                # print('after (DEC)', sum(orders))
                return action
            if cond_inc:
                num = (sum(orders) // env.trans_cap) + 1
                while sum(orders) < num * env.trans_cap:
                    # print('inc', flgs, abs(reorder - (inv_p + vals_inc)))
                    # 調整後の在庫量が最もSに近い商品を選択
                    idx = abs(reorder - (inv_p + vals_inc)).argmin()
                    # 調整可能な商品がなくなれば終了
                    if flgs[idx] == -1:
                        break
                    # inv_pがtriggerを上回っている場合は対象から外す
                    if inv_p[idx] > trigger[idx]:
                        flgs[idx] = -1
                        continue
                    orders[idx] = vals_inc[idx]
                    vals_inc[idx] += lots[idx]
                    action[idx] += 1
                    cnt += 1

                # print('after (INC)', sum(orders))
                return action

        return action

    def sim_scs_policy(self, params, add_d=False, capacitated=False, draw_result=False, write=False):
        ''' (S, c, s)方策をシミュレートする'''
        self.bench_params = params
        self.env_scs = copy.deepcopy(self.env)

        if add_d:
            params = params.reshape(self.num_products, 4)
            for i, node in enumerate(self.env_scs.nodes):
                node.set_scds_params(params[i])
        else:
            params = params.reshape(self.num_products, 3)
            for i, node in enumerate(self.env_scs.nodes):
                node.set_scs_params(params[i])

        self.env_scs.update_demand()
        self.env_scs.create_result_memory()  # 結果格納用のResultMemoryを作る
        episode_reward = 0

        for step in range(self.num_time):
            action = []

            # sを下回る商品が一つでもあれば必ず発注する
            flg = False

            # (s, c, d, s)方策で使用する集合
            set_below_c = []
            set_above_c = []
            numerator = []
            denominator = []

            for i, node in enumerate(self.env_scs.nodes):
                for ret in node.iRet:
                    if USE_LT_INV:
                        if ret.inv_lt[step] <= ret.scs_params['s']:
                            flg = True
                        if ret.inv_lt[step] <= ret.scs_params['c']:
                            set_below_c.append(i)
                        else:
                            set_above_c.append(i)
                            numerator.append(ret.scs_params['S']-ret.inv_lt[step])
                            denominator.append(ret.scs_params['S']-ret.scs_params['c'])
                    else:
                        if ret.inv_position[step] <= ret.scs_params['s']:
                             flg = True
                        if ret.inv_position[step] <= ret.scs_params['c']:
                            set_below_c.append(i)
                        else:
                            set_above_c.append(i)
                            numerator.append(ret.scs_params['S']-ret.inv_position[step])
                            denominator.append(ret.scs_params['S']-ret.scs_params['c'])

            # actionの決定
            for i, node in enumerate(self.env_scs.nodes):
                for ret in node.iRet:
                    action.extend(node.simulate_scs_policy(step, flg, add_d, numerator, denominator))
            if capacitated:
                action = self.modify_heuristic_capacitated(self.env_scs, action, step)

            action_converted = np.array(action).reshape(self.num_products, self.num_ret)

            done, reward, _ = self.env_scs.step(action_converted, step)
            if step > IGNORE_STEP:
                episode_reward += reward

        if draw_result:
            self.logger.info(f'EpisodeReward: {episode_reward}')
            self.env_scs.calc_episode_performance()
            self.env_scs.episode_performance_print(self.logger)

        if write:
            model = '(s, c, d, S)' if add_d else '(s, c, S)'
            self.output_result(model, episode_reward.item(), capacitated)

        return episode_reward

    def output_result(self, model, episode_reward, capacitated=False):
        self.ga_outfile = f'data/ga_result_{self.agent_initialized_time}_{self.scenario}.csv'

        file = self.ga_outfile
        if not os.path.isfile(file):
            with open(file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'scenario', 'var', 'ro', 'model', 'n_generations', 'parameters','num_samples', 'inv_cap', 'capacitated', 'param_new', 'episode_reward'])

        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.agent_initialized_time, self.scenario, self.env.s_m, self.env.ro, model, NGEN, self.bench_params, NUM_SAMPLE_GA,  self.env.inv_cap, capacitated, True, episode_reward])

    def reset(self, num_states):
        return np.zeros(num_states)

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def finish_episode(self, gamma, optimizer):
        ''' Actir-Critic '''
        eps = np.finfo(np.float32).eps.item()
        
        R = 0
        saved_actions = self.policy.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]

    def learn_policy_gradient(self,pred_fresh):
        def discount_rewards(rewards, gamma=0.99):
            r = np.array([gamma**i * rewards[i] 
                          for i in range(len(rewards))])
            # Reverse the array direction for cumsum and then
            # revert back to the original order
            r = r[::-1].cumsum()[::-1]
            return r - r.mean()
        
        self.num_states_per_product = 3 # state を変更したらこれも変更する
        self.num_states = self.num_states_per_product * self.num_products * self.num_ret
        self.num_actions_per_product = 3
        self.num_actions =  self.num_actions_per_product ** (self.num_products * self.num_ret)#商品数の数だけactionスペースは増える
    
        # self.pe = ActorPolicy(self.num_states, self.num_actions)
        self.policy = ActorCriticPolicy(self.num_states, self.num_actions)
        optimizer = optim.Adam(self.policy.parameters(), lr=3e-2)

        # logger
        log_str = "_AC_Nproducts{}_Nret{}_Nstates{}_Nactions{}".format(self.num_products, self.num_ret,self.num_states, self.num_actions)
        self._set_logger(log_str)
        self.logger.info('')

        '''シミュレーションの実行 '''
        for episode in range(NUM_EPISODES):  # 最大試行数分繰り返す
                    
            episode_reward = 0
            self.env_sim = copy.deepcopy(self.env) #RetailerとWHインスタンスの初期状態をコピーする
            self.env_sim.create_result_memory() #結果格納用のResultMemoryを作る
            if pred_fresh:
                self.env_sim.reflesh_pred()

            # 初期値を得る
            observation = self.reset(self.num_states)            
            state = observation  # 観測をそのまま状態sとして使用
            state = np.array(state)
            e_state = state

            for step in range(MAX_STEPS):  # 1エピソードのループ
                # print(state)
                action_origin = self.select_action(state)
                # action_probs = self.pe.predict(state).detach().numpy()
                # action_origin = np.random.choice(self.num_actions, p=action_probs)
                action = self.convert_action(action_origin, True) # Retailerレベルでの発注量を計算する                
                done, reward = self.env_sim.step(action, step)
                self.policy.rewards.append(-reward)

                if done : #最後
                    state_next = None
                    break;
                else:
                    state_next, e_state_next = self.get_next_state(step,self.env_sim.nodes, True)

                if step > IGNORE_STEP:
                    episode_reward += reward

                # 観測の更新
                state, e_state = state_next,e_state_next
                state = np.array(state)
                e_state = np.array(e_state)

            self.finish_episode(GAMMA, optimizer)
            
            if episode % 50 ==0:
                # self.logger.info('(%d) EpisodeReward: %d Loss: %d' % (episode, episode_reward, (np.array(self.brain.td_error_memory.memory)**2).mean()))
                self.logger.info('(%d) EpisodeReward: %d' % (episode, episode_reward))
                self.env_sim.calc_episode_performance()
                self.env_sim.episode_performance_print(self.logger)
            
    def learn_initialize(self, option_nn):
        # logger
        print('in initialize:', self.state_real)

        version = f"{self.scenario}_OptionNN{option_nn}_InvCap{self.env.inv_cap}_dFac{self.env.d_factor}_s_m{self.env.s_m}_ro{self.env.ro}"
        version += f"_Nproducts{self.num_products}_Nactions{self.num_actions_per_product}_Season{self.season}"
        version += f"_TransAlloc{self.op_trans_alloc}_CTrans{self.Ctrans}"

        self.version = version
        self._set_logger()
        self.logger.info(version)

        # header of result csv
        self.write_to_csv([
            'episode', 'episode_reward', 'loss','c_inv','c_pel','c_trans',
            'c_inv','c_pel','c_trans','trans_rate','ord_size'])
        self.write_to_csv_val([
            'episode', 'episode_reward', 'c_inv','c_pel','c_trans',
            'c_inv','c_pel','c_trans','trans_rate','ord_size'])

    def learn_multi(self, load_weights, option_nn, pred_fresh, imitation_learn, has_wh_const):
        '''Q-Learningのための初期設定 '''
        self.learn_initialize(option_nn)
        self.logger.info('')
        print_setting_learn(self.logger)

        self.option_nn = option_nn

        self.num_states = self.num_states_per_product

        if ESTIMATE_OTHERS:
            self.num_states += 1
        if self.env.inv_cap > 0 and ADD_INV_STATE:
            self.num_states += 1

        # agentごとに固定して学習を進める
        if USE_INDEPENDENT_LEARN:
            LEARNING_FREQ = self.num_products
        else:
            LEARNING_FREQ = 1

        # Benchmarkの計算
        # self.logger.info('Start Benchmarking...')
        # self.sim_s_q_policy(True, False, False, 0, 'container')
        # self.sim_s_q_policy(True, False, False, 0, 'palette')
        # self.logger.info('')

        # NNインスタンスの作成
        self.brains = [IndependentBrain(self.num_states_per_product, self.num_products, self.num_ret,
                                    self.num_actions_per_product, option_nn, self.num_states) for _ in
                      range(self.num_products)]
        self.logger.info(self.brains[0].main_q_network)

        '''シミュレーションの実行 '''
        self.best_episode_reward = 1_000_000
        self.best_episode_reward_valid = 1_000_000

        for episode in range(NUM_EPISODES):  # 最大試行数分繰り返す
            time_for_replay = 0
            time_for_get_next = 0
            st = time.time()

            # 課題：validationのみでの欠品大量発生（all 0 が選択されてしまう）
            # learning時も欠品が大量に発生するような状態を作り出す
            episode_zero = False
            episode_zero_freq = 1

            if 'step' in self.scenario:
                freq = 20 if episode < 1500 else 80
                if episode % freq == 5:
                    episode_zero = True
                    episode_zero_freq = random.choice([3, 5, 10])

            if EPSILON_CONSTANT:
                epsilon = START_EPSILON
            else:
                epsilon = START_EPSILON * (1 / (episode // DECAY_FREQ + 1)) + FINAL_EPSILON

            # validate every N episodes
            if episode % FREQ_VALIDATION == 0:
                [self.validate_multi(episode) for _ in range(NUM_VALIDATION)]

            if CLEAR_MEMORY:
                if episode % TARGET_UPDATE_FREQUENCY == 0:
                    print(f'({episode}) Before clear: {len(self.brains[0].memory)}')
                    for brain in self.brains:
                        brain.memory.clear_memory()
                        brain.td_error_memory.clear_memory()

            episode_reward = 0
            self.env_sim = copy.deepcopy(self.env)  # RetailerとWHインスタンスの初期状態をコピーする
            self.env_sim.update_demand()
            self.env_sim.create_result_memory()  # 結果格納用のResultMemoryを作る
            if pred_fresh:
                self.env_sim.reflesh_pred()

            # 初期値を得る
            observation = self.reset(self.num_states)
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0).to(device)
            states = [state] * self.num_products

            for step in range(MAX_STEPS):  # 1エピソードのループ
                random_joint_trial = 0
                # 全て同時に0であるケースも意図的に学習に含める必要がある
                val = np.random.uniform(0, 1)
                if epsilon > val:
                    # 0.1 > 0.05
                    random_joint_trial = 1
                elif 1.5 * epsilon > val:
                    # 0.15 > 0.12
                    random_joint_trial = 2

                if episode_zero:
                    # しばらく0発注を続け，そうではないタイミングではrandomにjoint-selectionを決める
                    if (step % episode_zero_freq) != 0:
                        random_joint_trial = 2
                    else:
                        random_joint_trial = 1

                action_origin = [brain.decide_action(state, episode, joint_random=random_joint_trial)[0]
                                 for state, brain in zip(states, self.brains)]
                action = np.array(action_origin).reshape(self.num_products, self.num_ret)

                if ESTIMATE_OTHERS:
                    # replay memoryには，確定した発注量を格納する
                    orders = [action_origin * node.iRet[0].lot for action_origin, node in
                              zip(action_origin, self.env.nodes)]
                    for i in range(self.num_products):
                        states[i][0, self.num_states - 1] = sum(orders) - orders[i]

                done, reward, reward_p = self.env_sim.step(action, step)

                if done:  # 最後
                    states_next = [None] * self.num_products
                else:
                    temp = time.time()
                    states_next, _ = self.get_next_state_multi(step, self.env_sim.nodes)
                    time_for_get_next += time.time() - temp

                if step > IGNORE_STEP:
                    episode_reward += reward

                    # メモリに経験を追加
                    for idx in range(self.num_products):
                        if (not USE_INDEPENDENT_LEARN) or (episode // TARGET_UPDATE_FREQUENCY) % LEARNING_FREQ == idx:
                            self.memorize_multi(idx, episode, step, states[idx],
                                            action_origin[idx].to(device),
                                            states_next[idx], -reward_p[idx].view(-1).to(device))

                    # 全てのエージェントのQ関数を更新する
                    # multi-agentのため個別に学習を進める
                    if step % FREQ_LEARN == 0:
                        st_replay = time.time()
                        if ENABLE_MULTI:
                            # print(f'Parallelism on {cpu_count()} CPU')
                            Parallel(n_jobs=-1)([delayed(brain.replay)(episode) for brain in self.brains])
                        else:
                            for idx, brain in enumerate(self.brains):
                                brain.replay(episode)
                        time_for_replay += time.time() - st_replay

                # 観測の更新
                states = copy.deepcopy(states_next)

                # 終了時の処理
                if done:
                    if episode % 50 == 0:
                        # TD誤差メモリの中身を更新する
                        # self.update_td_error_memory()
                        pass

                    # DDQNで追加、2試行に1度、Target Q-NetworkをMainと同じにコピーする
                    if episode % TARGET_UPDATE_FREQUENCY == 0:
                        for brain in self.brains:
                            brain.update_target_q_network()
                    break

            for brain in self.brains:
                brain.scheduler.step()

            # loss = (np.array(self.brain.td_error_memory.memory)**2).mean()
            loss = 0
            # file出力
            self.env_sim.calc_episode_performance()
            wi, ws, qi, qs, ic, pc, tc, trate, qo = self.env_sim.get_episode_performance()
            self.write_to_csv([episode, episode_reward.item(), loss,
                               np.array(ic).sum(), np.array(pc).sum(), np.array(tc).sum(),
                               ic, pc, tc, 0, list(np.round(np.array(qo), 2))])
            # list(np.round(np.array(trate), 2))

            # Best Episode更新
            if self.best_episode_reward > episode_reward.item():
                self.best_episode_reward = episode_reward.item()
                self.env_sim_best = copy.deepcopy(self.env_sim)
                with open(self.scm_best_path, mode='wb') as f:
                    pickle.dump(self.env_sim_best, f)

            # logger出力
            if episode % 5 == 0:
                # self.logger.info('(%d) EpisodeReward: %d Loss: %d MemUsed: %d Best: %d' % (
                self.logger.info('(%d) EpisodeReward: %d Epsilon: %f MemoryUsed:%d Best: %d' % (
                    episode, episode_reward.item(),
                    epsilon,
                    # loss,
                    psutil.virtual_memory().percent,
                    self.best_episode_reward))

            del self.env_sim
            gc.collect()

            if DEBUG_PRINT:
                print(f'Learning time.. Total : {time.time() - st:.3f}sec, '
                      f'Replay: {time_for_replay:.3f}sec, '
                      f'get next:{time_for_get_next:.3f}sec')

    def get_next_state_multi(self, step, nodes, is_validate=False):
        step = step + 1
        states = []

        # 直近の在庫合計量
        tot_inv = 0
        tot_inv_p = 0
        for node in nodes:
            for i, Ret in enumerate(node.iRet):
                tot_inv += Ret.inv[step]
                tot_inv_p += Ret.inv_position[step]

        for node in nodes:
            state = []
            for i, Ret in enumerate(node.iRet):
                # 1. 直近在庫
                state.append(Ret.inv[step])
                if USE_FORECAST:
                    # 2. LT後までの入荷予定
                    state.append(Ret.repl_plan[step: step + Ret.lt].sum())
                    # 3. LT後在庫（予測）
                    state.append(Ret.inv_lt[step])
                    # 4. LT後までの累積需要予測
                    pred_cum = Ret.pred[step + 1: step + Ret.lt + 1].sum()
                    state.append(pred_cum)
                    # 5. LT後から4週の累積需要予測
                    pred_cum_2 = Ret.pred[step + Ret.lt:step + Ret.lt + 5].sum()
                    state.append(pred_cum_2)
                else:
                    # 在庫ポジション
                    state.append(Ret.inv_position[step])

                # 全商品の直近合計在庫量を含める
                if self.env.inv_cap >0 and ADD_INV_STATE:
                    state.append(tot_inv)
                    # state.append(tot_inv_p)

                if ESTIMATE_OTHERS:
                    # 他商品の合計発注量
                    state.append(0)

                state = torch.from_numpy(np.array(state)).type(torch.FloatTensor)
                state = torch.unsqueeze(state, 0).to(device)
                states.append(state)

        # 次週の自分以外の商品の発注量を予測する（実質ここで行動を決定している）
        st = time.time()
        best_action = None
        if ESTIMATE_OTHERS:
            # Qの取得
            Qs_all = self.get_all_qs(states, is_validate)
            ed_qs = time.time()
            # 近似解
            best_action, best_orders, best_q = self.decide_approx_action_multi(states, step, nodes, Qs_all, is_validate)
            ed_approx = time.time()

            # 全探索
            if SELECT_OPTIMAL:
                best_action_exact, best_orders_exact, best_q_exact = self.decide_optimal_action_multi(Qs_all)
                if DEBUG_PRINT:
                    print(f'diff in best : {best_q_exact - best_q}, {best_action}, {best_action_exact}')
                    print(f'forward:{ed_qs-st:.2f}sec, approx selection:{ed_approx-ed_qs:.2f}sec, optimal selection:{time.time()-st: .2f}sec')
                best_orders = best_orders_exact
                best_action = best_action_exact
            if (step % 100 == 50) and DEBUG_PRINT:
                print(
                    f'forward:{ed_qs - st:.4f}sec, approx selection:{ed_approx - ed_qs:.4f}sec')

            for i in range(self.num_products):
                states[i][0, self.num_states - 1] = sum(best_orders) - best_orders[i]

        return states, best_action

    def get_all_qs(self, states, is_validate):
        # まずは全ての組み合わせに対して，先にNNからforward処理しておく
        max_order = sum([(self.num_actions_per_product-1) * node.iRet[0].lot for idx, node in enumerate(self.env.nodes)])
        states_all = [[copy.deepcopy(states[j].detach()) for _ in range(max_order)] for j in range(self.num_products)]
        for j in range(self.num_products):
            for i in range(max_order):
                states_all[j][i][0, self.num_states - 1] = i

        # forward処理
        Qs_all = []
        if ENABLE_MULTI_Q and is_validate:
            set_start_method('spawn', force=True)
            p = Pool(cpu_count())
            nets = [brain.main_q_network for brain in self.brains]

            Qs_all = p.map(unwrap_forward,
                           [(net, torch.cat(state).to(device)) for net, state in zip(
                    nets, states_all)])
            p.close()
        else:
            for j in range(self.num_products):
                # 行動選択は常にmain-networkを参照するように修正
                self.brains[j].main_q_network.eval()
                Qs = self.brains[j].main_q_network(torch.cat(states_all[j]).to(device))
                '''
                if is_validate:
                    self.brains[j].main_q_network.eval()
                    Qs = self.brains[j].main_q_network(torch.cat(states_all[j]).to(device))
                else:
                    self.brains[j].target_q_network.eval()
                    Qs = self.brains[j].target_q_network(torch.cat(states_all[j]).to(device))
                '''
                Qs_all.append(Qs)

        return Qs_all

    def decide_approx_action_multi(self, states, step, nodes, Qs_all, is_validate):
        def calc_total_q(orders):
            if Qs_all is None:
                q_this = 0
                for j in range(self.num_products):
                    states[j][0, self.num_states - 1] = sum(orders) - orders[j]
                    q_this += self.brains[j].decide_action(state=states[j], episode=0,
                                                           greedy=True, target=(not is_validate))[1]
            else:
                q_this = sum([Qs_all[j][sum(orders) - orders[j], actions[j]].item() for j in range(self.num_products)])
            return q_this

        def search_better_action(states, orders, sequential=False, start=0):
            ''' 現在のorderに対し，自分の行動を変化させて良い方向を探索する '''
            actions = [0] * self.num_products
            search_order = list(range(self.num_products))[start:] + list(range(self.num_products))[:start]

            for j in search_order:
                num = math.ceil(sum(orders) - orders[j])
                states[j][0, self.num_states - 1] = num
                actions[j] = Qs_all[j][num].argmax().item()
                if sequential:
                    orders[j] = actions[j] * self.env.nodes[j].iRet[0].lot
            # ordersの更新
            orders = [actions[i] * self.env.nodes[i].iRet[0].lot for i in range(self.num_products)]

            return actions, orders

        # 発注量0が初期状態
        orders = [0] * self.num_products
        actions = [0] * self.num_products
        best_q = calc_total_q(orders)
        best_action, best_orders, best_k, best_seq = actions, orders, -1, False

        # qty allocの場合は最初は初期値は平均的な発注量とする
        if self.op_trans_alloc == 'qty':
            for idx, node in enumerate(nodes):
                orders[idx] = node.iRet[0].pred[step]

        # ordersを初期解として探索する
        seq_improve_cnt = 0
        for k in range(LEVEL_K):
            actions, orders = search_better_action(states, orders, sequential=False)
            q_this = calc_total_q(orders)
            if q_this > best_q:
                best_q, best_action, best_orders, best_k, best_seq = q_this, actions, orders, k, False

            # sequentialに探索をして，良い点が見つければそこを次のスタート点とする
            for i in range(self.num_products):
                actions_seq, orders_seq = search_better_action(states, orders, sequential=True, start=i)
                q_this_seq = calc_total_q(orders_seq)
                # sequentialで改善した場合はaction, ordersを更新する
                if q_this_seq > q_this:
                    actions, orders, q_this = actions_seq, orders_seq, q_this_seq
                    seq_improve_cnt += 1

                if q_this > best_q:
                    best_q, best_action, best_orders, best_k, best_seq = q_this, actions, orders, k, True

        if (step % 100 == 50) and DEBUG_PRINT and best_k >= 0:
            print(f'best K: {best_k}, sequence: {best_seq}, action:{best_action}, seq_improve_cnt:{seq_improve_cnt}')

        return best_action, best_orders, best_q

    def decide_optimal_action_multi(self, Qs_all):
        ''' 全探索 '''
        candidates = self.num_actions_per_product ** self.num_products

        best_q = - float('inf')

        for option in range(candidates):
            s = str(np.base_repr(option, self.num_actions_per_product))
            s = '0' * (self.num_products - len(s)) + s
            action = [int(s) for s in list(s)]
            orders = [action * self.env.nodes[idx].iRet[0].lot for idx, action in enumerate(action)]

            tot = sum([Qs_all[j][sum(orders) - orders[j], action[j]].item() for j in range(self.num_products)])
            if tot > best_q:
                best_q = tot
                best_action = action
                best_orders = orders

        return best_action, best_orders, best_q

    def validate_multi(self, episode, draw=False):
        episode_reward = 0
        self.env_val = copy.deepcopy(self.env)  # RetailerとWHインスタンスの初期状態をコピーする
        self.env_val.update_demand()
        self.env_val.create_result_memory()  # 結果格納用のResultMemoryを作る

        observation = self.reset(self.num_states)
        state = observation  # 観測をそのまま状態sとして使用
        state = torch.from_numpy(state).type(torch.FloatTensor)  # NumPy変数をPyTorchのテンソルに変換
        state = torch.unsqueeze(state, 0).to(device)  # size 4をsize 1x4に変換
        states = [state] * self.num_products
        best_action = None

        for step in range(MAX_STEPS):  # 1エピソードのループ
            if best_action is None:
                action_origin = [brain.decide_action(state, episode)[0] for state, brain in zip(states, self.brains)]
            else:
                action_origin = best_action
            action = np.array(action_origin).reshape(self.num_products, self.num_ret)
            done, reward, reward_p = self.env_val.step(action, step)

            if done:  # 最後
                states = [None] * self.num_products
            else:
                states, best_action = self.get_next_state_multi(step, self.env_val.nodes, is_validate=True)
                # if DEBUG_PRINT:
                #     print(best_action)

            if step > IGNORE_STEP:
                episode_reward += reward

        self.env_val.calc_episode_performance()
        wi, ws, qi, qs, ic, pc, tc, trate, qo = self.env_val.get_episode_performance()
        self.write_to_csv_val([episode, episode_reward.item(),
                               np.array(ic).sum(), np.array(pc).sum(), np.array(tc).sum(),
                               ic, pc, tc, 0, list(np.round(np.array(qo), 2))])
        # list(np.round(np.array(trate), 2))

        # 記録を更新した場合は出力する（最終評価で使うわけではない）
        if self.best_episode_reward_valid > episode_reward.item():
            self.best_episode_reward_valid = episode_reward.item()
            with open(self.scm_val_path, mode='wb') as f:
                pickle.dump(self.env_val, f)

        if episode % 50 == 0:
            self.logger.info(F'EpisodeReward: {episode_reward.item():.0f} BEST : {self.best_episode_reward_valid:.0f}')
            self.env_val.episode_performance_print(self.logger)

        if draw:
            self.env_val.draw()
        del self.env_val

    def learn(self, load_weights,option_nn, pred_fresh, imitation_learn, has_wh_const):
        '''Q-Learningのための初期設定 '''
        if option_nn == 10:
            self.learn_multi(load_weights,option_nn, pred_fresh, imitation_learn, has_wh_const)
            return

        self.learn_initialize(option_nn)
        self.logger.info('')
        self.option_nn = option_nn

        ''' Brain の初期化'''
        if option_nn in [3,4,5,6,7]:#Action Branching
            self.brain = BranchedBrain(self.num_states_per_product, self.num_products, self.num_ret,
                                       self.num_actions_per_product, option_nn, self.num_states)
            self.logger.info(self.brain.main_q_network)
        elif option_nn ==10:
            self.brain = [BranchedBrain(self.num_states_per_product, self.num_products, self.num_ret,
                                       self.num_actions_per_product, option_nn, self.num_states) for _ in range(self.num_products)]
            self.logger.info(self.brain[0].main_q_network)
        else:
            self.brain = Brain(
                self.num_states, self.num_actions, self.num_states_per_product, self.num_actions_per_product, self.num_products, self.num_ret,option_nn)          
        if load_weights:
            self.load_weight(self.brain)

        '''シミュレーションの実行 '''
        self.best_episode_reward = 1_000_000
        self.best_episode_reward_valid = 1_000_000
        for episode in range(NUM_EPISODES):  # 最大試行数分繰り返す
            if episode % FREQ_VALIDATION == 0:
                [self.validate(episode, True, has_wh_const, False) for _ in range(NUM_VALIDATION)]

            # 全体学習モードと枝別学習モードを切り替える場合はメモリをクリアする
            if USE_INDEPENDENT_LEARN:
                if episode % (self.num_products*10) == 0:
                    print(f'({episode}) Before clear: {len(self.brain.memory)}')

                # 枝別の場合は学習対象枝ごとにメモリをクリアする
                if ((episode // (self.num_products*10)) % 2 == 1) and (episode % BRANCH_SWITCHING_FREQ == 0):
                    print(f'({episode}) Before clear: {len(self.brain.memory)}')

            st = time.time()
            time_learing = 0
            time_action=0
            time_step=0

            episode_reward = 0
            self.env_sim = copy.deepcopy(self.env) #RetailerとWHインスタンスの初期状態をコピーする
            self.env_sim.create_result_memory() #結果格納用のResultMemoryを作る
            if pred_fresh:
                self.env_sim.reflesh_pred()

            self.brain.lots = [node.iRet[0].lot for node in self.env_sim.nodes]

            # 初期値を得る
            observation = self.reset(self.num_states)
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0).to(device)
            if SAMPLING_LEARN:
                e_state = [state for i in range(10)]
            else:
                e_state = state

            for step in range(MAX_STEPS):  # 1エピソードのループ
                temp = time.time()
                # action_origin = self.brain.decide_action(e_state, episode, False, SAMPLING_LEARN) #beliefでactionを決める
                # 現時点でのWH在庫を上限として発注量を算出する
                action_origin = self.brain.decide_action(e_state, episode, False, SAMPLING_LEARN,
                                                         has_wh_const, np.trunc(self.env_sim.get_wh_inv(step)))
                if ESTIMATE_OTHERS:
                    # memoryに入れる際は，実際の発注量を格納する
                    palettes = self.brain.calc_palettes(action_origin[0])
                    state[0, self.num_states - 1] = palettes1

                if option_nn in [3, 4, 5, 6, 7]:  # Action Branching
                    action = np.array(action_origin).reshape(self.num_products,self.num_ret)
                else:
                    action = self.convert_action(action_origin, False) # Retailerレベルでの発注量を計算する
                time_action += time.time() - temp

                # actionを送ってその日の在庫受払計算を行う
                temp = time.time()
                done, reward, reward_p = self.env_sim.step(action, step)
                time_step += time.time() - temp

                # IGNORE_STEPまでは無視する（計算しないしMemoryにpushしない）
                if done : #最後
                    state_next = None
                else:
                    # ESTIMATE_OTHERSのときは合計予想発注量tとして0をいれる
                    state_next, e_state_next = self.get_next_state(step,self.env_sim.nodes, False, SAMPLING_LEARN)

                if step > IGNORE_STEP:
                    episode_reward += reward
                    # print(self.get_action_idx_from_list(action.flatten()))

                    # メモリに経験を追加
                    # rewardはコストであり,最小化問題にする
                    # メモリに入れるstateは真のstateとする
                    self.memorize(
                        episode,
                        step,
                        state,
                        action_origin.to(device),
                        state_next,
                        - reward.to(device),
                        - reward_p.to(device)
                    )

                    # TD誤差メモリにTD誤差を追加
                    ''' multi step learning対応で， Experienve Memoryへのmemorizeと同時に実施するよう修正 / sglab'''
                    # self.memorize_td_error(0)  # 本当はTD誤差を格納するが、0をいれておく

                    # Q関数を更新する
                    if step % FREQ_LEARN == 0:
                        temp = time.time()
                        self.update_q_function(episode)
                        time_learing += time.time() - temp

                    # print("({0}) 4) UPDATE Q FUNCTION: {1}".format(step, time.time() - start) + "[sec]")
                    # start = time.time()

                # 観測の更新
                state = state_next
                e_state = e_state_next

                # 終了時の処理
                if done:
                    if episode % 50 == 0:
                        # TD誤差メモリの中身を更新する
                        self.update_td_error_memory()

                    # DDQNで追加、2試行に1度、Target Q-NetworkをMainと同じにコピーする
                    if episode % TARGET_UPDATE_FREQUENCY == 0:
                        self.update_target_q_function()
                    break

            self.brain.scheduler.step()

            loss = 0
            self.env_sim.calc_episode_performance()

            # Best Episode更新
            if self.best_episode_reward > episode_reward.item():
                self.best_episode_reward = episode_reward.item()
                self.env_sim_best = copy.deepcopy(self.env_sim)
                with open(self.scm_best_path, mode='wb') as f:
                    pickle.dump(self.env_sim_best, f)

            # logger出力
            if episode % 10 ==0:
                for param_group in self.brain.optimizer.param_groups:
                    self.logger.info(f'lr:{param_group["lr"]}, epsilon:{START_EPSILON * (1 / (episode//100 + 1)) + FINAL_EPSILON}')
                self.logger.info('(%d) EpisodeReward: %d Loss: %d MemUsed: %d Best: %d' % (
                    episode, episode_reward.item(),loss, psutil.virtual_memory().percent,
                    self.best_episode_reward))

            # 学習済みパラメータを保存する
            if episode % 50 ==0:
                torch.save(self.brain.main_q_network.state_dict(), self.model_path)

            del self.env_sim
            gc.collect()

            # tot = time.time() - st
            # if episode < 5:
            #     print(f'{tot:.2}sec: {time_learing/tot:.0%}, {time_action/tot:.0%}, {time_step/tot:.0%}')
            #     print(f'{(episode)} target_branch:{(episode // BRANCH_SWITCHING_FREQ) % self.num_products}, current memory size:', len(self.brain.memory))

    def validate(self, episode, pred_fresh, has_wh_const, draw):
        episode_reward = 0
        self.env_val = copy.deepcopy(self.env)
        if pred_fresh:
            self.env_val.update_demand()
        self.env_val.create_result_memory()  # 結果格納用のResultMemoryを作る

        observation = self.reset(self.num_states)
        state = observation
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0).to(device)
        if self.sample_val:
            e_state = [state for i in range(NUM_SAMPLING)]
        else:
            e_state = state

        for step in range(MAX_STEPS):  # 1エピソードのループ
            action_origin = self.brain.decide_action(e_state, episode, True, self.sample_val,
                                                     has_wh_const, np.trunc(self.env_val.get_wh_inv(step)))
            if self.option_nn in [3, 4, 5, 6, 7]:  # Action Branching
                action = np.array(action_origin).reshape(self.num_products, self.num_ret)
            else:
                action = self.convert_action(action_origin, False)

            done, reward, reward_p = self.env_val.step(action, step)

            if done : #最後
                state_next = None
            else:
                # USE_ESTIMATE_OTHERSのときは合計予想発注量tとして0をいれる
                state_next, e_state_next = self.get_next_state(
                    step,self.env_val.nodes, False, self.sample_val, is_validate=True)

            if step > IGNORE_STEP:
                episode_reward += reward
            # 観測の更新
            state = state_next
            e_state = e_state_next

        self.env_val.calc_episode_performance()
        wi, ws, qi, qs, ic, pc, tc, trate, qo = self.env_val.get_episode_performance()
        self.write_to_csv_val([episode, episode_reward.item(),
                           np.array(ic).sum(), np.array(pc).sum(), np.array(tc).sum(),
                           ic, pc, tc, 0, list(np.round(np.array(qo), 2))])

        if self.best_episode_reward_valid > episode_reward.item():
            self.best_episode_reward_valid = episode_reward.item()

        if episode % 50 == 0:
            self.logger.info(F'EpisodeReward: {episode_reward.item():.0f} BEST : {self.best_episode_reward_valid:.0f}')
            self.env_val.episode_performance_print(self.logger)

        if draw:
            self.env_val.draw()
        with open(self.scm_val_path, mode='wb') as f:
            pickle.dump(self.env_val, f)

        del self.env_val

    def get_next_state(self, step, nodes, prob, sampling, is_validate=False):
        # print(self.state_real)
        step = step + 1

        if not self.state_real:
            state = []  # 真の状態:Experience Memoryに入れて学習させる
            e_state = []  # 期待される状態:Actionを決める際に用いる
            # training時はsamplingせずに期待値を用いる，inference時はsamplingする

            for node in nodes:
                for i, Ret in enumerate(node.iRet):
                    # 1. 直近在庫
                    state.append(Ret.inv[step])
                    state.append(Ret.inv_position[step])

                    # 2. LT後までの入荷予定
                    # state.append(Ret.repl_plan[step: step + Ret.lt].sum())
                    # 3. LT後在庫（予測）
                    # state.append(Ret.inv_lt[step])

                    # 4. LT後までの累積需要予測
                    # pred_cum = Ret.pred[step + 1: step + Ret.lt + 1].sum()
                    # demand_cum = Ret.demand[step + 1: step + Ret.lt + 1].sum()
                    # state.append(pred_cum)

                    # 5. LT後から4週の累積需要予測
                    # pred_cum_2 = Ret.pred[step + Ret.lt:step + Ret.lt + 5].sum()
                    # demand_cum_2 = Ret.demand[step + Ret.lt:step + Ret.lt + 5].sum()
                    # state.append(pred_cum_2)

                    if not self.obs_only:
                        # 6. LT後までの累積誤差(stateは実績の予測誤差）
                        state.append(pred_cum - demand_cum)
                        # 7. LT後からN週の累積誤差(stateは実績の予測誤差）
                        state.append(pred_cum_2 - demand_cum_2)

            # 予測誤差パラメータをもとにサンプリングする
            if sampling:
                for j in range(NUM_SAMPLING):
                    e_state_unit = []

                    for node in nodes:
                        for i, Ret in enumerate(node.iRet):
                            Ret.pred_temp = np.random.normal(loc=Ret.pred, scale=Ret.sd * Ret.error_rate, size=Ret.num_time + BUFFER)
                            Ret.pred_temp = np.where(Ret.pred_temp < 0, 0, Ret.pred_temp)
                            Ret.inv_lt_temp = Ret.inv[step] - Ret.pred_temp[step: step + Ret.lt + 1].sum() + Ret.repl_plan[step: step + Ret.lt].sum()

                    # 予測をupdateしたnodes_tempについて，e_stateをそれぞれ求める
                    for node in nodes:
                        for i, Ret in enumerate(node.iRet):
                            # 1. 直近在庫
                            e_state_unit.append(Ret.inv[step])
                            # 2. LT後までの入荷予定
                            e_state_unit.append(Ret.repl_plan[step: step + Ret.lt].sum())

                            if self.obs_only:
                                # 3. LT後在庫
                                e_state_unit.append(Ret.inv_lt_temp)

                                pred_cum = Ret.pred_temp[step + 1: step + Ret.lt + 1].sum()
                                pred_cum_2 = Ret.pred_temp[step + Ret.lt:step + Ret.lt + 5].sum()

                                # 4. LT後までの累積需要
                                e_state_unit.append(pred_cum)
                                # 5. LT後から4週の累積需要
                                e_state_unit.append(pred_cum_2)

                            else:
                                # 3. LT後在庫
                                e_state_unit.append(Ret.inv_lt[step])
                                pred_cum = Ret.pred[step + 1: step + Ret.lt + 1].sum()
                                pred_cum_2 = Ret.pred[step + Ret.lt:step + Ret.lt + 5].sum()
                                # 4. LT後までの累積需要
                                e_state_unit.append(pred_cum)
                                # 5. LT後から4週の累積需要
                                e_state_unit.append(pred_cum_2)
                                # 6. LT後までの累積誤差（誤差の予測：inference時は分布で与える）
                                e_state_unit.append(pred_cum * np.random.normal(loc=0, scale= (Ret.sd_ini / Ret.myu) * self.error_rate, size=1)[0] / np.sqrt(Ret.lt))
                                # 7. LT後からN週の累積誤差（誤差の予測：inference時は分布で与える）
                                e_state_unit.append(pred_cum_2 * np.random.normal(loc=0, scale=(Ret.sd_ini / Ret.myu) * self.error_rate, size=1)[0] / np.sqrt(4))

                    e_state_unit = torch.from_numpy(np.array(e_state_unit)).type(torch.FloatTensor)
                    e_state_unit = torch.unsqueeze(e_state_unit, 0).to(device)
                    e_state.append(e_state_unit)
                    # print(e_state)

            else:
                for node in nodes:
                    for i, Ret in enumerate(node.iRet):
                        # 1. 直近在庫
                        e_state.append(Ret.inv[step])
                        e_state.append(Ret.inv_position[step])
                        # 2. LT後までの入荷予定
                        # e_state.append(Ret.repl_plan[step: step + Ret.lt].sum())

                        # 3. LT後在庫
                        # e_state.append(Ret.inv_lt[step])

                        # pred_cum = Ret.pred[step + 1: step + Ret.lt + 1].sum()
                        # pred_cum_2 = Ret.pred[step + Ret.lt:step + Ret.lt + 5].sum()

                        # 4. LT後までの累積需要(stateは実績)
                        # e_state.append(pred_cum)
                        # 5. LT後から4週の累積需要(stateは実績)
                        # e_state.append(pred_cum_2)

                        if not self.obs_only:
                            # 6. LT後までの累積誤差(行動選択時には予測は合っている前提で行動を選択する）
                            e_state.append(0)
                            # 7. LT後からN週の累積誤差(行動選択時には予測は合っている前提で行動を選択する）
                            e_state.append(0)

                if ESTIMATE_OTHERS:
                    e_state.append(0)
                e_state = torch.from_numpy(np.array(e_state)).type(torch.FloatTensor)
                e_state = torch.unsqueeze(e_state, 0).to(device)

            if ESTIMATE_OTHERS:
                state.append(0)
            state = torch.from_numpy(np.array(state)).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0).to(device)

            # 学習する際には相手の予測はせずに，全員発注しない場合として作る
            if ESTIMATE_OTHERS and is_validate:
                palettes = self.brain.estimate_palettes(state)
                state[0, self.num_states-1] = palettes
                e_state[0, self.num_states-1] = palettes

            return state, e_state

        else:
            # !! POMDP対応
            # e_stateについては，予測誤差をパラメータとしてサンプリングする
            # 真のstate
            state = [] # 真の状態:Experience Memoryに入れて学習させる
            for node in nodes:
                for i, Ret in enumerate(node.iRet):
                    # 1. 直近在庫
                    state.append(Ret.inv[step])
                    # 2. 入荷予定
                    state.append(Ret.repl_plan[step: step + Ret.lt].sum())
                    # 3. LT後在庫
                    state.append(Ret.inv_lt_true[step])
                    # 4. LT後までの累積需要(stateは実績)
                    demand_cum = Ret.demand[step + 1: step + Ret.lt + 1].sum()
                    state.append(demand_cum)
                    # 5. LT後から4週の累積需要(stateは実績)
                    demand_cum_2 = Ret.demand[step + Ret.lt:step + Ret.lt + 5].sum()
                    state.append(demand_cum_2)

            # belief state : b(s)
            e_state = [] # 期待される状態:Actionを決める際に用いる

            # 予測誤差パラメータをもとにサンプリングする
            if sampling:
                for j in range(NUM_SAMPLING):
                    e_state_unit = []

                    # 現在の予測値を予測の期待値として，上振れ下振れシナリオでシミュレーションする
                    # そのために，現在の予測値に対して想定される予測誤差をパラメータとしてブレさせる
                    for node in nodes:
                        for i, Ret in enumerate(node.iRet):
                            Ret.pred_temp = np.random.normal(loc=Ret.pred, scale=Ret.sd * Ret.error_rate, size=Ret.num_time + BUFFER)
                            Ret.pred_temp = np.where(Ret.pred_temp < 0, 0, Ret.pred_temp)
                            Ret.inv_lt_temp = Ret.inv[step] - Ret.pred_temp[step: step + Ret.lt + 1].sum() + Ret.repl_plan[step: step + Ret.lt].sum()

                    # print("({0}) 2) reflesh {1}".format(step, time.time() - start) + "[sec]")
                    # start = time.time()

                    # 予測をupdateしたnodes_tempについて，e_stateをそれぞれ求める
                    for node in nodes:
                        for i, Ret in enumerate(node.iRet):
                            # 1. 直近在庫
                            e_state_unit.append(Ret.inv[step])
                            # 2. 入荷予定
                            e_state_unit.append(Ret.repl_plan[step: step + Ret.lt].sum())
                            # 3. LT後在庫
                            e_state_unit.append(Ret.inv_lt_temp)
                            # 4. LT後までの累積需要
                            pred_cum = Ret.pred_temp[step + 1: step + Ret.lt + 1].sum()
                            e_state_unit.append(pred_cum)
                            # 5. LT後から4週の累積需要
                            pred_cum_2 = Ret.pred_temp[step+Ret.lt:step+Ret.lt+5].sum()
                            e_state_unit.append(pred_cum_2)
                    e_state_unit = torch.from_numpy(np.array(e_state_unit)).type(torch.FloatTensor)
                    e_state_unit = torch.unsqueeze(e_state_unit, 0).to(device)
                    e_state.append(e_state_unit)

                    # print("({0}) 3) e_state append {1}".format(step, time.time() - start) + "[sec]")
                    # start = time.time()

            else:
                for node in nodes:
                    for i, Ret in enumerate(node.iRet):
                        # 1. 直近在庫
                        e_state.append(Ret.inv[step])
                        # 2. 入荷予定
                        e_state.append(Ret.repl_plan[step: step + Ret.lt].sum())
                        # 3. LT後在庫
                        e_state.append(Ret.inv_lt[step])
                        # 4. LT後までの累積需要
                        pred_cum = Ret.pred[step + 1: step + Ret.lt + 1].sum()
                        e_state.append(pred_cum)
                        # 5. LT後から4週の累積需要
                        pred_cum_2 = Ret.pred[step + Ret.lt:step + Ret.lt + 5].sum()
                        e_state.append(pred_cum_2)

                e_state = torch.from_numpy(np.array(e_state)).type(torch.FloatTensor)
                e_state = torch.unsqueeze(e_state, 0).to(device)

            state = torch.from_numpy(np.array(state)).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0).to(device)

            return state, e_state
                    
    def convert_action(self, action, prob):
        '''各stepにおいて，商品，拠点別の発注量を計算する '''
        # actionはnumpyでindexをproduct_idとして取得できる
        # action = self.brain.decide_action(state, episode)
        if self.option_nn == 0:
            # Branching Architectureではない場合
            is_per_product= False
        else:
            # Branching Architectureの場合
            is_per_product = True
        if prob:
            action = self.get_action_list_from_idx(action)
        else:
            action = self.get_action_list_from_idx(action.item(), is_per_product)
        return np.array(action).reshape(self.num_products,self.num_ret)
    
    def get_action_list_from_idx(self, action_idx, is_per_product):
        def Base_10_to_n(X, n):
            if (int(X/n)):
                return Base_10_to_n(int(X/n), n)+str(X%n)
            return str(X%n)
        # action.item()をaction_idxとして渡す
        # 124のようなaction_indexを渡して， [4,4,4]のようなリストを得る
        action_list = Base_10_to_n(action_idx, self.num_actions_per_product)
        ret = [int(a)  for a in action_list]
        if is_per_product:
            # productごとにbranchを分岐する場合
            while len(ret) < self.num_ret:
                ret.insert(0, 0)
        else:
            # branchを分岐しない場合（全てのproduct/retailerについて一つの大きなNNで表現する場合
            while len(ret) < self.num_ret * self.num_products:
                ret.insert(0, 0)
        return ret

    def get_action_idx_from_list(self, action_list):
        def Base_n_to_10(X,n):
            out = 0
            for i in range(1,len(str(X))+1):
                out += int(X[-i])*(n**(i-1))
            return out #int out
    # [4,4,4]のようなactionを渡して， 124のようなactionインデックスを得る
        s = ""
        for i in action_list:
            s = s + str(i)
        return Base_n_to_10(s, self.num_actions_per_product)

class Logger():
    ''' Agent ごとにloggerを生成する　'''
    def __init__(self, log_dir="", dir_name=""):
        self.log_dir = log_dir
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if dir_name:
            self.log_dir = os.path.join(self.log_dir, dir_name)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

        self._callback = tf.summary.FileWriter(self.log_dir)
        # K.callbacks.TensorBoard(self.log_dir)

    @property
    def writer(self):
        return self._callback#.writer

    def set_model(self, model):
        self._callback.set_model(model)

    def path_of(self, file_name):
        return os.path.join(self.log_dir, file_name)

    def describe(self, name, values, episode=-1, step=-1):
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = "{} is {} (+/-{})".format(name, mean, std)
        if episode > 0:
            print("At episode {}, {}".format(episode, desc))
        elif step > 0:
            print("At step {}, {}".format(step, desc))

    def plot(self, name, values, interval=10):
        indices = list(range(0, len(values), interval))
        means = []
        stds = []
        for i in indices:
            _values = values[i:(i + interval)]
            means.append(np.mean(_values))
            stds.append(np.std(_values))
        means = np.array(means)
        stds = np.array(stds)
        plt.figure()
        plt.title("{} History".format(name))
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds,
                         alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g",
                 label="{} per {} episode".format(name.lower(), interval))
        plt.legend(loc="best")
        plt.show()

    def write(self, index, name, value):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.tag = name
        summary_value.simple_value = value
        self.writer.add_summary(summary, index)
        self.writer.flush()

    def write_image(self, index, frames):
        # Deal with a 'frames' as a list of sequential gray scaled image.
        last_frames = [f[:, :, -1] for f in frames]
        if np.min(last_frames[-1]) < 0:
            scale = 127 / np.abs(last_frames[-1]).max()
            offset = 128
        else:
            scale = 255 / np.max(last_frames[-1])
            offset = 0
        channel = 1  # gray scale
        tag = "frames_at_training_{}".format(index)
        values = []

        for f in last_frames:
            height, width = f.shape
            array = np.asarray(f * scale + offset, dtype=np.uint8)
            image = Image.fromarray(array)
            output = io.BytesIO()
            image.save(output, format="PNG")
            image_string = output.getvalue()
            output.close()
            image = tf.Summary.Image(
                        height=height, width=width, colorspace=channel,
                        encoded_image_string=image_string)
            value = tf.Summary.Value(tag=tag, image=image)
            values.append(value)

        summary = tf.Summary(value=values)
        self.writer.add_summary(summary, index)
        self.writer.flush()

class Environment():
    # 1つの商品についての{WH, Retailer}の集合であるNodeを統合管理するクラス
    # env.node[p]で，商品pに対するRetailer, Warehouseが取得できる

    def __init__(self, k, num_time, num_products, myu, lt_retailer, lt_wh, lot_retailer, lot_wh, num_ret, inv_cap, trans_cap,
                 d_factor, season, s_m, ro, error_rate, op_trans_alloc, draw):
        self.num_time = num_time
        self.num_products = num_products        
        self.num_ret = num_ret
        self.season = season
        self.op_trans_alloc = op_trans_alloc
        self.inv_cap = inv_cap
        self.trans_cap = trans_cap

        self.myu = myu
        self.s_m = s_m
        self.ro = ro
        self.error_rate = error_rate
        self.d_factor = d_factor

        # 商品ごとにNodeを作成し，Nodeの中に拠点を配置し，　それぞれ需要やコスト原単価を設定する
        self.nodes = [Node(i, k, num_time, lt_retailer[i], lt_wh[i], lot_retailer[i], lot_wh[i], inv_cap) for i in range(num_products)]
        for i, node in enumerate(self.nodes):
            node.create_retailer(i, num_ret, myu[i], Cinv_ret, Cpel_ret, season, s_m, error_rate)
            node.create_warehouse(i, Cinv_wh, Cpel_wh)

        # 多次元正規分布で需要を生成する
        self.update_demand()

        for i, node in enumerate(self.nodes):
            if draw == True:
                node.draw_ini_ret()

        '''
        # Warehouse 在庫関連データ
        self.qty_inv_wh = np.zeros((num_products, num_time + BUFFER))
        self.qty_short_wh = np.zeros((num_products, num_time + BUFFER))

        # product / 拠点 / time ごとの在庫関連データ
        self.c_inv = np.zeros((num_products, num_ret, num_time + BUFFER))
        self.c_pel = np.zeros((num_products, num_ret, num_time + BUFFER))
        self.qty_inv = np.zeros((num_products, num_ret, num_time + BUFFER))
        self.qty_short = np.zeros((num_products, num_ret, num_time + BUFFER))
        
        # 拠点 / timeごとの輸送関連データ
        self.qty_trans = np.zeros((num_ret, num_time + BUFFER))
        self.c_trans = np.zeros((num_ret, num_time + BUFFER))
        
        # 合計コスト
        self.reward   = np.zeros(num_time + BUFFER) #　t=stepにおける総reward
        self.c_trans_all  = np.zeros(num_time + BUFFER) #　t=stepにおける輸送コスト
        '''

    def update_demand(self):
        demand = self.create_demand()
        for i, node in enumerate(self.nodes):
            node.iRet[0].demand = demand[:, i]
            node.iRet[0].demand = np.where(node.iRet[0].demand < 0, 0, node.iRet[0].demand)

    def create_demand(self):
        sigma = self.s_m * self.myu
        num = len(self.myu)
        cov = np.zeros((num, num))
        size = self.num_time + BUFFER

        for i in range(num):
            for j in range(num):
                cov[i, j] = sigma[i] * sigma[j]
                cov[i, j] *= self.ro ** abs(i-j)

        # print(cov)
        data = np.random.multivariate_normal(self.myu, cov, size=size)
        return data

    def set_num_actions(self, num_actions_per_product):
        for node in self.nodes:
            for ret in node.iRet:
                ret.num_actions_per_product = num_actions_per_product

    def create_result_memory(self):
        self.result = ResultMemory(self.num_products, self.num_ret, self.num_time, self.op_trans_alloc, self.inv_cap, self.trans_cap)

    '''
    def draw(self):
    for node in self.nodes:
        node.draw()
    '''

    def draw(self):
        fig, ax = plt.subplots(nrows=self.num_products, figsize=(12, self.num_products*3))
        plt.style.use('fivethirtyeight')
        for cnt, node in enumerate(self.nodes):
            for i in range(self.num_ret):
                ax[cnt].plot(node.iRet[i].inv, label='inventory', linewidth=1)
                ax[cnt].plot(node.iRet[i].inv_lt, label='inventory_lt', linewidth=1)
                # ax[cnt].plot(node.iRet[i].inv_position, label='inventory_position', linewidth=1)
                ax[cnt].plot(node.iRet[i].demand, label='demand', linewidth=1)
                ax[cnt].plot(node.iRet[i].short, label='short', linewidth=1)
                ax[cnt].plot(node.iRet[i].ord, label='ord', linewidth=1)
                # ax2 = ax[cnt].twinx()
                # ax2.plot(node.iRet[i].scs_flg, label='scs flg', linewidth=1)
                ax[cnt].set_title(f'{cnt}')
                plt.grid()
        plt.legend()

    def get_episode_performance(self):
        return self.result.get_episode_performance()

    def calc_episode_performance(self):
        self.result.calc_episode_performance(self.nodes)
        
    def calc_step_reward(self, step):
        return self.result.calc_step_reward(self.nodes, step)
        
    def episode_performance_print(self, logger):
        self.result.episode_performance_print(logger)

    def get_wh_inv(self, step):
        ''' stepにおける各商品のWH在庫を取得する '''
        inv = [node.iWH.inv[step]/node.lot_retailer for node in self.nodes]

        return np.array(inv)

    def step(self, action, step):
        ''' Retailerのactionを送って在庫の受払を計算する '''
        done =False
        
        for p, node in enumerate(self.nodes) :#商品ごとのループ
            # Retailer
            self.reflect_action(action[p], node, step)# 1-1. RetailerのActionをord, repl_planに反映させる
            # WH
            self.reflect_action_wh(node.iWH.get_action_s_q_policy(step), node, step)# 1-2. WH在庫は(s,Q)ポリシーにてord, repl_planに反映させる
            #Tentative ; WHでの在庫切れや納品遅れはないとする場合
            self.fix_retailer_repl(node, step)

            ''' 2. 在庫の受払を行う '''
            self.update_inventory_at_retailer(node, step)# 2-1. Retailerの受払計算
            self.update_inventory_at_warehouse(node, step)# 2-2. WHの受払計算

        ''' Rewardの計算：全Retailer Nodeについて '''
        reward, reward_p = self.calc_step_reward(step)
        reward = torch.FloatTensor([reward])
        reward_p = torch.FloatTensor(np.array(reward_p)).view(-1)
        # reward_p = torch.unsqueeze(reward_p, 0)
        
        if step == MAX_STEPS - 1:
            done =True
            
        # action branching向けに商品ごとのrewardを返す
        # 輸送コストは発注した商品に対して按分する，もしくは発注量に応じて商品ごとに按分する
        return done, reward, reward_p
    
    def reflect_action(self, action, node, step):
        # actionをもとに，　発注を確定し， LT後に入荷するよう，入荷予定を更新する
        for i, Ret in enumerate(node.iRet):
            Ret.action[step] =action[i]
            Ret.ord[step] =action[i]* Ret.lot
            Ret.repl_plan[step + Ret.lt -1] = Ret.ord[step]# 入荷予定に入れる

    def reflect_action_wh(self, action, node, step):
        # actionをもとに，　発注を確定し， LT後に入荷（利用可能，つまりdemand(step+LT)に対して引き当てできるタイミングで，入荷予定を更新する
        node.iWH.action[step] =action
        node.iWH.ord[step] =action* node.iWH.lot
        node.iWH.repl_plan[step + node.iWH.lt -1] = node.iWH.ord[step]# 入荷予定に入れる

    def fix_retailer_repl(self, node, step):
        # ''' tentative ''' 　#入荷予定はそのままの場合
        for i, Ret in enumerate(node.iRet):
            Ret.repl[step] =Ret.repl_plan[step]
            node.iWH.ship[step] +=  Ret.repl[step]
        
    def update_inventory_at_retailer(self, node, step):
        # 全ての拠点について在庫受払を計算する
        qty_short_at_retailer = 0
        for i, Ret in enumerate(node.iRet):
            Ret.ship[step] = min(Ret.inv[step], Ret.demand[step]) #在庫を上限として，需要そのまま
            Ret.short[step] = Ret.demand[step] - Ret.ship[step] #欠品
            # qty_short_at_retailer = qty_short_at_retailer + Ret.short[step]
            Ret.proceed(step)
            
    def update_inventory_at_warehouse(self, node, step):
        # WHでは在庫遅れはないとする
        node.iWH.repl[step] = node.iWH.repl_plan[step]
        # 受払を計算
        node.iWH.proceed(step)

    def reflesh_pred(self):
        ''' Retailerでの将来予測値を毎回書き換える'''
        for node in self.nodes:
            node.reflesh_pred()
                
