from .config_learn import USE_LT_INV
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

BUFFER = 10
IGNORE_STEP = 20

# -- logi setting
# TRANS_CAPACITY = 20 #　# of palettes in 20ft container

# -- cost setting
Cinv_ret, Cpel_ret = 0.02, 1
Cinv_wh, Cpel_wh = 0, 0 # 一旦Retailerの評価のみ
Ctrans = 1 #輸送コスト(Capacityあたり)

# - fixed inventory cost setting
# INV_CAPACITY = 50 #　# of palettes in retailer warehouse
# Cinv_ret_fix = Cinv_ret * INV_CAPACITY * 0.7 #(Capacityまで)


# cost reference
# コンテナ20ftで26m3，パレット1つで大体1.7m3→20ftコンテナに15-20パレくらい
# 輸送コスト：コンテナ20ftで100,000円（中国からの海上運賃，THC，ドレージ）
# 在庫コスト：¥200/m3・日でとすると，1,400¥m3・週　→ パレット体積当たりだと2,000￥/m3・週

class SCM():
    def __init__(self, p, id, num, lt, lot,k):
        self.p = p # product_id
        self.id = id # ret_id
        
        ''' Parameter Setting '''
        self.lt = lt
        self.lot = lot
        self.num_time = num
        self.k = k

        ''' 各種時系列データの初期化を行う'''
        self.action = np.zeros(num + BUFFER) #　actionそのもの（ロット単位の発注）
        self.inv = np.zeros(num + BUFFER) #　t=stepにおける在庫
        self.inv_lt = np.zeros(num + BUFFER) #　t=stepにおけるLT後時点の在庫
        self.inv_position = np.zeros(num + BUFFER) #　t=stepにおける在庫＋発注残
        self.inv_lt_true = np.zeros(num + BUFFER) #　t=stepにおけるLT後時点の真の在庫
        self.ord = np.zeros(num + BUFFER) #　t=stepにおける発注
        self.demand = np.zeros(num + BUFFER) #　t=stepにおける需要
        self.ship = np.zeros(num + BUFFER) #　t=stepにおける出荷
        self.repl = np.zeros(num + BUFFER) #　t=stepにおける入荷
        self.repl_plan = np.zeros(num + BUFFER) #　t=stepにおける入荷予定（発注した際に更新する）
        self.short = np.zeros(num + BUFFER) #　t=stepにおける欠品
        self.scs_flg = np.zeros(num + BUFFER)

        ''' Cost時系列データ'''
        self.c_inv = np.zeros(num + BUFFER) #　t=stepにおける欠品
        self.c_pel = np.zeros(num + BUFFER) #　t=stepにおける欠品
        self.cost   = np.zeros(num + BUFFER) #　t=stepにおける欠品
        
    def set_cost_unit_price(self, c_i, c_p):
        self.c_i = c_i
        self.c_p = c_p

    def proceed(self, step):
        ''' 在庫の受払計算を行う'''
        # inv[step]はt=stepにおける頭在庫であるので，t=step-1の末在庫でもある
        self.inv[step+1] = self.inv[step] - self.ship[step] + self.repl[step]
        self.inv[step+1] = max(self.inv[step+1], 0) #ネガティブ在庫はなし
        # LT後在庫は直近在庫に対して，出荷（予測）を引いて，　入荷（予定）を足すことで計算する
        # step + LT時点での需要を差し引いた時に安全在庫を下回ると，そこで欠品する可能性が生じる
        self.inv_lt[step+1]  = self.inv[step+1] - self.pred[step+1 : step+self.lt+2].sum() + self.repl_plan[step+1 : step+self.lt+1].sum()
        # 真のLT後在庫は直近在庫に対して，出荷の実績を受払にて計算する
        self.inv_lt_true[step+1]  = self.inv[step+1] - self.demand[step+1 : step+self.lt+2].sum() + self.repl_plan[step+1 : step+self.lt+1].sum()
        # 在庫ポジション
        self.inv_position[step+1] = self.inv[step+1] + self.repl_plan[step+1 : step+self.lt+1].sum()

    '''
    def get_action_s_q_policy(self, step):
        # [現在庫] + [発注済み数量] が安全在庫を切ったら発注する
        oo = self.repl_plan[step:].sum() #発注済み数量
        if  (oo + self.inv[step]) < self.safe_inv_sub:
            self.ord[step] = math.ceil((self.safe_inv_sub - (oo + self.inv[step]))/self.lot) * self.lot
        else:
            self.ord[step] = 0
        self.repl_plan[step + self.lt] = self.ord[step]# 入荷予定に入れる
    '''

    def get_action_s_q_policy(self, step):
        ''' 安全在庫水準に基づいて発注量を求める '''
        # [現在庫] + [発注済み数量] が安全在庫を切ったら発注する
        oo = self.repl_plan[step:].sum() #発注済み数量
        if  (oo + self.inv[step]) < self.safe_inv_sub:
            action = math.ceil((self.safe_inv_sub - (oo + self.inv[step]))/self.lot)
        else:
            action = 0
        return action 
        
    def draw_initial_setting(self, ax):
        ax.plot(self.demand, label='demand')
        ax.plot(self.pred, label = 'pred')
        ax.plot(np.ones(self.num_time) * self.safe_inv , label ='safe inv')
        ax.set_title(self.type + '_' + str(self.id) + '/ d:' + str(self.demand.sum()))
        ax.legend()
        
    def calc_reward(self, step):
        ''' t=stepにおけるコストを算出する '''
        # ロットサイズ + (LT＋10日)分を上限とする
        # self.c_inv[step] = np.clip(self.inv[step],a_min=0, a_max=(self.lot + self.myu*(self.lt+10))) * self.c_i
        self.c_inv[step] = self.inv[step]* self.c_i
        self.c_pel[step] = self.short[step] * self.c_p
        self.cost[step] = self.c_inv[step] + self.c_pel[step]        

    def mod_reward_inv(self, step, c_inv_rate, c_inv_zero, num, c_inv):
        self.c_inv[step] = c_inv_zero
        # if c_inv_zero > 0:
        # equal allocation
        # self.c_inv[step] += c_inv / num
        # else:
        # qty-based allocation
        self.c_inv[step] += self.inv[step] * c_inv_rate * Cinv_ret

        self.cost[step] = self.c_inv[step] + self.c_pel[step]
        # if step % 100 == 50:
        #    print(f"({step}), {c_inv_zero:.3f}, {self.inv[step]:.1f}, {self.c_inv[step]:.3f}, {c_inv_rate:.3f}")

class Retailer(SCM):
    def __init__(self, p, id, num, lt, lot, k):
        ''' 初期化 '''
        super().__init__(p, id, num, lt, lot, k)
        self.type = 'Retailer'

    def initialize(self, myu, season,s_m, error_rate):
        ''' 需要の生成 '''
        self.myu=myu
        self.season = season
        self.error_rate = error_rate
        self.s_m = s_m

        np.random.seed(self.id + self.p*10) #拠点ごとの需要を固定するために，　Seedを拠点ごとに固定する

        # Poisson : パレット単位だとかなり小さくなるのでケース単位にした上で割り戻す
        # self.demand = np.random.poisson(myu * 10, self.num_time + BUFFER)/10
        # Gaussian
        self.demand = np.random.normal(loc = myu, scale = myu * s_m, size = self.num_time + BUFFER)
        self.demand = np.where(self.demand < 0, 0, self.demand)

        # avoid negative value
        self.sd_ini = np.std(self.demand)
        # normal : パレット単位だとかなり小さくなる
        # self.demand = np.random.normal(loc=myu, scale=0.2, size=MAX_STEPS)

        # season=Trueであれば，需要は最初の2倍まで単調増加する
        if season:
            for i,d in enumerate(self.demand):
                self.demand[i] = d + 2* myu * (i /self.num_time)
        
        self.sd = np.std(self.demand)
        self.pred = self.demand + np.random.normal(loc   = 0,scale = self.sd * self.error_rate, size  = self.num_time + BUFFER)
        self.pred = np.where(self.pred<0,0,self.pred)
        self.error = self.pred - self.demand
        self.sd_error = np.std(self.error)
        self.safe_inv = self.k * self.sd_error * np.sqrt(self.lt)
        self.safe_inv_sub = self.k * self.sd * np.sqrt(self.lt) + self.lt * myu
        self.inv[0] = myu * 10 #初期在庫: 10days
        self.inv_lt[0] = myu * 10 #初期在庫
        self.scs_params={}
        self.ss_periodic_params={}
        self.ss_qs_params={}

    ''' Retailer : LT後在庫が安全在庫水準になるように発注量を求める '''    
    '''
    def get_action_s_q_policy_at_lt_inv(self, step):
        if  self.inv_lt[step] < self.safe_inv:
            self.ord[step] = math.ceil((self.safe_inv - self.inv_lt[step])/self.lot) * self.lot
        else:
            self.ord[step] = 0
        self.repl_plan[step + self.lt] = self.ord[step]# 入荷予定に入れる
    '''

    def get_action_scs_policy(self, step:int, flg:bool, add_d, numerator, denominator):
        ''' Retailer : (S, c, s)方策で発注量を決定する '''
        # add_d = Trueの場合は(S, c, d, s)方策

        if flg:
            self.scs_flg[step] = 1

        adjuster = 0
        if add_d and len(numerator)>0:
            adjuster = max([(self.scs_params['S'] - self.scs_params['d']) * numer / denom for
                            numer, denom in zip(numerator, denominator)])
            # print(numerator, denominator, adjuster)

        if not USE_LT_INV:
            cond1 = (self.inv_position[step] <= self.scs_params['s'])
            cond2 = (self.inv_position[step] <= self.scs_params['c'] and flg)
            if cond1 or cond2:
                action = self.scs_params['S'] - self.inv_position[step] - adjuster
                action = max(action, 0)
                action = max(action // self.lot, 1)
            else:
                action = 0
            if not action >= 0:
                action = 0
            assert action >= 0, f'action is {action}{type(action)} action should be zero or positive'
            return action
        else:
            cond1 = (self.inv_lt[step] <= self.scs_params['s'])
            cond2 = (self.inv_lt[step] <= self.scs_params['c'] and flg)
            if cond1 or cond2:
                action = self.scs_params['S'] - self.inv_lt[step] - adjuster
                action = max(action // self.lot, 1)
            else:
                action = 0
            return action

    def get_action_qs_policy(self, step:int):
        ''' Retailer : QS方策で発注量を決定する '''
        assert not USE_LT_INV, 'USE_LT_INV should be False'

        cond1 = self.inv_position[step] <= self.ss_qs_params['S']
        if cond1:
            action = self.ss_qs_params['S'] - self.inv_position[step]
            action = max(action // self.lot, 1)
        else:
            action = 0
        return action

    def get_action_ss_periodic_policy(self, step:int):
        ''' Retailer : P(S, s)方策で発注量を決定する '''
        if not USE_LT_INV:
            cond1 = self.inv_position[step] <= self.ss_periodic_params['s']
            if cond1:
                action = self.ss_periodic_params['S'] - self.inv_position[step]
                action = max(action // self.lot, 1)
            else:
                action = 0
            return action
        else:
            cond1 = self.inv_lt[step] <= self.ss_periodic_params['s']
            if cond1:
                action = self.ss_periodic_params['S'] - self.inv_lt[step]
                action = max(action // self.lot, 1)
            else:
                action = 0
            return action

    def get_action_s_q_policy_at_lt_inv(self, step):
        ''' Retailer : LT後在庫が安全在庫水準になるように発注量を求める '''
        if  self.inv_lt[step] < self.safe_inv:
            action=math.ceil((self.safe_inv - self.inv_lt[step])/self.lot)
        else:
            action=0
        return action

    def get_action_s_s_policy_at_lt_inv(self, step, trans_cost_mode):
        ''' Retailer : 発注量は単位期間コストが最低となる発注量を都度選択する '''
        # trans_cost_mode = 'container' : 輸送コストはコンテナあたりCtransかかる
        # trans_cost_mode = 'palette' : 輸送コストは発注量に対して、paletteあたりで Ctrans / TRANS_CAPACITY

        best_action = 0

        # 発注するかどうかの条件はs，Qと同じ
        if  self.inv_lt[step] < self.safe_inv:
            best_expected_unit_cost = np.inf

            # 全てのactionを試す
            for s in range(self.num_actions_per_product-1):
                inv_lt_temp = np.zeros(self.num_time + BUFFER)
                action = s+1

                # 将来在庫について，このロットを発注した場合として更新する
                inv_lt_temp[step + self.lt] = self.inv_lt[step] + (action * self.lot)
                for t in range(step+self.lt, self.num_time):
                    inv_lt_temp[t+1] = inv_lt_temp[t] - self.demand[t]

                # print(inv_lt_temp)

                # LT以降，将来在庫が安全在庫を下回るポイントを抽出する
                t = np.min(np.where(inv_lt_temp[step+self.lt:] < self.safe_inv))

                # 輸送コストを計算する
                if trans_cost_mode == 'container':
                    c_trans = math.ceil(action * self.lot / TRANS_CAPACITY) * Ctrans
                elif trans_cost_mode =='palette':
                    c_trans = action * self.lot * (Ctrans / TRANS_CAPACITY)

                # 次回の入荷ポイントまで，在庫保管コストを計算する
                c_inv_temp_inv = np.where(inv_lt_temp<0, 0,inv_lt_temp)
                c_inv_temp_sum = c_inv_temp_inv[step + self.lt:step + self.lt + t + 1].sum()
                c_inv = c_inv_temp_sum * self.c_i

                # 次回の入荷ポイントまで，欠品コストを計算する
                c_inv_temp_short = np.where(inv_lt_temp<0, inv_lt_temp,0)
                c_inv_temp_short_sum = - c_inv_temp_short[step + self.lt:step + self.lt + t + 1].sum()
                c_short = c_inv_temp_short_sum * self.c_p

                # 単位期間コストを計算する
                expected_unit_cost = (c_inv + c_trans + c_short)/(t+1)

                # print(step,action, self.safe_inv, t,c_inv,c_trans,c_short, expected_unit_cost)

                # 単位期間コストがよければそれを採用する
                if best_expected_unit_cost > expected_unit_cost:
                    best_expected_unit_cost = expected_unit_cost
                    best_action = action

            # print('best action:', step, best_action)

        return best_action

class WH(SCM):
    def __int__(self, p, id, num, lt, lot, k):
        ''' 初期化 '''
        super().__init__(p, id, num, lt, lot, k)
        self.type = 'Warehouse'

    def initialize(self, iRet):
        ''' 需要の生成 '''
        self.all_demand = np.zeros(self.num_time + BUFFER)
        self.pred = np.zeros(self.num_time + BUFFER)
        for Ret in iRet:
            self.all_demand = self.all_demand + Ret.demand
            self.pred = self.pred + Ret.pred
        self.myu=self.all_demand.mean()
        self.sd = np.std(self.all_demand)
        self.safe_inv_sub = self.k * self.sd * np.sqrt(self.lt) + self.lt * self.all_demand.mean()
        self.inv[0] = self.all_demand.mean() * 10 #初期在庫
        self.safe_inv = self.all_demand.mean() * 20 # for test purpose
        self.safe_inv_sub = self.all_demand.mean() * 20
        
    def get_action_s_q_policy_at_lt_inv_at_wh(self, step):
        ''' Warehouse : Ret + WH LT後のエシェロン在庫が安全在庫水準になるように発注量を求める '''
        # Retailerも含めた総将来在庫を計算する必要がある．
        # 必要量は，　総在庫量がトータル安全在庫を切る場合
        
        print()
        
        
class Node():
    # RetailerインスタンスとWarehouseインスタンスの集合を保持するクラス
    # 商品ごとに一つ作成される
    
    def __init__(self, product, k, num_time, lt_retailer, lt_wh, lot_retailer, lot_wh, inv_cap):
        # Parameter Setting
        self.product = product
        self.k = k
        self.num_time = num_time
        self.lt_retailer, self.lt_wh =lt_retailer, lt_wh
        self.lot_retailer, self.lot_wh = lot_retailer, lot_wh
        self.inv_cap = inv_cap

        self.short_at_retailer = np.zeros(num_time + BUFFER) #　Retailer欠品
        self.short_at_wh = np.zeros(num_time + BUFFER) #　WH欠品
        self.total_cost = np.zeros(num_time + BUFFER) #　コスト
                
    def create_retailer(self, p, num_ret, myu,c_i, c_p, season, s_m, error_rate):
        ''' Retailer インスタンスの生成'''
        self.num_ret = num_ret
        self.iRet = [0] * num_ret
        for i in range(num_ret):
            self.iRet[i] = Retailer(i, p, self.num_time , self.lt_retailer, self.lot_retailer, self.k)
            self.iRet[i].initialize(myu, season, s_m, error_rate)
            self.iRet[i].set_cost_unit_price(c_i,c_p)
    
    def draw_ini_ret(self):
        # fig, ax = plt.subplots(ncols=3, figsize=(12,4))
        fig, ax = plt.subplots(ncols=self.num_ret, figsize=(12,4))
        for i in range(self.num_ret):
            if self.num_ret > 1:
                self.iRet[i].draw_initial_setting(ax[i])
            else:
                self.iRet[i].draw_initial_setting(ax)
    
    def create_warehouse(self, i, c_i,c_p):
        ''' WH インスタンスの生成'''
        self.iWH = WH(i, 1, self.num_time, self.lt_wh, self.lot_wh, self.k )
        self.iWH.initialize(self.iRet)
        self.iWH.set_cost_unit_price(c_i,c_p)

    def reflesh_pred(self):
        ''' Retailerでの将来予測値を毎回書き換える'''
        self.iWH.pred = np.zeros(self.num_time + BUFFER)
        for i, Ret in enumerate(self.iRet):
            Ret.pred =  np.random.normal(loc   = Ret.demand,scale = Ret.sd * Ret.error_rate, size  = self.num_time + BUFFER)
            Ret.pred = np.where(Ret.pred<0, 0, Ret.pred)
            self.iWH.pred = self.iWH.pred + Ret.pred

    def simulate_s_q_policy(self, step, with_pred):
        action = []
        for Ret in self.iRet:
            if with_pred:
                action.append(Ret.get_action_s_q_policy_at_lt_inv(step))
            else:
                action.append(Ret.get_action_s_q_policy(step)) #iRet.ord/repl_planが更新される
        return action

    def simulate_s_s_policy(self, step, with_pred, trans_cost_mode):
        action = []
        for Ret in self.iRet:
            action.append(Ret.get_action_s_s_policy_at_lt_inv(step, trans_cost_mode))
        return action

    def set_scds_params(self, params):
        s, c, d, S = params
        for i, Ret in enumerate(self.iRet):
            Ret.scs_params['s'] = s
            Ret.scs_params['c'] = c
            Ret.scs_params['d'] = d
            Ret.scs_params['S'] = S
            Ret.param_reorder_point = S

    def set_scs_params(self, params):
        s, c, S = params
        for i, Ret in enumerate(self.iRet):
            Ret.scs_params['s'] = s
            Ret.scs_params['c'] = c
            Ret.scs_params['S'] = S
            Ret.param_reorder_point = S
            Ret.param_trigger_point = c
            Ret.param_mustorder_point = s

    def simulate_scs_policy(self, step, flg, add_d, numerator, denominator):
        action = []
        for Ret in self.iRet:
            action.append(Ret.get_action_scs_policy(step, flg, add_d, numerator, denominator))
        return action

    def set_ss_periodic_params(self, params):
        s, S = params
        for i, Ret in enumerate(self.iRet):
            Ret.ss_periodic_params['s'] = s
            Ret.ss_periodic_params['S'] = S
            Ret.param_reorder_point = S
            Ret.param_trigger_point = s
            Ret.param_mustorder_point = s

    def simulate_ss_periodic_policy(self, step):
        action = []
        for Ret in self.iRet:
            action.append(Ret.get_action_ss_periodic_policy(step))
        return action

    def set_qs_params(self, params):
        for i, Ret in enumerate(self.iRet):
            Ret.ss_qs_params['S'] = params
            Ret.param_reorder_point = params

    def simulate_qs_policy(self, step):
        action = []
        for Ret in self.iRet:
            action.append(Ret.get_action_qs_policy(step))
        return action

    '''
    def simulate_s_q_policy(self, step, with_pred):
        # (s,Q) policy for Retailer(s) and WH
        # 1. 発注量の決定
        
        # 1-1. RetailerのAction
        # Retailerレベルでの発注量を計算する
        for Ret in self.iRet:
            if with_pred:
                Ret.get_action_s_q_policy_at_lt_inv(step)
            else:
                Ret.get_action_s_q_policy(step) #iRet.ord/repl_planが更新される
        
        # 1-2. WHのAction
        # -- 1) WHレベルでの発注量を計算する
        self.iWH.get_action_s_q_policy(step)        
        # -- 2) 各Retailerへの出荷処理
        # --iRetに対して順番にその日のrepl_planを参照し，　在庫がある分だけreplとして更新する
        shipment_from_wh = 0
        qty_short_at_wh = 0
        
        for Ret in self.iRet:
            Ret.repl[step] = min(self.iWH.inv[step], Ret.repl_plan[step])
            self.iWH.inv[step] = self.iWH.inv[step] - Ret.repl[step]
            shipment_from_wh =  shipment_from_wh + Ret.repl[step]
            qty_short_at_wh = qty_short_at_wh  +  (Ret.repl_plan[step] - Ret.repl[step])
        self.iWH.ship[step] =  shipment_from_wh
        self.iWH.short[step] =  qty_short_at_wh

        # 2. 受払の計算: self.inv[step+1] = self.inv[step] - self.ship[step] + self.repl[step]

        # 2-1. Retailerの受払計算
        # -- 1) 入荷処理(repl): 1-2で実施済み
        # --2) 出荷処理(ship)
        # --3)　受払処理
        qty_short_at_retailer = 0
        for Ret in self.iRet:
            Ret.ship[step] = min(Ret.inv[step], Ret.demand[step]) #在庫を上限として，需要そのまま
            Ret.short[step] = Ret.demand[step] - Ret.ship[step] #欠品
            qty_short_at_retailer = qty_short_at_retailer + Ret.short[step]
            Ret.proceed(step)

        # 2-2. WHの受払計算
        # -- 1) 入荷処理(repl):
        self.iWH.repl[step] = self.iWH.repl_plan[step]
        # --2) 出荷処理(ship): 1-2で実施済み
        # --3)　受払処理
        self.iWH.proceed(step)
        
        # print('step:{}/ shipment to retailer : {} / short_at_wh : {} / short_at_retailer:{}'.format(
        #     step, shipment_from_wh, qty_short_at_wh,qty_short_at_retailer))
        self.short_at_retailer[step] = qty_short_at_retailer
        self.short_at_wh[step] = qty_short_at_wh
    '''
        
    '''
        # rewardの計算
         if step > IGNORE_STEP:
            for Ret in self.iRet:
                Ret.calc_reward(step)
                self.total_cost[step] = self.total_cost[step] + Ret.cost[step]
            self.iWH.calc_reward(step)        
            self.total_cost[step] = self.total_cost[step] + self.iWH.cost[step]
    '''
                    
    def draw(self):
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(12,4))
        
        for i in range(self.num_ret):
            axL.plot(self.iRet[i].inv, label='Retail inventory_' + str(i))
            axL.plot(self.iRet[i].demand, label='Retail demand_' + str(i))
            axL.plot(self.iRet[i].short, label='Retail short_' + str(i))
            axL.plot(self.iRet[i].ord, label='Retail ord_' + str(i))
        axL.legend()
        axL.set_title('Retailer')
        axR.plot(self.iWH.inv, label='WH inventory')
        axR.plot(self.iWH.ship, label='WH shipment')
        axR.legend()
        axR.set_title('Warehouse')
        
class ResultMemory():
    ''' 計算結果格納用 '''
    def __init__(self, num_products, num_ret, num_time, op_trans_alloc, inv_cap, trans_cap):
        self.num_time = num_time
        self.op_trans_alloc = op_trans_alloc

        # 在庫キャパシティに応じた在庫コスト設定
        self.inv_cap = inv_cap
        self.trans_cap = trans_cap
        self.Cinv_ret_fix = Cinv_ret * self.inv_cap * 0.7 #(Capacityまで)

        self.num_products = num_products
        self.num_ret = num_ret
        self.num_nodes = int(num_products * num_ret)

        # Warehouse 在庫関連データ
        self.qty_inv_wh = np.zeros((num_products, num_time + BUFFER))
        self.qty_short_wh = np.zeros((num_products, num_time + BUFFER))

        # product / 拠点 / time ごとの在庫関連データ
        self.c_inv = np.zeros((num_products, num_ret, num_time + BUFFER))
        self.c_pel = np.zeros((num_products, num_ret, num_time + BUFFER))
        self.qty_inv = np.zeros((num_products, num_ret, num_time + BUFFER))
        self.qty_short = np.zeros((num_products, num_ret, num_time + BUFFER))
        self.qty_ord = np.zeros((num_products, num_ret, num_time + BUFFER))

        # 拠点 / timeごとの輸送/在庫関連データ
        self.qty_trans = np.zeros((num_ret, num_time + BUFFER))
        self.c_trans = np.zeros((num_ret, num_time + BUFFER))
        self.qty_inv_ret_sum = np.zeros((num_ret, num_time + BUFFER))
        self.c_inv_ret_sum = np.zeros((num_ret, num_time + BUFFER))

        # 合計コスト
        self.reward   = np.zeros(num_time + BUFFER) #　t=stepにおける総reward
        self.c_trans_all  = np.zeros(num_time + BUFFER) #　t=stepにおける輸送コスト
        self.c_inv_all  = np.zeros(num_time + BUFFER) #　t=stepにおける輸送コスト

    def calc_step_reward(self, nodes, step):
        # t=stepにおけるコスト（在庫，欠品，輸送）を計算して，合計(reward)を返す
        reward = 0
        reward_p = np.zeros((self.num_products, self.num_ret)) #商品×拠点ごとのreward 

        # 計算に向けた集計処理（輸送量と在庫量）
        for p, node in enumerate(nodes):#商品ループ
            for i, Ret in enumerate(node.iRet): #拠点ループ
                # Retailerコスト：在庫コスト(c_inv)と欠品コスト(c_pel)を計算
                Ret.calc_reward(step)
                # 輸送コスト算出のためにWH-Retailerへの商品合計の数量を計算する
                self.qty_trans[i,step] += Ret.repl[step]
                # 在庫コスト算出のためにRetailer在庫の全商品の合計在庫数量を計算する
                self.qty_inv_ret_sum[i,step] += Ret.inv[step]

        # 在庫コスト（拠点別の商品横断）
        # INV_CAPを0としておけば全て変動費扱い
        # c_inv = np.array([self.Cinv_ret_fix + max(0, q - self.inv_cap) * Cinv_ret for q in self.qty_inv_ret_sum[:,step]])
        c_inv = np.array([max(0, q - self.inv_cap) * Cinv_ret for q in self.qty_inv_ret_sum[:,step]])
        c_inv_rate = np.array([np.where(q > 0, max(0, q - self.inv_cap)/q, 0) for q in self.qty_inv_ret_sum[:,step]])

        # if step % 100 == 50:
        #     print(f"({step}) {c_inv}, {c_inv_rate}")

        # 在庫保管単価を物量で割り戻して算出
        '''
        for i, q in enumerate(self.qty_inv_ret_sum[:,step]):
            if q > 0:
                c_inv[i] = c_inv[i] / q
            else:
                c_inv[i] = 0
        '''

        # 修正した在庫保管単価を元に再計算する
        for p, node in enumerate(nodes):#商品ループ
            for i, Ret in enumerate(node.iRet): #拠点ループ
                # 全商品の総計在庫量が0の場合は、単純にfixed costを商品数で按分する
                # 固定分のコストは全商品で均等に按分する
                Ret.mod_reward_inv(step, c_inv_rate[i], self.Cinv_ret_fix / self.num_products,
                                   self.num_products, c_inv[i])
                node.total_cost[step] += Ret.cost[step]

                # Branchごとのrewardを保持する（ここではc_inv + c_pel)
                reward_p[p, i] = Ret.cost[step]

            # WHコスト
            node.iWH.calc_reward(step)
            node.total_cost[step] += node.iWH.cost[step]

            reward += node.total_cost[step]

        # 輸送コスト（商品横断）
        # TRANS_CAPACITYで離散化した
        c_trans = np.array([math.ceil(q) for q in self.qty_trans[:,step]/self.trans_cap]) * Ctrans
        reward +=  c_trans.sum()

        # 商品ごとのAllocation
        # 輸送コストについては、発注するかどうかに関わらず商品ごとに平等に割り振る
        # 在庫コストについては、商品別の在庫水準に比例させる（単価自体を割り戻した上で単純に実績に乗じることで算出する）
        for p, node in enumerate(nodes):#商品ループ
            for i, Ret in enumerate(node.iRet): #拠点ループ
                if c_trans[i]>0:
                    if self.op_trans_alloc =='qty':
                        # 商品ごとの輸送量に応じて輸送コストを商品ごとに按分する（当該商品で発注していなければゼロになる）
                        reward_p[p,i] += c_trans[i] * (Ret.repl[step]/self.qty_trans[i, step])
                    elif self.op_trans_alloc =='all':
                        # 商品数で単純に按分する（当該商品が発注していなくてもコスト按分される）
                        reward_p[p,i] += c_trans[i] / self.num_products

        # resultに計算結果を反映
        self.reward[step] = reward
        self.c_trans[:, step] = c_trans
        self.c_trans_all[step] = c_trans.sum() #全Retの合計
        
        # print(step, ":", reward, reward_p, c_trans, self.qty_trans[:,step])

        # action branching向けに商品ごとのrewardを返す
        # 輸送コストは発注した商品に対して按分する，もしくは発注量に応じて商品ごとに按分する
        return reward, reward_p

    def calc_episode_performance(self, nodes):
        # 各種コストを一括で管理する        
        for p, node in enumerate(nodes):
            # Warehouse関連
            self.qty_inv_wh[p, :] = node.iWH.inv
            self.qty_short_wh[p, :] = node.iWH.short
            
            # Retailer関連            
            for i, Ret in enumerate(node.iRet):
                self.c_inv[p, i,:] = Ret.c_inv
                self.c_pel[p, i,:] = Ret.c_pel
                self.qty_inv[p, i,:] = Ret.inv
                self.qty_short[p, i,:] = Ret.short
                self.qty_ord[p, i,:] = Ret.ord

    def get_episode_performance(self):
        # Warehouse 関連
        wh_inv = [i for i in self.qty_inv_wh[:, IGNORE_STEP:self.num_time].mean(axis=1).flatten()]
        wh_short = [i for i in self.qty_short_wh[:, IGNORE_STEP:self.num_time].sum(axis=1).flatten()]

        # Retailer 関連
        ret_inv = [i for i in self.qty_inv[:, :, IGNORE_STEP:self.num_time].mean(axis=2).flatten()]
        ret_short = [i for i in self.qty_short[:, :, IGNORE_STEP:self.num_time].sum(axis=2).flatten()]

        # cost
        ret_c = [i for i in self.c_inv[:, :, IGNORE_STEP:self.num_time].sum(axis=2).flatten()]
        ret_p = [i for i in self.c_pel[:, :, IGNORE_STEP:self.num_time].sum(axis=2).flatten()]
        ret_t = [i for i in self.c_trans[:, IGNORE_STEP:self.num_time].sum(axis=1).flatten()]

        # KPI
        # ret_trans_rate = [(qty_trans[qty_trans>0]/TRANS_CAPACITY).mean() for qty_trans in self.qty_trans]
        ret_trans_rate = [(qty_trans[qty_trans > 0] / self.trans_cap) for qty_trans in self.qty_trans]
        ret_trans_round = [np.array(pd.Series(ret_trans_rate).map(lambda x: math.ceil(x))) for ret_trans_rate in
                           ret_trans_rate]
        # ret_trans_rate = [(ret_trans_rate / (ret_trans_round+0.00001)).mean() for ret_trans_rate, ret_trans_round in
        #  zip(ret_trans_rate, ret_trans_round)]

        ret_ord = [qty_ord[qty_ord>0].mean() for qty_ord in self.qty_ord[:,:,IGNORE_STEP:self.num_time]]

        # round
        wh_inv = self.return_rounded(wh_inv,1)
        wh_short = self.return_rounded(wh_short,1)
        ret_inv = self.return_rounded(ret_inv,1)
        ret_short = self.return_rounded(ret_short,1)
        ret_c = self.return_rounded(ret_c,1)
        ret_p = self.return_rounded(ret_p,1)
        ret_t = self.return_rounded(ret_t,1)
        # ret_trans_rate = self.return_rounded(ret_trans_rate,2)
        ret_ord = self.return_rounded(ret_ord,2)

        return wh_inv, wh_short, ret_inv, ret_short, ret_c, ret_p, ret_t, 0, ret_ord

    def return_rounded(self, ret, round_digit):
        return list(np.round(np.array(ret), round_digit))


    def episode_performance_print(self, logger):
        wi, ws, qi, qs, ic, pc, tc, trate, qo = self.get_episode_performance()
        logger.info('Retailer...  Inv(mean):{} /Short:{} /  InvCost:{} / PelCost:{} / TransCost:{} / TransRate:{} / AvgOrd:{}'.format(
            qi, qs, ic, pc, tc, trate, qo))
        logger.info('Warehouse... Inv(mean) : {} /Short : {}'.format(wi, ws))