from config import *
from agent import *
import argparse
import subprocess as sp
import requests
import sys
from config_learn import NUM_VALIDATION

parser = argparse.ArgumentParser(description='SCM RL')

# demand setting
parser.add_argument('--scenario', default='ex1', help='Scenario : ex1 tp ex5')
parser.add_argument('--season', default=False, help='season')
parser.add_argument('--num_ret', default=1, help='number of retailers')
parser.add_argument('--error_rate', default=0.5, type=float,help='error rate')
parser.add_argument('--s_m', default=0.6, type=float,help='sigma over myu for demand')
parser.add_argument('--ro', default=0, type=float,help='correlation')
parser.add_argument('--d', default=1.0, type=float,help='demand factor')
parser.add_argument('--invcap', default=0, type=float,help='warehouse capacity')

# cost setting
parser.add_argument('--op_trans_alloc', default='all', help='trans cost allocation option')

# NN
parser.add_argument('--nn_option', default=10, help='select NN architecture / 0:DQN 3:Branched DQN')
parser.add_argument('--num_actions_per_product', default=4, help='number of actions per node')

# heuristic
parser.add_argument('--ga', default=None, help='compute scs or scds or ss')
parser.add_argument('--force_uncap', action='store_true', help='not use adjustment logic')

args = parser.parse_args()

# print('state_real=' + str(args.state_real))
# print('obs_only=' + str(args.obs_only))
# print('sample_val=' + str(args.sample_val))

print('scenario='+ str(args.scenario))
print('d factor='+ str(args.d))

print('num_ret='+ str(args.num_ret))
print('num_actions_per_product=' + str(args.num_actions_per_product))
print('nn_option=' + str(args.nn_option))
print('season=' + str(args.season))
print('sigma over myu=' + str(args.s_m))
print('error_rate=' + str(args.error_rate))
print('op_trans_alloc=' + str(args.op_trans_alloc))


scenario = args.scenario
num_ret = int(args.num_ret)
nn_option = int(args.nn_option)
season = args.season
error_rate = args.error_rate # // degree of forecast accuracy from 0(complete) to 1(no contribution)
s_m = args.s_m
d_factor = args.d
op_trans_alloc = args.op_trans_alloc # // all:equal allocation，qty: order quantity-based allocation
num_actions_per_product = int(args.num_actions_per_product)

# obs_only = args.obs_only
# sample_val = args.sample_val
# state_real = args.state_real
# True  : true state in experience memory
# False : observation in experience memory

# Experimental Setting
num_products, season, myu, lot_retailer, lot_wh, inv_cap, trans_cap = get_ex_setting(scenario)
if args.invcap > 0:
    inv_cap = args.invcap

assert not season, 'season should be False'

lt_retailer = [LT_RET for i in range(num_products)]
lt_wh = [LT_WH for i in range(num_products)]

myu = np.array(myu) * d_factor

env = Environment(k, MAX_STEPS, num_products, myu, lt_retailer, lt_wh, lot_retailer, lot_wh, num_ret, inv_cap, trans_cap,
                  d_factor, season, s_m, args.ro, error_rate, op_trans_alloc, False)
env.set_num_actions(num_actions_per_product)

if args.ga is not None:
    agent = Agent(scenario, env,num_actions_per_product, state_real, obs_only,sample_val, op_trans_alloc)
    agent.version = 'ga'
    agent._set_logger()

    # F-EOP
    # agent.sim_s_q_policy(True, False,False,0, trans_cost_mode='container')
    # simple
    # agent.sim_s_q_policy(True, False,False,0, trans_cost_mode='palette')

    # コンテナサイズ制約をかけるかどうか
    # コンテナサイズを設定している場合に制約をつける
    capacitated = False if 'cap' in scenario else True
    capacitated_val = False if 'cap' in scenario else True
    if args.force_uncap:
        capacitated = False

    # scs GA
    best_param = np.array(agent.optimize_ga(args.ga, capacitated))
    if args.ga == 'scs':
        [agent.sim_scs_policy(best_param, capacitated=capacitated_val, draw_result=True, write=True) for _ in range(NUM_VALIDATION)]
    elif args.ga == 'scds':
        [agent.sim_scs_policy(best_param, capacitated=capacitated_val, draw_result=True, add_d=True, write=True) for _ in range(NUM_VALIDATION)]
    elif args.ga == 'qs':
        [agent.sim_qs_policy(best_param, capacitated=capacitated_val, draw_result=True, write=True) for _ in range(NUM_VALIDATION)]
    elif args.ga == 'ss':
        [agent.sim_ss_periodic_policy(best_param, capacitated=capacitated_val, draw_result=True, write=True) for _ in range(NUM_VALIDATION)]

    if UploadToGCP:
        str_command = 'gsutil -m cp -r ' + agent.ga_outfile + GCP_bucket + 'ga/'
        print(str_command)
        sp.call(str_command, shell=True)
        str_command = 'gsutil -m cp -r ' + agent.LOG + GCP_bucket + 'ga/'
        print(str_command)
        sp.call(str_command, shell=True)

    del agent
    sys.exit()

agent = Agent(scenario, env, num_actions_per_product, state_real=False, obs_only=True, sample_val=False, op_trans_alloc='all')
agent.learn(load_weights=False, option_nn=nn_option, pred_fresh=True, imitation_learn=False, has_wh_const=False)

try:
    if UploadToGCP:
    # upload to gcloud
        # str_command = 'gsutil -m cp -r ' + agent.file + GCP_bucket + scenario +'/'
        # print(str_command)
        # sp.call(str_command, shell=True)

        str_command = 'gsutil -m cp -r ' + agent.file_val + GCP_bucket + 'experiments/'
        print(str_command)
        sp.call(str_command, shell=True)

        str_command = 'gsutil -m cp -r ' + agent.LOG + GCP_bucket + 'experiments/'
        print(str_command)
        sp.call(str_command, shell=True)

        # str_command = 'gsutil -m cp -r ' + agent.model_path + GCP_bucket + scenario +'/'
        # print(str_command)
        # sp.call(str_command, shell=True)

        # str_command = 'gsutil -m cp -r ' + agent.scm_best_path + GCP_bucket + scenario +'/'
        # print(str_command)
        # sp.call(str_command, shell=True)

        # str_command = 'gsutil -m cp -r ' + agent.scm_val_path + GCP_bucket + scenario +'/'
        # print(str_command)
        # sp.call(str_command, shell=True)

except Exception as e:
    if UseLineNotification == True:
        message = "Abnormal termination:" + agent.version + "_" + str(e)
        payload = {"message": message}
        r = requests.post(url ,headers = headers ,params=payload)
    print(e)
else:
    if UseLineNotification == True:
        message = "Normal termination:" + agent.version
        payload = {"message" :  message}
        r = requests.post(url ,headers = headers ,params=payload)
