import torch

DEBUG_PRINT = False
DEBUG_GA = False

ESTIMATE_OTHERS = True

# Learning episodes/steps
NUM_EPISODES = 4000
MAX_STEPS = 200
TARGET_UPDATE_FREQUENCY = 10
NUM_SAMPLING = 300
FREQ_VALIDATION = 50
FREQ_LEARN = 1
SAMPLING_LEARN = False
BRANCH_SWITCHING_FREQ = 10
USE_INDEPENDENT_LEARN = False
CLEAR_MEMORY = False
HYSTERETIC_BETA = 0.4
ENABLE_GRADIENT_CLIPPING = False
LEVEL_K = 5
E_GREEDY_JOINT = True
SELECT_OPTIMAL = False
DECAY_FREQ = 30
USE_FORECAST = False
NUM_VALIDATION = 12
ADD_INV_STATE = True
ENABLE_MULTI = False
ENABLE_MULTI_Q = False
BOLTZMAN_TEMP_DECAY = 0.98
BOLTZMAN_TEMP_MIN = 10

assert not ENABLE_MULTI, 'multi processing is under construction'
assert not ENABLE_MULTI_Q, 'multi processing is under construction'

# NN Architecture
FUNCTION_APPROX_LINEAR = False
NN_LAYER = 4
assert NN_LAYER == 4, 'nn layers should be 4'
assert not FUNCTION_APPROX_LINEAR, 'FUNCTION_APPROX_LINEAR should be False'

if USE_FORECAST:
    n_fc1, n_fc2, n_fc3 = 64, 32, 32
else:
    n_fc1, n_fc2, n_fc3 = 64, 32, 32
    # n_fc1, n_fc2, n_fc3 = 16, 8, 8

# scs GA parameters
CXPB, MUTPB, NGEN = 0.5, 0.2, 100  # 交差確率、突然変異確率、進化計算のループ回数
NGEN = 5 if DEBUG_GA else 100

NUM_POPULATION = 2 if DEBUG_GA else 50
NUM_SAMPLE_GA = 6
USE_LT_INV = True if USE_FORECAST else False

# Safe Factor
k = 3.1

# Learning setting
GAMMA = 0.995
BATCH_SIZE = 32
# CAPACITY = 10000
# CAPACITY = 180 * BRANCH_SWITCHING_FREQ * 10
# CAPACITY = 180 * TARGET_UPDATE_FREQUENCY
CAPACITY = 10000
TD_ERROR_EPSILON = 0.0001  # 誤差に加えるバイアス

# 学習率
LEARNING_RATE = 0.0001
LR_DECAY = 1
CLIP = 0.25 # Gradient Clipping
START_EPSILON = 0.1
FINAL_EPSILON = 0.0001
EPSILON_CONSTANT = False

NSTEP = 4  # for multi step learning　# 基本的には，　LT +1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_setting_ga(logger):
    logger.info(f' -- GA PARAM .. USE_LT_INV:{USE_LT_INV}')


def print_setting_learn(logger):
    logger.info(f' --- Parameter Settings --- ')
    logger.info(f'ESTIMATE_OTHERS:{ESTIMATE_OTHERS}')
    logger.info(f'TARGET_UPDATE_FREQUENCY:{TARGET_UPDATE_FREQUENCY}')
    logger.info(f'USE_INDEPENDENT_LEARN:{USE_INDEPENDENT_LEARN}')
    logger.info(f'CLEAR_MEMORY:{CLEAR_MEMORY}')
    logger.info(f'HYSTERETIC_BETA:{HYSTERETIC_BETA}')
    logger.info(f'ENABLE_GRADIENT_CLIPPING:{ENABLE_GRADIENT_CLIPPING}')
    logger.info(f'CAPACITY:{CAPACITY}')
    logger.info(f'NUM_EPISODES:{NUM_EPISODES}')
    logger.info(f'LEVEL_K: {LEVEL_K}')
    logger.info(f'E_GREEDY_JOINT: {E_GREEDY_JOINT}')
    logger.info(f'SELECT_OPTIMAL: {SELECT_OPTIMAL}')
    logger.info(f'DECAY_FREQ: {DECAY_FREQ}')
    logger.info(f'EPSILON_CONSTANT: {EPSILON_CONSTANT}')
    logger.info(f'USE_FORECAST: {USE_FORECAST}')
    logger.info(f'ADD_INV_STATE: {ADD_INV_STATE}')
    logger.info(f'BOLTZMAN_TEMP_DECAY: {BOLTZMAN_TEMP_DECAY}')
    logger.info(f'BOLTZMAN_TEMP_MIN: {BOLTZMAN_TEMP_MIN}')

    logger.info(f' --- Function Approximation --- ')
    logger.info(f'FUNCTION_APPROX_LINEAR: {FUNCTION_APPROX_LINEAR}')
    logger.info(f'NN_LAYER: {NN_LAYER}')
