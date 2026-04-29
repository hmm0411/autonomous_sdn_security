GAMMA = 0.95 # giam discount factor -> tranh overestimation reward tuong lai
LR_DQN = 1e-4 # LR = 5e-4 # learning rate cao -> nhanh chong gap local minima, tuy nhien co the khong on dinh
BATCH_SIZE = 128 # batch size lon -> on dinh hon, nhung can du bo nho, smoothing
BUFFER_SIZE = 100000
TARGET_UPDATE = 500 # target update nho -> on dinh hon, nhung co the cham hon trong viec gap local minima

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 4000

STATE_DIM = 9
ACTION_DIM = 5

MAX_EPISODES = 5000
MAX_STEPS = 1000
# MAX_EPISODES = 50
# MAX_STEPS = 200
WINDOW_SIZE = 50

clip_eps = 0.15
entropy_coef = 0.02
value_coef = 0.5
LR_PPO = 2e-4
