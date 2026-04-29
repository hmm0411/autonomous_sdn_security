from prometheus_client import start_http_server, Gauge

reward_g = Gauge('rl_reward', 'Reward')
model_g = Gauge('model_selected', '0=DQN,1=PPO')

start_http_server(9000)

def update_metrics(state, reward, model):
    try:
        reward_g.set(reward)

        if model == "DQN":
            model_g.set(0)
        else:
            model_g.set(1)
    except:
        pass