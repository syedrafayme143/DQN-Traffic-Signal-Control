import numpy as np
from utils import *
import os
from utils import import_train_configuration
from shutil import copyfile
from collections import deque
import matplotlib.pyplot as plt

# Load configuration
config = import_train_configuration(config_file='training_settings.ini')
if not config['is_train']:
    config = import_test_configuration(config_file_path=config['test_model_path_name'])
print(config)

# Initialize environment
if config['is_train']:
    from Environment.SUMO_train import SUMO
    env = SUMO()

print('State shape: ', env.num_states)
print('Number of actions: ', env.action_space.n)

# Initialize agent
if config['agent_type'] == 'DQN':
    from dqn_agent import DQNAgent
    agent = DQNAgent(env.num_states, env.action_space.n, config['hidden_dim'], config['fixed_action_space'], env.TL_list,
                     config['memory_size_max'], config['batch_size'], config['gamma'], config['tau'],
                     config['learning_rate'], config['target_update'])

# Create training or testing result folder
if config['is_train']:
    path = set_train_path(config['models_path_name'])
    print('Training results will be saved in:', path)
    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
else:
    test_path, plot_path = set_test_path(config['test_model_path_name'])
    print('Test results will be saved in:', plot_path)

# DQN Reinforcement Learning Training Loop
def DQNRL(n_episodes=config['total_episodes'], max_t=config['max_steps'] + 1000,
          eps_start=config['eps_start'], eps_end=config['eps_end'], eps_decay=config['eps_decay'],
          single_agent=config['single_agent']):

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    episode_start = 1 if config['is_train'] else n_episodes + 1
    episode_end = n_episodes + 1 if config['is_train'] else n_episodes + 11

    for i_episode in range(episode_start, episode_end):
        env._TrafficGen.generate_routefile(model_path=env.model_path, model_id=env.model_id, seed=i_episode)
        state = env.reset()
        score = 0

        for t in range(max_t):
            # TASK 4: Correct RL loop
            action = agent.act(np.array(state), eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += sum(list(reward.values()))
            if done:
                break

        scores_window.append(score)
        scores.append(score)

        # TASK 5: Epsilon decay
        eps = max(eps_end, eps_decay * eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    env.reset()
    env.close()
    return scores

# TASK 7: Train DQN agent
if 'DQN' in config['agent_type']:
    scores = DQNRL()

# Plot results
import plotly.express as px
fig = px.line(x=np.arange(len(scores)), y=scores)
fig.show()
fig.write_html(os.path.join(path if config['is_train'] else plot_path, 'training_reward.html'))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(os.path.join(path if config['is_train'] else plot_path, 'training_reward.png'))

