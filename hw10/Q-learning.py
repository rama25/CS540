import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)


    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        upcoming_st = env.reset()

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):

            current_st = upcoming_st

            if random.uniform(0, 1) < EPSILON:
                ac = env.action_space.sample()
            else:
                ac = np.argmax(
                    np.array([Q_table[(current_st, i)] for i in range(env.action_space.n)]))

            upcoming_st, episode_reward, done, information = env.step(ac)

            if done:
                q_fin = episode_reward - Q_table[current_st, ac]
                Q_table[current_st, ac] += LEARNING_RATE * q_fin
            else:
                q_maximum = episode_reward + np.max(np.array([Q_table[(upcoming_st, i)] for i in range(env.action_space.n)])) \
                        - Q_table[current_st, ac]
                Q_table[current_st, ac] += LEARNING_RATE * q_maximum

        EPSILON *= EPSILON_DECAY

        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 

        
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################