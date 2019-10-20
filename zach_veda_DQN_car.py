import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

#credit to https://github.com/gsurma

# from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class Model(tf.keras.Model):
    def __init__(self,observation_space,action_space):
        super(Model, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.d1 = Dense(24, input_shape=(observation_space,), activation="relu")
        self.d2 = Dense(24, activation="relu")
        self.d3 = Dense(action_space, activation="linear")


    def call(self, states):
        return self.d3(self.d2(self.d1(states)))

    def loss(self,inputs, labels):
        return tf.keras.losses.mean_squared_error(inputs,labels)


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = []
        self.EBU = True
        self.beta = 0.5
        self.episode_memory = []

        # self.model = Sequential()


        self.model = Model(observation_space,action_space)
        # self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        # self.model.add(Dense(24, activation="relu"))
        # self.model.add(Dense(self.action_space, activation="linear"))
        # self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def add_episode(self):
        self.episode_memory.append([])

    def episode_remember(self, state, action, reward, next_state, done):
        self.episode_memory[-1].append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self,q_hat):
        if not self.EBU:
            exit()
            # if len(self.memory) < BATCH_SIZE:
            #     return

            # batch = random.sample(self.memory, BATCH_SIZE)
            # for state, action, reward, state_next, terminal in batch:
            #     q_update = reward
            #     if not terminal:
            #         q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            #     q_values = self.model.predict(state)
            #     q_values[0][action] = q_update
            #     self.model.fit(state, q_values, verbose=0)
            # self.exploration_rate *= EXPLORATION_DECAY
            # self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        else:
            episode = random.sample(self.episode_memory, 1)[0]
            T=len(episode)

            episode = np.array(episode)

            #print(episode)
            next_actions = episode[:,1]

            next_rewards = episode[:,2]
            next_states = np.stack(episode[:,3])[0]
            #print(next_states)

            cur_states =  np.stack(episode[:,0])[0]

            q_tilde = q_hat.model(next_states)
            y = np.zeros(T)
            y[-1] = next_rewards[-1]

            for k in range(T-2,0,-1):
                cur_action = next_actions[k+1]
                q_tilde[k][cur_action]= self.beta * y[k+1] + (1-self.beta) * q_tilde[k][cur_action]
                y[k] = next_rewards[k] + GAMMA * np.max(q_tilde[k,:])
            with tf.GradientTape() as tape:
                q_values = self.model(cur_states)
                print(next_actions)
                QA_values = tf.gather(q_values,tf.convert_to_tensor(next_actions, dtype=tf.int32))
                losses = self.model.loss(QA_values, y)

            gradients = tape.gradient(losses, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # print(q_values)
            # print(QA_values)
            # exit()

            # self.model.fit(QA_values, y, verbose=0)
            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)




def cartpole():
    env = gym.make(ENV_NAME)
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    dqn_solver_old = DQNSolver(observation_space, action_space)

    run = 0
    while True:
        run += 1
        state = env.reset()
        dqn_solver.add_episode()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            dqn_solver.episode_remember(state, action, reward, state_next, terminal)
     
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                # score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay(dqn_solver_old)
        
        dqn_solver_old.model.set_weights(dqn_solver.model.get_weights()) 
if __name__ == "__main__":
    cartpole()
