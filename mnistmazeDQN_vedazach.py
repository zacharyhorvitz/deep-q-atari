import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import gzip
# from google.colab import drive

#credit to https://github.com/gsurma

# from scores.score_logger import ScoreLogger
GAMMA = 0.9
LEARNING_RATE = 0.01 #0.001

MEMORY_SIZE = 100000000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0
#EXPLORATION_DECAY = 0.999 #0.995


class DQNetwork():
  def __init__(self, actions, input_shape,
               minibatch_size=32,
               learning_rate=0.00025,
               discount_factor=0.99,
               dropout_prob=0.1,
               load_path=None,
               logger=None):

      # Parameters
      self.actions = actions  # Size of the network output
      self.discount_factor = discount_factor  # Discount factor of the MDP
      self.minibatch_size = minibatch_size  # Size of the training batches
      self.learning_rate = learning_rate  # Learning rate
      self.dropout_prob = dropout_prob  # Probability of dropout
      self.logger = logger
      self.training_history_csv = 'training_history.csv'

      if self.logger is not None:
          self.logger.to_csv(self.training_history_csv, 'Loss,Accuracy')

      # Deep Q Network as defined in the DeepMind article on Nature
      # Ordering channels first: (samples, channels, rows, cols)

 

      self.model = Sequential()

     # First convolutional layer
      # self.model.add(Conv2D(32, 8, strides=(4, 4),
      #                       padding='valid',
      #                       activation='relu',
      #                       input_shape=input_shape,
      #                       data_format='channels_first'))

      # Second convolutional layer
      self.model.add(Conv2D(64, 4, strides=(3, 3),
                            padding='valid',
                            activation='relu',
                            input_shape=input_shape,
                            data_format='channels_first'))

      # Third convolutional layer
      self.model.add(Conv2D(64, 3, strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            input_shape=input_shape,
                            data_format='channels_first'))

      # Flatten the convolution output
      self.model.add(Flatten())

      # First dense layer
      self.model.add(Dense(512, input_shape=input_shape, activation='relu'))

      # Output layer
      self.model.add(Dense(self.actions))

      # Load the network weights from saved model
      if load_path is not None:
          self.load(load_path)

      self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

      self.model.compile(loss='mean_squared_error',
                         optimizer='rmsprop',
                         metrics=['accuracy'])


     
# class Model(tf.keras.Model):
#     def __init__(self,observation_space,action_space):
#         super(Model, self).__init__()
#         self.optimizer = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#         self.d1 = Dense(24, input_shape=(observation_space,), activation="relu")
#         self.d2 = Dense(24, activation="relu")
#         self.d3 = Dense(action_space, activation="linear")
        
#     @tf.function
#     def call(self, states):
#         return self.d3(self.d2(self.d1(states)))

#     def loss(self,inputs, labels):
#         return tf.keras.losses.mean_squared_error(inputs,labels)


  def loss(self,inputs, labels):
           return tf.keras.losses.mean_squared_error(inputs,labels)



class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        
        self.observation_space = observation_space
        self.action_space = action_space

        self.action_space = action_space
        self.memory = []
        self.EBU = False #True
        self.beta = 1 #0.5
        self.episode_memory = []

        self.network = DQNetwork(action_space,[1,28,28]) #Model(observation_space,action_space)
        self.old_network  = DQNetwork(action_space,[1,28,28]) # Model(observation_space,action_space)
        self.network.model.build()
        self.old_network.model.build()
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def add_episode(self):
        self.episode_memory.append([])

    def episode_remember(self, state, action, reward, next_state, done):
        self.episode_memory[-1].append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values =  self.network.model(state).numpy()
        return np.argmax(q_values[0])

    def experience_replay(self):
       
          episode = random.sample(self.episode_memory, 1)[0]
          T=len(episode)
          episode = np.array(episode)
          
          actions = episode[:,1]
          next_rewards = episode[:,2]
          next_states = np.squeeze(np.stack(episode[:,3]),axis=1)

          print(next_states.shape)

          cur_states =  np.squeeze(np.stack(episode[:,0]),axis=1)
          #print(len(next_states),len(next_states[0]))
          q_tilde_temp = self.old_network.model(tf.convert_to_tensor(next_states,tf.float32)) #self.old_model(next_states)
          q_tilde = q_tilde_temp.numpy()

          y = np.zeros(T)
          y[-1] = next_rewards[-1]

          with tf.GradientTape() as tape:
            if not self.EBU:
              # exit()
              q_values = self.network.model(cur_states)
              QA_values = tf.gather(tf.reshape(q_values,[-1]),tf.convert_to_tensor(np.arange(len(actions))+actions, dtype=tf.int32))

              next_QA_values = np.max(q_tilde,axis=1)
              # print("maxes",next_QA_values)
              # print("r",next_rewards)
              # losses = self.network.loss(QA_values, np.array(next_rewards) + GAMMA * next_QA_values)
              discounted_rewards =np.array(next_rewards,dtype=np.float32)+ GAMMA * next_QA_values
              discounted_rewards = tf.convert_to_tensor(discounted_rewards,dtype=tf.float32)
              # print(discounted_rewards)
              losses = self.network.loss(QA_values,discounted_rewards)

            else:
              for k in range(T-2,0,-1):
                cur_action = actions[k]
                q_tilde[k][cur_action] = self.beta * y[k+1] + (1-self.beta) * q_tilde[k][cur_action]
                y[k] = next_rewards[k] + GAMMA * np.max(q_tilde[k,])
              q_values = self.network.model(cur_states)
              QA_values = tf.gather(tf.reshape(q_values,[-1]),tf.convert_to_tensor(np.arange(len(actions))+actions, dtype=tf.int32))


              losses = self.network.loss(QA_values, y)

          gradients = tape.gradient(losses, self.network.model.trainable_variables)
          self.network.optimizer.apply_gradients(zip(gradients, self.network.model.trainable_variables))

          # self.exploration_rate *= EXPLORATION_DECAY
          # self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


# from tensorflow.examples.tutorials.mnist import input_data

class Maze_rand_stochastic(object):
  class observation_space(object):
    def __init__(self):
      self.shape = [28, 28, 2]

  class action_space(object):
    def __init__(self):
      self.n = 4

  class spec(object):
    def __init__(self):
      self.timestep_limit = 1000

  def __init__(self, height = 10, width = 10, wall = np.zeros((10,10)), opt_len = 10 + 10 - 2):
    if height < 0 or height > 10 or width < 0 or width > 10:
      raise Exception('Height and width must be in [0, 10].')

    self.height = height
    self.width = width
    self.wall = wall
    self.opt_len = opt_len
    
    self.observation_space = Maze_rand_stochastic.observation_space()
    self.action_space = Maze_rand_stochastic.action_space()
    self.spec = Maze_rand_stochastic.spec()

    self.images,self.labels = get_data("MNIST/train-images-idx3-ubyte.gz","MNIST/train-labels-idx1-ubyte.gz",55000)
    
    #print(self.images.shape,self.labels.shape)
    # self.Dataset = input_data.read_data_sets(path)
    # self.images = self.Dataset.train.images
    # self.labels = self.Dataset.train.labels

    def _get_idx(labels, idx_len):
      idx = []
      for i in range(10):
       # print(np.where(labels == i)[0][:idx_len])
        idx.append(np.where(labels == i)[0][:idx_len])
      # print(np.array(idx))
      # for x in idx:
      #   print(len(x))
      # return idx
      return np.stack(idx)
    self.idx = _get_idx(self.labels, 4963) #4987)
    # print(self.idx.shape)

  def _get_observation(self, state):
    idxs = self.idx[(state, np.random.randint(self.idx.shape[1], size = 2))]
    return np.transpose(self.images[idxs, :].reshape(-1, 28, 28), (1, 2, 0))

  def reset(self):
    self.state = np.array([0, 0])
    return self._get_observation(self.state)

  def step(self, action):
    if 1: 
      #print self.state
      temp_state = np.array([self.state[0],self.state[1]])
      #print '=' * 10
      #print temp_state, self.state
      stochasticity = np.random.rand()
      
      if action == 0:
        if stochasticity < 0.8:
          temp_state[0] += 1
        elif stochasticity < 0.9:
          temp_state[1] += 1
        else:
          temp_state[1] -= 1
          
      elif action == 1:
        if stochasticity < 0.8:
          temp_state[0] -= 1
        elif stochasticity < 0.9:
          temp_state[1] += 1
        else:
          temp_state[1] -= 1
        
      elif action == 2:
        if stochasticity < 0.8:
          temp_state[1] -= 1
        elif stochasticity < 0.9:
          temp_state[0] += 1
        else:
          temp_state[0] -= 1
        
      elif action == 3:
        if stochasticity < 0.8:
          temp_state[1] += 1
        elif stochasticity < 0.9:
          temp_state[0] += 1
        else:
          temp_state[0] -= 1
        
      else:
        raise ValueError('Action should be one of 0, 1, 2, 3.')
    #print temp_state, self.state
    temp_state = np.clip(temp_state, 0, [self.height - 1, self.width - 1]) # if the agent crashes into outer wall
    #print temp_state, self.state
    
    if self.wall[temp_state[0],temp_state[1]] == 1: #if the agent crashes into inner wall
      temp_state = self.state
      
    if np.array_equal(self.state, temp_state):
      reward = -1.0; done = False
    else:
      self.state = temp_state
      reward = 0.0; done = False

    if self.state[0] == self.height - 1 and self.state[1] == self.width - 1:
      reward = 1000.0
      done = True
    #print action, self.state, reward
    return self._get_observation(self.state), reward, done, None


def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files, 
    normalizes the inputs, and returns (NumPy array of inputs, NumPy 
    array of labels). Read the data of the file into a buffer and use 
    np.frombuffer to turn the data into a NumPy array. Keep in mind that 
    each file has a header of a certain size. This method should be called
    within the main function of the model.py file to get BOTH the train and
    test data. If you change this method and/or write up separate methods for 
    both train and test data, we will deduct points.
    :param inputs_file_path: file path for inputs, something like 
    'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, something like 
    'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather 
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs as float32 and labels as int8
    """

    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_examples)
        data = np.frombuffer(buf, dtype=np.uint8) / 255.0
        inputs = data.reshape(num_examples, 784)

    with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_examples)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int8)


def mnist_maze():
    # path = 'MNIST'

    env = Maze_rand_stochastic(height = 10, width = 10, wall = np.zeros((10,10)), opt_len = 10 + 10 - 2)

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    dqn_solver.old_network.model.set_weights(dqn_solver.network.model.get_weights()) 

    run_lengths = []

    run = 0
    total_steps = 0
    while total_steps < 200000:
        run += 1
        state = env.reset()
        dqn_solver.add_episode()
        state = np.reshape(np.transpose(state,[2,0,1]),[-1,2,28,28])#np.reshape(state, [-1, observation_space])
        step = 0
        while step < 200:
            total_steps+=1
            step += 1
            dqn_solver.exploration_rate = (1/(200000**2))*(total_steps-200000)**2
           # self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

            #print(step)
            action = dqn_solver.act(tf.convert_to_tensor(state,dtype=tf.float32))
            state_next, reward, terminal, info = env.step(action)
       #     print(state_next.shape)
            state_next = np.reshape(np.transpose(state_next,[2,0,1]),[-1,2,28,28])
            reward = reward if not terminal else -reward
          #  state_next = np.reshape(state_next, [-1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            dqn_solver.episode_remember(tf.convert_to_tensor(state,dtype=tf.float32), action, reward, tf.convert_to_tensor(state_next,dtype=tf.float32), terminal)
            # print(self.episode_memory)
            # exit()
            dqn_solver.experience_replay()
            state = state_next

            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                break
        print(step)
        run_lengths.append(step)
        dqn_solver.old_network.model.set_weights(dqn_solver.network.model.get_weights()) 
        if len(run_lengths[-10:]) >= 10:
          print(sum(run_lengths[-10:])/(len(run_lengths[-10:])))

if __name__ == "__main__":
    mnist_maze()
