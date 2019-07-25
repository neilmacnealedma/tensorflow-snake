
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import keras
import game
import numpy as np
import random
import rl
import rl.agents
import rl.memory
import rl.policy

POPULATION = 50
STEPS = 200

class SnakeInputLayer(keras.layers.Layer):

  def __init__(self):
    self.output_dim = 8196
    super(SnakeInputLayer, self).__init__(input_shape=(1, 404))
    self.board_input = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), input_shape=(20, 20), activation='relu', dtype=tf.float32, name='board-input')
    self.pooling_layer = keras.layers.MaxPooling2D((2, 2), 1, name='pooling')
    self.flatten_layer = keras.layers.Flatten(name='flatten')
    self.direction_input = keras.layers.Dense(4, activation='relu', dtype=tf.float32, name='direction-input')

  def call(self, inputs):
    tensor = inputs
    tensor = tf.reshape(tensor, [-1, 404])
    board_tensor, direction_tensor = tf.split(tensor, (400, 4), 1)
    board_tensor = tf.reshape(board_tensor, (1, 20, 20, 1))
    board_tensor = self.board_input(board_tensor)
    board_tensor = self.pooling_layer(board_tensor)
    board_tensor = self.flatten_layer(board_tensor)
    direction_tensor = self.direction_input(direction_tensor)
    return tf.concat([board_tensor, direction_tensor], 1)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)

  def get_config(self):
    return {}

def create_model():
  print("Creating new model")
  model = keras.models.Sequential([
    SnakeInputLayer(),
    keras.layers.Dense(512, activation='relu', name='hidden-512'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation='relu', name='hidden-256'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4, activation='softmax', name='output')
  ])
  memory = rl.memory.SequentialMemory(limit=50000, window_length=1)
  policy = rl.policy.BoltzmannQPolicy()
  dqn = rl.agents.dqn.DQNAgent(model=model,
  memory=memory,
  target_model_update=1e-2,
  policy=policy,
  nb_actions=4,
  custom_model_objects={'SnakeInputLayer': SnakeInputLayer})

  dqn.compile(keras.optimizers.Adam(lr=1e-3), metrics=['mae'])

  return dqn

class BoardEnv():

  def __init__(self, board):
    self.board = board

  def reset(self):
    self.board.reset()
    return self.create_obs()

  def create_obs(self):
    return self.board.create_tf_input()

  def step(self, action):
    self.board.ai_control(action)
    done, reward = self.board.update()
    return self.create_obs(), reward, done, {}

  def render(self, mode):
    print("RENDERING YEET")

class AI:
  def __init__(self):
    self.model = create_model()

  def train(self, board, steps):
    self.model.fit(BoardEnv(board), steps)

  def show_game(self, board):
    self.model.test(BoardEnv(board), nb_episodes=5, visualize=True)
