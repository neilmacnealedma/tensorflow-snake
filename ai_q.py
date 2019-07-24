
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
    super(SnakeInputLayer, self).__init__(input_shape=(404,))
    self.board_input = keras.layers.Conv2D(filters=32, kernel_size=(4, 4), input_shape=(20, 20), activation='relu', dtype=tf.float32, name='board-input')
    self.pooling_layer = keras.layers.MaxPooling2D((2, 2), 1, name='pooling')
    self.flatten_layer = keras.layers.Flatten(name='flatten')
    self.direction_input = keras.layers.Dense(4, activation='relu', dtype=tf.float32, name='direction-input')

  def call(self, inputs):
    tensor = inputs
    print(tensor)
    board_tensor, direction_tensor = tf.split(tensor, (400, 4), 1)
    board_tensor = tf.reshape(board_tensor, (1, 20, 20, 1))
    board_tensor = self.board_input(board_tensor)
    board_tensor = self.pooling_layer(board_tensor)
    board_tensor = self.flatten_layer(board_tensor)
    direction_tensor = self.direction_input(direction_tensor)
    return tf.concat([board_tensor, direction_tensor], 1)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)

def create_model():
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
  custom_model_objects={SnakeInputLayer: SnakeInputLayer})

  dqn.compile(keras.optimizers.Adam(lr=1e-3), metrics=['mae'])

  return dqn

def mutate(good_model, bad_model):
  bad_model.set_weights(good_model.get_weights())

  weights = bad_model.trainable_weights
  for i in range(len(weights)):
    weights[i].assign(weights[i] + random.randint(-100, 100) * 0.002)

class AI:

  def __init__(self):
    self.models = []
    self.scores = []
    self.generation = 0

    for _ in range(POPULATION):
      self.models.append(create_model())
      self.scores.append(0)

  def train_once(self, board):
    print("Started training generation {}".format(self.generation))
    all_steps = []
    for i in range(POPULATION):
      board = game.Board(board.width, board.height, None)
      model = self.models[i]
      steps = []
      for _ in range(STEPS):
        input = board.create_tf_input()
        prediction = model.predict(np.array([input]))[0]
        direction = np.argmax(prediction)
        steps.append([input, direction])
        board.ai_control(direction)
        needs_die = board.update()
        if needs_die:
          break
      score = board.get_score()
      self.scores[i] = score
      all_steps.append(steps)
      print("Completed snake, which got score {}".format(score))

    i = 0
    for score in sorted(self.scores):
      if i == int(len(self.scores) / 2):
        median_score = score
        break
      i += 1

    print("Mutating models")
    good_models = []
    good_scores = []
    bad_models = []
    bad_scores = []
    for i in range(POPULATION):
      score = self.scores[i]
      model = self.models[i]
      if score >= median_score and len(good_models) < POPULATION / 2:
        good_models.append(model)
        good_scores.append(score)
      else:
        bad_models.append(model)
        bad_scores.append(score)
    print(good_models)
    print(bad_models)
    for i in range(len(good_models)):
      print("Mutating model " + str(i))
      mutate(good_models[i], bad_models[i])
    print("Finished mutations")

    self.models = good_models + bad_models
    self.scores = good_scores + bad_scores

    max_score = 0
    for i in range(POPULATION):
      score = self.scores[i]
      if score > max_score:
        max_score = score
        self.best_model_index = i
    self.models[self.best_model_index].save_model("models/gen-{}.model".format(self.generation))
    print("Finished training generation {}".format(self.generation))
    self.generation += 1

  def control(self, board):
    model = self.models[self.best_model_index]
    input = board.create_tf_input()
    prediction = model.predict(np.array([input]))[0]
    direction = np.argmax(prediction)
    board.ai_control(direction)
