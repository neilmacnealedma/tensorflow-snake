
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import game
import numpy as np
import random

POPULATION = 50
STEPS = 200

class SnakeModel(tf.keras.Model):

  def __init__(self):
    super(SnakeModel, self).__init__()
    self.board_input = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), input_shape=(20, 20), activation='relu', dtype=tf.float32, name='board-input')
    self.flatten_layer = tf.keras.layers.Flatten(name='flatten')
    self.direction_input = tf.keras.layers.Dense(4, activation='relu', dtype=tf.float32, name='direction-input')
    self.hidden_layers = [
      tf.keras.layers.Dense(512, activation='relu', name='hidden-512'),
      tf.keras.layers.Dropout(0.8),
      tf.keras.layers.Dense(256, activation='relu', name='hidden-256'),
      tf.keras.layers.Dropout(0.8),
      tf.keras.layers.Dense(128, activation='relu', name='hidden-128'),
      tf.keras.layers.Dropout(0.8),
      tf.keras.layers.Dense(4, activation='softmax', name='output')
    ]

  def call(self, inputs):
    tensor = inputs
    board_tensor, direction_tensor = tf.split(tensor, (400, 4), 1)
    board_tensor = tf.reshape(board_tensor, (1, 20, 20, 1))
    board_tensor = self.board_input(board_tensor)
    board_tensor = self.flatten_layer(board_tensor)
    direction_tensor = self.direction_input(direction_tensor)
    tensor = tf.concat([board_tensor, direction_tensor], 1)
    for layer in self.hidden_layers:
      tensor = layer(tensor)
    return tensor

def create_model():
  model = SnakeModel()

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

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
    print("Finished training generation {}".format(self.generation))
    self.generation += 1

  def control(self, board):
    model = self.models[self.best_model_index]
    input = board.create_tf_input()
    prediction = model.predict(np.array([input]))[0]
    direction = np.argmax(prediction)
    board.ai_control(direction)
