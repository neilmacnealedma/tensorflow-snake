
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import game
import numpy as np
import random

POPULATION = 10
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

def mutate(model):
  new_model = SnakeModel()

  new_model.predict(np.empty((1, 404)))
  new_model.set_weights(model.get_weights())

  weights = new_model.trainable_weights
  for i in range(len(weights)):
    weights[i].assign(weights[i] + random.randint(-10, 10) * 0.01)

  new_model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return new_model

class AI:

  def __init__(self):
    self.models = []
    self.scores = []

    for _ in range(POPULATION):
      self.models.append(create_model())
      self.scores.append(0)

  def train_once(self, board):
    print("Started training generation {}")
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
    succesful_models = []
    succesful_scores = []
    succesful_steps = []
    for i in range(POPULATION):
      score = self.scores[i]
      model = self.models[i]
      steps = all_steps[i]
      if score >= median_score and len(succesful_models) < POPULATION:
        new_model = mutate(model)
        succesful_models.append(model)
        succesful_models.append(new_model)
        succesful_steps.append(steps)
        succesful_steps.append([])
        succesful_scores.append(score)
        succesful_scores.append(0)
    print("Finished mutations")

    self.models = succesful_models
    self.scores = succesful_scores

    max_score = 0
    for i in range(POPULATION):
      score = succesful_scores[i]
      if score > max_score:
        max_score = score
        self.best_model_index = i
    print("Finished training generation {}")

  def control(self, board):
    model = self.models[self.best_model_index]
    input = board.create_tf_input()
    prediction = model.predict(np.array([input]))[0]
    direction = np.argmax(prediction)
    board.ai_control(direction)
