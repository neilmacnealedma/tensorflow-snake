
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import game
import numpy as np

POPULATION = 20
STEPS = 200

def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(20, 20) + 4, activation='relu', dtype=tf.float32),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(4, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

class AI:

  def __init__(self):
    self.models = []
    self.scores = []

    for _ in range(POPULATION):
      self.models.append(create_model())
      self.scores.append(0)

  def train_once(self, board):
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

    succesful_models = []
    succesful_scores = []
    succesful_steps = []
    for i in range(POPULATION):
      score = self.scores[i]
      model = self.models[i]
      steps = all_steps[i]
      if score >= median_score and len(succesful_models) < POPULATION:
        succesful_models.append(model)
        succesful_models.append(model)
        succesful_steps.append(steps)
        succesful_steps.append(steps)
        succesful_scores.append(score)
        succesful_scores.append(score)

    for i in range(POPULATION):
      score = succesful_scores[i]
      model = succesful_models[i]
      steps = succesful_steps[i]
      X = np.empty((len(steps), 20 * 20 + 4))
      Y = np.empty((len(steps), 1))
      i = 0
      for step in steps:
        X[i] = step[0]
        Y[i] = [step[1]]
        i += 1
      model.fit(X, Y)

  def control(self, board):
    model = self.models[self.best_model_index]
    input = board.create_tf_input()
    prediction = model.predict(np.array([input]))[0]
    direction = np.argmax(prediction)
    board.ai_control(direction)
