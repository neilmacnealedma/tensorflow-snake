
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import game
import numpy as np

POPULATION = 10
STEPS = 200

def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20 * 20 + 4, activation='relu', dtype=tf.float64),
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
    self.all_steps = []
    self.scores = []

    for _ in range(POPULATION):
      self.models.append(create_model())
      self.scores.append(0)
      self.all_steps.append([])

  def train_once(self, board, generations):
    for i in range(POPULATION):
      board = game.Board(board.width, board.height, None)
      model = self.models[i]
      steps = self.all_steps[i]
      for _ in range(STEPS):
        input = board.create_tf_input()
        prediction = model.predict(np.array([input]))[0]
        direction = np.argmax(prediction)
        steps.append([input, direction])
        board.ai_control(direction)
        needs_die = board.update()
        if needs_die:
          break
      scores[i] = board.get_score()

    succesful_models = []

  def control(board):
    pass
