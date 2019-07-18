
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import game

POPULATION = 100
STEPS = 200

def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(20, 20)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='ftrl',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

def train(board, generations):
  models = []
  steps = []
  scores = []

  for _ in range(POPULATION):
    models.append(create_model)
    scores.append(0)
    steps.append([])

  for _ in range(generations):
    tmp_board = game.Board(board.width, board.height, None)

    for i in range(POPULATION):
      model = models[i]
      for _ in range(STEPS):
        X = tmp_board.create_tf_input()
        prediction = model.predict(X)
