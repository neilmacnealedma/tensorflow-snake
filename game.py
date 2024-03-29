
import pygame
import random
import numpy as np
import collections
import math

class Board:

  def __init__(self, width, height, display):
    self.width = width
    self.height = height
    self.snake = Snake(int(width / 2), int(height / 2), self)
    self.apple_x = random.randrange(0, width)
    self.apple_y = random.randrange(0, height)
    self.tile_width = 20
    self.tile_height = 20
    self.display = display
    self.num_apples = 0
    self.time_survived = 0
    self.time_since_last_apple = 0
    self.got_apple_flag = False
    self.reward = 1

  def reset(self):
    self.snake = Snake(int(self.width / 2), int(self.height / 2), self)
    self.apple_x = random.randrange(0, self.width)
    self.apple_y = random.randrange(0, self.height)
    self.num_apples = 0
    self.time_survived = 0
    self.time_since_last_apple = 0
    self.got_apple_flag = False

  def manual_control(self):
    keys = pygame.key.get_pressed()
    new_dir = -1
    if keys[pygame.K_w]:
      new_dir = 0
    if keys[pygame.K_d]:
      new_dir = 1
    if keys[pygame.K_s]:
      new_dir = 2
    if keys[pygame.K_a]:
      new_dir = 3
    if new_dir != -1:
      self.ai_control(new_dir)

  def ai_control(self, direction):
    if self.snake.direction == (direction + 2) % 4:
      self.reward = -50
    else:
      # dist to apple, where closer is higher score
      self.reward = math.sqrt((self.snake.head_x - self.apple_x) ** 2 + (self.snake.head_y - self.apple_y) ** 2) * -1 + 30
      self.snake.direction = direction

  def update(self):
    needs_die = self.snake.update()
    self.time_survived += 1
    self.time_since_last_apple += 1
    if self.time_since_last_apple > 400:
      return True, 0
    return_val = needs_die or self.snake.head_x < 0 or self.snake.head_x > self.width - 1 or self.snake.head_y < 0 or self.snake.head_y > self.height - 1
    reward = self.reward
    if self.got_apple_flag:
      reward += 100
      self.got_apple_flag = False
    return return_val, reward

  def draw(self):
    self.display.fill((0, 0, 0))
    self.snake.draw()
    self.draw_tile((self.apple_x, self.apple_y), (255, 0, 0))

  def draw_tile(self, pos, color):
    self.display.fill(color, (pos[0] * self.tile_width, pos[1] * self.tile_height, self.tile_width, self.tile_height))

  def new_apple(self):
    self.num_apples += 1
    self.apple_x = random.randrange(0, self.width)
    self.apple_y = random.randrange(0, self.height)
    self.time_since_last_apple = 0
    self.got_apple_flag = True

  def create_tf_input(self):
    arr = np.zeros((self.width * self.height + 4), dtype=np.float32)
    for seg in self.snake.segments:
      if seg.x < self.width and seg.y < self.height and seg.x >= 0 and seg.y >= 0:
        arr[seg.y * self.width + seg.x] = 0.5
    arr[self.apple_y * self.width + self.apple_x] = 1
    arr[self.width * self.height + self.snake.direction] = 1
    return arr

  def get_score(self):
    return self.time_survived + self.num_apples * 100

class Snake:

  def __init__(self, x, y, board):
    self.segments = collections.deque()
    self.segments.append(Segment(x, y))
    self.segments.append(Segment(x - 1, y))
    self.segments.append(Segment(x - 2, y))
    self.segments.append(Segment(x - 3, y))
    self.segments.append(Segment(x - 4, y))
    self.head_x = x
    self.head_y = y
    self.direction = 1
    self.board = board

  def update(self):
    if self.direction == 0:
      self.head_y -= 1
    elif self.direction == 1:
      self.head_x += 1
    elif self.direction == 2:
      self.head_y += 1
    elif self.direction == 3:
      self.head_x -= 1
    else:
      raise "WTF direction wrong"
    for seg in self.segments:
      if seg.x == self.head_x and seg.y == self.head_y:
        return True
    self.segments.appendleft(Segment(self.head_x, self.head_y))
    if self.board.apple_x == self.head_x and self.board.apple_y == self.head_y:
      self.board.new_apple()
    else:
      self.segments.pop()
    return False

  def draw(self):
    for seg in self.segments:
      self.board.draw_tile((seg.x, seg.y), (255, 255, 255))

class Segment:

  def __init__(self, x, y):
    self.x = x
    self.y = y
