
import pygame
import game
import ai as ai_module
import ai_q as ai_q_module

pygame.init()

display = pygame.display.set_mode((800,600))
pygame.display.set_caption('Snake')
clock = pygame.time.Clock()

def manual_game():
  crashed = False
  frame = 0
  while not crashed:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        crashed = True

    board.manual_control()
    if frame >= 20:
      if board.update():
        crashed = True
      frame = 0
    frame += 1
    board.draw()
    pygame.display.update()
    clock.tick(60)

def ai_game():
  ai = ai_module.AI()

  needs_train = True
  while needs_train:
    crashed = False
    frame = 0
    board = game.Board(20, 20, display)
    ai.train_once(board)
    needs_train = False
    while not crashed:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          crashed = True

      if frame >= 5:
        ai.control(board)
        if board.update():
          crashed = True
          needs_train = True
        frame = 0
      frame += 1
      board.draw()
      pygame.display.update()
      clock.tick(60)

def ai_q_game():
  board = game.Board(20, 20, display)
  ai = ai_q_module.AI(board)

  while True:
    ai.train(10000)
    ai.show_game()

ai_q_game()
