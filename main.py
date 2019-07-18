
import pygame
import game
import ai

pygame.init()

display = pygame.display.set_mode((800,600))
pygame.display.set_caption('Snake')
clock = pygame.time.Clock()

board = game.Board(20, 20, display)

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
  ai.train(board, generations=5)

  crashed = False
  frame = 0
  while not crashed:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        crashed = True

    if frame >= 20:
      ai.control(board)
      if board.update():
        crashed = True
      frame = 0
    frame += 1
    board.draw()
    pygame.display.update()
    clock.tick(60)

ai_game()
