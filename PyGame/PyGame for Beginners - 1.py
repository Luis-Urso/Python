###
# PyGame Tutorial for Beginners - 1
# based on: https://medium.com/iothincvit/pygame-for-beginners-234da7d3c56f


import pygame

pygame.init()

screen = pygame.display.set_mode((500,500))

done=False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done= True
            
    pygame.display.flip()
    