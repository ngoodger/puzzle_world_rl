import pygame
import pymunk
import random
import pymunk.pygame_util
import math
from pygame.locals import *
from pygame.color import *
from random import random
import time
import numpy as np 

PUZZLE_SIZE = 90
PIECE_COUNT = 2
PIECE_SIZE = 10
FRAME_SIZE = 64 
VELOCITY_SCALE = 10.
MASS = 100
SURFACE_ELASTICITY = 0.9
SURFACE_FRICTION = 0.9
STEP_SIZE = 1.
DISPLAY = True
FEATURES_PER_PIECE_COUNT = 6
TOTAL_FEATURE_COUNT = FEATURES_PER_PIECE_COUNT * (PIECE_COUNT * PIECE_COUNT)
POSITION_OFFSET = FRAME_SIZE / (PIECE_COUNT + 1)

class PuzzleWorld():

    def reset(self):
        for i in range(PIECE_COUNT):
            for j in range(PIECE_COUNT):
                piece = self.pieces[i + j * PIECES_COUNT]
                x_rand_offset = PIECE_SIZE * (random() - 0.5)
                y_rand_offset = PIECE_SIZE * (random() - 0.5)
                piece.body.position = POSITION_OFFSET * (i + 1) + x_rand_offset, offset * (j + 1) + y_rand_offset
                piece.body.angular_velocity = (random() - 0.5) * math.pi
                piece.body.angle = (random() - 0.5) * math.pi
                piece.body.velocity = VELOCITY_SCALE * (random() - 0.5), VELOCITY_SCALE * (random() - 0.5)

    def __init__(self, feature_type="PREDEFINED"):
        screen_draw = pygame.Surface((FRAME_SIZE, FRAME_SIZE))
        self.feature_type = feature_type

        self.obsv = np.zeros(TOTAL_FEATURE_COUNT) 

        def get_vertices_world(shape):
            new = [v.rotated(shape.body.angle) + shape.body.position for v in shape.get_vertices()]
            return new

        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        screen = pygame.display.set_mode((FRAME_SIZE, FRAME_SIZE))
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        ###
        ###  Create screen borders
        ###
        static_lines = [pymunk.Segment(space.static_body, (0.0, 0.0), (FRAME_SIZE, 0.0), 1.0)
                        ,pymunk.Segment(space.static_body, (0.0, 0.0), (0.0, FRAME_SIZE), 1.0)
                        ,pymunk.Segment(space.static_body, (FRAME_SIZE, FRAME_SIZE), (FRAME_SIZE, 0.0), 1.0)
                        ,pymunk.Segment(space.static_body, (FRAME_SIZE, FRAME_SIZE), (0.0, FRAME_SIZE), 1.0)
        ] 
        for line in static_lines:
            line.elasticity = SURFACE_ELASTICITY 
            line.friction = SURFACE_FRICTION 
            line.group = 1
        space.add(static_lines)

        ###
        ###  Create pieces 
        ###
        pieces = []
        half_size  = PIECE_SIZE / 2.
        FRAME_SIZE * (random() - 1.)
        for i in range(PIECE_COUNT * PIECE_COUNT):
            vertices = [(- half_size,
                         - half_size),
                         (- half_size,
                         + half_size),
                        (+ half_size,
                         + half_size),
                        (+ half_size ,
                         - half_size )]
            inertia = pymunk.moment_for_poly(MASS, vertices)
            body = pymunk.Body(MASS, inertia)
            piece = pymunk.Poly(body, vertices)
            piece.elasticity = SURFACE_ELASTICITY 
            piece.friction = SURFACE_FRICTION 
            space.add(body, piece)
            pieces.append(piece)
        self.reset()

def step(force):
    space.step(STEP_SIZE)
    # Apply force to first piece only.
    pieces[0].body.apply_force_at_local_point(force[0], force[1])
    if DISPLAY:
        screen.fill(THECOLORS["white"])
        space.debug_draw(draw_options)
        pygame.display.flip()
    if self.feature_type = "PREDEFINED":
        # predict the next state based on current state.
        for i, piece in enumerate(pieces):
            self.obsv[0 + i * FEATURES_PER_PIECE_COUNT] = piece.body.position[0]
            self.obsv[1 + i * FEATURES_PER_PIECE_COUNT ] = piece.body.position[1]
            self.obsv[2 + i * FEATURES_PER_PIECE_COUNT] = piece.body.velocity[0]
            self.obsv[3 + i * FEATURES_PER_PIECE_COUNT] = piece.body.velocity[1]
            self.obsv[4 + i * FEATURES_PER_PIECE_COUNT] = piece.body.angle
            self.obsv[5 + i * FEATURES_PER_PIECE_COUNT] = piece.body.angular_velocity
    else:
        screen_draw.fill(THECOLORS["black"])
        ### Draw stuff
        body_vertices = get_vertices_world(piece)
        pygame.draw.polygon(screen_draw, (255, 255, 255), [body_vertices[0], body_vertices[1],body_vertices[2], body_vertices[3]], 1)
        test = pygame.surfarray.array3d(screen_draw)
        screen.blit(screen_draw, (0, 0))
    return self.obsv
