import os
import neat
import math
import graphviz
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from utilities import *
from constants import *

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class Agent:
    def __init__(self, beehive_pos):
        self.beehive_position = beehive_pos
        self.start_position = beehive_pos
        self.energy = 4000
        # Initialize memory grid
        self.grid_cols = WIDTH // GRID_SIZE
        self.grid_rows = HEIGHT // GRID_SIZE
        self.memory_grid = np.zeros((self.grid_rows, self.grid_cols))
        self.reset()
        
    def update_memory_grid(self):
        # Convert current position to grid coordinates
        grid_x = int(self.x // GRID_SIZE)
        grid_y = int(self.y // GRID_SIZE)
        
        # Ensure grid coordinates are within bounds
        grid_x = max(0, min(grid_x, self.grid_cols - 1))
        grid_y = max(0, min(grid_y, self.grid_rows - 1))
        
        # Mark current position with maximum value (1.0)
        self.memory_grid[grid_y, grid_x] = 1.0
        
        # Apply memory decay to all cells
        self.memory_grid *= MEMORY_DECAY
        
    def get_local_memory(self):
        """Returns memory values for the 8 surrounding grid cells plus current cell"""
        grid_x = int(self.x // GRID_SIZE)
        grid_y = int(self.y // GRID_SIZE)
        
        memory_values = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                y = grid_y + dy
                x = grid_x + dx
                if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                    memory_values.append(self.memory_grid[y, x])
                else:
                    memory_values.append(0.0)
        
        return memory_values
    
    def update(self, action, reward_pos):
        # Previous update logic
        rotate = action[0]
        move_forward = action[1] > 0
        
        self.angle = (self.angle + rotate * ROTATION_SPEED) % 360
        
        if move_forward:
            dx = math.cos(math.radians(self.angle)) * MOVE_SPEED
            dy = math.sin(math.radians(self.angle)) * MOVE_SPEED
            new_x = self.x + dx
            new_y = self.y + dy
            
            if 0 <= new_x <= WIDTH:
                self.x = new_x
            if 0 <= new_y <= HEIGHT:
                self.y = new_y

            movement_cost = 25
            if self.carrying_reward:  # carrying pollen = more energy required, increases urgency
                movement_cost *= 1.5
            self.energy -= movement_cost
            
            # Update memory grid after movement
            self.update_memory_grid()

        self.path_positions.append((self.x, self.y))
        self.steps_taken += 1
        
        if self.vision_active and distance((self.x, self.y), reward_pos) <= VISION_RANGE:
            self.reward_position = reward_pos
            if not self.found_reward:
                self.found_reward = True
        
        if self.reward_position and not self.carrying_reward:
            if distance((self.x, self.y), reward_pos) <= AGENT_SIZE + REWARD_SIZE:
                self.carrying_reward = True
        
        if self.carrying_reward and distance((self.x, self.y), self.beehive_position) <= AGENT_SIZE + BEEHIVE_SIZE:
            self.returned_home = True
            return True
            
        if self.energy <= 0:
            return True
            
        return False
        
    def reset(self):
        self.x = self.start_position[0]
        self.y = self.start_position[1]
        self.angle = random.randint(0, 360)
        self.vision_active = True
        self.reward_position = None
        self.carrying_reward = False
        self.path_positions = [(self.x, self.y)]
        self.steps_taken = 0
        self.found_reward = False
        self.returned_home = False
        # Reset memory grid
        self.memory_grid.fill(0)
        
    def get_sensor_data(self, reward_pos):
        """
        Returns enhanced sensor data including spatial memory:
        17 inputs total:
        - Current heading vector (2)
        - Path integration vector (2)
        - Target vector (2, zeroed when not visible)
        - Binary state flags (2)
        - Local memory grid values (9)
        """

        # Previous sensor calculations
        heading_x = math.cos(math.radians(self.angle))
        heading_y = math.sin(math.radians(self.angle))
        
        path_x, path_y = 0, 0
        if len(self.path_positions) > 1:
            path_window = self.path_positions[-20:]
            for i in range(1, len(path_window)):
                dx = (path_window[i][0] - path_window[i-1][0]) / MOVE_SPEED
                dy = (path_window[i][1] - path_window[i-1][1]) / MOVE_SPEED
                path_x += dx
                path_y += dy
        
        if self.vision_active:
            dx = reward_pos[0] - self.x
            dy = reward_pos[1] - self.y
            distance_to_target = math.sqrt(dx**2 + dy**2) / max(WIDTH, HEIGHT)
            angle_to_target = math.degrees(math.atan2(dy, dx)) - self.angle
            angle_to_target = (angle_to_target + 360) % 360
            target_angle = angle_to_target / 180.0
            target_distance = distance_to_target
        else:
            target_angle = 0
            target_distance = 0
        
        # Get local memory values
        memory_values = self.get_local_memory()
        
        return [
            heading_x, heading_y,      # Current heading vector
            path_x, path_y,           # Integrated path vector
            target_angle, target_distance,  # Target angle and distance
            1.0 if self.carrying_reward else 0.0,  # Carrying state
            1.0 if self.vision_active else 0.0,    # Vision state
            *memory_values            # Local memory grid values (9 inputs)
        ]


