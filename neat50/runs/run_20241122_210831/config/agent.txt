# neat50/agent.py

import os
import neat
import math
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
        self.energy_default = 4000
        self.energy = 4000
        # Initialize memory grid with reduced complexity
        self.grid_cols = WIDTH // GRID_SIZE
        self.grid_rows = HEIGHT // GRID_SIZE
        self.memory_grid = np.zeros((self.grid_rows, self.grid_cols))
        self.reset()
        
    def update_memory_grid(self):
        """Update memory grid with current position and apply decay"""
        grid_x = int(self.x // GRID_SIZE)
        grid_y = int(self.y // GRID_SIZE)
        
        # Ensure grid coordinates are within bounds
        grid_x = max(0, min(grid_x, self.grid_cols - 1))
        grid_y = max(0, min(grid_y, self.grid_rows - 1))
        
        # Mark current position with maximum value
        self.memory_grid[grid_y, grid_x] = 1.0
        
        # Apply memory decay
        self.memory_grid *= MEMORY_DECAY
        
    def get_local_memory(self):
        """Get memory values for surrounding cells"""
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
        # Handle rotation and movement
        rotate = action[0]  
        move_forward = action[1] > 0
        
        # Update angle with normalized rotation
        self.angle = (self.angle + rotate * ROTATION_SPEED) % 360
        
        if move_forward:
            # Calculate new position
            dx = math.cos(math.radians(self.angle)) * MOVE_SPEED
            dy = math.sin(math.radians(self.angle)) * MOVE_SPEED
            new_x = self.x + dx
            new_y = self.y + dy
            
            # Boundary checking
            if 0 <= new_x <= WIDTH:
                self.x = new_x
            if 0 <= new_y <= HEIGHT:
                self.y = new_y

            # Energy cost for movement
            movement_cost = 25
            if self.carrying_reward:  # Higher cost when carrying reward
                movement_cost *= 1.5
            self.energy -= movement_cost
            
            # Update memory grid after movement
            self.update_memory_grid()

        # Track path and steps
        self.path_positions.append((self.x, self.y))
        self.steps_taken += 1
        
        # Vision and reward detection
        if self.vision_active and distance((self.x, self.y), reward_pos) <= VISION_RANGE:
            self.reward_position = reward_pos
            if not self.found_reward:
                self.found_reward = True
        
        # Reward pickup
        if self.reward_position and not self.carrying_reward:
            if distance((self.x, self.y), reward_pos) <= AGENT_SIZE + REWARD_SIZE:
                self.carrying_reward = True
        
        # Return to hive
        if self.carrying_reward and distance((self.x, self.y), self.beehive_position) <= AGENT_SIZE + BEEHIVE_SIZE:
            self.returned_home = True
            return True
            
        # Energy depletion check
        if self.energy <= 0:
            return True
            
        return False
        
    def reset(self):
        """Reset agent state for new episode"""
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
        self.memory_grid.fill(0)
        
    def get_sensor_data(self, reward_pos):
        """
        Get sensor data including spatial memory
        Returns 17 inputs total:
        - Current heading vector (2)
        - Path integration vector (2)
        - Target vector (2, zeroed when not visible)
        - Binary state flags (2)
        - Local memory grid values (9)
        """
        # Heading vector
        heading_x = math.cos(math.radians(self.angle))
        heading_y = math.sin(math.radians(self.angle))
        
        # Path integration
        path_x, path_y = 0, 0
        if len(self.path_positions) > 1:
            path_window = self.path_positions[-20:]  # Recent path memory
            for i in range(1, len(path_window)):
                dx = (path_window[i][0] - path_window[i-1][0]) / MOVE_SPEED
                dy = (path_window[i][1] - path_window[i-1][1]) / MOVE_SPEED
                path_x += dx
                path_y += dy
        
        # Target vector when visible
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
            target_angle, target_distance,  # Target vector
            1.0 if self.carrying_reward else 0.0,  # Carrying state
            1.0 if self.vision_active else 0.0,    # Vision state
            *memory_values            # Local memory grid values
        ]