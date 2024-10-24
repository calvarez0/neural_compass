import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Constants
WIDTH, HEIGHT = 800, 800
BEE_SIZE = 10
POLLEN_SIZE = 5
BEE_SPEED = 5
VISION_ANGLE = 60  # degrees
VISION_RANGE = 150  # pixels
NUM_ACTIONS = 16  # 8 directions for movement, 8 for gaze rotation
GAMMA = 0.99
LR = 0.001

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Helper functions
def get_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def rotate_vector(vector, angle):
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    return vector[0] * cos_a - vector[1] * sin_a, vector[0] * sin_a + vector[1] * cos_a

# Neural Network for the Bee
class BeeNet(nn.Module):
    def __init__(self):
        super(BeeNet, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_ACTIONS)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Bee agent class
class Bee:
    def __init__(self):
        self.position = [WIDTH // 2, HEIGHT // 2]
        self.direction = 0  # degrees
        self.gaze_direction = 0  # degrees
        self.model = BeeNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = []

    def move(self, action):
        if action < 8:
            # Move the bee's body in one of the 8 directions
            direction_vectors = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
            dx, dy = direction_vectors[action]
            self.position[0] += dx * BEE_SPEED
            self.position[1] += dy * BEE_SPEED
        else:
            # Rotate the bee's gaze
            self.gaze_direction = (self.gaze_direction + (action - 8) * 45) % 360

    def see_pollen(self, pollen_pos):
        distance = get_distance(self.position, pollen_pos)
        if distance <= VISION_RANGE:
            vector_to_pollen = (pollen_pos[0] - self.position[0], pollen_pos[1] - self.position[1])
            gaze_vector = rotate_vector((1, 0), self.gaze_direction)
            angle_diff = np.degrees(np.arccos(
                np.dot(gaze_vector, vector_to_pollen) /
                (np.linalg.norm(gaze_vector) * np.linalg.norm(vector_to_pollen))
            ))
            return angle_diff <= VISION_ANGLE / 2
        return False

    def remember_hive(self):
        # Calculate a vector-based memory or 2D accelerometer
        pass

    def get_state(self, pollen_pos):
        distance = get_distance(self.position, pollen_pos) / max(WIDTH, HEIGHT)
        angle_diff = abs(self.gaze_direction - self.direction) / 360.0
        bee_x, bee_y = self.position[0] / WIDTH, self.position[1] / HEIGHT
        pollen_x, pollen_y = pollen_pos[0] / WIDTH, pollen_pos[1] / HEIGHT
        return torch.FloatTensor([distance, angle_diff, bee_x, bee_y, pollen_x, pollen_y])

    def act(self, state):
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def update(self, state, action, reward, next_state):
        self.optimizer.zero_grad()
        q_values = self.model(state)
        max_next_q_values = self.model(next_state).max()
        target = reward + GAMMA * max_next_q_values
        loss = nn.functional.mse_loss(q_values[action], target)
        loss.backward()
        self.optimizer.step()

# Environment setup
class Environment:
    def __init__(self):
        self.bee = Bee()
        self.pollen = [random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)]

    def reset(self):
        self.bee.position = [WIDTH // 2, HEIGHT // 2]
        self.bee.direction = 0
        self.bee.gaze_direction = 0
        self.pollen = [random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)]

    def render(self):
        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, (255, 255, 0), self.pollen, POLLEN_SIZE)
        pygame.draw.circle(screen, (0, 0, 0), self.bee.position, BEE_SIZE)
        pygame.display.flip()

    def step(self):
        state = self.bee.get_state(self.pollen)
        action = self.bee.act(state)
        self.bee.move(action)

        reward = -0.01
        if self.bee.see_pollen(self.pollen):
            reward += 1.0
        if get_distance(self.bee.position, self.pollen) < 10:
            reward += 10.0
            self.reset()

        next_state = self.bee.get_state(self.pollen)
        self.bee.update(state, action, reward, next_state)

# Main loop
env = Environment()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    env.step()
    env.render()
    clock.tick(60)

pygame.quit()
