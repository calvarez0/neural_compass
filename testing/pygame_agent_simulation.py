import pygame
import math
import random
import numpy as np

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
AGENT_SIZE = 10
REWARD_SIZE = 10
VISION_RANGE = 150  # Radius of vision
ROTATION_SPEED = 10  # Degrees per frame
MOVE_SPEED = 5  # Pixels per frame
EPISODES = 1000

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Biologically Plausible Fly Simulation")

# Helper functions
def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def angle_to_vector(angle):
    radian = math.radians(angle)
    return math.cos(radian), math.sin(radian)

def normalize_angle(angle):
    return angle % 360

# Agent class
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = random.randint(0, 360)
        self.vision_active = True
        self.memory = None  # Memory for reward vector
        self.explored_positions = set()  # Track explored positions

    def draw(self):
        pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), AGENT_SIZE)
        if self.vision_active:
            end_x = self.x + VISION_RANGE * math.cos(math.radians(self.angle))
            end_y = self.y + VISION_RANGE * math.sin(math.radians(self.angle))
            pygame.draw.line(screen, GREEN, (self.x, self.y), (end_x, end_y), 2)

    def move_forward(self):
        direction = angle_to_vector(self.angle)
        new_x = self.x + direction[0] * MOVE_SPEED
        new_y = self.y + direction[1] * MOVE_SPEED

        # Enforce boundaries
        if 0 <= new_x <= WIDTH - AGENT_SIZE:
            self.x = new_x
        if 0 <= new_y <= HEIGHT - AGENT_SIZE:
            self.y = new_y

    def rotate_left(self):
        self.angle = normalize_angle(self.angle - ROTATION_SPEED)

    def rotate_right(self):
        self.angle = normalize_angle(self.angle + ROTATION_SPEED)

    def detect_reward(self, reward_pos):
        if self.vision_active and distance((self.x, self.y), reward_pos) <= VISION_RANGE:
            dx = reward_pos[0] - self.x
            dy = reward_pos[1] - self.y
            self.memory = (dx, dy)  # Store relative position to the reward
            return True
        return False

    def navigate_using_memory(self):
        if self.memory:
            target_dx, target_dy = self.memory
            target_angle = math.degrees(math.atan2(target_dy, target_dx))
            target_angle = normalize_angle(target_angle)

            if abs(target_angle - self.angle) > ROTATION_SPEED:
                if (target_angle - self.angle + 360) % 360 > 180:
                    self.rotate_left()
                else:
                    self.rotate_right()
            else:
                self.move_forward()

# Reward class
class Reward:
    def __init__(self):
        self.x = random.randint(REWARD_SIZE, WIDTH - REWARD_SIZE)
        self.y = random.randint(REWARD_SIZE, HEIGHT - REWARD_SIZE)

    def draw(self):
        pygame.draw.circle(screen, RED, (self.x, self.y), REWARD_SIZE)

# Q-learning agent implementation
class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((360, 3))  # Angle (0-359), actions (rotate left, rotate right, move forward)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 1.0  # Exploration-exploitation tradeoff

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2])  # Explore: 0 = rotate left, 1 = rotate right, 2 = move forward
        else:
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def reduce_epsilon(self):
        self.epsilon = max(0.1, self.epsilon * 0.995)

# Main loop
def main():
    clock = pygame.time.Clock()
    q_agent = QLearningAgent()

    for episode in range(EPISODES):
        agent = Agent(WIDTH // 2, HEIGHT // 2)
        reward = Reward()
        done = False
        steps = 0

        while not done and steps < 1000:
            current_angle = int(agent.angle)
            action = q_agent.choose_action(current_angle)

            if action == 0:
                agent.rotate_left()
            elif action == 1:
                agent.rotate_right()
            elif action == 2:
                agent.move_forward()

            # Exploration rewards for new positions
            current_position = (int(agent.x), int(agent.y))
            if current_position not in agent.explored_positions:
                reward_value = 10  # Reward for exploring
                agent.explored_positions.add(current_position)
            else:
                reward_value = -1  # Slight penalty for revisiting
            
            # Detect and remember reward position
            if agent.detect_reward((reward.x, reward.y)):
                print("Reward found!")
                agent.vision_active = False  # Turn off vision once reward is seen
                reward_value = 100
                done = True  # End episode upon reward detection
            
            q_agent.update_q_value(current_angle, action, reward_value, int(agent.angle))
            if not agent.vision_active:
                agent.navigate_using_memory()
            
            steps += 1
            screen.fill(BLACK)
            agent.draw()
            reward.draw()
            pygame.display.flip()
            clock.tick(60)

        q_agent.reduce_epsilon()

    pygame.quit()

if __name__ == "__main__":
    main()
