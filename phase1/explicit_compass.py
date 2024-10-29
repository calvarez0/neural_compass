import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Constants remain the same
WIDTH, HEIGHT = 800, 800
AGENT_SIZE = 10
REWARD_SIZE = 10
BEEHIVE_SIZE = 20
VISION_RANGE = 250
ROTATION_SPEED = 10
MOVE_SPEED = 5
# EPISODES = 1000

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = random.randint(0, 360)
        self.vision_active = True
        self.beehive_position = (x, y)
        self.reward_position = None
        self.carrying_reward = False
        self.movement_vectors = []  # Store actual movement vectors for navigation
        self.path_positions = [(x, y)]  # Store positions for visualization

    def draw(self, ax):
        agent_circle = patches.Circle((self.x, self.y), AGENT_SIZE, color='green')
        ax.add_patch(agent_circle)
        
        if self.vision_active:
            end_x = self.x + VISION_RANGE * math.cos(math.radians(self.angle))
            end_y = self.y + VISION_RANGE * math.sin(math.radians(self.angle))
            ax.plot([self.x, end_x], [self.y, end_y], 'g-')

        # Draw the path
        if len(self.path_positions) > 1:
            path_x = [p[0] for p in self.path_positions]
            path_y = [p[1] for p in self.path_positions]
            ax.plot(path_x, path_y, 'b-', alpha=0.5)

    def move_forward(self):
        # Calculate the movement vector
        dx = math.cos(math.radians(self.angle)) * MOVE_SPEED
        dy = math.sin(math.radians(self.angle)) * MOVE_SPEED
        
        # Store the original position
        old_x, old_y = self.x, self.y
        
        # Update position with boundary check
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Only update position if within bounds
        moved = False
        if 0 <= new_x <= WIDTH:
            self.x = new_x
            moved = True
        if 0 <= new_y <= HEIGHT:
            self.y = new_y
            moved = True
            
        # Only store the movement vector if we actually moved
        if moved:
            actual_dx = self.x - old_x
            actual_dy = self.y - old_y
            if actual_dx != 0 or actual_dy != 0:  # Only store non-zero movements
                self.movement_vectors.append((actual_dx, actual_dy))
                self.path_positions.append((self.x, self.y))

    def rotate_randomly(self):
        if random.random() < 0.5:
            self.rotate_left()
        else:
            self.rotate_right()

    def rotate_left(self):
        self.angle = (self.angle - ROTATION_SPEED) % 360

    def rotate_right(self):
        self.angle = (self.angle + ROTATION_SPEED) % 360

    def detect_reward(self, reward_pos):
        if self.vision_active and distance((self.x, self.y), reward_pos) <= VISION_RANGE:
            self.reward_position = reward_pos
            print("Reward detected and memorized!")
            return True
        return False

    def move_towards_reward(self):
        if self.reward_position:
            dx = self.reward_position[0] - self.x
            dy = self.reward_position[1] - self.y
            target_angle = math.degrees(math.atan2(dy, dx)) % 360
            
            if abs(target_angle - self.angle) > ROTATION_SPEED:
                if (target_angle - self.angle + 360) % 360 > 180:
                    self.rotate_left()
                else:
                    self.rotate_right()
            else:
                self.move_forward()

    def return_to_beehive(self):
        if self.movement_vectors:
            # Calculate the total displacement vector
            total_dx = sum(v[0] for v in self.movement_vectors)
            total_dy = sum(v[1] for v in self.movement_vectors)
            
            # Calculate the angle back to the beehive (opposite direction)
            target_angle = (math.degrees(math.atan2(-total_dy, -total_dx))) % 360
            
            if abs(target_angle - self.angle) > ROTATION_SPEED:
                if (target_angle - self.angle + 360) % 360 > 180:
                    self.rotate_left()
                else:
                    self.rotate_right()
            else:
                self.move_forward()

    def reset_for_new_search(self):
        self.vision_active = True
        self.reward_position = None
        self.carrying_reward = False
        self.movement_vectors = []
        self.path_positions = [(self.x, self.y)]

class Beehive:
    def __init__(self):
        self.x = random.randint(BEEHIVE_SIZE, WIDTH - BEEHIVE_SIZE)
        self.y = random.randint(BEEHIVE_SIZE, HEIGHT - BEEHIVE_SIZE)

    def draw(self, ax):
        beehive_rect = patches.Rectangle((self.x - BEEHIVE_SIZE // 2, self.y - BEEHIVE_SIZE // 2), 
                                       BEEHIVE_SIZE, BEEHIVE_SIZE, color='brown')
        ax.add_patch(beehive_rect)

class Reward:
    def __init__(self):
        self.reposition()

    def reposition(self):
        self.x = random.randint(REWARD_SIZE, WIDTH - REWARD_SIZE)
        self.y = random.randint(REWARD_SIZE, HEIGHT - REWARD_SIZE)

    def draw(self, ax):
        reward_circle = patches.Circle((self.x, self.y), REWARD_SIZE, color='red')
        ax.add_patch(reward_circle)

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

# Main function is the only thing that needs to change
def main():
    fig, ax = plt.subplots()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')
    
    beehive = Beehive()
    agent = Agent(beehive.x, beehive.y)
    reward = Reward()

    def update(frame):
        ax.clear()
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_aspect('equal')
        
        if agent.vision_active:
            if agent.detect_reward((reward.x, reward.y)):
                agent.vision_active = False
            else:
                agent.rotate_randomly()
                agent.move_forward()
        
        if agent.reward_position and not agent.carrying_reward:
            agent.move_towards_reward()
            if distance((agent.x, agent.y), (reward.x, reward.y)) <= AGENT_SIZE + REWARD_SIZE:
                agent.carrying_reward = True
                print("Reward picked up!")

        if agent.carrying_reward:
            agent.return_to_beehive()
            if distance((agent.x, agent.y), (beehive.x, beehive.y)) <= AGENT_SIZE + BEEHIVE_SIZE:
                agent.carrying_reward = False
                print("Reward delivered to the beehive!")
                # Reset agent for new search and reposition reward
                agent.reset_for_new_search()
                reward.reposition()

        agent.draw(ax)
        beehive.draw(ax)
        reward.draw(ax)

    # Modified animation parameters to run indefinitely
    ani = FuncAnimation(fig, update, frames=None, repeat=True, interval=2)
    plt.show()

if __name__ == "__main__":
    main()