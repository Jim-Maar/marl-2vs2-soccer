# Box 2D pygame Soccer Environment for Gymnasium

import pygame
import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)
import os
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from typing import Tuple, Dict, Any, List
from pathlib import Path
from datetime import datetime
# Check if we need to use a virtual display
try:
    # Try to use xvfb if available
    os.environ.setdefault('SDL_VIDEODRIVER', 'x11')
    pygame.display.init()
except pygame.error:
    # Fall back to dummy driver if no display is available
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    pygame.display.init()

# Pygame initialization
pygame.init()

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800

GAME_WIDTH = 30
GAME_HEIGHT = 40

PPM = SCREEN_WIDTH / GAME_WIDTH  # pixels per meter

FPS = 20
PLAYER_SIZE = 1.5  # meters
BALL_RADIUS = 2.0  # meters
PLAYER_SPEED = 9.0
WALL_THICKNESS = 1.0
GOAL_WIDTH = 15.0  # meters in width
MAXIMUM_VELOCITY = 50.0  # maximum velocity for observation space#
REAKISTIC_MAXIMUM_VELOCITY = 12.0
SPAWNING_RADIUS = 3.0  # random spawn radius in meters

# Physics hyperparams
PLAYER_DENSITY = 1.0
PLAYER_FRICTION = 0.3
BALL_DENSITY = 1.0
BALL_FRICTION = 0.3
BALL_RESTITUTION = 0.8

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Actions
UP = 0
UP_RIGHT = 1
RIGHT = 2
DOWN_RIGHT = 3
DOWN = 4
DOWN_LEFT = 5
LEFT = 6
UP_LEFT = 7
NO_OP = 8

SIMILAR_ACTION_PAIRS = [
    (UP, UP_RIGHT),
    (UP, UP_LEFT),
    (DOWN, DOWN_RIGHT),
    (DOWN, DOWN_LEFT),
    (RIGHT, UP_RIGHT),
    (RIGHT, DOWN_RIGHT),
    (LEFT, UP_LEFT),
    (LEFT, DOWN_LEFT),
]

DEFAULT_REWARD_SPECIFICATION = {
    "goal": 10.0,
    "stay_in_field": 0.02,
    "smoothness": 0.05,
    "stay_own_half": 0.02,
    "winning_the_ball_and_passing": 1.5,
}

'''DEBUG_REWARD_SPECIFICATION = {
    "goal": 10.0,
    "winning_the_ball_and_passing": 1.0,
    "stay_in_field": 0.05,
    "smoothness": 0.05,
    "dist_to_ball": 0.05,
    "dist_to_goal": 0.05,
    "velocity_to_goal": 0.05,
    "velocity_to_ball": 0.05,
    "stay_own_half": 0.05,
}'''

DEBUG_REWARD_SPECIFICATION = {
    "dist_to_ball": 1.0,
}
# DEFAULT_REWARD_SPECIFICATION = DEBUG_REWARD_SPECIFICATION

class SoccerContactListener(Box2D.b2ContactListener):
    def __init__(self, env):
        Box2D.b2ContactListener.__init__(self)
        self.env = env
    
    def BeginContact(self, contact):
        # Check if contact involves the ball
        body_a = contact.fixtureA.body
        body_b = contact.fixtureB.body

        # Determine which body is the ball and which is the player
        if body_a.userData is not None and body_b.userData is not None:
            if body_a.userData.get('type') == 'ball' and 'team' in body_b.userData:
                # body_a is ball, body_b is player
                self.env.ball_touched[body_b.userData['team']] = True
                self.env.ball_toucher = body_b.userData['id']
            elif body_b.userData.get('type') == 'ball' and 'team' in body_a.userData:
                # body_b is ball, body_a is player
                self.env.ball_touched[body_a.userData['team']] = True
                self.env.ball_toucher = body_a.userData['id']

    def EndContact(self, contact):
        pass
    
    def PreSolve(self, contact, oldManifold):
        pass
    
    def PostSolve(self, contact, impulse):
        pass

class Soccer(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode=None, video_log_freq=100, env_id="Soccer-v0", seed=1, reward_specification=DEFAULT_REWARD_SPECIFICATION):
        super().__init__()
        
        # Environment parameters
        self.num_agents = 4
        self.team_size = 2
        self.num_teams = self.num_agents // self.team_size
        self.max_steps = 600
        self.video_log_freq = video_log_freq
        self.render_mode = render_mode
        self.env_id = env_id
        self.seed = seed
        self.reward_specification = reward_specification
        self.local_position_history = []

        # Observation and action spaces
        # Each agent observes: 
        # - own position (2) and velocity (2)
        # - teammate position (2) and velocity (2)
        # - enemy positions (2*2) and velocities (2*2)
        # - ball position (2) and velocity (2)
        # = 20 values total
        self.observation_space = Box(
            low=np.array([[0.0 if i % 4 <= 1 else -MAXIMUM_VELOCITY 
                          for i in range(20)] for _ in range(self.num_agents)]),
            high=np.array([[0.0 if i % 4 <= 1 else MAXIMUM_VELOCITY 
                           for i in range(20)] for _ in range(self.num_agents)]),
            dtype=np.float32
        )
        
        # 9 actions for each agent: UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT, NO_OP
        self.action_space = MultiDiscrete([9, 9, 9, 9])
        
        # Initialize pygame if rendering is needed
        if self.render_mode is not None:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Box2D Soccer")
            self.clock = pygame.time.Clock()
        
        # Initialize Box2D world
        self.world = world(gravity=(0, 0), doSleep=True)
        
        # Set up contact listener
        self.contact_listener = SoccerContactListener(self)
        self.world.contactListener = self.contact_listener
        
        # For video recording
        self.frames = []
        self.step_count = 0
        self.episode_count = 0
        
        # Reset to initialize everything
        self.reset()

    def add_to_ball_toucher_history(self, agent_idx):
        self.ball_toucher_history.append(agent_idx)
        if len(self.ball_toucher_history) > 3:
            self.ball_toucher_history.pop(0)

    def add_to_action_history(self, actions):
        self.action_history.append(actions)
        if len(self.action_history) > 3:
            self.action_history.pop(0)

    def add_to_local_position_history(self, local_position):
        self.local_position_history.append(local_position)
        if len(self.local_position_history) > 3:
            self.local_position_history.pop(0)
    
    def create_boundaries(self):
        # Create walls and goals
        # Left wall
        self.world.CreateStaticBody(
            position=(0, GAME_HEIGHT/2),
            shapes=polygonShape(box=(WALL_THICKNESS, GAME_HEIGHT/2)),
        )
        
        # Right wall
        self.world.CreateStaticBody(
            position=(GAME_WIDTH, GAME_HEIGHT/2),
            shapes=polygonShape(box=(WALL_THICKNESS, GAME_HEIGHT/2)),
        )
        
        # Top wall (with goal opening)
        self.create_goal_wall(True)  # Top
        
        # Bottom wall (with goal opening)
        self.create_goal_wall(False)  # Bottom
    
    def create_goal_wall(self, is_top):
        wall_width = (GAME_WIDTH - GOAL_WIDTH) / 2
        y_pos = 0 if is_top else GAME_HEIGHT
        
        # Left part
        self.world.CreateStaticBody(
            position=(wall_width/2, y_pos),
            shapes=polygonShape(box=(wall_width/2, WALL_THICKNESS)),
        )
        
        # Right part
        self.world.CreateStaticBody(
            position=(GAME_WIDTH - wall_width/2, y_pos),
            shapes=polygonShape(box=(wall_width/2, WALL_THICKNESS)),
        )
    
    def create_players(self):
        # Create 4 players (2 per team)
        # Team 1: Players 0 and 1 (RED team - bottom)
        # Team 2: Players 2 and 3 (BLUE team - top)
        
        self.players = []
        
        # Default positions
        default_positions = [
            (GAME_WIDTH/4, GAME_HEIGHT/6),         # Team 1 - Player 0 (bottom left)
            (3*GAME_WIDTH/4, GAME_HEIGHT/6),       # Team 1 - Player 1 (bottom right)
            (GAME_WIDTH/4, 5*GAME_HEIGHT/6),       # Team 2 - Player 2 (top left)
            (3*GAME_WIDTH/4, 5*GAME_HEIGHT/6),     # Team 2 - Player 3 (top right)
        ]
        
        # Add randomness to positions
        for i, (x, y) in enumerate(default_positions):
            # Add random offset within SPAWNING_RADIUS
            random_x = x + self.np_random.uniform(-SPAWNING_RADIUS, SPAWNING_RADIUS)
            random_y = y + self.np_random.uniform(-SPAWNING_RADIUS, SPAWNING_RADIUS)
            
            # Ensure players stay within bounds
            random_x = max(PLAYER_SIZE, min(GAME_WIDTH - PLAYER_SIZE, random_x))
            random_y = max(PLAYER_SIZE, min(GAME_HEIGHT - PLAYER_SIZE, random_y))
            
            # Create player
            player = self.world.CreateDynamicBody(
                position=(random_x, random_y),
                fixtures=Box2D.b2FixtureDef(
                    shape=polygonShape(box=(PLAYER_SIZE/2, PLAYER_SIZE/2)),
                    density=PLAYER_DENSITY,
                    friction=PLAYER_FRICTION,
                ),
            )
            player.userData = {"team": i // self.team_size, "id": i}
            self.players.append(player)
    
    def create_ball(self):
        self.ball = self.world.CreateDynamicBody(
            position=(GAME_WIDTH/2, GAME_HEIGHT/2),
            fixtures=Box2D.b2FixtureDef(
                shape=circleShape(radius=BALL_RADIUS),
                density=BALL_DENSITY,
                friction=BALL_FRICTION,
                restitution=BALL_RESTITUTION,
            ),
        )
        self.ball.userData = {"type": "ball"}
        self.reset_ball()
    
    def reset_ball(self):
        self.ball.position = (GAME_WIDTH/2, GAME_HEIGHT/2)
        self.ball.linearVelocity = (0, 0)
        self.ball.angularVelocity = 0
        self.ball_touched = [False for _ in range(self.num_teams)]
        self.last_ball_toucher = None # last agent that touched the ball, isnt reset when noone touches it
        self.ball_toucher_history = [] # history of agents that touched the ball, noone is None
        self.ball_toucher = None # last agent that touched the ball, is reset when noone touches it
    
    def check_goal(self):
        ball_pos = self.ball.position
        if ball_pos.y < 0:  # Bottom goal (Team 2 scores)
            self.score[1] += 1
            return 1  # Team 2 scored
        elif ball_pos.y > GAME_HEIGHT:  # Top goal (Team 1 scores)
            self.score[0] += 1
            return 0  # Team 1 scored
        return -1  # No goal
    
    def get_local_position(self, pos, agent_id, normalize=False):
        """Convert global position to local position for the given agent"""
        # For soccer, we'll define local coordinates as:
        # Team 1 (bottom): y-axis points up
        # Team 2 (top): y-axis points down
        # Left players: x-axis points right
        # Right players: x-axis points left
        
        x, y = pos
        # Team 1 (bottom)
        if agent_id == 0:  # Bottom left
            new_pos = np.array([x, y])
        elif agent_id == 1:  # Bottom right
            new_pos = np.array([GAME_WIDTH - x, y])
        # Team 2 (top)
        elif agent_id == 2:  # Top left
            new_pos = np.array([x, GAME_HEIGHT - y])
        elif agent_id == 3:  # Top right
            new_pos = np.array([GAME_WIDTH - x, GAME_HEIGHT - y])
        if normalize:
            new_pos[0] = new_pos[0] / GAME_WIDTH
            new_pos[1] = new_pos[1] / GAME_HEIGHT
        return new_pos
    
    def get_local_velocity(self, vel, agent_id, normalize=False):
        """Convert global velocity to local velocity for the given agent"""
        vx, vy = vel
        
        # Team 1 (bottom)
        if agent_id == 0:  # Bottom left
            new_vel = np.array([vx, vy])
        elif agent_id == 1:  # Bottom right
            new_vel = np.array([-vx, vy])
        # Team 2 (top)
        elif agent_id == 2:  # Top left
            new_vel = np.array([vx, -vy])
        elif agent_id == 3:  # Top right
            new_vel = np.array([-vx, -vy])
        if normalize:
            new_vel = new_vel / REAKISTIC_MAXIMUM_VELOCITY
        return new_vel
        
    def get_global_velocity(self, vel, agent_id):
        return self.get_local_velocity(vel, agent_id) # works because function is s

    def get_observations(self):
        """Get observations for all agents"""
        observations = []
        
        for i in range(self.num_agents):
            agent_observation = []
            team_index = i // self.team_size
            team_start = team_index * self.team_size
            team_end = (team_index + 1) * self.team_size
            
            # Get teammate and enemy indices
            teammate_indices = list(range(team_start, i)) + list(range(i + 1, team_end))
            enemy_indices = list(range(0, team_start)) + list(range(team_end, self.num_agents))
            
            # Add own position and velocity (local coordinates)
            own_pos = self.get_local_position(
                (self.players[i].position.x, self.players[i].position.y), 
                i,
                normalize=True
            )
            own_vel = self.get_local_velocity(
                (self.players[i].linearVelocity.x, self.players[i].linearVelocity.y),
                i,
                normalize=True
            )
            agent_observation.extend(own_pos)
            agent_observation.extend(own_vel)
            
            # Add teammate positions and velocities (local coordinates)
            for j in teammate_indices:
                teammate_pos = self.get_local_position(
                    (self.players[j].position.x, self.players[j].position.y),
                    i,
                    normalize=True
                )
                teammate_vel = self.get_local_velocity(
                    (self.players[j].linearVelocity.x, self.players[j].linearVelocity.y),
                    i,
                    normalize=True
                )
                agent_observation.extend(teammate_pos)
                agent_observation.extend(teammate_vel)
            
            # Add enemy positions and velocities (local coordinates)
            for j in enemy_indices:
                enemy_pos = self.get_local_position(
                    (self.players[j].position.x, self.players[j].position.y),
                    i,
                    normalize=True
                )
                enemy_vel = self.get_local_velocity(
                    (self.players[j].linearVelocity.x, self.players[j].linearVelocity.y),
                    i,
                    normalize=True
                )
                agent_observation.extend(enemy_pos)
                agent_observation.extend(enemy_vel)
            
            # Add ball position and velocity (local coordinates)
            ball_pos = self.get_local_position(
                (self.ball.position.x, self.ball.position.y),
                i,
                normalize=True
            )
            ball_vel = self.get_local_velocity(
                (self.ball.linearVelocity.x, self.ball.linearVelocity.y),
                i,
                normalize=True
            )
            agent_observation.extend(ball_pos)
            agent_observation.extend(ball_vel)
            
            # Add to observations
            observations.append(np.array(agent_observation, dtype=np.float32))
        
        return np.array(observations, dtype=np.float32)
    
    def get_goal_reward(self, goal_scored):
        team_rewards = np.zeros(self.num_teams)
        if goal_scored >= 0:  # A goal was scored
            scoring_team = goal_scored
            for team in range(self.num_teams):
                if team == scoring_team:
                    team_rewards[team] += 1  # Big reward for scoring team
                else:
                    team_rewards[team] += -0.5  # Penalty for conceding team
        return team_rewards
    
    def get_stay_in_field_reward(self, agent_idx):
        """Calculate reward for staying in the field"""
        player = self.players[agent_idx]
        # Check if player is in the field
        if player.position.x > 0 and player.position.x < GAME_WIDTH and \
            player.position.y > 0 and player.position.y < GAME_HEIGHT:
            return 1.0
        else:
            return -1.0
    
    def are_actions_similar(self, action1, action2):
        """Check if two actions are similar"""
        return (action1, action2) in SIMILAR_ACTION_PAIRS or (action2, action1) in SIMILAR_ACTION_PAIRS

    def get_smoothness_reward(self, agent_idx=None):
        """Calculate reward for smoothness of actions"""
        '''if len(self.action_history) < 2:
            return 0.0
            
        current_action = self.action_history[-1][agent_idx]
        previous_action = self.action_history[-2][agent_idx]
        if self.are_actions_similar(current_action, previous_action):
            return 1.0
        else:
            return -1.0'''
        if len(self.local_position_history) < 2:
            return 0.0
        previous_position = self.local_position_history[0][agent_idx]
        this_position = self.local_position_history[-1][agent_idx]
        dist = np.linalg.norm(this_position - previous_position)
        expected_dist = PLAYER_SIZE / FPS * 3
        if dist >= expected_dist / 2:
            return 1.0
        else:
            return -1.0
        
    def get_winning_the_ball_and_passing_reward(self):
        """Calculate reward for winning the ball and passing"""
        team_rewards = np.zeros(self.num_teams)
        if self.ball_toucher is None:
            return team_rewards
        if len(self.ball_toucher_history) < 3: # less than 3 steps in
            return team_rewards
        if not (self.ball_toucher_history[-2] is None and self.ball_toucher_history[-1] is None):
            return team_rewards
        if self.last_ball_toucher == self.ball_toucher:
            return team_rewards
        for team in range(self.num_teams):
            last_ball_toucher_team = self.last_ball_toucher // self.team_size
            if self.ball_touched[team]:
                team_rewards[team] += 1
            elif last_ball_toucher_team == team:
                team_rewards[team] += -0.5
        return team_rewards

    def get_normalized_distance(self, vector1 : Box2D.b2Vec2, vector2 : Box2D.b2Vec2, max_distance = GAME_WIDTH + GAME_HEIGHT):
        distance = np.sqrt((vector1.x - vector2.x)**2 + (vector1.y - vector2.y)**2)
        normalized_distance = distance / max_distance
        return normalized_distance

    def get_dist_to_ball_reward(self, agent_idx, only_beginning=True):
        """Calculate reward for distance to ball"""
        if only_beginning:
            return 0.0
        
        normalized_distance = self.get_normalized_distance(self.players[agent_idx].position, self.ball.position)
        return 1.0 - normalized_distance
    
    def get_dist_to_goal_reward(self):
        """Calculate reward for distance to goal"""
        team_rewards = np.zeros(self.num_teams)
        for team in range(self.num_teams):
            pos_of_enemy_goal = Box2D.b2Vec2(GAME_WIDTH / 2, GAME_HEIGHT if team == 0 else 0)
            normalized_distance = self.get_normalized_distance(self.ball.position, pos_of_enemy_goal)
            team_rewards[team] = 1.0 - normalized_distance
        return team_rewards
    
    def get_normalized_dot_product(self, vector1 : Box2D.b2Vec2, vector2 : Box2D.b2Vec2, max_value):
        vector2_magnitude = np.sqrt(vector2.x**2 + vector2.y**2)
        dot_product = (vector1.x * vector2.x + vector1.y * vector2.y) / (vector2_magnitude + 1e-6)
        normalized_dot_product = dot_product / max_value
        return normalized_dot_product

    def get_velocity_to_ball_reward(self, agent_idx):
        """Calculate reward for velocity towards the ball"""
        player = self.players[agent_idx]
        normalized_dot_product = self.get_normalized_dot_product(player.linearVelocity, self.ball.position - player.position, REAKISTIC_MAXIMUM_VELOCITY)
        return normalized_dot_product

    def get_velocity_to_goal_reward(self):
        """Calculate reward for velocity towards the goal"""
        team_rewards = np.zeros(self.num_teams)
        for team in range(self.num_teams):
            pos_of_enemy_goal = Box2D.b2Vec2(GAME_WIDTH / 2, GAME_HEIGHT if team == 0 else 0)
            normalized_dot_product = self.get_normalized_dot_product(self.ball.linearVelocity, pos_of_enemy_goal - self.ball.position, REAKISTIC_MAXIMUM_VELOCITY)
            team_rewards[team] = normalized_dot_product
        return team_rewards

    def get_stay_own_half_reward(self, agent_idx):
        """Calculate reward for staying in own half defined as the cirle with the center of the own goal is the center of the circle and the radius as the distance to the ball"""
        player = self.players[agent_idx]
        team = agent_idx // self.team_size
        pos_of_own_goal = Box2D.b2Vec2(GAME_WIDTH / 2, 0 if team == 0 else GAME_HEIGHT)
        radius = self.get_normalized_distance(pos_of_own_goal, self.ball.position, max_distance = 1)
        dist_to_own_goal = self.get_normalized_distance(player.position, pos_of_own_goal, max_distance = 1)
        if dist_to_own_goal < radius:
            return 1.0
        else:
            return 0.0
        
    def calculate_rewards(self, goal_scored):
        """Calculate rewards for all agents"""
        rewards = np.zeros(self.num_agents)
        team_rewards = np.zeros(self.num_teams)
        if "goal" in self.reward_specification:
            team_rewards += self.reward_specification["goal"] * self.get_goal_reward(goal_scored)
        if "winning_the_ball_and_passing" in self.reward_specification:
            team_rewards += self.reward_specification["winning_the_ball_and_passing"] * self.get_winning_the_ball_and_passing_reward()
        if "velocity_to_goal" in self.reward_specification:
            team_rewards += self.reward_specification["velocity_to_goal"] * self.get_velocity_to_goal_reward()
        if "dist_to_goal" in self.reward_specification:
            team_rewards += self.reward_specification["dist_to_goal"] * self.get_dist_to_goal_reward()
        # calculate rewards
        for team_idx in range(self.num_teams):
            team_start = team_idx * self.team_size
            team_end = (team_idx + 1) * self.team_size
            for agent_idx in range(team_start, team_end):
                if "stay_in_field" in self.reward_specification:
                    team_rewards[team_idx] += 1/self.team_size * self.reward_specification["stay_in_field"] * self.get_stay_in_field_reward(agent_idx)
                if "smoothness" in self.reward_specification:
                    team_rewards[team_idx] += 1/self.team_size * self.reward_specification["smoothness"] * self.get_smoothness_reward(agent_idx)
                if "velocity_to_ball" in self.reward_specification:
                    team_rewards[team_idx] += 1/self.team_size * self.reward_specification["velocity_to_ball"] * self.get_velocity_to_ball_reward(agent_idx)
                # Some rewards should only be calculated for one player
                if "dist_to_ball" in self.reward_specification:
                    team_rewards[team_idx] = max(team_rewards[team_idx], self.reward_specification["dist_to_ball"] * self.get_dist_to_ball_reward(agent_idx))
                if "stay_own_half" in self.reward_specification:
                    team_rewards[team_idx] = max(team_rewards[team_idx], self.reward_specification["stay_own_half"] * self.get_stay_own_half_reward(agent_idx))
        # Distribute team rewards to individual agents
        # Currently the reward needs to be the same for all agents in a tean!!!
        for i in range(self.num_agents):
            team = i // self.team_size
            rewards[i] = team_rewards[team]
        return rewards
    
    def process_action_to_velocity(self, action):
        """Convert action to local velocity vector"""
        local_vel = [0.0, 0.0]
        
        # Calculate velocity based on action
        if action == UP:
            local_vel[1] = PLAYER_SPEED
        elif action == UP_RIGHT:
            local_vel[0] = PLAYER_SPEED * 0.7071  # 1/sqrt(2) for diagonal movement
            local_vel[1] = PLAYER_SPEED * 0.7071
        elif action == RIGHT:
            local_vel[0] = PLAYER_SPEED
        elif action == DOWN_RIGHT:
            local_vel[0] = PLAYER_SPEED * 0.7071
            local_vel[1] = -PLAYER_SPEED * 0.7071
        elif action == DOWN:
            local_vel[1] = -PLAYER_SPEED
        elif action == DOWN_LEFT:
            local_vel[0] = -PLAYER_SPEED * 0.7071
            local_vel[1] = -PLAYER_SPEED * 0.7071
        elif action == LEFT:
            local_vel[0] = -PLAYER_SPEED
        elif action == UP_LEFT:
            local_vel[0] = -PLAYER_SPEED * 0.7071
            local_vel[1] = PLAYER_SPEED * 0.7071
        # NO_OP: vel remains (0, 0)
        
        return local_vel

    def step(self, actions):
        """Take a step in the environment with the given actions"""

        self.add_to_action_history(actions)

        # Process actions for each agent
        for i, action in enumerate(actions):
            player = self.players[i]
            local_vel = self.process_action_to_velocity(action)
            global_vel = self.get_global_velocity(local_vel, i)
            player.linearVelocity = Box2D.b2Vec2(global_vel[0], global_vel[1])
        
        if self.ball_toucher is not None:
            self.last_ball_toucher = self.ball_toucher
        self.ball_toucher = None

        # Update physics
        self.world.Step(1.0/FPS, 6, 2)

        self.add_to_ball_toucher_history(self.ball_toucher)
        local_position = np.array([
            self.get_local_position(
                (self.players[i].position.x, self.players[i].position.y),
                i,
                normalize=False
            ) for i in range(self.num_agents)
        ])
        self.add_to_local_position_history(local_position)
        # Check for goals
        goal_scored = self.check_goal()
        
        # Get observations
        observations = self.get_observations()
        
        # Calculate rewards
        rewards = self.calculate_rewards(goal_scored)
        
        # Check if episode is done
        self.step_count += 1
        terminated = goal_scored >= 0  # Episode ends if a goal is scored
        truncated = self.step_count >= self.max_steps  # Or if max steps reached
        
        # Reset if needed
        if terminated or truncated:
            # if self.render_mode is not None and (self.episode_count % self.video_log_freq == 0):
            #     self.save_video()
            
            # Don't actually reset here, just prepare for the next reset
            if terminated:
                self.reset_ball()
        
        # Format rewards like in mappo_selfplay_test
        info = {"other_reward": rewards[1:]}
        
        return observations, rewards[0], terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.action_history = []
        
        # Clear the world
        for body in self.world.bodies:
            self.world.DestroyBody(body)
        
        # Reset step counter
        self.step_count = 0
        self.episode_count += 1
        
        # Create boundaries, players, and ball
        self.create_boundaries()
        self.create_players()
        self.create_ball()
        
        # Reset score
        self.score = [0, 0]  # [team1_score, team2_score]
        
        # Clear frames for new episode
        self.frames = []
        
        # Get initial observations
        observations = self.get_observations()
        
        return observations, {}
    
    def render(self, mode="rgb_array"):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        self.screen.fill(BLACK)
        
        # Draw walls
        # Left wall
        pygame.draw.rect(self.screen, WHITE, (0, 0, WALL_THICKNESS * PPM, SCREEN_HEIGHT))
        # Right wall
        pygame.draw.rect(self.screen, WHITE, (SCREEN_WIDTH - WALL_THICKNESS * PPM, 0, WALL_THICKNESS * PPM, SCREEN_HEIGHT))
        
        # Draw goals and top/bottom walls
        goal_width_pixels = GOAL_WIDTH * PPM  # Convert goal width to pixels
        wall_width = (SCREEN_WIDTH - goal_width_pixels) / 2
        
        # Top walls
        pygame.draw.rect(self.screen, WHITE, (0, 0, wall_width, WALL_THICKNESS * PPM))  # Left part
        pygame.draw.rect(self.screen, WHITE, (wall_width + goal_width_pixels, 0, wall_width, WALL_THICKNESS * PPM))  # Right part
        
        # Bottom walls
        pygame.draw.rect(self.screen, WHITE, (0, SCREEN_HEIGHT - WALL_THICKNESS * PPM, wall_width, WALL_THICKNESS * PPM))  # Left part
        pygame.draw.rect(self.screen, WHITE, (wall_width + goal_width_pixels, SCREEN_HEIGHT - WALL_THICKNESS * PPM, wall_width, WALL_THICKNESS * PPM))  # Right part
        
        # Draw goal lines in a different color
        pygame.draw.rect(self.screen, GREEN, (wall_width, 0, goal_width_pixels, 2))  # Top goal line
        pygame.draw.rect(self.screen, GREEN, (wall_width, SCREEN_HEIGHT - 2, goal_width_pixels, 2))  # Bottom goal line
        
        # Draw players
        for player in self.players:
            pos = (int(player.position.x * PPM), int(player.position.y * PPM))
            team = player.userData["team"]
            color = RED if team == 0 else BLUE
            pygame.draw.rect(self.screen, color, 
                            (pos[0] - PLAYER_SIZE*PPM/2, pos[1] - PLAYER_SIZE*PPM/2, 
                            PLAYER_SIZE*PPM, PLAYER_SIZE*PPM))
        
        # Draw ball
        ball_pos = (int(self.ball.position.x * PPM), int(self.ball.position.y * PPM))
        pygame.draw.circle(self.screen, WHITE, ball_pos, int(BALL_RADIUS * PPM))
        
        # Update display
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Return rgb array
        return pygame.surfarray.array3d(self.screen)
    
    def save_video(self):
        """Save recorded frames as a video"""
        if not self.frames:
            return
        
        try:
            import numpy as np
            import imageio
            video_dir = Path(__file__).parent / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            soccer_dir = video_dir / f"{self.env_id}__MAPPOSoccer__seed{self.seed}__{current_datetime}"
            soccer_dir.mkdir(parents=True, exist_ok=True)
            filename = soccer_dir / f"rl_video_episode_{self.episode_count}.mp4"
            frames_array = np.array(self.frames)
            imageio.mimsave(filename, frames_array, fps=FPS)
            print(f"Recording saved as {filename}")
            
            # Clear frames after saving
            self.frames = []
        except ImportError:
            print("Could not save video: imageio and/or numpy not installed")
    
    def close(self):
        """Close the environment"""
        if self.render_mode is not None:
            pygame.quit()