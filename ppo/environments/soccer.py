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

FPS = 30
PLAYER_SIZE = 1.5  # meters
BALL_RADIUS = 0.8  # meters
PLAYER_SPEED = 9.0
WALL_THICKNESS = 1.0
GOAL_WIDTH = 15.0  # meters in width

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
DOWN = 1
LEFT = 2
RIGHT = 3
NO_OP = 4

class Soccer(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode=None, video_log_freq=100, env_id="Soccer-v0", seed=1):
        super().__init__()
        
        # Environment parameters
        self.num_agents = 4
        self.team_size = 2
        self.num_teams = self.num_agents // self.team_size
        self.max_steps = 300
        self.video_log_freq = video_log_freq
        self.render_mode = render_mode
        self.env_id = env_id
        self.seed = seed
        
        # Observation and action spaces
        # Each agent observes: own position (2), teammate position (2), 
        # enemy positions (2*2), ball position (2) = 10 values
        self.observation_space = Box(
            low=np.array([[0.0 for _ in range(10)] for _ in range(self.num_agents)]),
            high=np.array([[float(max(GAME_WIDTH, GAME_HEIGHT)) for _ in range(10)] for _ in range(self.num_agents)]),
            dtype=np.float32
        )
        
        # 5 actions for each agent: UP, DOWN, LEFT, RIGHT, NO_OP
        self.action_space = MultiDiscrete([5, 5, 5, 5])
        
        # Initialize pygame if rendering is needed
        if self.render_mode is not None:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Box2D Soccer")
            self.clock = pygame.time.Clock()
        
        # Initialize Box2D world
        self.world = world(gravity=(0, 0), doSleep=True)
        
        # For video recording
        self.frames = []
        self.step_count = 0
        self.episode_count = 0
        
        # Reset to initialize everything
        self.reset()
    
    def create_boundaries(self):
        # Create walls and goals
        # Left wall
        self.world.CreateStaticBody(
            position=(0, SCREEN_HEIGHT/(2*PPM)),
            shapes=polygonShape(box=(WALL_THICKNESS, SCREEN_HEIGHT/(2*PPM))),
        )
        
        # Right wall
        self.world.CreateStaticBody(
            position=(SCREEN_WIDTH/PPM, SCREEN_HEIGHT/(2*PPM)),
            shapes=polygonShape(box=(WALL_THICKNESS, SCREEN_HEIGHT/(2*PPM))),
        )
        
        # Top wall (with goal opening)
        self.create_goal_wall(True)  # Top
        
        # Bottom wall (with goal opening)
        self.create_goal_wall(False)  # Bottom
    
    def create_goal_wall(self, is_top):
        wall_width = (SCREEN_WIDTH/PPM - GOAL_WIDTH) / 2
        y_pos = 0 if is_top else SCREEN_HEIGHT/PPM
        
        # Left part
        self.world.CreateStaticBody(
            position=(wall_width/2, y_pos),
            shapes=polygonShape(box=(wall_width/2, WALL_THICKNESS)),
        )
        
        # Right part
        self.world.CreateStaticBody(
            position=(SCREEN_WIDTH/PPM - wall_width/2, y_pos),
            shapes=polygonShape(box=(wall_width/2, WALL_THICKNESS)),
        )
    
    def create_players(self):
        # Create 4 players (2 per team)
        # Team 1: Players 0 and 1 (RED team - bottom)
        # Team 2: Players 2 and 3 (BLUE team - top)
        
        self.players = []
        
        # Team 1 - Player 0 (bottom left)
        player0 = self.world.CreateDynamicBody(
            position=(SCREEN_WIDTH/(4*PPM), SCREEN_HEIGHT/(6*PPM)),
            fixtures=Box2D.b2FixtureDef(
                shape=polygonShape(box=(PLAYER_SIZE/2, PLAYER_SIZE/2)),
                density=1.0,
                friction=0.3,
            ),
        )
        player0.userData = {"team": 0, "id": 0}
        self.players.append(player0)
        
        # Team 1 - Player 1 (bottom right)
        player1 = self.world.CreateDynamicBody(
            position=(3*SCREEN_WIDTH/(4*PPM), SCREEN_HEIGHT/(6*PPM)),
            fixtures=Box2D.b2FixtureDef(
                shape=polygonShape(box=(PLAYER_SIZE/2, PLAYER_SIZE/2)),
                density=1.0,
                friction=0.3,
            ),
        )
        player1.userData = {"team": 0, "id": 1}
        self.players.append(player1)
        
        # Team 2 - Player 2 (top left)
        player2 = self.world.CreateDynamicBody(
            position=(SCREEN_WIDTH/(4*PPM), 5*SCREEN_HEIGHT/(6*PPM)),
            fixtures=Box2D.b2FixtureDef(
                shape=polygonShape(box=(PLAYER_SIZE/2, PLAYER_SIZE/2)),
                density=1.0,
                friction=0.3,
            ),
        )
        player2.userData = {"team": 1, "id": 2}
        self.players.append(player2)
        
        # Team 2 - Player 3 (top right)
        player3 = self.world.CreateDynamicBody(
            position=(3*SCREEN_WIDTH/(4*PPM), 5*SCREEN_HEIGHT/(6*PPM)),
            fixtures=Box2D.b2FixtureDef(
                shape=polygonShape(box=(PLAYER_SIZE/2, PLAYER_SIZE/2)),
                density=1.0,
                friction=0.3,
            ),
        )
        player3.userData = {"team": 1, "id": 3}
        self.players.append(player3)
    
    def create_ball(self):
        self.ball = self.world.CreateDynamicBody(
            position=(SCREEN_WIDTH/(2*PPM), SCREEN_HEIGHT/(2*PPM)),
            fixtures=Box2D.b2FixtureDef(
                shape=circleShape(radius=BALL_RADIUS),
                density=0.1,
                friction=0.3,
                restitution=0.8,
            ),
        )
        self.ball.userData = {"type": "ball"}
        self.ball_touched = False
        self.last_ball_toucher = None
    
    def reset_ball(self):
        self.ball.position = (SCREEN_WIDTH/(2*PPM), SCREEN_HEIGHT/(2*PPM))
        self.ball.linearVelocity = (0, 0)
        self.ball.angularVelocity = 0
        self.ball_touched = False
        self.last_ball_toucher = None
    
    def check_goal(self):
        ball_pos = self.ball.position
        if ball_pos.y < 0:  # Bottom goal (Team 2 scores)
            self.score[1] += 1
            return 1  # Team 2 scored
        elif ball_pos.y > SCREEN_HEIGHT/PPM:  # Top goal (Team 1 scores)
            self.score[0] += 1
            return 0  # Team 1 scored
        return -1  # No goal
    
    def get_local_position(self, pos, agent_id):
        """Convert global position to local position for the given agent"""
        # For soccer, we'll define local coordinates as:
        # Team 1 (bottom): y-axis points up
        # Team 2 (top): y-axis points down
        # Left players: x-axis points right
        # Right players: x-axis points left
        
        x, y = pos
        
        # Team 1 (bottom)
        if agent_id == 0:  # Bottom left
            return np.array([x, y])
        elif agent_id == 1:  # Bottom right
            return np.array([SCREEN_WIDTH/PPM - x, y])
        # Team 2 (top)
        elif agent_id == 2:  # Top left
            return np.array([x, SCREEN_HEIGHT/PPM - y])
        elif agent_id == 3:  # Top right
            return np.array([SCREEN_WIDTH/PPM - x, SCREEN_HEIGHT/PPM - y])
    
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
            
            # Add own position (local coordinates)
            own_pos = self.get_local_position(
                (self.players[i].position.x, self.players[i].position.y), 
                i
            )
            agent_observation.append(own_pos)
            
            # Add teammate positions (local coordinates)
            for j in teammate_indices:
                teammate_pos = self.get_local_position(
                    (self.players[j].position.x, self.players[j].position.y),
                    i
                )
                agent_observation.append(teammate_pos)
            
            # Add enemy positions (local coordinates)
            for j in enemy_indices:
                enemy_pos = self.get_local_position(
                    (self.players[j].position.x, self.players[j].position.y),
                    i
                )
                agent_observation.append(enemy_pos)
            
            # Add ball position (local coordinates)
            ball_pos = self.get_local_position(
                (self.ball.position.x, self.ball.position.y),
                i
            )
            agent_observation.append(ball_pos)
            
            # Flatten and add to observations
            observations.append(np.concatenate(agent_observation))
        
        return np.array(observations, dtype=np.float32)
    
    def calculate_rewards(self, goal_scored):
        """Calculate rewards for all agents"""
        rewards = np.zeros(self.num_agents)
        
        if goal_scored >= 0:  # A goal was scored
            scoring_team = goal_scored
            for i in range(self.num_agents):
                team = i // self.team_size
                if team == scoring_team:
                    rewards[i] = 10.0  # Big reward for scoring team
                else:
                    rewards[i] = -10.0  # Penalty for conceding team
            return rewards
        for team in range(self.num_teams):
            team_start = team * self.team_size
            team_end = (team + 1) * self.team_size
            
            # Target y-coordinate for the goal
            target_y = SCREEN_HEIGHT/PPM if team == 0 else 0
            
            if self.ball_touched:
                # Reward based on ball distance to opponent's goal
                ball_distance = abs(self.ball.position.y - target_y) / (SCREEN_HEIGHT/PPM)
                team_reward = 0.1 * (1.0 - ball_distance)  # Higher reward when ball is closer to goal
            else:
                # Reward based on closest player distance to ball
                min_distance = float('inf')
                for i in range(team_start, team_end):
                    player = self.players[i]
                    distance = np.sqrt(
                        (player.position.x - self.ball.position.x)**2 + 
                        (player.position.y - self.ball.position.y)**2
                    )
                    min_distance = min(min_distance, distance)
                
                # Normalize distance and convert to reward
                normalized_distance = min_distance / (np.sqrt((SCREEN_WIDTH/PPM)**2 + (SCREEN_HEIGHT/PPM)**2))
                team_reward = 0.05 * (1.0 - normalized_distance)  # Higher reward when player is closer to ball
            
            # Assign the same reward to all team members
            for i in range(team_start, team_end):
                rewards[i] = team_reward
        
        return rewards
    
    def step(self, actions):
        """Take a step in the environment with the given actions"""
        # Process actions for each agent
        for i, action in enumerate(actions):
            player = self.players[i]
            vel = Box2D.b2Vec2(0, 0)
            
            if action == UP:
                vel.y = PLAYER_SPEED
            elif action == DOWN:
                vel.y = -PLAYER_SPEED
            elif action == LEFT:
                vel.x = -PLAYER_SPEED
            elif action == RIGHT:
                vel.x = PLAYER_SPEED
            # NO_OP: vel remains (0, 0)
            
            player.linearVelocity = vel
        
        # Update physics
        self.world.Step(1.0/FPS, 6, 2)
        
        # Check for ball contact
        for player in self.players:
            # Simple distance-based check for ball contact
            distance = np.sqrt(
                (player.position.x - self.ball.position.x)**2 + 
                (player.position.y - self.ball.position.y)**2
            )
            if distance < (PLAYER_SIZE/2 + BALL_RADIUS):
                self.ball_touched = True
                self.last_ball_toucher = player.userData["id"]
        
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
        
        # Render if needed
        # if self.render_mode is not None:
        #     self.render()
        #     # Save frame for video if needed
        #     if self.episode_count % self.video_log_freq == 0:
        #         frame_copy = self.screen.copy()
        #         self.frames.append(pygame.surfarray.array3d(frame_copy))
        
        # Format rewards like in mappo_selfplay_test
        info = {"other_reward": rewards[1:]}
        
        return observations, rewards[0], terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
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
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = f"{self.score[0]} - {self.score[1]}"
        text_surface = font.render(score_text, True, WHITE)
        self.screen.blit(text_surface, (SCREEN_WIDTH/2 - 30, 10))
        
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