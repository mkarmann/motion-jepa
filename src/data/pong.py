import torch
import numpy as np
from torch.utils.data import Dataset

class PongDataset(Dataset):
    state_description = [
        'left player y-position', # range [0, 1]
        'right player y-position', # range [0, 1]
        'ball x-position', # range [0, 1]
        'ball y-position', # range [0, 1]
        'ball x-velocity', # range [0, 1]
        'ball y-velocity', # range [0, 1]
        'score left player, normalized to [0, 1]',     # range 0 to 99 points is normalized to [0, 1.0]
        'score right player, normalized to [0, 1]',   # range 0 to 99 points is normalized to [0, 1.0]
    ]
    action_description = [
        'left player y-velocity', # range [0, 1]
        'right player y-velocity' # range [0, 1]
    ]
    state_groups = (
        ('Bat positions', (0, 1)),
        ('Ball position', (2, 3)),
        ('Scores', (6, 7)),
    )

    def __init__(self, img_size=224, episode_length=16, size=10000, ball_radius=0.04, bat_height=0.25, seed=None):
        """
        Dataset for ping pong game.
        :param img_size: dimensions of the output image (img_size, img_size)
        :param episode_length: how many images per episode
        :param size: How many samples in this dataset (for dataloader epochs)
        :param ball_radius: radius of the ball
        :param bat_height: height of the bat
        :param seed: Seed for the random number generator
        """
        super().__init__()
        self.name = 'Pong'
        self.state_dim = len(self.state_description)
        self.action_dim = len(self.action_description)
        self.img_size = img_size
        self.episode_length = episode_length
        self.size = size
        self.ball_radius = ball_radius
        self.bat_height = bat_height
        self.seed = seed

        self.grid_x, self.grid_y = torch.meshgrid(torch.arange(img_size, dtype=torch.float32), torch.arange(img_size, dtype=torch.float32), indexing='ij')
        self.rng = np.random.default_rng(seed)
        
        # Upgraded to a smoother 7x5 pixel representation for digits 0-9
        digits = [
            [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0], # 0
            [0,0,1,0,0, 0,1,1,0,0, 1,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0], # 1
            [0,1,1,1,0, 1,0,0,0,1, 0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 1,1,1,1,1], # 2
            [0,1,1,1,0, 1,0,0,0,1, 0,0,0,0,1, 0,0,1,1,0, 0,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0], # 3
            [0,0,0,1,0, 0,0,1,1,0, 0,1,0,1,0, 1,0,0,1,0, 1,1,1,1,1, 0,0,0,1,0, 0,0,0,1,0], # 4
            [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,0, 0,0,0,0,1, 0,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0], # 5
            [0,0,1,1,0, 0,1,0,0,0, 1,0,0,0,0, 1,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0], # 6
            [1,1,1,1,1, 0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0], # 7
            [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0], # 8
            [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,1, 0,0,0,0,1, 0,0,0,1,0, 0,1,1,0,0], # 9
        ]
        self.score_atlas = torch.tensor(digits, dtype=torch.float32).view(10, 7, 5)

        temp = self.generate_states(size, seed)
        self.states = temp['states']
        self.actions = temp['actions']

    def __len__(self):
        return self.size

    def to(self, device):
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)
        self.score_atlas = self.score_atlas.to(device)
        return self

    def simulate(self, actions, start_state, background_start_state=None, generator=None):
        """actions (B, episode_length, 2), start_state (B, 8) -> states (B, episode_length, 8)."""
        prev_state = start_state
        b, t, _ = actions.shape
        states = torch.zeros(b, t, 8, device=actions.device, dtype=actions.dtype)
        states[:, 0] = start_state

        ball_speed = 0.1
        player_bat_speed = 0.1
        opponent_bat_speed = 0.1
        
        for i in range(actions.shape[1] - 1):
            action = actions[:, i]
            next_state = prev_state.clone()

            # Move the left player bat up or down
            next_state[:, 0] += (action[:, 0] - 0.5) * 2 * player_bat_speed
            next_state[:, 0] = torch.clamp(next_state[:, 0], 0, 1)

            # Move the right player bat up or down
            next_state[:, 1] += (action[:, 1] - 0.5) * 2 * opponent_bat_speed
            next_state[:, 1] = torch.clamp(next_state[:, 1], 0, 1)

            # Move the ball
            next_state[:, 2] += (next_state[:, 4] - 0.5) * 2 * ball_speed
            next_state[:, 3] += (next_state[:, 5] - 0.5) * 2 * ball_speed

            # Check if the ball is out of bounds on the top or bottom
            above_mask = next_state[:, 3] < 0
            below_mask = next_state[:, 3] > 1
            next_state[:, 3] = torch.where(above_mask, -next_state[:, 3], next_state[:, 3])
            next_state[:, 3] = torch.where(below_mask, 2.0 - next_state[:, 3], next_state[:, 3])
            next_state[:, 5] = torch.where(above_mask, 1 - next_state[:, 5], next_state[:, 5])
            next_state[:, 5] = torch.where(below_mask, 1 - next_state[:, 5], next_state[:, 5])

            # Calculate offsets for hit detection
            global_ball_y = next_state[:, 3] * (1 - 2 * self.ball_radius) + self.ball_radius
            player_bat_y = next_state[:, 0] * (1 - self.bat_height) + self.bat_height * 0.5
            opponent_bat_y = next_state[:, 1] * (1 - self.bat_height) + self.bat_height * 0.5
            
            ball_offset_p = global_ball_y - player_bat_y
            ball_offset_o = global_ball_y - opponent_bat_y

            # Check if the ball is hit by the player bat
            hit_player = (next_state[:, 2] <= 0.07) & (torch.abs(ball_offset_p) < self.ball_radius + self.bat_height * 0.5) & (next_state[:, 4] < 0.5)

            # Check if the ball is hit by the opponent bat
            hit_opp = (next_state[:, 2] >= 0.93) & (torch.abs(ball_offset_o) < self.ball_radius + self.bat_height * 0.5) & (next_state[:, 4] > 0.5)
            
            hit_any = hit_player | hit_opp

            # --- HITTING PHYSICS ---
            # Increase x-speed on hit (max divergence from 0.5 is 0.5, meaning min/max velocities of 0.0/1.0)
            current_vx_offset = torch.abs(next_state[:, 4] - 0.5)
            new_vx_offset = torch.clamp(current_vx_offset * 1.2 + 0.05, 0.0, 0.5)
            next_state[:, 4] = torch.where(hit_player, 0.5 + new_vx_offset, next_state[:, 4]) # Bounce right
            next_state[:, 4] = torch.where(hit_opp, 0.5 - new_vx_offset, next_state[:, 4])    # Bounce left

            # Angle y-velocity based on where it hit the bat
            offset = torch.where(hit_player, ball_offset_p, ball_offset_o)
            new_vy = 0.5 + torch.clamp((offset / (self.bat_height * 0.5)) * 0.5, -0.5, 0.5)
            next_state[:, 5] = torch.where(hit_any, new_vy, next_state[:, 5])

            # --- SCORING & RESETS ---
            score_opp = next_state[:, 2] < 0.0
            score_player = next_state[:, 2] > 1.0
            score_any = score_player | score_opp

            # Add points, capped at 1.0 (representing 99)
            next_state[:, 6] = torch.clamp(next_state[:, 6] + score_player.float() * (1.0 / 99.0), 0.0, 1.0)
            next_state[:, 7] = torch.clamp(next_state[:, 7] + score_opp.float() * (1.0 / 99.0), 0.0, 1.0)

            # Put ball in middle, heading straight (y=0.5)
            next_state[:, 2] = torch.where(score_any, 0.5, next_state[:, 2])
            next_state[:, 3] = torch.where(score_any, 0.5, next_state[:, 3])
            next_state[:, 5] = torch.where(score_any, 0.5, next_state[:, 5])
            
            # Serve to the opponent of whoever scored
            serve_speed = 0.2
            next_state[:, 4] = torch.where(score_player, 0.5 + serve_speed, next_state[:, 4]) # Player scored -> Serve right (to opponent)
            next_state[:, 4] = torch.where(score_opp, 0.5 - serve_speed, next_state[:, 4])    # Opponent scored -> Serve left (to player)

            states[:, i + 1] = next_state
            prev_state = next_state
            
        return {'states': states}

    def generate_states(self, B=1, seed=None):
        """Generate (B, episode_length, 8) states and (B, episode_length, 2) actions."""
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            torch.manual_seed(seed)

        start_state = torch.rand((B, 8), generator=generator)
        start_state[:, 6:8] = torch.randint(0, 100, (B, 2), generator=generator) / 99.0
        actions = torch.rand((B, self.episode_length, self.action_dim), generator=generator)
        out = self.simulate(actions, start_state)
        out['actions'] = actions
        return out

    def draw_states(self, states, background_states=None):
        device = states.device
        dtype = states.dtype
        N = states.shape[0]

        # Normalized coordinate grids mapped to current device
        y_grid = (self.grid_x.unsqueeze(0) / self.img_size).to(device).to(dtype)
        x_grid = (self.grid_y.unsqueeze(0) / self.img_size).to(device).to(dtype)
        
        # Exact 1-pixel thickness in normalized coordinates
        px = 1.0 / self.img_size

        # --- AESTHETICS (Outlines & Net) ---
        # 1-pixel borders along the map edges.
        # Float compares at ``1 - px`` disagree on CPU vs CUDA for the last
        # row/col (CUDA includes them, CPU does not). OR-ing the integer last
        # indices matches CUDA and is a no-op on CUDA.
        border_mask = (y_grid < px) | (y_grid > 1.0 - px) | (x_grid < px) | (x_grid > 1.0 - px)
        border_mask = (
            border_mask
            | (self.grid_x.unsqueeze(0).to(device) == (self.img_size - 1))
            | (self.grid_y.unsqueeze(0).to(device) == (self.img_size - 1))
        )
        # Dashed middle line: 1-pixel thin, segmented vertically
        center_line = (torch.abs(x_grid - 0.5) < px) & ((y_grid * 30) % 1.0 < 0.5)
        
        # Combine structural elements (draw them slightly dimmer so gameplay elements pop)
        map_structure = torch.clamp(border_mask.float() + center_line.float(), 0.0, 1.0) * 0.3

        # --- GAMEPLAY ELEMENTS ---
        # 1. Bats
        player_center_y = (states[:, 0] * (1 - self.bat_height) + self.bat_height * 0.5).view(N, 1, 1)
        opp_center_y = (states[:, 1] * (1 - self.bat_height) + self.bat_height * 0.5).view(N, 1, 1)

        player_bat = (torch.abs(x_grid - 0.05) <= 0.02) & (torch.abs(y_grid - player_center_y) <= self.bat_height * 0.5)
        opp_bat = (torch.abs(x_grid - 0.95) <= 0.02) & (torch.abs(y_grid - opp_center_y) <= self.bat_height * 0.5)

        # 2. Ball
        ball_center_x = (states[:, 2] * (1 - 2 * self.ball_radius) + self.ball_radius).view(N, 1, 1)
        ball_center_y = (states[:, 3] * (1 - 2 * self.ball_radius) + self.ball_radius).view(N, 1, 1)
        ball = (x_grid - ball_center_x)**2 + (y_grid - ball_center_y)**2 <= self.ball_radius**2

        # 3. Scores (Split into Tens and Units for 00 to 99 range)
        player_score = torch.round(states[:, 6] * 99).long().clamp(0, 99).view(N, 1, 1)
        opp_score = torch.round(states[:, 7] * 99).long().clamp(0, 99).view(N, 1, 1)

        p1_tens = player_score // 10
        p1_ones = player_score % 10
        p2_tens = opp_score // 10
        p2_ones = opp_score % 10

        atlas = self.score_atlas.to(device).to(dtype)   

        # Calculate bounding boxes that strictly respect the 7:5 aspect ratio of the atlas
        score_h = 0.12
        score_w = score_h * (5.0 / 7.0)  # Prevents distortion!
        y_start = 0.05
        gap = 0.01 # Small gap between digits

        score_y_idx = torch.clamp(((y_grid - y_start) / score_h * 7).long(), 0, 6)

        # --- Player 1 score (Centered as a block around x=0.35) ---
        p1_tens_x_start = 0.35 - score_w - (gap / 2)
        score1_tens_x_idx = torch.clamp(((x_grid - p1_tens_x_start) / score_w * 5).long(), 0, 4)
        box1_tens_mask = (y_grid >= y_start) & (y_grid < y_start + score_h) & (x_grid >= p1_tens_x_start) & (x_grid < p1_tens_x_start + score_w)

        p1_ones_x_start = 0.35 + (gap / 2)
        score1_ones_x_idx = torch.clamp(((x_grid - p1_ones_x_start) / score_w * 5).long(), 0, 4)
        box1_ones_mask = (y_grid >= y_start) & (y_grid < y_start + score_h) & (x_grid >= p1_ones_x_start) & (x_grid < p1_ones_x_start + score_w)

        # --- Player 2 score (Centered as a block around x=0.65) ---
        p2_tens_x_start = 0.65 - score_w - (gap / 2)
        score2_tens_x_idx = torch.clamp(((x_grid - p2_tens_x_start) / score_w * 5).long(), 0, 4)
        box2_tens_mask = (y_grid >= y_start) & (y_grid < y_start + score_h) & (x_grid >= p2_tens_x_start) & (x_grid < p2_tens_x_start + score_w)

        p2_ones_x_start = 0.65 + (gap / 2)
        score2_ones_x_idx = torch.clamp(((x_grid - p2_ones_x_start) / score_w * 5).long(), 0, 4)
        box2_ones_mask = (y_grid >= y_start) & (y_grid < y_start + score_h) & (x_grid >= p2_ones_x_start) & (x_grid < p2_ones_x_start + score_w)

        # Project pixels
        p1_pixels = atlas[p1_tens, score_y_idx, score1_tens_x_idx] * box1_tens_mask + \
                    atlas[p1_ones, score_y_idx, score1_ones_x_idx] * box1_ones_mask
                    
        p2_pixels = atlas[p2_tens, score_y_idx, score2_tens_x_idx] * box2_tens_mask + \
                    atlas[p2_ones, score_y_idx, score2_ones_x_idx] * box2_ones_mask

        # Combine interactive foreground elements (Bright White)
        foreground = torch.clamp(player_bat.float() + opp_bat.float() + ball.float() + p1_pixels + p2_pixels, 0.0, 1.0)
        
        # Layer foreground over the structure map cleanly
        composite = torch.clamp(map_structure + foreground, 0.0, 1.0)

        # Expand mask into 3 color channels (RGB grayscale values from 0.0 to 1.0)
        images = torch.stack([composite, composite, composite], dim=1).contiguous()
        return images

    def __getitem__(self, item):
        states = self.states[item]
        actions = self.actions[item]
        images = self.draw_states(states)
        return {
            'task': self.name,
            'images': images, # (T, C, H, W) color range [0-1] in RGB
            'actions': actions, # (T, 2) actions in range [0-1]
            'states': states, # (T, 8) in range [0-1]
            'state_description': self.state_description,
            'action_description': self.action_description,
        }