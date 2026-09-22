import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

WOOD_PATH = Path(__file__).resolve().parents[2] / 'images' / 'wood_texture.png'
WOOD_CROP = 224
_HOLE_Y = 0.5
_HOLE_X_STATE = 0.32


def _load_wood_texture(path=WOOD_PATH):
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class GolfDataset(Dataset):
    state_description = [
        'ball x-position',  # range [0, 1]
        'ball y-position',  # range [0, 1]
        'cross angle',  # range [0, 1] → quarter turn (``+`` 4-fold symmetry)
        'cross velocity',  # range [0, 1] → [-1, 1] × angular_velocity
        'blue bar top offset',  # [0, 1] wraps in x
        'blue bar top velocity',  # [0, 1] → [-1, 1] × bar_speed
        'blue bar bot offset',  # [0, 1] wraps in x
        'blue bar bot velocity',  # [0, 1] → [-1, 1] × bar_speed
        'green bar offset',  # [0, 1] wraps in y (left horizontal bar)
        'green bar velocity',  # [0, 1] → [-1, 1] × bar_speed
        'wood crop x',  # [0, 1] top-left of the 224 crop
        'wood crop y',
    ]
    action_description = [
        'ball x-velocity',  # range [0, 1]
        'ball y-velocity',  # range [0, 1]
    ]
    # Same groups / dims as the probing-table Golf columns (bars = blue + green).
    state_groups = (
        ('Ball position', (0, 1)),
        ('Red cross', (2,)),
        ('Bars', (4, 6, 8)),
        ('Wood crop', (10, 11)),
    )
    bar_speed = 0.045
    wood_crop = WOOD_CROP
    hole_y = _HOLE_Y

    def __init__(
        self,
        img_size=224,
        episode_length=16,
        size=10000,
        ball_radius=0.04,
        obstacle_angular_velocity=0.035,
        seed=None,
    ):
        super().__init__()
        self.blue_ys = (0.065, 0.935)
        self.blue_half_len = 0.10
        self.green_cx = 0.085
        self.green_half_len = 0.10

        self.name = 'Golf'
        self.state_dim = len(self.state_description)
        self.action_dim = len(self.action_description)
        self.img_size = img_size
        self.episode_length = episode_length
        self.size = 0
        self.ball_radius = ball_radius
        self.obstacle_angular_velocity = obstacle_angular_velocity
        self.seed = seed

        self.border_x = 0.025
        self.border_y = 0.22
        self.hole_radius = 0.045
        self.obstacle_arm = 0.215
        self.obstacle_half_thick = 0.030
        self.ball_speed = 0.045
        self.reset_sx = 1.0
        self.reset_sy = 0.5
        self.obstacle_cx = 0.5
        self.obstacle_cy = 0.5

        self.grid_x, self.grid_y = torch.meshgrid(
            torch.arange(img_size, dtype=torch.float32),
            torch.arange(img_size, dtype=torch.float32),
            indexing='ij',
        )
        self.rng = np.random.default_rng(seed)
        temp = self.generate_states(0, seed)
        self.states = temp['states']
        self.actions = temp['actions']

        self.border_x = 0.025
        self.border_y = 0.025
        self.obstacle_arm = 0.27
        self.obstacle_cx = 0.46
        ix = self.grid_y.long()
        iy = self.grid_x.long()
        wall_tex = torch.zeros(1, 3, img_size, img_size)
        wh = (ix * 5 + iy * 11) % 6
        wall_tex[0, :, wh == 0] -= 0.035
        wall_tex[0, :, wh == 2] += 0.025
        wall_tex[0, 1, wh == 4] -= 0.02
        self._wall_detail = wall_tex
        self.size = size
        temp = self.generate_states(size, seed)
        self.states = temp['states']
        self.actions = temp['actions']
        self.wood_texture = _load_wood_texture()

    def __len__(self):
        return self.size

    def to(self, device):
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)
        self.wood_texture = self.wood_texture.to(device)
        self._wall_detail = self._wall_detail.to(device)
        return self

    def _obstacle_cos_sin(self, states):
        theta = states[:, 2] * (0.5 * math.pi)
        return torch.cos(theta), torch.sin(theta)

    def _obstacle_omega(self, states):
        return (states[:, 3] * 2.0 - 1.0) * self.obstacle_angular_velocity * 4.0

    def _bar_omega(self, vel):
        return (vel * 2.0 - 1.0) * self.bar_speed

    def _play_lo_hi(self):
        x_lo = self.border_x + self.ball_radius
        x_hi = 1.0 - self.border_x - self.ball_radius
        y_lo = self.border_y + self.ball_radius
        y_hi = 1.0 - self.border_y - self.ball_radius
        return x_lo, x_hi, y_lo, y_hi

    def _ball_xy(self, states):
        x_lo, x_hi, y_lo, y_hi = self._play_lo_hi()
        bx = x_lo + states[:, 0] * (x_hi - x_lo)
        by = y_lo + states[:, 1] * (y_hi - y_lo)
        return bx, by

    def _hole_xy(self, states):
        x_lo = self.border_x + self.hole_radius
        y_lo = self.border_y + self.hole_radius
        y_hi = 1.0 - self.border_y - self.hole_radius
        x_hi = self.obstacle_cx - self.obstacle_arm - self.hole_radius - 0.02
        x_hi = max(x_hi, x_lo + 1e-3)
        n = states.shape[0]
        hx = x_lo + _HOLE_X_STATE * (x_hi - x_lo)
        hy = y_lo + self.hole_y * (y_hi - y_lo)
        return (
            torch.full((n,), hx, device=states.device, dtype=states.dtype),
            torch.full((n,), hy, device=states.device, dtype=states.dtype),
        )

    def _obstacle_hit(self, bx, by, cos_t, sin_t):
        dx = bx - self.obstacle_cx
        dy = by - self.obstacle_cy
        lx = cos_t * dx + sin_t * dy
        ly = -sin_t * dx + cos_t * dy
        arm = self.obstacle_arm + self.ball_radius
        thick = self.obstacle_half_thick + self.ball_radius
        return ((lx.abs() <= arm) & (ly.abs() <= thick)) | ((ly.abs() <= arm) & (lx.abs() <= thick))

    def _wrap_axis_hit(self, ball_a, ball_b, center_a, rail_bs, half_a, half_b):
        hit = torch.zeros_like(ball_a, dtype=torch.bool)
        pad_a = half_a + self.ball_radius
        pad_b = half_b + self.ball_radius
        for shift in (-1.0, 0.0, 1.0):
            along = (ball_a - (center_a + shift)).abs() <= pad_a
            for rail in rail_bs:
                hit = hit | (along & ((ball_b - rail).abs() <= pad_b))
        return hit

    def _blue_hit(self, bx, by, off_top, off_bot):
        return (
            self._wrap_axis_hit(bx, by, off_top, (self.blue_ys[0],), self.obstacle_half_thick, self.blue_half_len)
            | self._wrap_axis_hit(bx, by, off_bot, (self.blue_ys[1],), self.obstacle_half_thick, self.blue_half_len)
        )

    def _green_hit(self, bx, by, offset):
        hit = torch.zeros_like(bx, dtype=torch.bool)
        pad_x = self.green_half_len + self.ball_radius
        pad_y = self.obstacle_half_thick + self.ball_radius
        along_x = (bx - self.green_cx).abs() <= pad_x
        for shift in (-1.0, 0.0, 1.0):
            hit = hit | (along_x & ((by - (offset + shift)).abs() <= pad_y))
        return hit

    def _any_hit(self, states):
        cos_t, sin_t = self._obstacle_cos_sin(states)
        bx, by = self._ball_xy(states)
        return (
            self._obstacle_hit(bx, by, cos_t, sin_t)
            | self._blue_hit(bx, by, states[:, 4], states[:, 6])
            | self._green_hit(bx, by, states[:, 8])
        )

    def simulate(self, actions, start_state, background_start_state=None, generator=None):
        del background_start_state
        if start_state.shape[-1] < self.state_dim:
            pad = torch.rand(
                *start_state.shape[:-1], self.state_dim - start_state.shape[-1],
                device=start_state.device, dtype=start_state.dtype, generator=generator,
            )
            start_state = torch.cat([start_state, pad], dim=-1)
        prev_state = start_state
        b, t, _ = actions.shape
        states = torch.zeros(b, t, self.state_dim, device=actions.device, dtype=actions.dtype)
        states[:, 0] = start_state

        x_lo, x_hi, y_lo, y_hi = self._play_lo_hi()
        x_span = x_hi - x_lo
        y_span = y_hi - y_lo
        crop = start_state[..., -2:]

        for i in range(t - 1):
            action = actions[:, i]
            blocked = self._any_hit(prev_state)
            next_state = prev_state.clone()
            next_state[:, 0] = (prev_state[:, 0] + (action[:, 0] - 0.5) * 2.0 * self.ball_speed / x_span).clamp(0.0, 1.0)
            next_state[:, 1] = (prev_state[:, 1] + (action[:, 1] - 0.5) * 2.0 * self.ball_speed / y_span).clamp(0.0, 1.0)
            next_state[:, 2] = torch.remainder(prev_state[:, 2] + self._obstacle_omega(prev_state), 1.0)
            next_state[:, 3] = start_state[:, 3]
            next_state[:, 4] = torch.remainder(prev_state[:, 4] + self._bar_omega(start_state[:, 5]), 1.0)
            next_state[:, 5] = start_state[:, 5]
            next_state[:, 6] = torch.remainder(prev_state[:, 6] + self._bar_omega(start_state[:, 7]), 1.0)
            next_state[:, 7] = start_state[:, 7]
            next_state[:, 8] = torch.remainder(prev_state[:, 8] + self._bar_omega(start_state[:, 9]), 1.0)
            next_state[:, 9] = start_state[:, 9]
            next_state[:, 10] = crop[:, 0]
            next_state[:, 11] = crop[:, 1]
            next_state = torch.where(blocked.unsqueeze(-1), prev_state, next_state)
            next_state[:, 10] = crop[:, 0]
            next_state[:, 11] = crop[:, 1]

            states[:, i + 1] = next_state
            prev_state = next_state

        states[..., -2:] = crop.unsqueeze(1).expand(-1, t, -1)
        return {'states': states}

    def generate_states(self, B=1, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            torch.manual_seed(seed)

        start_state = torch.rand((B, self.state_dim), generator=generator)
        bad = self._any_hit(start_state)
        for _ in range(64):
            if not bool(bad.any()):
                break
            idx = bad.nonzero(as_tuple=False).squeeze(-1)
            start_state = start_state.clone()
            start_state[idx] = torch.rand((idx.numel(), self.state_dim), generator=generator)
            bad = bad.clone()
            bad[idx] = self._any_hit(start_state[idx])

        actions = torch.rand((B, self.episode_length, self.action_dim), generator=generator)
        out = self.simulate(actions, start_state, generator=generator)
        out['actions'] = actions
        return out

    def _blue_mask(self, x_grid, y_grid, center, rail):
        n = center.shape[0]
        mask = torch.zeros(n, *x_grid.shape[-2:], device=x_grid.device, dtype=torch.bool)
        half_x = self.obstacle_half_thick
        half_y = self.blue_half_len
        along_y = (y_grid - rail).abs() <= half_y
        for shift in (-1.0, 0.0, 1.0):
            c = (center + shift).view(n, 1, 1)
            mask = mask | (((x_grid - c).abs() <= half_x) & along_y)
        return mask

    def _green_mask(self, x_grid, y_grid, center):
        n = center.shape[0]
        mask = torch.zeros(n, *x_grid.shape[-2:], device=x_grid.device, dtype=torch.bool)
        along_x = (x_grid - self.green_cx).abs() <= self.green_half_len
        for shift in (-1.0, 0.0, 1.0):
            c = (center + shift).view(n, 1, 1)
            mask = mask | (along_x & ((y_grid - c).abs() <= self.obstacle_half_thick))
        return mask

    def _wood_floors(self, states):
        device, dtype = states.device, states.dtype
        tex = self.wood_texture.to(device=device, dtype=dtype)
        _, th, tw = tex.shape
        crop = self.wood_crop
        max_x = max(tw - crop, 0)
        max_y = max(th - crop, 0)
        x0 = (states[:, -2].to(dtype) * max_x).long().clamp(0, max_x)
        y0 = (states[:, -1].to(dtype) * max_y).long().clamp(0, max_y)
        yy = y0[:, None, None] + torch.arange(crop, device=device).view(1, crop, 1)
        xx = x0[:, None, None] + torch.arange(crop, device=device).view(1, 1, crop)
        crops = tex[:, yy, xx].permute(1, 0, 2, 3).contiguous()
        if self.img_size != crop:
            crops = F.interpolate(
                crops, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False,
            )
        return crops.clamp(0.0, 1.0)

    def draw_states(self, states, background_states=None):
        del background_states
        device = states.device
        dtype = states.dtype
        n = states.shape[0]
        shadow_ox = 0.012
        shadow_oy = 0.014

        y_grid = (self.grid_x.unsqueeze(0) / self.img_size).to(device=device, dtype=dtype)
        x_grid = (self.grid_y.unsqueeze(0) / self.img_size).to(device=device, dtype=dtype)

        images = self._wood_floors(states)
        border_c = torch.tensor([0.28, 0.16, 0.08], device=device, dtype=dtype).view(1, 3, 1, 1)
        border_lo = torch.tensor([0.16, 0.09, 0.04], device=device, dtype=dtype).view(1, 3, 1, 1)

        px = 1.0 / self.img_size
        border_mask = (
            (x_grid < self.border_x)
            | (x_grid > 1.0 - self.border_x)
            | (y_grid < self.border_y)
            | (y_grid > 1.0 - self.border_y)
        )
        images = torch.where(border_mask.unsqueeze(1), border_c, images)
        field_x = (x_grid >= self.border_x) & (x_grid <= 1.0 - self.border_x)
        field_y = (y_grid >= self.border_y) & (y_grid <= 1.0 - self.border_y)
        inner_lip = border_mask & (
            (((x_grid >= self.border_x - px * 2) & (x_grid < self.border_x)) & field_y)
            | (((x_grid > 1.0 - self.border_x) & (x_grid <= 1.0 - self.border_x + px * 2)) & field_y)
            | (((y_grid >= self.border_y - px * 2) & (y_grid < self.border_y)) & field_x)
            | (((y_grid > 1.0 - self.border_y) & (y_grid <= 1.0 - self.border_y + px * 2)) & field_x)
        )
        images = torch.where(inner_lip.unsqueeze(1), border_lo, images)
        images = torch.where(
            border_mask.unsqueeze(1),
            (images + self._wall_detail.to(device=device, dtype=dtype)).clamp(0.0, 1.0),
            images,
        )

        def _paint(mask, color):
            nonlocal images
            c = color.view(1, 3, 1, 1) if color.ndim == 1 else color
            images = torch.where(mask.unsqueeze(1), c, images)
            return images

        def _shadow(mask, alpha=0.4):
            nonlocal images
            m = mask.to(dtype).unsqueeze(1)
            images = images * (1.0 - alpha * m)
            return images

        shade_w = 0.035
        d_wall = torch.minimum(
            torch.minimum(x_grid - self.border_x, (1.0 - self.border_x) - x_grid),
            torch.minimum(y_grid - self.border_y, (1.0 - self.border_y) - y_grid),
        )
        field = ~border_mask
        wall_shade = field & (d_wall < shade_w) & (d_wall >= 0)
        shade_a = ((1.0 - d_wall / shade_w).clamp(0.0, 1.0) * 0.4) * wall_shade.to(dtype)
        images = images * (1.0 - shade_a.unsqueeze(1))

        hx, hy = self._hole_xy(states)
        hx = hx.view(n, 1, 1)
        hy = hy.view(n, 1, 1)
        dist2 = (x_grid - hx) ** 2 + (y_grid - hy) ** 2
        _shadow(dist2 <= (self.hole_radius * 1.25) ** 2, alpha=0.45)
        _paint(dist2 <= self.hole_radius ** 2, torch.tensor([0.08, 0.08, 0.08], device=device, dtype=dtype))
        _paint(dist2 <= (self.hole_radius * 0.55) ** 2, torch.tensor([0.02, 0.02, 0.02], device=device, dtype=dtype))

        blue_top = states[:, 4]
        blue_bot = states[:, 6]
        green = states[:, 8]
        _shadow(self._blue_mask(x_grid, y_grid, blue_top + shadow_ox, self.blue_ys[0]), alpha=0.45)
        _shadow(self._blue_mask(x_grid, y_grid, blue_bot + shadow_ox, self.blue_ys[1]), alpha=0.45)
        _paint(self._blue_mask(x_grid, y_grid, blue_top, self.blue_ys[0]), torch.tensor([0.15, 0.40, 0.90], device=device, dtype=dtype))
        _paint(self._blue_mask(x_grid, y_grid, blue_bot, self.blue_ys[1]), torch.tensor([0.15, 0.40, 0.90], device=device, dtype=dtype))
        _shadow(self._green_mask(x_grid, y_grid, green + shadow_oy), alpha=0.45)
        _paint(self._green_mask(x_grid, y_grid, green), torch.tensor([0.20, 0.72, 0.22], device=device, dtype=dtype))

        cos_t, sin_t = self._obstacle_cos_sin(states)
        c = cos_t.view(n, 1, 1)
        s = sin_t.view(n, 1, 1)

        def _cross_at(cx, cy):
            dx = x_grid - cx
            dy = y_grid - cy
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            return (
                (lx.abs() <= self.obstacle_arm) & (ly.abs() <= self.obstacle_half_thick)
            ) | (
                (ly.abs() <= self.obstacle_arm) & (lx.abs() <= self.obstacle_half_thick)
            )

        _shadow(_cross_at(self.obstacle_cx + shadow_ox, self.obstacle_cy + shadow_oy), alpha=0.45)
        _paint(
            _cross_at(self.obstacle_cx, self.obstacle_cy),
            torch.tensor([0.85, 0.12, 0.10], device=device, dtype=dtype),
        )

        bx, by = self._ball_xy(states)
        bx = bx.view(n, 1, 1)
        by = by.view(n, 1, 1)
        ball_shadow = (x_grid - (bx + shadow_ox)) ** 2 + (y_grid - (by + shadow_oy)) ** 2 <= (self.ball_radius * 1.05) ** 2
        _shadow(ball_shadow, alpha=0.4)
        ball = (x_grid - bx) ** 2 + (y_grid - by) ** 2 <= self.ball_radius ** 2
        _paint(ball, torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype))
        return images.contiguous()

    def __getitem__(self, item):
        states = self.states[item]
        actions = self.actions[item]
        images = self.draw_states(states)
        return {
            'task': self.name,
            'images': images,
            'actions': actions,
            'states': states,
            'state_description': self.state_description,
            'action_description': self.action_description,
        }
