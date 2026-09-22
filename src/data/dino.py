import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

# Golden-ratio conjugate: low-discrepancy steps under mod-1 (same as flappy gap).
SPEED_STEP = 0.5 * (math.sqrt(5.0) - 1.0)

# Mean of the three cloud parallax speeds (base scroll rate).
_CLOUD_SCROLL = 0.022
N_HEARTS = 10
_HEARTS_PER_ROW = 5
_CLOUD_Y_SHIFT = 0.08
# Parallax: highest / mid / lowest cloud. Multipliers of `_CLOUD_SCROLL`.
_CLOUD_SPEED_MULTS = (0.45, 1.0, 1.85)
_SCROLL_DIMS = (0, 8, 9)
_STATE_DIM = 10

_WHITE = (255, 255, 255, 255)
# Classic 7×6 pixel heart (one cell = one pixel).
_HEART_FILLED = frozenset({
    (1, 0), (2, 0), (4, 0), (5, 0),
    (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
    (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
    (1, 3), (2, 3), (3, 3), (4, 3), (5, 3),
    (2, 4), (3, 4), (4, 4),
    (3, 5),
})
_HEART_OUTLINE = frozenset(
    (x, y) for x, y in _HEART_FILLED
    if any((x + dx, y + dy) not in _HEART_FILLED for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)))
)


def _rgba(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _crop_alpha(tex: torch.Tensor, thresh: float = 0.05) -> torch.Tensor:
    opaque = tex[3] > thresh
    rows = opaque.any(dim=1).nonzero(as_tuple=False)
    cols = opaque.any(dim=0).nonzero(as_tuple=False)
    if rows.numel() == 0 or cols.numel() == 0:
        return tex
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return tex[:, r0:r1, c0:c1]


def _scale(tex: torch.Tensor, height: int) -> torch.Tensor:
    _, h0, w0 = tex.shape
    width = max(8, round(height * w0 / max(h0, 1)))
    return F.interpolate(tex.unsqueeze(0), size=(height, width), mode='nearest').squeeze(0)


def _fill_transparent_rgb(tex: torch.Tensor, rgb) -> torch.Tensor:
    fill = torch.tensor(rgb, dtype=tex.dtype).view(3, 1, 1)
    tex = tex.clone()
    tex[:3] = torch.where(tex[3:4] > 0, tex[:3], fill.expand_as(tex[:3]))
    return tex


def _boxes(img_size, grid_w, grid_h, out_h, parts, fill_rgb):
    img = Image.new('RGBA', (grid_w, grid_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def box(x0, y0, x1, y1, color):
        d.rectangle((x0, y0, x1, y1), fill=color)

    for part in parts:
        box(*part)
    tex = _scale(_crop_alpha(_rgba(img)), max(10, round(img_size * out_h)))
    return _fill_transparent_rgb(tex, fill_rgb)


def _make_dino(img_size: int) -> torch.Tensor:
    body, dark, eye, belly = (18, 18, 16, 255), (8, 8, 7, 255), (4, 4, 4, 255), (28, 28, 24, 255)
    return _boxes(img_size, 48, 40, 0.11, [
        (2, 20, 10, 24, body), (0, 18, 6, 22, body),
        (10, 16, 28, 28, body), (12, 22, 26, 30, belly),
        (24, 10, 32, 20, body), (28, 6, 42, 16, body), (30, 8, 40, 14, belly), (38, 8, 40, 10, eye),
        (40, 10, 46, 15, body), (42, 12, 46, 13, dark),
        (26, 20, 32, 23, body), (30, 22, 34, 24, dark),
        (12, 28, 18, 36, body), (12, 34, 18, 38, dark),
        (20, 28, 26, 36, body), (20, 34, 26, 38, dark),
    ], [18 / 255, 18 / 255, 16 / 255])


def _make_cactus(img_size: int) -> torch.Tensor:
    green, dark, light = (16, 22, 16, 255), (8, 12, 8, 255), (24, 30, 22, 255)
    return _boxes(img_size, 28, 48, 0.14, [
        (11, 4, 17, 47, green), (11, 4, 13, 47, dark), (15, 4, 17, 47, light),
        (4, 16, 12, 20, green), (4, 10, 8, 20, green), (4, 10, 6, 20, dark),
        (16, 22, 24, 26, green), (20, 14, 24, 26, green), (22, 14, 24, 26, light),
        (12, 2, 16, 6, green),
    ], [16 / 255, 22 / 255, 16 / 255])


def _make_heart(img_size: int, pixels) -> torch.Tensor:
    """Integer-scaled 7×6 pixel heart (avoids the 8px-wide floor in ``_scale``)."""
    img = Image.new('RGBA', (7, 6), (0, 0, 0, 0))
    for x, y in pixels:
        img.putpixel((x, y), _WHITE)
    tex = _rgba(img)
    cell = max(1, round(img_size * 0.075 / 6.0))
    if cell > 1:
        tex = tex.repeat_interleave(cell, dim=1).repeat_interleave(cell, dim=2)
    return _fill_transparent_rgb(tex, [1.0, 1.0, 1.0])


def _make_heart_strips(img_size: int, n_hearts: int = N_HEARTS) -> torch.Tensor:
    """HUD strips for 0–n filled hearts, wrapped into rows of ``_HEARTS_PER_ROW``."""
    filled = _make_heart(img_size, _HEART_FILLED)
    empty = _make_heart(img_size, _HEART_OUTLINE)
    _, sh, sw = filled.shape
    gap_px = max(1, round(img_size * 0.010))
    n_cols = min(_HEARTS_PER_ROW, n_hearts)
    n_rows = (n_hearts + n_cols - 1) // n_cols
    strip_w = n_cols * sw + (n_cols - 1) * gap_px
    strip_h = n_rows * sh + (n_rows - 1) * gap_px
    strips = filled.new_zeros(n_hearts + 1, 4, strip_h, strip_w)
    for n_filled in range(n_hearts + 1):
        for i in range(n_hearts):
            row, col = divmod(i, n_cols)
            x0 = col * (sw + gap_px)
            y0 = row * (sh + gap_px)
            strips[n_filled, :, y0:y0 + sh, x0:x0 + sw] = filled if i < n_filled else empty
    return strips


def _make_cloud_layers(img_size: int) -> torch.Tensor:
    """Three x-tilable clouds as separate RGBA tiles (same silhouettes as Dino)."""
    s = img_size
    fill = (214, 220, 238, 235)

    def layer(ellipses):
        img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        def ellipse(cx, cy, rx, ry):
            for dx in (-s, 0, s):
                draw.ellipse((cx + dx - rx, cy - ry, cx + dx + rx, cy + ry), fill=fill)

        for args in ellipses:
            ellipse(*args)
        return _fill_transparent_rgb(_rgba(img), [c / 255.0 for c in fill[:3]])

    # Compact 3-lobe (highest).
    compact = layer([
        (0.18 * s, 0.28 * s + 0.010 * s, 0.095 * s, 0.030 * s),
        (0.18 * s - 0.048 * s, 0.28 * s - 0.004 * s, 0.042 * s, 0.028 * s),
        (0.18 * s + 0.012 * s, 0.28 * s - 0.016 * s, 0.046 * s, 0.032 * s),
        (0.18 * s + 0.055 * s, 0.28 * s - 0.002 * s, 0.036 * s, 0.026 * s),
    ])
    # Two fat lobes / peanut (lowest).
    k = 0.78
    peanut = layer([
        (0.52 * s - 0.038 * s * k, 0.42 * s, 0.070 * s * k, 0.052 * s * k),
        (0.52 * s + 0.042 * s * k, 0.42 * s + 0.006 * s * k, 0.058 * s * k, 0.042 * s * k),
        (0.52 * s, 0.42 * s - 0.018 * s * k, 0.040 * s * k, 0.032 * s * k),
    ])
    # Long low train (mid height).
    train = layer([
        (0.84 * s, 0.34 * s + 0.010 * s, 0.17 * s, 0.032 * s),
        (0.84 * s - 0.112 * s, 0.34 * s - 0.002 * s, 0.042 * s, 0.030 * s),
        (0.84 * s - 0.038 * s, 0.34 * s - 0.018 * s, 0.050 * s, 0.038 * s),
        (0.84 * s + 0.042 * s, 0.34 * s - 0.010 * s, 0.044 * s, 0.032 * s),
        (0.84 * s + 0.118 * s, 0.34 * s + 0.004 * s, 0.038 * s, 0.026 * s),
    ])
    # Back-to-front: high/slow, mid, low/fast.
    return torch.stack([compact, train, peanut], dim=0)


class DinoDataset(Dataset):
    state_description = [
        'cloud 0 scroll [0, 1] (highest, slowest)',
        'dino y-position [0, 1] (0 = ground, 1 = jump apex)',
        'dino y-velocity [0, 1] (0.5 = rest, >0.5 up, <0.5 down)',
        'cactus x-position [0, 1]',
        'cactus x-velocity [0, 1] (leftward speed [0.5, 1.5] times base speed)',
        'hearts [0, 1] (0-10 as n/10; 0 = frozen)',
        'cloud height [0, 1] (0 = higher, 1 = lower)',
        'sky color [0, 1] (0 = cooler, 1 = warmer)',
        'cloud 1 scroll [0, 1] (mid height)',
        'cloud 2 scroll [0, 1] (lowest, fastest)',
    ]
    action_description = [
        'jump [0, 1] (<0.5 none, ≥0.5 jump from ground)',
    ]
    state_groups = (
        ('Cactus', (3,)),
        ('Clouds', (0, 6, 8, 9)),
        ('Dino', (1,)),
        ('Hearts', (5,)),
        ('Sky', (7,)),
    )

    def __init__(
        self,
        img_size=224,
        episode_length=16,
        size=10000,
        jump_speed=0.18,
        gravity=0.05,
        cactus_speed=0.055,
        scroll_speed=_CLOUD_SCROLL,
        warmup_steps=64,
        seed=None,
    ):
        super().__init__()
        self.name = 'Dino'
        self.state_dim = _STATE_DIM
        self.action_dim = 1
        self.img_size = img_size
        self.episode_length = episode_length
        self.size = size
        self.jump_speed = float(jump_speed)
        self.gravity = float(gravity)
        y, vy, apex = 0.0, self.jump_speed, 0.0
        for _ in range(64):
            y += vy
            vy -= self.gravity
            if y > apex:
                apex = y
            if y <= 0.0:
                break
        self.apex = apex
        self.cactus_speed = float(cactus_speed)
        self.cactus_speed_lo = 0.5 * self.cactus_speed
        self.cactus_speed_hi = 1.5 * self.cactus_speed
        self.scroll_speed = float(scroll_speed)
        self.warmup_steps = max(1, int(warmup_steps))
        self.seed = seed
        self.dino_x = 0.14
        self.dino_half_w = 0.028
        self.sky_top = (0.015, 0.02, 0.07)
        self.sky_mid = (0.12, 0.12, 0.38)
        self.sky_horizon = (0.72, 0.58, 0.88)
        self.ground_color = (0.06, 0.05, 0.04)
        self.ground_thickness = 0.07
        self.ground_cy = 1.0 - self.ground_thickness
        self.jump_range = 0.40 * self.apex
        self.dino = _make_dino(img_size)
        self.cactus = _make_cactus(img_size)
        cactus_h = self.cactus.shape[1] / float(img_size)
        cactus_w = self.cactus.shape[2] / float(img_size)
        self.cactus_half_w = 0.45 * cactus_w
        self.clear_y = max(0.25, (cactus_h - 0.03) / self.jump_range)
        self.heart_strips = _make_heart_strips(self.img_size)
        sw = int(self.heart_strips.shape[-1])
        self._heart_px = self.img_size - round(0.03 * self.img_size) - sw
        self._heart_py = round(0.028 * self.img_size)
        self.cloud_layers = _make_cloud_layers(self.img_size)
        self.background = self.cloud_layers[0]
        out = self.generate_states(size, seed)
        self.states, self.actions = out['states'], out['actions']

    def __len__(self):
        return self.size

    def to(self, device):
        self.dino = self.dino.to(device)
        self.cactus = self.cactus.to(device)
        self.heart_strips = self.heart_strips.to(device)
        self.cloud_layers = self.cloud_layers.to(device)
        self.background = self.cloud_layers[0]
        return self

    def _cactus_step(self, vel):
        """Map cactus velocity state [0, 1] → leftward speed per frame."""
        return self.cactus_speed_lo + vel * (self.cactus_speed_hi - self.cactus_speed_lo)

    def _heart_count(self, hearts):
        return torch.round(hearts * N_HEARTS).long().clamp(0, N_HEARTS)

    def _pad_state(self, start_state):
        d = start_state.shape[-1]
        if d > _STATE_DIM:
            return start_state[..., :_STATE_DIM]
        need = _STATE_DIM - d
        if need <= 0:
            return start_state
        extra = start_state.new_empty(*start_state.shape[:-1], need)
        extra.uniform_(0.0, 1.0)
        return torch.cat([start_state, extra], dim=-1)

    def generate_states(self, B=1, seed=None):
        self.state_dim = _STATE_DIM
        g = None
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
            torch.manual_seed(seed)

        warm0 = torch.zeros(B, self.state_dim)
        warm0[:, 0] = torch.rand(B, generator=g)
        warm0[:, 2] = 0.5
        warm0[:, 3] = torch.rand(B, generator=g)
        warm0[:, 4] = torch.rand(B, generator=g)
        warm0[:, 5] = 1.0
        warm0[:, 6] = torch.rand(B, generator=g)
        warm0[:, 7] = torch.rand(B, generator=g)
        warm0[:, 8] = torch.rand(B, generator=g)
        warm0[:, 9] = torch.rand(B, generator=g)
        warm_act = torch.randint(0, 2, (B, self.warmup_steps, 1), generator=g).to(warm0.dtype)
        warm_states = self.simulate(warm_act, warm0)['states']

        t0 = torch.randint(0, self.warmup_steps, (B,), generator=g)
        start = warm_states[torch.arange(B), t0].clone()
        start[:, 5] = torch.randint(1, N_HEARTS + 1, (B,), generator=g).to(start.dtype) / float(N_HEARTS)
        start[:, 6] = torch.rand(B, generator=g)
        start[:, 7] = torch.rand(B, generator=g)

        actions = torch.randint(0, 2, (B, self.episode_length, 1), generator=g).to(start.dtype)
        out = self.simulate(actions, start)
        out['actions'] = actions
        return out

    def simulate(self, actions, start_state, background_start_state=None, generator=None):
        self.state_dim = _STATE_DIM
        start_state = self._pad_state(start_state)
        prev = start_state
        b, t, _ = actions.shape
        states = torch.zeros(b, t, self.state_dim, device=actions.device, dtype=actions.dtype)
        states[:, 0] = start_state
        speeds = self.scroll_speed * torch.tensor(
            _CLOUD_SPEED_MULTS, device=actions.device, dtype=actions.dtype,
        )
        for i in range(t - 1):
            alive = self._heart_count(prev[:, 5]) > 0
            s = prev.clone()
            for dim, speed in zip(_SCROLL_DIMS, speeds):
                s[:, dim] = (s[:, dim] + speed) % 1.0

            y = s[:, 1] * self.apex
            vy = (s[:, 2] - 0.5) * 2.0 * self.jump_speed
            on_ground = (y <= 1e-6) & (vy <= 1e-6)
            vy = torch.where(on_ground & (actions[:, i, 0] >= 0.5), torch.full_like(vy, self.jump_speed), vy)
            y = y + vy
            vy = vy - self.gravity
            landed = y <= 0.0
            y = torch.where(landed, torch.zeros_like(y), y)
            vy = torch.where(landed, torch.zeros_like(vy), vy)
            s[:, 1] = y / self.apex
            s[:, 2] = (0.5 + 0.5 * (vy / self.jump_speed)).clamp(0.0, 1.0)

            x = s[:, 3] - self._cactus_step(s[:, 4])
            wrap = x < 0.0
            x = torch.where(wrap, torch.ones_like(x), x)
            vel = torch.where(wrap, torch.remainder(s[:, 4] + SPEED_STEP, 1.0), s[:, 4])
            s[:, 3] = x
            s[:, 4] = vel

            hit = ((x - self.dino_x).abs() < self.cactus_half_w + self.dino_half_w) & (s[:, 1] < self.clear_y)
            hearts = self._heart_count(s[:, 5]).to(dtype=s.dtype)
            fatal = hit & (hearts <= 1)
            recover = hit & ~fatal
            s[:, 1] = torch.where(recover, torch.zeros_like(s[:, 1]), s[:, 1])
            s[:, 2] = torch.where(recover, torch.full_like(s[:, 2], 0.5), s[:, 2])
            s[:, 3] = torch.where(recover, torch.ones_like(x), s[:, 3])
            s[:, 4] = torch.where(recover, torch.remainder(s[:, 4] + SPEED_STEP, 1.0), s[:, 4])
            hearts = torch.where(hit, hearts - 1.0, hearts).clamp(min=0.0)
            s[:, 5] = hearts / float(N_HEARTS)

            s = torch.where(alive.unsqueeze(-1), s, prev)
            states[:, i + 1] = s
            prev = s
        return {'states': states}

    def _scroll_cloud_layers(self, scrolls, h, w, y_shift):
        """Scroll each cloud tile independently. ``scrolls`` is (N, K)."""
        n, k = scrolls.shape
        device, dtype = scrolls.device, scrolls.dtype
        layers = self.cloud_layers.to(device=device, dtype=dtype)
        tile_w = layers.shape[-1]
        ys = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        nk = n * k
        scroll = scrolls.reshape(nk, 1, 1)
        dy = (y_shift.unsqueeze(1).expand(n, k).reshape(nk, 1, 1) - 0.5) * 4.0 * _CLOUD_Y_SHIFT
        sample_x = ((grid_x + 1.0) * 0.5 * w + scroll * tile_w + 0.5) / tile_w * 2.0 - 1.0
        sample_x = torch.remainder(sample_x + 1.0, 2.0) - 1.0
        sample_y = grid_y.expand(nk, -1, -1) - dy
        grid = torch.stack([sample_x, sample_y], dim=-1)
        premult = torch.cat([layers[:, :3] * layers[:, 3:4], layers[:, 3:4]], dim=1)
        premult = premult.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(nk, 4, layers.shape[2], layers.shape[3])
        sampled = torch.nn.functional.grid_sample(
            premult, grid, mode='bilinear', padding_mode='border', align_corners=False,
        )
        a = sampled[:, 3:4].clamp(0.0, 1.0)
        rgb = sampled[:, :3] / a.clamp(min=1e-6)
        return rgb.view(n, k, 3, h, w), a.view(n, k, 1, h, w)

    def _sky_colors(self, tint, device, dtype):
        """Per-sample sky palette. tint=0.5 matches Dino5 defaults."""
        t = tint.view(-1, 1).to(device=device, dtype=dtype)
        default = torch.tensor(
            [self.sky_top, self.sky_mid, self.sky_horizon],
            device=device, dtype=dtype,
        )
        delta = torch.tensor(
            [
                (-0.02, 0.02, 0.04),
                (-0.05, 0.04, 0.08),
                (-0.14, 0.08, 0.06),
            ],
            device=device, dtype=dtype,
        )
        palette = default.unsqueeze(0) + (2.0 * t.view(-1, 1, 1) - 1.0) * delta.unsqueeze(0)
        return palette.clamp(0.0, 1.0).unbind(dim=1)

    def _blit(self, images, sprite, cx, feet_cy):
        """Alpha-composite ``sprite`` with feet at ``(cx, feet_cy)``, batched over N."""
        n, _, h, w = images.shape
        sprite = sprite.to(device=images.device, dtype=images.dtype)
        _, sh, sw = sprite.shape
        px = (cx * w - sw * 0.5).round().long()
        py = (feet_cy * h - sh).round().long()
        ys = torch.arange(h, device=images.device)
        xs = torch.arange(w, device=images.device)
        sy = ys.view(1, h, 1) - py.view(n, 1, 1)
        sx = xs.view(1, 1, w) - px.view(n, 1, 1)
        valid = (sy >= 0) & (sy < sh) & (sx >= 0) & (sx < sw)
        patch = sprite[:, sy.clamp(0, sh - 1), sx.clamp(0, sw - 1)].permute(1, 0, 2, 3)
        a = patch[:, 3:4] * valid.unsqueeze(1).to(dtype=patch.dtype)
        return patch[:, :3] * a + images * (1.0 - a)

    def _draw_hearts(self, images, hearts):
        n = images.shape[0]
        hud = self.heart_strips[self._heart_count(hearts.reshape(n))]
        hud = hud.to(device=images.device, dtype=images.dtype)
        _, _, sh, sw = hud.shape
        py, px = self._heart_py, self._heart_px
        patch = images[:, :, py:py + sh, px:px + sw]
        a = hud[:, 3:4]
        images = images.clone()
        images[:, :, py:py + sh, px:px + sw] = hud[:, :3] * a + patch * (1.0 - a)
        return images

    def draw_states(self, states, background_states=None):
        n, device, dtype = states.shape[0], states.device, states.dtype
        h = w = self.img_size
        tint = states[:, 7] if states.shape[-1] > 7 else torch.full((n,), 0.5, device=device, dtype=dtype)
        cloud_y = states[:, 6] if states.shape[-1] > 6 else torch.full((n,), 0.5, device=device, dtype=dtype)
        top, mid, bot = self._sky_colors(tint, device, dtype)
        t = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1)
        u = t.pow(0.85)
        split = 0.48
        lo = (u / split).clamp(0.0, 1.0)
        hi = ((u - split) / (1.0 - split)).clamp(0.0, 1.0)
        top = top.view(n, 3, 1, 1)
        mid = mid.view(n, 3, 1, 1)
        bot = bot.view(n, 3, 1, 1)
        images = top + (mid - top) * lo + (bot - mid) * hi
        images = images.expand(n, 3, h, w).clone()
        scrolls = torch.stack(
            [states[:, d] if states.shape[-1] > d else states[:, 0] for d in _SCROLL_DIMS],
            dim=1,
        )
        rgb, a = self._scroll_cloud_layers(scrolls, h, w, y_shift=cloud_y)
        for i in range(rgb.shape[1]):
            images = rgb[:, i] * a[:, i] + images * (1.0 - a[:, i])
        gy0 = round(self.ground_cy * h)
        ground = torch.tensor(self.ground_color, device=device, dtype=dtype).view(1, 3, 1, 1)
        images[:, :, gy0:h, :] = ground
        feet_g = torch.full((n,), self.ground_cy, device=device, dtype=dtype)
        images = self._blit(images, self.cactus, states[:, 3], feet_g)
        feet = self.ground_cy - states[:, 1] * self.jump_range
        cx = torch.full((n,), self.dino_x, device=device, dtype=dtype)
        images = self._blit(images, self.dino, cx, feet)
        if states.shape[-1] > 5:
            hearts = states[:, 5]
        else:
            hearts = torch.ones(n, device=device, dtype=dtype)
        return self._draw_hearts(images, hearts).clamp(0.0, 1.0)

    def __getitem__(self, item):
        states, actions = self.states[item], self.actions[item]
        return {
            'images': self.draw_states(states),
            'states': states,
            'actions': actions,
            'state_description': self.state_description,
            'action_description': self.action_description,
        }
