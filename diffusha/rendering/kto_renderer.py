"""Rendering functions for KTO lunar lander visualization.

Ported from poslathian/lunar_lander/game.py drawing functions.
Provides terrain, obstacle, lander, trajectory, and progress bar rendering.
"""

import math
import numpy as np
import pygame
from typing import Optional, Tuple, List, Dict

# ── Colors ────────────────────────────────────────────────────────────────
SKY = (0, 0, 0)
GROUND = (50, 50, 50)
LANDER_FILL = (128, 102, 230)
LANDER_EDGE = (77, 77, 128)
FLAME_MAIN = (255, 160, 40)
FLAME_SIDE = (255, 80, 60)
FLAG_POLE = (255, 255, 255)
FLAG_COLOR = (204, 204, 0)
PAD_LINE = (200, 200, 200)
TRACE_COL = (80, 80, 120)
HUD_TEXT = (180, 180, 180)
SUCCESS_TEXT = (100, 255, 100)
FAIL_TEXT = (255, 100, 100)
OBSTACLE_COL = (180, 60, 60)
PLAN_PATH = (60, 100, 160)
PLAN_DOT = (180, 220, 255)
PROGRESS_BG = (40, 40, 40)
PROGRESS_FG = (100, 180, 255)

# Lander geometry (body coordinates in pixels)
LANDER_BODY = [(-14, 17), (-17, 0), (-17, -10), (17, -10), (17, 0), (14, 17)]
LANDER_LEGS = [(-16, -22, -10, -20), (16, 22, -10, -20)]


class KTORenderer:
    """Pygame renderer for KTO lunar lander environment.

    Handles all visualization: terrain, obstacles, lander, trajectory,
    progress bar, and HUD.
    """

    def __init__(self, scale: float = 30.0, viewport_w: int = 600,
                 viewport_h: int = 400):
        """Initialize renderer.

        Args:
            scale: Pixels per world unit
            viewport_w: Screen width in pixels
            viewport_h: Screen height in pixels
        """
        self.scale = scale
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen pixels.

        Args:
            x: World x-coordinate
            y: World y-coordinate

        Returns:
            (screen_x, screen_y) in pixels
        """
        sx = int(x * self.scale)
        sy = int(self.viewport_h - y * self.scale)
        return sx, sy

    def body_to_screen(self, body_x: float, body_y: float, world_x: float,
                       world_y: float, cos_theta: float, sin_theta: float
                       ) -> Tuple[int, int]:
        """Transform body-frame coordinate to screen space.

        Args:
            body_x, body_y: Body-frame coordinates (pixels)
            world_x, world_y: Lander position in world frame
            cos_theta, sin_theta: Lander orientation

        Returns:
            (screen_x, screen_y) in pixels
        """
        # Rotate body coordinate by theta
        rotated_x = body_x * cos_theta - body_y * sin_theta
        rotated_y = body_x * sin_theta + body_y * cos_theta

        # Translate to world frame and convert to screen
        world_x_final = world_x + rotated_x / self.scale
        world_y_final = world_y + rotated_y / self.scale

        return self.world_to_screen(world_x_final, world_y_final)

    def draw_terrain(self, surface: pygame.Surface, terrain_xs: np.ndarray,
                     terrain_ys: np.ndarray, pad_x: float, pad_y: float):
        """Draw terrain as filled polygons with landing pad line.

        Args:
            surface: Pygame surface to draw on
            terrain_xs: Terrain x-coordinates (world units)
            terrain_ys: Terrain y-coordinates (world units)
            pad_x: Landing pad center x-coordinate
            pad_y: Landing pad y-coordinate
        """
        # Draw terrain segments
        points = list(zip(terrain_xs, terrain_ys))
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            pygame.draw.polygon(surface, GROUND, [
                self.world_to_screen(x1, y1),
                self.world_to_screen(x2, y2),
                self.world_to_screen(x2, 0),
                self.world_to_screen(x1, 0)
            ])

        # Draw landing pad line
        pad_half_width = 2.0  # World units
        pygame.draw.line(
            surface, PAD_LINE,
            self.world_to_screen(pad_x - pad_half_width, pad_y),
            self.world_to_screen(pad_x + pad_half_width, pad_y),
            3
        )

    def draw_flags(self, surface: pygame.Surface, pad_x: float, pad_y: float):
        """Draw flag poles at landing pad edges.

        Args:
            surface: Pygame surface to draw on
            pad_x: Landing pad center x-coordinate
            pad_y: Landing pad y-coordinate
        """
        pad_half_width = 2.0
        flag_height = 50  # Pixels

        for flag_x in (pad_x - pad_half_width, pad_x + pad_half_width):
            base = self.world_to_screen(flag_x, pad_y)
            top = (base[0], base[1] - flag_height)

            # Pole
            pygame.draw.line(surface, FLAG_POLE, base, top, 2)

            # Flag (triangle)
            pygame.draw.polygon(surface, FLAG_COLOR, [
                top,
                (top[0], top[1] + 10),
                (top[0] + 25, top[1] + 5)
            ])

    def draw_obstacles(self, surface: pygame.Surface,
                       obstacles: List[Tuple[float, float, float]]):
        """Draw circular obstacles.

        Args:
            surface: Pygame surface to draw on
            obstacles: List of (cx, cy, radius) tuples in world units
        """
        for cx, cy, r in obstacles:
            screen_x, screen_y = self.world_to_screen(cx, cy)
            screen_radius = int(r * self.scale)
            pygame.draw.circle(surface, OBSTACLE_COL, (screen_x, screen_y),
                             screen_radius)

    def draw_lander(self, surface: pygame.Surface, x: float, y: float,
                    theta: float, Fm: float, Fs: float):
        """Draw lander with flame effects.

        Args:
            surface: Pygame surface to draw on
            x, y: Lander position (world units)
            theta: Lander angle (radians)
            Fm: Main engine thrust [N]
            Fs: Side engine thrust [N]
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        def body_to_screen(bx, by):
            return self.body_to_screen(bx, by, x, y, cos_theta, sin_theta)

        # Draw body
        body_pts = [body_to_screen(*v) for v in LANDER_BODY]
        pygame.draw.polygon(surface, LANDER_FILL, body_pts)
        pygame.draw.aalines(surface, LANDER_EDGE, True, body_pts)

        # Draw legs
        for xi, xo, yt, yb in LANDER_LEGS:
            pygame.draw.polygon(surface, LANDER_EDGE, [
                body_to_screen(xi, yt),
                body_to_screen(xo, yt),
                body_to_screen(xo, yb),
                body_to_screen(xi, yb)
            ])

        # Main engine flame
        if Fm > 0.5:
            jitter_x = np.random.uniform(-2, 2)
            flame_length = 8 + np.clip(Fm, 0, None) * 3
            pygame.draw.polygon(surface, FLAME_MAIN, [
                body_to_screen(-5, -10),
                body_to_screen(5, -10),
                body_to_screen(jitter_x, -10 - flame_length)
            ])

        # Side engine flame
        if abs(Fs) > 0.1:
            direction = np.sign(Fs)
            jitter_y = np.random.uniform(-1, 1)
            flame_length = 5 + abs(Fs) * 4
            pygame.draw.polygon(surface, FLAME_SIDE, [
                body_to_screen(direction * 17, 8),
                body_to_screen(direction * 17, -2),
                body_to_screen(direction * (17 + flame_length), 3 + jitter_y)
            ])

    def draw_trace(self, surface: pygame.Surface,
                   trace: List[Tuple[float, float]]):
        """Draw lander trajectory trace as dots.

        Args:
            surface: Pygame surface to draw on
            trace: List of (x, y) positions in world units
        """
        # Draw every 3rd point for performance
        for i in range(0, len(trace), 3):
            screen_pos = self.world_to_screen(*trace[i])
            pygame.draw.circle(surface, TRACE_COL, screen_pos, 2)

    def draw_planned_trajectory(self, surface: pygame.Surface,
                               plan: Optional[Dict[str, np.ndarray]],
                               constraint_xy: Optional[np.ndarray]):
        """Draw KTO planned trajectory as blue line with constraint dots.

        Args:
            surface: Pygame surface to draw on
            plan: Plan dict with keys ['x', 'y', 'theta', ...] or None
            constraint_xy: (N, 2) array of constraint points or None
        """
        if plan is None:
            return

        # Draw smooth path (every 3rd point for performance)
        if 'x' in plan and 'y' in plan:
            pts = [self.world_to_screen(plan['x'][i], plan['y'][i])
                   for i in range(0, len(plan['x']), 3)]
            if len(pts) > 1:
                pygame.draw.lines(surface, PLAN_PATH, False, pts, 1)

        # Draw constraint sample points
        if constraint_xy is not None:
            for x, y in constraint_xy:
                screen_pos = self.world_to_screen(x, y)
                pygame.draw.circle(surface, PLAN_DOT, screen_pos, 3)

    def draw_progress_bar(self, surface: pygame.Surface, font: pygame.font.Font,
                         phase: str, frac: float, lander_pos: Optional[Tuple[float, float, float]] = None):
        """Draw planning progress bar with phase label.

        Args:
            surface: Pygame surface to draw on
            font: Pygame font for text
            phase: Current planning phase (e.g., 'warmstart', 'obstacles')
            frac: Progress fraction [0, 1]
            lander_pos: Optional (x, y, theta) to draw static lander preview
        """
        bar_w, bar_h = 300, 20
        bar_x = (self.viewport_w - bar_w) // 2
        bar_y = self.viewport_h // 2

        # Phase label
        label = font.render(f"Planning: {phase} ...", True, HUD_TEXT)
        surface.blit(label, (bar_x, bar_y - 24))

        # Background bar
        pygame.draw.rect(surface, PROGRESS_BG, (bar_x, bar_y, bar_w, bar_h))

        # Progress fill
        fill_w = max(1, int(bar_w * frac))
        pygame.draw.rect(surface, PROGRESS_FG, (bar_x, bar_y, fill_w, bar_h))

        # Draw lander at start position (static preview)
        if lander_pos is not None:
            x, y, theta = lander_pos
            self.draw_lander(surface, x, y, theta, 0, 0)

    def draw_hud(self, surface: pygame.Surface, font: pygame.font.Font,
                 time: float, state: Dict[str, float], Fm: float, Fs: float,
                 control_mode: str, solve_time: float = 0.0):
        """Draw heads-up display with state info.

        Args:
            surface: Pygame surface to draw on
            font: Pygame font for text
            time: Episode time in seconds
            state: Lander state dict
            Fm: Main thrust [N]
            Fs: Side thrust [N]
            control_mode: 'teleop', 'heuristic', or 'kto'
            solve_time: KTO solve time (if applicable)
        """
        mode_label = control_mode.upper()
        lines = [
            f"[{mode_label}]  t={time:.2f}s",
            f"pos ({state['x']:.1f}, {state['y']:.1f})  "
            f"vel ({state['vx']:.1f}, {state['vy']:.1f})",
            f"Fm={Fm:.1f}  Fs={Fs:.2f}  theta={np.degrees(state['theta']):.1f}deg",
        ]

        if control_mode == 'kto' and solve_time > 0:
            lines[0] += f"  solve={solve_time:.2f}s"

        for row, text in enumerate(lines):
            surface.blit(font.render(text, True, HUD_TEXT), (10, 10 + row * 18))

    def draw_banner(self, surface: pygame.Surface, font: pygame.font.Font,
                   text: str, success: bool = False):
        """Draw centered banner text (for episode end).

        Args:
            surface: Pygame surface to draw on
            font: Pygame font for text
            text: Banner text
            success: True for green (landed), False for red (crashed)
        """
        color = SUCCESS_TEXT if success else FAIL_TEXT
        img = font.render(text, True, color)
        rect = img.get_rect(center=(self.viewport_w // 2, self.viewport_h // 2))

        # Black background
        pygame.draw.rect(surface, (0, 0, 0), rect.inflate(40, 20))
        surface.blit(img, rect)
