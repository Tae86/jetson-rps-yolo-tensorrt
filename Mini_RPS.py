import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
import random

# ============================================================
# 1. 모델 로드
# ============================================================
model = YOLO("best.engine", task="detect")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', 640, 480 + 100)

ansToText = {0: 'scissors', 1: 'rock', 2: 'paper'}

# ============================================================
# 1-1. SINGLE 모드용 컴퓨터 제스처 이미지
# ⚠️ 파일명이 다르면 여기 경로만 수정하면 돼
# ============================================================
IMAGE_PATHS = {
    'rock': 'rock.png',
    'paper': 'paper.png',
    'scissors': 'scissors.png',
}


def load_gesture_images(paths):
    imgs = {}
    for key, path in paths.items():
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            print(f"[WARN] '{key}' 이미지를 '{path}'에서 못 찾음 - 대체 화면으로 표시함")
        imgs[key] = im
    return imgs


GESTURE_IMAGES = load_gesture_images(IMAGE_PATHS)


def rgb(r, g, b):
    """가독성을 위해 RGB로 적고 내부에서 BGR로 변환"""
    return (b, g, r)


# ============================================================
# 2. 디자인 테마 - 딥스페이스 팔레트
# ============================================================
COLOR_BG_TOP    = rgb(15, 8, 30)     # 짙은 인디고
COLOR_BG_BOT    = rgb(30, 12, 45)    # 자주빛 네뷸라
COLOR_ACCENT    = rgb(0, 230, 255)   # 시안 네온
COLOR_TEXT      = rgb(235, 235, 245)
COLOR_TEXT_DIM  = rgb(150, 145, 170)
COLOR_DRAW      = rgb(210, 210, 220)

PLAYER_COLOR = {
    1: rgb(0, 230, 255),    # P1 - 시안
    2: rgb(255, 70, 200),   # P2 - 마젠타
}
PLAYER_LABEL = {1: "PLAYER 1", 2: "PLAYER 2"}

CLASS_COLOR = {
    'scissors': rgb(255, 90, 160),   # 플라즈마 핑크
    'rock':     rgb(210, 210, 220),  # 문록(moon rock) 실버
    'paper':    rgb(120, 190, 255),  # 아이스 블루
}
CLASS_LABEL = {'scissors': 'SCISSORS', 'rock': 'ROCK', 'paper': 'PAPER'}

FLAVOR = {
    ('rock', 'scissors'): "ROCK CRUSHES SCISSORS",
    ('scissors', 'rock'): "ROCK CRUSHES SCISSORS",
    ('scissors', 'paper'): "SCISSORS CUTS PAPER",
    ('paper', 'scissors'): "SCISSORS CUTS PAPER",
    ('paper', 'rock'): "PAPER COVERS ROCK",
    ('rock', 'paper'): "PAPER COVERS ROCK",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# ============================================================
# 3. 그리기 유틸
# ============================================================

def overlay_rect(img, pt1, pt2, color, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def gradient_rect(img, pt1, pt2, color_top, color_bottom, alpha=0.8):
    x1, y1 = pt1
    x2, y2 = pt2
    height = max(y2 - y1, 1)
    width = max(x2 - x1, 1)
    grad = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(3):
        grad[:, :, i] = np.linspace(color_top[i], color_bottom[i], height).reshape(-1, 1).astype(np.uint8)
    roi = img[y1:y2, x1:x2]
    if roi.shape[:2] == grad.shape[:2]:
        cv2.addWeighted(grad, alpha, roi, 1 - alpha, 0, roi)
        img[y1:y2, x1:x2] = roi


def apply_glow(frame, glow_canvas, intensity=0.8, downscale=2):
    h, w = frame.shape[:2]
    small = cv2.resize(glow_canvas, (w // downscale, h // downscale))
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=6, sigmaY=6)
    blurred = cv2.resize(small, (w, h))
    out = frame.astype(np.float32) + blurred.astype(np.float32) * intensity
    np.clip(out, 0, 255, out=out)
    frame[:] = out.astype(np.uint8)


def draw_star(img, x, y, size, color, thickness=1):
    """4방향 반짝이는 별(스파클) - 폰트 대신 벡터로 그려서 확실하게 렌더링"""
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
    half = int(size * 0.5)
    cv2.line(img, (x - half, y - half), (x + half, y + half), color, 1, cv2.LINE_AA)
    cv2.line(img, (x - half, y + half), (x + half, y - half), color, 1, cv2.LINE_AA)


def draw_planet(img, cx, cy, r, color, ring=True):
    """장식용 미니 행성 아이콘"""
    cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    if ring:
        cv2.ellipse(img, (cx, cy), (int(r * 1.8), int(r * 0.6)), -20, 0, 360, color, 1, cv2.LINE_AA)


class Starfield:
    """지정된 사각형 영역 안에서만 반짝이는 별 배경 (상단/하단 바 전용)"""

    def __init__(self, rect, num_stars=24):
        self.rect = rect  # (x1, y1, x2, y2)
        x1, y1, x2, y2 = rect
        self.stars = []
        for _ in range(num_stars):
            self.stars.append({
                'x': random.randint(x1, x2),
                'y': random.randint(y1, y2),
                'r': random.choice([1, 1, 1, 2]),
                'phase': random.uniform(0, math.pi * 2),
                'speed': random.uniform(1.5, 3.5),
            })

    def draw(self, img, t):
        for s in self.stars:
            b = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * s['speed'] + s['phase']))
            c = tuple(int(ch * b) for ch in COLOR_TEXT)
            cv2.circle(img, (s['x'], s['y']), s['r'], c, -1, cv2.LINE_AA)


class AmbientStars:
    """카메라 화면 전체에 아주 은은하게 떠다니는 별가루 (천천히 위로 떠오르며 반짝임)"""

    def __init__(self, w, h, top_bar_h, bottom_bar_h, num_stars=26):
        self.w, self.h = w, h
        self.top = top_bar_h
        self.bottom = h - bottom_bar_h
        self.stars = []
        for _ in range(num_stars):
            self.stars.append({
                'x': random.uniform(0, w),
                'y': random.uniform(self.top, self.bottom),
                'r': random.choice([1, 1, 2]),
                'phase': random.uniform(0, math.pi * 2),
                'speed': random.uniform(1.0, 2.2),
                'drift': random.uniform(-9, -18),   # 초당 위로 떠오르는 속도(px/s)
            })

    def update_and_draw(self, img, dt, t):
        for s in self.stars:
            s['y'] += s['drift'] * dt
            if s['y'] < self.top:
                s['y'] = self.bottom
                s['x'] = random.uniform(0, self.w)

            b = 0.20 + 0.35 * (0.5 + 0.5 * math.sin(t * s['speed'] + s['phase']))
            color = tuple(int(ch * b) for ch in COLOR_TEXT)
            cv2.circle(img, (int(s['x']), int(s['y'])), s['r'], color, -1, cv2.LINE_AA)


class Meteor:
    """짧게 스쳐 지나가는 유성 하나"""

    def __init__(self, w, h, top_bar_h):
        self.x = random.uniform(w * 0.55, w * 1.05)
        self.y = random.uniform(top_bar_h + 15, h * 0.28)
        self.vx = -random.uniform(220, 340)   # 초당 이동 픽셀 (왼쪽 아래로)
        self.vy = random.uniform(70, 130)
        self.life = random.uniform(0.5, 0.85)  # 존재 시간(초) - 짧고 굵게
        self.age = 0.0
        self.length = random.randint(28, 48)
        self.color = rgb(225, 235, 255)

    def step(self, dt, w, h):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.age += dt
        alive = (self.age < self.life and self.x > -60 and self.y < h * 0.4)
        return alive

    def draw(self, img):
        remain = max(0.0, 1 - self.age / self.life)
        alpha = 0.75 * remain
        norm = math.hypot(self.vx, self.vy) or 1
        tail_x = int(self.x - self.vx / norm * self.length)
        tail_y = int(self.y - self.vy / norm * self.length)
        overlay = img.copy()
        cv2.line(overlay, (int(self.x), int(self.y)), (tail_x, tail_y), self.color, 2, cv2.LINE_AA)
        cv2.circle(overlay, (int(self.x), int(self.y)), 2, self.color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


class MeteorShower:
    """화면 상단 영역에서 여러 유성이 동시에 스쳐가도록 관리"""

    def __init__(self, w, h, top_bar_h, max_concurrent=3):
        self.w, self.h, self.top_bar_h = w, h, top_bar_h
        self.max_concurrent = max_concurrent
        self.meteors = []
        self.timer = 0.0
        self.next_spawn = random.uniform(0.25, 0.6)

    def update_and_draw(self, img, dt):
        self.timer += dt
        if self.timer >= self.next_spawn and len(self.meteors) < self.max_concurrent:
            self.meteors.append(Meteor(self.w, self.h, self.top_bar_h))
            self.timer = 0.0
            self.next_spawn = random.uniform(0.25, 0.8)

        alive = []
        for m in self.meteors:
            if m.step(dt, self.w, self.h):
                m.draw(img)
                alive.append(m)
        self.meteors = alive


class UFO:
    """화면 상단부를 천천히 가로지르는 UFO (가끔, 은은하게)"""

    def __init__(self, w, h, top_bar_h):
        self.w, self.h, self.top_bar_h = w, h, top_bar_h
        self.active = False
        self.x = 0.0
        self.y = top_bar_h + 30
        self.speed = 0.0
        self.direction = 1
        self.timer = 0.0
        self.next_spawn = random.uniform(9.0, 16.0)

    def _spawn(self):
        self.active = True
        self.direction = random.choice([1, -1])
        self.x = -60.0 if self.direction == 1 else self.w + 60.0
        self.y = random.uniform(self.top_bar_h + 26, self.h * 0.26)
        self.speed = random.uniform(65, 100) * self.direction

    def update_and_draw(self, img, dt, t):
        if not self.active:
            self.timer += dt
            if self.timer >= self.next_spawn:
                self._spawn()
                self.timer = 0.0
            return

        self.x += self.speed * dt
        bob = math.sin(t * 4.0) * 3
        self._draw(img, int(self.x), int(self.y + bob), t)

        out_of_bounds = (self.direction == 1 and self.x > self.w + 60) or \
                         (self.direction == -1 and self.x < -60)
        if out_of_bounds:
            self.active = False
            self.next_spawn = random.uniform(14.0, 24.0)

    def _draw(self, img, cx, cy, t):
        overlay = img.copy()
        body_color = rgb(190, 195, 210)
        dome_color = rgb(120, 220, 255)
        cv2.ellipse(overlay, (cx, cy), (20, 7), 0, 0, 360, body_color, -1, cv2.LINE_AA)
        cv2.ellipse(overlay, (cx, cy - 6), (9, 9), 0, 180, 360, dome_color, -1, cv2.LINE_AA)
        for i, dx in enumerate([-13, 0, 13]):
            blink = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * 6 + i * 2.1))
            light_color = tuple(int(ch * blink) for ch in rgb(0, 255, 255))
            cv2.circle(overlay, (cx + dx, cy + 4), 2, light_color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)


class SatelliteFlyby:
    """태양광 패널이 달린 인공위성이 가끔 천천히 가로질러 지나감"""

    def __init__(self, w, h, top_bar_h):
        self.w, self.h, self.top_bar_h = w, h, top_bar_h
        self.active = False
        self.x = 0.0
        self.y = top_bar_h + 30
        self.speed = 0.0
        self.direction = 1
        self.timer = 0.0
        self.next_spawn = random.uniform(10.0, 18.0)

    def _spawn(self):
        self.active = True
        self.direction = random.choice([1, -1])
        self.x = -40.0 if self.direction == 1 else self.w + 40.0
        self.y = random.uniform(self.top_bar_h + 22, self.h * 0.24)
        self.speed = random.uniform(35, 55) * self.direction  # UFO보다 느리고 일정하게

    def update_and_draw(self, img, dt):
        if not self.active:
            self.timer += dt
            if self.timer >= self.next_spawn:
                self._spawn()
                self.timer = 0.0
            return

        self.x += self.speed * dt
        self._draw(img, int(self.x), int(self.y))

        out_of_bounds = (self.direction == 1 and self.x > self.w + 40) or \
                         (self.direction == -1 and self.x < -40)
        if out_of_bounds:
            self.active = False
            self.next_spawn = random.uniform(14.0, 24.0)

    def _draw(self, img, cx, cy):
        overlay = img.copy()
        body_color = rgb(205, 208, 215)
        panel_color = rgb(60, 90, 170)
        border_color = rgb(20, 20, 25)

        bw, bh = 8, 5
        cv2.rectangle(overlay, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2),
                      body_color, -1, cv2.LINE_AA)

        pw, ph = 12, 6
        for side in (-1, 1):
            px1 = cx + side * (bw // 2 + 2)
            px2 = px1 + side * pw
            x_lo, x_hi = min(px1, px2), max(px1, px2)
            y_lo, y_hi = cy - ph // 2, cy + ph // 2
            cv2.rectangle(overlay, (x_lo, y_lo), (x_hi, y_hi), panel_color, -1, cv2.LINE_AA)
            cv2.rectangle(overlay, (x_lo, y_lo), (x_hi, y_hi), border_color, 1, cv2.LINE_AA)
            for gx in range(x_lo + 3, x_hi, 3):
                cv2.line(overlay, (gx, y_lo), (gx, y_hi), border_color, 1, cv2.LINE_AA)

        cv2.line(overlay, (cx, cy - bh // 2), (cx, cy - bh // 2 - 4), body_color, 1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy - bh // 2 - 4), 1, body_color, -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)


class SpaceshipFlyby:
    """작은 우주선이 엔진 불꽃을 내뿜으며 가끔 빠르게 지나감"""

    def __init__(self, w, h, top_bar_h):
        self.w, self.h, self.top_bar_h = w, h, top_bar_h
        self.active = False
        self.x = 0.0
        self.y = top_bar_h + 30
        self.speed = 0.0
        self.direction = 1
        self.timer = 0.0
        self.next_spawn = random.uniform(15.0, 26.0)

    def _spawn(self):
        self.active = True
        self.direction = random.choice([1, -1])
        self.x = -40.0 if self.direction == 1 else self.w + 40.0
        self.y = random.uniform(self.top_bar_h + 20, self.h * 0.30)
        self.speed = random.uniform(140, 210) * self.direction  # UFO/위성보다 훨씬 빠름

    def update_and_draw(self, img, dt, t):
        if not self.active:
            self.timer += dt
            if self.timer >= self.next_spawn:
                self._spawn()
                self.timer = 0.0
            return

        self.x += self.speed * dt
        self._draw(img, int(self.x), int(self.y), t)

        out_of_bounds = (self.direction == 1 and self.x > self.w + 40) or \
                         (self.direction == -1 and self.x < -40)
        if out_of_bounds:
            self.active = False
            self.next_spawn = random.uniform(18.0, 30.0)

    def _draw(self, img, cx, cy, t):
        overlay = img.copy()
        body_color = rgb(235, 235, 240)
        fin_color = rgb(255, 70, 70)
        window_color = rgb(120, 220, 255)
        flame_color = rgb(255, 180, 70)

        length, width = 20, 7
        d = self.direction
        nose = (cx + (length // 2) * d, cy)
        tail_top = (cx - (length // 2) * d, cy - width // 2)
        tail_bot = (cx - (length // 2) * d, cy + width // 2)
        pts = np.array([nose, tail_top, tail_bot], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], body_color, cv2.LINE_AA)

        fin_len = 6
        fin_tip = (tail_top[0] - int(fin_len * d * 0.4), tail_top[1] - 4)
        fin_pts = np.array([tail_top, (tail_top[0], tail_top[1] - 4), fin_tip], dtype=np.int32)
        cv2.fillPoly(overlay, [fin_pts], fin_color, cv2.LINE_AA)

        window_x = cx - (length // 5) * d
        cv2.circle(overlay, (int(window_x), cy), 2, window_color, -1, cv2.LINE_AA)

        flame_len = 7 + int(5 * (0.5 + 0.5 * math.sin(t * 22)))
        tail_x = cx - (length // 2) * d
        flame_tip = (tail_x - flame_len * d, cy)
        flame_pts = np.array([tail_top, tail_bot, flame_tip], dtype=np.int32)
        cv2.fillPoly(overlay, [flame_pts], flame_color, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)


class FallingMeteor:
    """위에서 아래로 실제로 낙하하는 메테오 1개 (거의 수직, 살짝 흔들리며 떨어짐)"""

    def __init__(self, w, h, top_bar_h, ground_y):
        self.x = random.uniform(w * 0.12, w * 0.88)
        self.y = float(top_bar_h + 4)
        self.vx = random.uniform(-45, 45)
        self.vy = random.uniform(520, 680)   # 빠르게 낙하 (화면을 1초 이내 통과)
        self.length = random.randint(46, 70)
        self.color = rgb(255, 190, 120)      # 불타는 주황빛
        self.ground_y = ground_y
        self.landed = False
        self.impact_life = 0.35
        self.impact_age = 0.0

    def step(self, dt):
        if not self.landed:
            self.x += self.vx * dt
            self.y += self.vy * dt
            if self.y >= self.ground_y:
                self.y = self.ground_y
                self.landed = True
                self.impact_age = 0.0
            return True
        else:
            self.impact_age += dt
            return self.impact_age < self.impact_life

    def draw(self, img):
        if not self.landed:
            norm = math.hypot(self.vx, self.vy) or 1
            tail_x = int(self.x - self.vx / norm * self.length)
            tail_y = int(self.y - self.vy / norm * self.length)
            overlay = img.copy()
            cv2.line(overlay, (int(self.x), int(self.y)), (tail_x, tail_y), self.color, 3, cv2.LINE_AA)
            cv2.circle(overlay, (int(self.x), int(self.y)), 3, self.color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        else:
            remain = max(0.0, 1 - self.impact_age / self.impact_life)
            radius = int(6 + (1 - remain) * 26)
            alpha = 0.6 * remain
            overlay = img.copy()
            cv2.circle(overlay, (int(self.x), int(self.y)), radius, self.color, 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


class FallingMeteorEvent:
    """아주 가끔(20~35초에 한 번꼴) 메테오 하나가 떨어지는 이벤트를 관리"""

    def __init__(self, w, h, top_bar_h, bottom_bar_h):
        self.w, self.h = w, h
        self.top_bar_h = top_bar_h
        self.ground_y = h - bottom_bar_h
        self.meteor = None
        self.timer = 0.0
        self.next_spawn = 10.0

    def update_and_draw(self, img, dt):
        if self.meteor is None:
            self.timer += dt
            if self.timer >= self.next_spawn:
                self.meteor = FallingMeteor(self.w, self.h, self.top_bar_h, self.ground_y)
                self.timer = 0.0
            return

        alive = self.meteor.step(dt)
        if alive:
            self.meteor.draw(img)
        else:
            self.meteor = None
            self.next_spawn = 10.0


def build_vignette(h, w, strength=0.45):
    """화면 가장자리를 은은하게 어둡게 (시네마틱 우주 느낌)"""
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt(((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2)
    mask = np.clip(1 - strength * np.clip(dist - 0.55, 0, None), 0, 1)
    return mask.astype(np.float32)


def apply_vignette(frame, vignette_mask):
    for c in range(3):
        frame[:, :, c] = (frame[:, :, c].astype(np.float32) * vignette_mask).astype(np.uint8)


def draw_corner_box(img, glow, x1, y1, x2, y2, color, thickness=3, length=20):
    pts = [
        ((x1, y1), (x1 + length, y1)), ((x1, y1), (x1, y1 + length)),
        ((x2, y1), (x2 - length, y1)), ((x2, y1), (x2, y1 + length)),
        ((x1, y2), (x1 + length, y2)), ((x1, y2), (x1, y2 - length)),
        ((x2, y2), (x2 - length, y2)), ((x2, y2), (x2, y2 - length)),
    ]
    for p1, p2 in pts:
        cv2.line(glow, p1, p2, color, thickness + 4, cv2.LINE_AA)
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)


def draw_pill_label(img, glow, text, x, y, bg_color, border_color, text_color=(15, 15, 15)):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.52, 2)
    pad_x, pad_y = 10, 6
    x1, y1 = x, y - th - pad_y * 2
    x2, y2 = x + tw + pad_x * 2, y
    cv2.rectangle(glow, (x1, y1), (x2, y2), border_color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)
    cv2.putText(img, text, (x1 + pad_x, y2 - pad_y), FONT, 0.52, text_color, 2, cv2.LINE_AA)


def draw_center_banner(img, glow, starfield, t, main_text, sub_lines, accent_color, pulse=0.0):
    h, w = img.shape[:2]
    (mw, mh), _ = cv2.getTextSize(main_text, FONT_BOLD, 1.05, 3)
    panel_w = max(mw + 90, 300)
    line_count = len(sub_lines)
    panel_h = 56 + line_count * 20
    x1 = (w - panel_w) // 2
    y1 = 14
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    overlay_rect(img, (x1 + 5, y1 + 7), (x2 + 5, y2 + 7), (0, 0, 0), alpha=0.35)
    gradient_rect(img, (x1, y1), (x2, y2), COLOR_BG_TOP, COLOR_BG_BOT, alpha=0.85)
    starfield.draw(img, t)  # 배너 패널 안에도 별 반짝임

    line_h = 3 + int(2 * (0.5 + 0.5 * math.sin(pulse)))
    cv2.rectangle(img, (x1, y1), (x2, y1 + line_h), accent_color, -1)
    cv2.rectangle(glow, (x1, y1), (x2, y1 + line_h + 6), accent_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), accent_color, 1, cv2.LINE_AA)

    tx = x1 + (panel_w - mw) // 2
    ty = y1 + 38
    cv2.putText(glow, main_text, (tx, ty), FONT_BOLD, 1.05, accent_color, 4, cv2.LINE_AA)
    cv2.putText(img, main_text, (tx, ty), FONT_BOLD, 1.05, accent_color, 2, cv2.LINE_AA)

    ly = y1 + 58
    for line in sub_lines:
        (sw, sh), _ = cv2.getTextSize(line, FONT, 0.46, 1)
        sx = x1 + (panel_w - sw) // 2
        cv2.putText(img, line, (sx, ly), FONT, 0.46, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        ly += 20

    return y2


def draw_score_bar(img, x, y, width, ratio, color, label, count):
    height = 10
    cv2.rectangle(img, (x, y), (x + width, y + height), (55, 40, 60), -1, cv2.LINE_AA)
    fill_w = int(width * ratio)
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + height), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + width, y + height), (110, 100, 120), 1, cv2.LINE_AA)
    cv2.putText(img, f"{label} {count}", (x, y - 4), FONT, 0.42, COLOR_TEXT, 1, cv2.LINE_AA)


# ============================================================
# 3-1. 화면 레이아웃 / 메뉴·모드전환 버튼
# ============================================================
W, H = 640, 480  # 카메라 해상도 고정값 (버튼 좌표 계산용)

BTN_W, BTN_H = 210, 72
SINGLE_BTN_RECT = (W // 2 - BTN_W - 16, H // 2 - BTN_H // 2, W // 2 - 16, H // 2 + BTN_H // 2)
MULTI_BTN_RECT = (W // 2 + 16, H // 2 - BTN_H // 2, W // 2 + 16 + BTN_W, H // 2 + BTN_H // 2)
MENU_BTN_RECT = (8, 34, 78, 56)  # 게임 중 좌상단 '메뉴로' 버튼


def point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(img, rect, text, border_color, fill_alpha=0.55):
    x1, y1, x2, y2 = rect
    overlay_rect(img, (x1, y1), (x2, y2), COLOR_BG_TOP, alpha=fill_alpha)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, FONT_BOLD, 0.6, 2)
    tx = x1 + ((x2 - x1) - tw) // 2
    ty = y1 + ((y2 - y1) + th) // 2
    cv2.putText(img, text, (tx, ty), FONT_BOLD, 0.6, border_color, 2, cv2.LINE_AA)


def draw_fit_image(img, rect, cv_image, fallback_color, fallback_label):
    """이미지를 rect 영역 안에 비율 유지하며 그림. 이미지 없으면 컬러 placeholder + 텍스트"""
    x1, y1, x2, y2 = rect
    pw, ph = x2 - x1, y2 - y1

    if cv_image is not None:
        ih, iw = cv_image.shape[:2]
        scale = min(pw / iw, ph / ih) * 0.88
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(cv_image, (nw, nh))
        ox = x1 + (pw - nw) // 2
        oy = y1 + (ph - nh) // 2
        img[oy:oy + nh, ox:ox + nw] = resized
        cv2.rectangle(img, (ox - 2, oy - 2), (ox + nw + 2, oy + nh + 2), COLOR_ACCENT, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x1 + 14, y1 + 14), (x2 - 14, y2 - 14), fallback_color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(fallback_label, FONT_BOLD, 0.8, 2)
        tx = x1 + (pw - tw) // 2
        ty = y1 + (ph + th) // 2
        cv2.putText(img, fallback_label, (tx, ty), FONT_BOLD, 0.8, fallback_color, 2, cv2.LINE_AA)


def draw_computer_panel(img, rect, phase, countdown_int, cpu_gesture):
    """SINGLE 모드 왼쪽 패널: 대기 / 카운트다운 / 공개된 컴퓨터 제스처"""
    x1, y1, x2, y2 = rect
    gradient_rect(img, (x1, y1), (x2, y2), COLOR_BG_TOP, COLOR_BG_BOT, alpha=0.9)
    cv2.rectangle(img, (x1, y1), (x2, y2), PLAYER_COLOR[2], 1, cv2.LINE_AA)

    label_y = y1 + 22
    cv2.putText(img, "CPU", (x1 + 14, label_y), FONT_BOLD, 0.55, PLAYER_COLOR[2], 2, cv2.LINE_AA)

    inner_rect = (x1 + 10, label_y + 12, x2 - 10, y2 - 10)

    if phase == 'result' and cpu_gesture is not None:
        draw_fit_image(img, inner_rect, GESTURE_IMAGES.get(cpu_gesture), CLASS_COLOR.get(cpu_gesture, COLOR_ACCENT),
                        CLASS_LABEL.get(cpu_gesture, cpu_gesture))
    elif phase == 'countdown':
        (tw, th), _ = cv2.getTextSize(str(countdown_int), FONT_BOLD, 2.4, 5)
        tx = x1 + ((x2 - x1) - tw) // 2
        ty = y1 + ((y2 - y1) + th) // 2
        cv2.putText(img, str(countdown_int), (tx, ty), FONT_BOLD, 2.4, COLOR_ACCENT, 5, cv2.LINE_AA)
    else:
        text = "?"
        (tw, th), _ = cv2.getTextSize(text, FONT_BOLD, 2.4, 5)
        tx = x1 + ((x2 - x1) - tw) // 2
        ty = y1 + ((y2 - y1) + th) // 2
        cv2.putText(img, text, (tx, ty), FONT_BOLD, 2.4, COLOR_TEXT_DIM, 5, cv2.LINE_AA)


# ============================================================
# 4. 판정 로직
# ============================================================
RPS_RULES = {
    ('rock', 'scissors'): 'rock',
    ('scissors', 'rock'): 'rock',
    ('scissors', 'paper'): 'scissors',
    ('paper', 'scissors'): 'scissors',
    ('paper', 'rock'): 'paper',
    ('rock', 'paper'): 'paper',
}


def judge_rps(h1, h2):
    if h1 == h2:
        return None  # 무승부
    return RPS_RULES.get((h1, h2))  # 이긴 쪽의 gesture 문자열, 못 찾으면 None


def judge_single(player_label, cpu_label):
    """SINGLE 모드: 'you' / 'cpu' / None(무승부) 반환"""
    if player_label == cpu_label:
        return None
    winner_gesture = judge_rps(player_label, cpu_label)
    if winner_gesture is None:
        return None
    return 'you' if winner_gesture == player_label else 'cpu'


# ============================================================
# 5. Player 1 / Player 2 위치 추적 (자리 바뀌어도 정체성 유지)
# ============================================================
prev_pos = {1: None, 2: None}


def assign_players(boxes_info):
    """boxes_info: [{'cx','cy','label','conf','xyxy'}, ...] (길이 2)"""
    global prev_pos
    b0, b1 = boxes_info

    if prev_pos[1] is None or prev_pos[2] is None:
        ordered = sorted(boxes_info, key=lambda b: b['cx'])  # 처음엔 왼쪽=P1, 오른쪽=P2
        assignment = {1: ordered[0], 2: ordered[1]}
    else:
        def d(a, p):
            return (a['cx'] - p[0]) ** 2 + (a['cy'] - p[1]) ** 2

        cost_straight = d(b0, prev_pos[1]) + d(b1, prev_pos[2])
        cost_swap = d(b0, prev_pos[2]) + d(b1, prev_pos[1])
        assignment = {1: b0, 2: b1} if cost_straight <= cost_swap else {1: b1, 2: b0}

    prev_pos[1] = (assignment[1]['cx'], assignment[1]['cy'])
    prev_pos[2] = (assignment[2]['cx'], assignment[2]['cy'])
    return assignment


# ============================================================
# 6. 상태 변수
# ============================================================
app_state = 'menu'  # 'menu' | 'single' | 'multi'

score = {1: 0, 2: 0, 'draw': 0}
round_active = False
last_main_text = None
last_sub_lines = []
last_accent = COLOR_ACCENT

score_single = {'you': 0, 'cpu': 0, 'draw': 0}
single_phase = 'idle'  # 'idle' | 'countdown' | 'result'
single_countdown_timer = 0.0
single_no_hand_timer = 0.0
computer_gesture = None
player_gesture = None
single_result_text = None
single_result_sub = []
single_accent = COLOR_ACCENT

flash_timer = 0
FLASH_FRAMES = 10


def reset_multi_round():
    global round_active, last_main_text, last_sub_lines, prev_pos
    round_active = False
    last_main_text = None
    last_sub_lines = []
    prev_pos = {1: None, 2: None}


def reset_single_round():
    global single_phase, single_countdown_timer, single_no_hand_timer
    global computer_gesture, player_gesture, single_result_text, single_result_sub
    single_phase = 'idle'
    single_countdown_timer = 0.0
    single_no_hand_timer = 0.0
    computer_gesture = None
    player_gesture = None
    single_result_text = None
    single_result_sub = []


def on_mouse(event, x, y, flags, param):
    global app_state
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if app_state == 'menu':
        if point_in_rect(x, y, SINGLE_BTN_RECT):
            app_state = 'single'
            reset_single_round()
        elif point_in_rect(x, y, MULTI_BTN_RECT):
            app_state = 'multi'
            reset_multi_round()
    else:
        if point_in_rect(x, y, MENU_BTN_RECT):
            app_state = 'menu'


cv2.setMouseCallback('cam', on_mouse)

prev_time = time.time()
fps = 0.0
t0 = time.time()
last_frame_time = t0

top_bar_h = 30
bottom_bar_h = 56
starfield_top = None
starfield_bottom = None
vignette_mask = None
meteor_shower = None
ufo = None
falling_meteor_event = None
ambient_stars = None
satellite = None
spaceship = None

# ============================================================
# 7. 메인 루프
# ============================================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]

    if starfield_top is None:
        starfield_top = Starfield((0, 0, w, top_bar_h), num_stars=18)
        starfield_bottom = Starfield((0, h - bottom_bar_h, w, h), num_stars=18)
        vignette_mask = build_vignette(h, w)
        meteor_shower = MeteorShower(w, h, top_bar_h, max_concurrent=3)
        ufo = UFO(w, h, top_bar_h)
        falling_meteor_event = FallingMeteorEvent(w, h, top_bar_h, bottom_bar_h)
        ambient_stars = AmbientStars(w, h, top_bar_h, bottom_bar_h, num_stars=26)
        satellite = SatelliteFlyby(w, h, top_bar_h)
        spaceship = SpaceshipFlyby(w, h, top_bar_h)

    glow = np.zeros_like(frame)
    now = time.time()
    dt = now - last_frame_time
    last_frame_time = now
    t = now - t0
    pulse = t * 3.0

    # ---- 배경 우주 연출 (모든 화면 공통) ----
    ambient_stars.update_and_draw(frame, dt, t)
    meteor_shower.update_and_draw(frame, dt)
    ufo.update_and_draw(frame, dt, t)
    satellite.update_and_draw(frame, dt)
    spaceship.update_and_draw(frame, dt, t)
    falling_meteor_event.update_and_draw(frame, dt)

    num_hands = 0

    # ================================================================
    # A. 메뉴 화면
    # ================================================================
    if app_state == 'menu':
        frame[:] = (frame.astype(np.float32) * 0.35).astype(np.uint8)
        apply_vignette(frame, vignette_mask)

        gradient_rect(frame, (0, 0), (w, top_bar_h), COLOR_BG_TOP, COLOR_BG_BOT, alpha=0.85)
        starfield_top.draw(frame, t)
        draw_planet(frame, 18, top_bar_h // 2, 6, COLOR_ACCENT, ring=True)
        cv2.putText(frame, "COSMIC RPS ARENA", (34, 20), FONT, 0.55, COLOR_ACCENT, 1, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize("SELECT GAME MODE", FONT_BOLD, 0.7, 2)
        cv2.putText(frame, "SELECT GAME MODE", ((w - tw) // 2, H // 2 - BTN_H // 2 - 30),
                    FONT_BOLD, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

        draw_button(frame, SINGLE_BTN_RECT, "SINGLE", PLAYER_COLOR[1])
        draw_button(frame, MULTI_BTN_RECT, "MULTI", PLAYER_COLOR[2])

        hint = "Click a mode, or press 1 / 2"
        (hw, hh), _ = cv2.getTextSize(hint, FONT, 0.45, 1)
        cv2.putText(frame, hint, ((w - hw) // 2, H // 2 + BTN_H // 2 + 34),
                    FONT, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    # ================================================================
    # B. 게임 화면 (SINGLE / MULTI 공통 - 추론 실행)
    # ================================================================
    else:
        results = model(frame, imgsz=320, conf=0.5, iou=0.45, device=0, verbose=False)
        boxes = results[0].boxes
        num_hands = len(boxes) if boxes is not None else 0

        boxes_info = []
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                label = ansToText.get(cls_id, "unknown")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes_info.append({
                    'cx': (x1 + x2) / 2, 'cy': (y1 + y2) / 2,
                    'label': label, 'conf': conf, 'xyxy': (x1, y1, x2, y2),
                })

        # ------------------------------------------------------------
        # B-1. MULTI 모드
        # ------------------------------------------------------------
        if app_state == 'multi':
            p1 = p2 = None
            if num_hands == 2:
                assignment = assign_players(boxes_info)
                p1, p2 = assignment[1], assignment[2]
                for pid, info in [(1, p1), (2, p2)]:
                    x1, y1, x2, y2 = info['xyxy']
                    pcolor = PLAYER_COLOR[pid]
                    ccolor = CLASS_COLOR.get(info['label'], COLOR_ACCENT)
                    draw_corner_box(frame, glow, x1, y1, x2, y2, pcolor)
                    pill_text = f"P{pid} - {CLASS_LABEL.get(info['label'], info['label'])} {info['conf']*100:.0f}%"
                    draw_pill_label(frame, glow, pill_text, x1, y1 - 6, ccolor, pcolor)
            else:
                prev_pos[1] = None
                prev_pos[2] = None
                for info in boxes_info:
                    x1, y1, x2, y2 = info['xyxy']
                    ccolor = CLASS_COLOR.get(info['label'], COLOR_ACCENT)
                    draw_corner_box(frame, glow, x1, y1, x2, y2, ccolor)
                    pill_text = f"{CLASS_LABEL.get(info['label'], info['label'])} {info['conf']*100:.0f}%"
                    draw_pill_label(frame, glow, pill_text, x1, y1 - 6, ccolor, ccolor)

            current_main = None
            current_sub = []
            current_accent = COLOR_ACCENT
            winning_player = None

            if p1 and p2:
                winner_gesture = judge_rps(p1['label'], p2['label'])
                if winner_gesture is None:
                    current_main = "STALEMATE"
                    current_sub = ["SAME SIGNAL DETECTED",
                                   f"P1: {p1['label'].upper()}   VS   P2: {p2['label'].upper()}"]
                    current_accent = COLOR_DRAW
                else:
                    winning_player = 1 if p1['label'] == winner_gesture else 2
                    current_main = f"{PLAYER_LABEL[winning_player]} WINS!"
                    flavor = FLAVOR.get((p1['label'], p2['label']), "")
                    current_sub = [flavor, f"P1: {p1['label'].upper()}   VS   P2: {p2['label'].upper()}"]
                    current_accent = PLAYER_COLOR[winning_player]

            if num_hands == 2 and current_main and not round_active:
                round_active = True
                if winning_player:
                    score[winning_player] += 1
                else:
                    score['draw'] += 1
                last_main_text = current_main
                last_sub_lines = current_sub
                last_accent = current_accent
                flash_timer = FLASH_FRAMES
            elif num_hands < 2:
                round_active = False

            if num_hands == 2 and current_main:
                draw_center_banner(frame, glow, starfield_top, t, current_main, current_sub, current_accent, pulse)
            elif last_main_text:
                draw_center_banner(frame, glow, starfield_top, t, last_main_text,
                                    ["LAST ROUND"], (110, 105, 120), pulse)
            else:
                draw_center_banner(frame, glow, starfield_top, t, "AWAITING PLAYERS",
                                    ["SHOW 2 HANDS TO BEGIN"], COLOR_TEXT_DIM, pulse)

        # ------------------------------------------------------------
        # B-2. SINGLE 모드
        # ------------------------------------------------------------
        else:  # app_state == 'single'
            player_box = max(boxes_info, key=lambda b: b['conf']) if boxes_info else None

            if player_box is not None:
                x1, y1, x2, y2 = player_box['xyxy']
                pcolor = PLAYER_COLOR[1]
                ccolor = CLASS_COLOR.get(player_box['label'], COLOR_ACCENT)
                draw_corner_box(frame, glow, x1, y1, x2, y2, pcolor)
                pill_text = f"YOU - {CLASS_LABEL.get(player_box['label'], player_box['label'])} {player_box['conf']*100:.0f}%"
                draw_pill_label(frame, glow, pill_text, x1, y1 - 6, ccolor, pcolor)

            # ---- 라운드 상태 머신 ----
            if single_phase == 'idle':
                if player_box is not None:
                    single_phase = 'countdown'
                    single_countdown_timer = 3.0
                    single_no_hand_timer = 0.0
                    computer_gesture = None

            elif single_phase == 'countdown':
                if player_box is None:
                    single_no_hand_timer += dt
                    if single_no_hand_timer > 0.5:
                        single_phase = 'idle'
                else:
                    single_no_hand_timer = 0.0
                    single_countdown_timer -= dt
                    if single_countdown_timer <= 0:
                        computer_gesture = random.choice(['rock', 'paper', 'scissors'])
                        player_gesture = player_box['label']
                        winner_side = judge_single(player_gesture, computer_gesture)

                        if winner_side == 'you':
                            score_single['you'] += 1
                            single_result_text = "YOU WIN!"
                            single_accent = PLAYER_COLOR[1]
                        elif winner_side == 'cpu':
                            score_single['cpu'] += 1
                            single_result_text = "CPU WINS!"
                            single_accent = PLAYER_COLOR[2]
                        else:
                            score_single['draw'] += 1
                            single_result_text = "DRAW"
                            single_accent = COLOR_DRAW

                        flavor = FLAVOR.get((player_gesture, computer_gesture), "")
                        single_result_sub = [flavor,
                                              f"YOU: {player_gesture.upper()}   CPU: {computer_gesture.upper()}"]
                        flash_timer = FLASH_FRAMES
                        last_accent = single_accent
                        single_phase = 'result'

            elif single_phase == 'result':
                if player_box is None:
                    single_phase = 'idle'

            # ---- 중앙 배너 ----
            if single_phase == 'idle':
                draw_center_banner(frame, glow, starfield_top, t, "SHOW YOUR HAND",
                                    ["VS COMPUTER"], COLOR_TEXT_DIM, pulse)
            elif single_phase == 'countdown':
                draw_center_banner(frame, glow, starfield_top, t, "GET READY",
                                    [f"REVEALING IN {math.ceil(single_countdown_timer)}..."], COLOR_ACCENT, pulse)
            else:
                draw_center_banner(frame, glow, starfield_top, t, single_result_text,
                                    single_result_sub, single_accent, pulse)

        # ------------------------------------------------------------
        # B-3. 공통 후처리 (글로우 / 플래시 / 비네트)
        # ------------------------------------------------------------
        apply_glow(frame, glow, intensity=0.8)

        if flash_timer > 0:
            alpha = 0.25 * (flash_timer / FLASH_FRAMES)
            flash_layer = np.full_like(frame, last_accent, dtype=np.uint8)
            cv2.addWeighted(flash_layer, alpha, frame, 1 - alpha, 0, frame)
            flash_timer -= 1

        apply_vignette(frame, vignette_mask)

        # ---- SINGLE 모드 왼쪽 CPU 패널 (비네트 뒤에 그려서 선명하게) ----
        if app_state == 'single':
            panel_rect = (4, top_bar_h + 4, w // 2 - 4, h - bottom_bar_h - 4)
            countdown_int = max(1, math.ceil(single_countdown_timer))
            draw_computer_panel(frame, panel_rect, single_phase, countdown_int, computer_gesture)
            cv2.line(frame, (w // 2, top_bar_h), (w // 2, h - bottom_bar_h), COLOR_ACCENT, 1, cv2.LINE_AA)

        # ---- 상단 타이틀 바 ----
        gradient_rect(frame, (0, 0), (w, top_bar_h), COLOR_BG_TOP, COLOR_BG_BOT, alpha=0.85)
        starfield_top.draw(frame, t)
        draw_planet(frame, 18, top_bar_h // 2, 6, COLOR_ACCENT, ring=True)
        cv2.putText(frame, "COSMIC RPS ARENA", (34, 20), FONT, 0.55, COLOR_ACCENT, 1, cv2.LINE_AA)
        draw_star(frame, w - 40, top_bar_h // 2, 6, COLOR_TEXT)
        active_now = (num_hands == 2) if app_state == 'multi' else (num_hands >= 1)
        dot_color = (80, 255, 120) if active_now else (100, 95, 115)
        cv2.circle(frame, (w - 16, top_bar_h // 2), 5, dot_color, -1, cv2.LINE_AA)

        # ---- 모드 전환(메뉴로) 버튼 ----
        draw_button(frame, MENU_BTN_RECT, "MENU", COLOR_ACCENT, fill_alpha=0.6)

        # ---- 하단 스코어 패널 ----
        gradient_rect(frame, (0, h - bottom_bar_h), (w, h), COLOR_BG_BOT, COLOR_BG_TOP, alpha=0.85)
        starfield_bottom.draw(frame, t)

        bar_width = 130
        by = h - bottom_bar_h + 18
        if app_state == 'multi':
            m = max(1, max(score.values()))
            draw_score_bar(frame, 12, by, bar_width, score[1] / m, PLAYER_COLOR[1], "P1", score[1])
            draw_score_bar(frame, 12 + bar_width + 16, by, bar_width, score[2] / m, PLAYER_COLOR[2], "P2", score[2])
            draw_score_bar(frame, 12 + (bar_width + 16) * 2, by, 70, score['draw'] / m, COLOR_DRAW, "DRAW", score['draw'])
        else:
            m = max(1, max(score_single.values()))
            draw_score_bar(frame, 12, by, bar_width, score_single['you'] / m, PLAYER_COLOR[1], "YOU", score_single['you'])
            draw_score_bar(frame, 12 + bar_width + 16, by, bar_width, score_single['cpu'] / m, PLAYER_COLOR[2], "CPU", score_single['cpu'])
            draw_score_bar(frame, 12 + (bar_width + 16) * 2, by, 70, score_single['draw'] / m, COLOR_DRAW, "DRAW", score_single['draw'])

        if dt > 0:
            fps = 1.0 / dt
        fps_text = f"{fps:4.1f} FPS"
        (fw, fh), _ = cv2.getTextSize(fps_text, FONT, 0.45, 1)
        cv2.putText(frame, fps_text, (w - fw - 10, h - 8), FONT, 0.45, COLOR_ACCENT, 1, cv2.LINE_AA)

    cv2.imshow("cam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        app_state = 'menu'
    elif key == ord('r'):
        if app_state == 'multi':
            score = {1: 0, 2: 0, 'draw': 0}
            last_main_text = None
            last_sub_lines = []
        elif app_state == 'single':
            score_single = {'you': 0, 'cpu': 0, 'draw': 0}
            single_result_text = None
    elif app_state == 'menu' and key == ord('1'):
        app_state = 'single'
        reset_single_round()
    elif app_state == 'menu' and key == ord('2'):
        app_state = 'multi'
        reset_multi_round()

cap.release()
cv2.destroyAllWindows()