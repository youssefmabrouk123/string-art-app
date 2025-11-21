class HQStringArt:
    def __init__(self, pins=400, size=900):
        self.pins = pins
        self.size = size
        self.radius = size // 2 - 12
        self.center = (size // 2, size // 2)

    # --------------------------------------------------------------------
    # 1) PIN GENERATION
    # --------------------------------------------------------------------
    def make_pins(self):
        ang = np.linspace(0, 2*np.pi, self.pins, endpoint=False)
        return [
            (
                int(self.center[0] + self.radius * np.cos(a)),
                int(self.center[1] + self.radius * np.sin(a))
            )
            for a in ang
        ]

    # --------------------------------------------------------------------
    # 2) IMAGE PREPROCESSING — optimized for string art
    # --------------------------------------------------------------------
    def preprocess(self, img):
        img = img.convert("L")
        img = ImageOps.fit(img, (self.size, self.size), Image.LANCZOS)

        arr = np.array(img).astype(np.float32)

        # Increased contrast for better edge tracking
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        arr = clahe.apply(arr.astype(np.uint8)).astype(np.float32)

        # Gamma correction (CRITICAL)
        arr = (arr / 255.0) ** 0.65 * 255.0

        # Invert: dark = more strings needed
        arr = 255 - arr

        # Circular mask
        mask = np.zeros_like(arr, dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius + 5, 255, -1)
        arr = arr * (mask / 255)

        self.target = arr
        self.work = arr.copy()

    # --------------------------------------------------------------------
    # 3) Bresenham for fast line extraction
    # --------------------------------------------------------------------
    def line_pts(self, x0, y0, x1, y1):
        pts = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            pts.append((y0, x0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return pts

    # --------------------------------------------------------------------
    # 4) OPTIMIZED SOLVER
    # --------------------------------------------------------------------
    def solve(self, max_lines=30000):

        pins = self.make_pins()
        seq = [0]
        cur = 0

        # Precompute lines between every pair (major speed upgrade)
        cache = {}

        for i in range(self.pins):
            for j in range(i+1, self.pins):
                x0, y0 = pins[i]
                x1, y1 = pins[j]
                pts = self.line_pts(x0, y0, x1, y1)
                cache[(i, j)] = pts
                cache[(j, i)] = pts

        for k in range(max_lines):

            best_score = -1
            best_pin = None
            best_pts = None

            # Try pin distances only between 15–200
            for d in range(15, self.pins // 2):
                p = (cur + d) % self.pins
                pts = cache[(cur, p)]
                local_vals = [self.work[y, x] for y, x in pts]
                score = np.mean(local_vals)

                if score > best_score:
                    best_score = score
                    best_pin = p
                    best_pts = pts

            # Subtract thread darkness
            for y, x in best_pts:
                self.work[y, x] = max(self.work[y, x] - 12.5, 0)

            seq.append(best_pin)
            cur = best_pin

        self.sequence = seq

    # --------------------------------------------------------------------
    # 5) High-quality rendering
    # --------------------------------------------------------------------
    def render(self, thickness=2):
        img = Image.new("L", (self.size, self.size), 255)
        draw = ImageDraw.Draw(img)

        pins = self.make_pins()

        for i in range(len(self.sequence)-1):
            a = pins[self.sequence[i]]
            b = pins[self.sequence[i+1]]
            draw.line([a, b], fill=0, width=thickness)

        return img
