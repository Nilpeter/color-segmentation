import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from CTkColorPicker import AskColor
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISPLAY_MAX_WIDTH = 1448
SIM_WIDTH = 2112
SIM_HEIGHT = 500

# Simulated shape positions / sizes (approximate real PRE_TEST captures)
SIM_CIRCLE_CENTER = (500, 250)
SIM_CIRCLE_RADIUS = 100
SIM_RECT_TL = (1400, 150)
SIM_RECT_BR = (1700, 350)

# Default simulated colours (BGR)
DEFAULT_SLAVE_BGR = (255, 255, 0)   # Cyan
DEFAULT_MASTER_BGR = (0, 255, 255)  # Yellow
DEFAULT_BG_BGR = (240, 240, 240)


# ---------------------------------------------------------------------------
# NumericInput – small entry widget with .get()/.set() interface
# ---------------------------------------------------------------------------

class NumericInput(ctk.CTkFrame):
    """Compact numeric entry with clamping, matching CTkSlider .get()/.set() API."""

    def __init__(self, master, from_=0, to=255, default=0, width=60,
                 command=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._min = from_
        self._max = to
        self._command = command

        self._var = ctk.StringVar(value=str(int(default)))
        self._entry = ctk.CTkEntry(self, textvariable=self._var,
                                    width=width, height=28,
                                    justify="center")
        self._entry.pack()

        # Validate / trigger on every key release and focus-out
        self._entry.bind("<KeyRelease>", self._on_change)
        self._entry.bind("<FocusOut>", self._on_change)

    def get(self):
        try:
            v = int(self._var.get())
        except (ValueError, tk.TclError):
            v = self._min
        return max(self._min, min(self._max, v))

    def set(self, value):
        self._var.set(str(int(max(self._min, min(self._max, value)))))

    def _on_change(self, _event=None):
        if self._command:
            self._command(self.get())


# ---------------------------------------------------------------------------
# Pure functions (no Tkinter dependency)
# ---------------------------------------------------------------------------

def color_to_gray(image, r_weight, g_weight, b_weight):
    """Weighted RGB to greyscale conversion.

    Parameters
    ----------
    image : np.ndarray  – BGR uint8 image (any size).
    r_weight, g_weight, b_weight : float – channel weights (0-100 scale).

    Returns
    -------
    np.ndarray – single-channel uint8 greyscale image.
    """
    total = r_weight + g_weight + b_weight
    if total == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    B, G, R = cv2.split(image)
    grey = (R.astype(np.float64) * r_weight +
            G.astype(np.float64) * g_weight +
            B.astype(np.float64) * b_weight) / total
    return np.clip(grey, 0, 255).astype(np.uint8)


def compute_contrast(grey_image, region_mask, bg_mask):
    """Median contrast between a region and the background.

    Returns abs(median_region - median_background).

    Alternative formulas (for future use):
        Michelson: (max - min) / (max + min)
        Weber:     (target - bg) / bg
    """
    region_pixels = grey_image[region_mask > 0]
    bg_pixels = grey_image[bg_mask > 0]
    if region_pixels.size == 0 or bg_pixels.size == 0:
        return 0.0
    return abs(float(np.median(region_pixels)) - float(np.median(bg_pixels)))


def segment_image(image):
    """Contour-based segmentation using HSV saturation to detect colored shapes
    (1 rectangle = Master, 1 circle = Slave outermost) on a grey background.

    Returns
    -------
    dict or None – keys 'master', 'slave', 'background' each containing
                   'mask' (uint8, 255=inside) and optionally 'contour'.
                   Returns None when required shapes are not found.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Colored shapes have high saturation; grey background has low saturation
    _, sat_binary = cv2.threshold(hsv[:, :, 1], 50, 255, cv2.THRESH_BINARY)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sat_binary = cv2.morphologyEx(sat_binary, cv2.MORPH_OPEN, kernel)
    sat_binary = cv2.morphologyEx(sat_binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(sat_binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = image.shape[:2]
    min_area = (h * w) * 0.001  # ignore tiny noise

    rectangles = []
    circles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if len(approx) == 4 and circularity < 0.75:
            rectangles.append(cnt)
        elif circularity > 0.6:
            circles.append(cnt)

    master_cnt = rectangles[0] if rectangles else None
    slave_cnt = None
    if circles:
        # Pick the outermost (largest bounding radius)
        circles.sort(key=lambda c: cv2.minEnclosingCircle(c)[1], reverse=True)
        slave_cnt = circles[0]

    if master_cnt is None or slave_cnt is None:
        return None  # trigger ROI fallback

    master_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(master_mask, [master_cnt], -1, 255, cv2.FILLED)

    slave_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(slave_mask, [slave_cnt], -1, 255, cv2.FILLED)

    union = cv2.bitwise_or(master_mask, slave_mask)
    bg_mask = cv2.bitwise_not(union)

    return {
        "master": {"contour": master_cnt, "mask": master_mask},
        "slave":  {"contour": slave_cnt,  "mask": slave_mask},
        "background": {"mask": bg_mask},
    }


def build_simulated_masks(h, w):
    """Build masks from known simulated shape coordinates (no detection)."""
    master_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(master_mask, SIM_RECT_TL, SIM_RECT_BR, 255, cv2.FILLED)

    slave_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(slave_mask, SIM_CIRCLE_CENTER, SIM_CIRCLE_RADIUS, 255, cv2.FILLED)

    union = cv2.bitwise_or(master_mask, slave_mask)
    bg_mask = cv2.bitwise_not(union)

    return {
        "master": {"contour": None, "mask": master_mask},
        "slave":  {"contour": None, "mask": slave_mask},
        "background": {"mask": bg_mask},
    }


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SEG_TEST \u2013 Segmentation Tool")
        self.resizable(True, True)

        # State
        self.original_image = None      # full-res BGR (loaded or simulated)
        self.current_color_image = None  # after region color adjustment
        self.grey_image = None
        self.composite_image = None
        self.segments = None             # dict from segment_image / build_simulated_masks
        self._debounce_id = None
        self._baseline_grey_contrast = {"master": 0.0, "slave": 0.0}
        self._baseline_comp_contrast = {"left": 0.0, "right": 0.0, "master": 0.0, "slave": 0.0}
        self._master_in_left = True     # updated by _detect_half_mapping
        self._roi_mode = False
        self._roi_callback = None
        self._roi_start = None
        self._roi_rect_id = None

        # Image references (prevent GC)
        self._tk_color = None
        self._tk_grey = None
        self._tk_comp = None

        self._build_gui()

        # Auto-start in Simulated mode
        self.mode_var.set("Simulated")
        self.after(100, self._generate_simulated_image)

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------
    def _build_gui(self):
        row = 0

        # --- Top bar: Load button + radio buttons + blend checkboxes ---
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        ctk.CTkButton(top_frame, text="Load Image", command=self._load_image).pack(side="left", padx=5)

        self.mode_var = ctk.StringVar(value="Capture")
        ctk.CTkRadioButton(top_frame, text="Capture", variable=self.mode_var,
                            value="Capture", command=self._on_mode_change
                            ).pack(side="left", padx=5)
        ctk.CTkRadioButton(top_frame, text="Simulated", variable=self.mode_var,
                            value="Simulated", command=self._on_mode_change
                            ).pack(side="left", padx=5)

        ctk.CTkLabel(top_frame, text="   ").pack(side="left")

        self.blend_master_var = ctk.BooleanVar(value=True)
        self.blend_slave_var = ctk.BooleanVar(value=True)
        self.blend_bg_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(top_frame, text="Blend Master", variable=self.blend_master_var,
                         command=self._on_top_slider_change).pack(side="left", padx=3)
        ctk.CTkCheckBox(top_frame, text="Blend Slave", variable=self.blend_slave_var,
                         command=self._on_top_slider_change).pack(side="left", padx=3)
        ctk.CTkCheckBox(top_frame, text="Blend BG", variable=self.blend_bg_var,
                         command=self._on_top_slider_change).pack(side="left", padx=3)

        row += 1

        # --- Divider between toolbar and content ---
        separator = ctk.CTkFrame(self, height=2, fg_color="grey70")
        separator.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5)
        row += 1

        # --- Color image (framed) + 9 sliders ---
        color_frame = ctk.CTkFrame(self, border_width=2, fg_color="grey30")
        color_frame.grid(row=row, column=0, padx=5, pady=5, sticky="nw")

        self.color_canvas = tk.Canvas(color_frame, width=DISPLAY_MAX_WIDTH, height=343,
                                      bg="grey30", highlightthickness=0)
        self.color_canvas.pack()
        # Mouse bindings for ROI drawing
        self.color_canvas.bind("<ButtonPress-1>", self._roi_press)
        self.color_canvas.bind("<B1-Motion>", self._roi_motion)
        self.color_canvas.bind("<ButtonRelease-1>", self._roi_release)

        slider_frame_color = ctk.CTkFrame(self, fg_color="transparent")
        slider_frame_color.grid(row=row, column=1, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(slider_frame_color, text="Color Image",
                     font=("Arial", 14, "bold")).pack(anchor="center", pady=(0, 5))

        color_slider_inner = ctk.CTkFrame(slider_frame_color, fg_color="transparent")
        color_slider_inner.pack(anchor="w")

        self.master_r, self.master_g, self.master_b = self._make_slider_row(
            color_slider_inner, "Master", 0, default_r=0, default_g=255, default_b=255)
        self.slave_r, self.slave_g, self.slave_b = self._make_slider_row(
            color_slider_inner, "Slave", 1, default_r=255, default_g=255, default_b=0)
        self.bg_r, self.bg_g, self.bg_b = self._make_slider_row(
            color_slider_inner, "BG", 2, default_r=240, default_g=240, default_b=240)

        ctk.CTkButton(slider_frame_color, text="Save Color", command=self._save_color,
                       height=52, font=("Arial", 22)).pack(fill="x", side="bottom", pady=(8, 0))

        row += 1

        # --- Divider between color and mono ---
        ctk.CTkFrame(self, height=2, fg_color="grey50").grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=5)
        row += 1

        # --- Greyscale image + 3 weight sliders (horizontal row) + contrast below ---
        grey_image_frame = ctk.CTkFrame(self, border_width=2, fg_color="grey30")
        grey_image_frame.grid(row=row, column=0, padx=5, pady=5, sticky="nw")
        self.grey_label = ctk.CTkLabel(grey_image_frame, text="", fg_color="grey30")
        self.grey_label.pack()

        grey_right_frame = ctk.CTkFrame(self, fg_color="transparent")
        grey_right_frame.grid(row=row, column=1, padx=5, pady=5, sticky="nsew")

        ctk.CTkLabel(grey_right_frame, text="Mono Image",
                     font=("Arial", 14, "bold")).pack(anchor="center", pady=(0, 5))

        grey_slider_frame = ctk.CTkFrame(grey_right_frame, fg_color="transparent")
        grey_slider_frame.pack(anchor="w")

        ctk.CTkLabel(grey_slider_frame, text="Grey", width=50).grid(
            row=0, column=0, sticky="w", pady=(26, 0))

        grey_r_frame = ctk.CTkFrame(grey_slider_frame, fg_color="transparent")
        grey_r_frame.grid(row=0, column=1, padx=4)
        ctk.CTkLabel(grey_r_frame, text="R", width=20).pack()
        self.grey_r = NumericInput(grey_r_frame, from_=0, to=100, default=33,
                                    command=lambda val: self._on_grey_slider_change())
        self.grey_r.pack()

        grey_g_frame = ctk.CTkFrame(grey_slider_frame, fg_color="transparent")
        grey_g_frame.grid(row=0, column=2, padx=4)
        ctk.CTkLabel(grey_g_frame, text="G", width=20).pack()
        self.grey_g = NumericInput(grey_g_frame, from_=0, to=100, default=33,
                                    command=lambda val: self._on_grey_slider_change())
        self.grey_g.pack()

        grey_b_frame = ctk.CTkFrame(grey_slider_frame, fg_color="transparent")
        grey_b_frame.grid(row=0, column=3, padx=4)
        ctk.CTkLabel(grey_b_frame, text="B", width=20).pack()
        self.grey_b = NumericInput(grey_b_frame, from_=0, to=100, default=33,
                                    command=lambda val: self._on_grey_slider_change())
        self.grey_b.pack()

        info_frame_grey = ctk.CTkFrame(grey_right_frame, fg_color="transparent")
        info_frame_grey.pack(anchor="w", pady=(8, 0))
        ctk.CTkLabel(info_frame_grey, text="Contrast", width=70,
                     font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="e", padx=(0, 10))
        self.contrast_master_label = ctk.CTkLabel(info_frame_grey, text="Master: --", width=110)
        self.contrast_master_label.grid(row=0, column=1, sticky="w", padx=5)
        self.contrast_slave_label = ctk.CTkLabel(info_frame_grey, text="Slave: --", width=110)
        self.contrast_slave_label.grid(row=0, column=2, sticky="w", padx=5)

        ctk.CTkLabel(info_frame_grey, text="\u0394C", width=70,
                     font=("Arial", 12, "bold")).grid(row=1, column=0, sticky="e", padx=(0, 10), pady=(6, 0))
        self.delta_master_label = ctk.CTkLabel(info_frame_grey, text="Master: --", width=110)
        self.delta_master_label.grid(row=1, column=1, sticky="w", padx=5, pady=(6, 0))
        self.delta_slave_label = ctk.CTkLabel(info_frame_grey, text="Slave: --", width=110)
        self.delta_slave_label.grid(row=1, column=2, sticky="w", padx=5, pady=(6, 0))

        ctk.CTkButton(grey_right_frame, text="Save Mono", command=self._save_grey,
                       height=52, font=("Arial", 22)).pack(fill="x", side="bottom", pady=(8, 0))

        grey_action_frame = ctk.CTkFrame(grey_right_frame, fg_color="transparent")
        grey_action_frame.pack(fill="x", side="bottom", pady=(8, 0))
        ctk.CTkButton(grey_action_frame, text="Reset", command=self._reset_grey_weights,
                       height=52, font=("Arial", 22)).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(grey_action_frame, text="Optimize", command=self._optimize_weights,
                       height=52, font=("Arial", 22)).pack(side="left", expand=True, fill="x", padx=(4, 0))

        self.optimize_progress = ctk.CTkProgressBar(grey_right_frame)
        self.optimize_progress.set(0)
        # Hidden by default; pack_forget keeps it invisible until optimization runs

        row += 1

        # --- Divider between mono and composite ---
        ctk.CTkFrame(self, height=2, fg_color="grey50").grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=5)
        row += 1

        # --- Composite image + 2x3 weight sliders (horizontal rows) + contrast below ---
        comp_image_frame = ctk.CTkFrame(self, border_width=2, fg_color="grey30")
        comp_image_frame.grid(row=row, column=0, padx=5, pady=5, sticky="nw")
        self.comp_label = ctk.CTkLabel(comp_image_frame, text="", fg_color="grey30")
        self.comp_label.pack()

        comp_right_frame = ctk.CTkFrame(self, fg_color="transparent")
        comp_right_frame.grid(row=row, column=1, padx=5, pady=5, sticky="nsw")

        ctk.CTkLabel(comp_right_frame, text="Composite Image",
                     font=("Arial", 14, "bold")).pack(anchor="center", pady=(0, 5))

        comp_slider_frame = ctk.CTkFrame(comp_right_frame, fg_color="transparent")
        comp_slider_frame.pack(anchor="w")

        # Row 0 = Master (always)
        self.comp_row0_label = ctk.CTkLabel(comp_slider_frame, text="Master", width=50)
        self.comp_row0_label.grid(row=0, column=0, sticky="w", pady=(26, 0))

        comp_lr_frame = ctk.CTkFrame(comp_slider_frame, fg_color="transparent")
        comp_lr_frame.grid(row=0, column=1, padx=4)
        ctk.CTkLabel(comp_lr_frame, text="R", width=20).pack()
        self.comp_left_r = NumericInput(comp_lr_frame, from_=0, to=100, default=33,
                                         command=lambda val: self._on_comp_slider_change())
        self.comp_left_r.pack()

        comp_lg_frame = ctk.CTkFrame(comp_slider_frame, fg_color="transparent")
        comp_lg_frame.grid(row=0, column=2, padx=4)
        ctk.CTkLabel(comp_lg_frame, text="G", width=20).pack()
        self.comp_left_g = NumericInput(comp_lg_frame, from_=0, to=100, default=33,
                                         command=lambda val: self._on_comp_slider_change())
        self.comp_left_g.pack()

        comp_lb_frame = ctk.CTkFrame(comp_slider_frame, fg_color="transparent")
        comp_lb_frame.grid(row=0, column=3, padx=4)
        ctk.CTkLabel(comp_lb_frame, text="B", width=20).pack()
        self.comp_left_b = NumericInput(comp_lb_frame, from_=0, to=100, default=33,
                                         command=lambda val: self._on_comp_slider_change())
        self.comp_left_b.pack()

        # Row 1 = Slave (always)
        self.comp_row1_label = ctk.CTkLabel(comp_slider_frame, text="Slave", width=50)
        self.comp_row1_label.grid(row=1, column=0, sticky="w", pady=(26, 0))

        comp_rr_frame = ctk.CTkFrame(comp_slider_frame, fg_color="transparent")
        comp_rr_frame.grid(row=1, column=1, padx=4)
        ctk.CTkLabel(comp_rr_frame, text="R", width=20).pack()
        self.comp_right_r = NumericInput(comp_rr_frame, from_=0, to=100, default=33,
                                          command=lambda val: self._on_comp_slider_change())
        self.comp_right_r.pack()

        comp_rg_frame = ctk.CTkFrame(comp_slider_frame, fg_color="transparent")
        comp_rg_frame.grid(row=1, column=2, padx=4)
        ctk.CTkLabel(comp_rg_frame, text="G", width=20).pack()
        self.comp_right_g = NumericInput(comp_rg_frame, from_=0, to=100, default=33,
                                          command=lambda val: self._on_comp_slider_change())
        self.comp_right_g.pack()

        comp_rb_frame = ctk.CTkFrame(comp_slider_frame, fg_color="transparent")
        comp_rb_frame.grid(row=1, column=3, padx=4)
        ctk.CTkLabel(comp_rb_frame, text="B", width=20).pack()
        self.comp_right_b = NumericInput(comp_rb_frame, from_=0, to=100, default=33,
                                          command=lambda val: self._on_comp_slider_change())
        self.comp_right_b.pack()

        info_frame_comp = ctk.CTkFrame(comp_right_frame, fg_color="transparent")
        info_frame_comp.pack(anchor="w", pady=(8, 0))
        ctk.CTkLabel(info_frame_comp, text="Contrast", width=70,
                     font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="e", padx=(0, 10))
        self.contrast_left_label = ctk.CTkLabel(info_frame_comp, text="Master: --", width=110)
        self.contrast_left_label.grid(row=0, column=1, sticky="w", padx=5)
        self.contrast_right_label = ctk.CTkLabel(info_frame_comp, text="Slave: --", width=110)
        self.contrast_right_label.grid(row=0, column=2, sticky="w", padx=5)

        ctk.CTkLabel(info_frame_comp, text="\u0394C", width=70,
                     font=("Arial", 12, "bold")).grid(row=1, column=0, sticky="e", padx=(0, 10), pady=(6, 0))
        self.delta_left_label = ctk.CTkLabel(info_frame_comp, text="Master: --", width=110)
        self.delta_left_label.grid(row=1, column=1, sticky="w", padx=5, pady=(6, 0))
        self.delta_right_label = ctk.CTkLabel(info_frame_comp, text="Slave: --", width=110)
        self.delta_right_label.grid(row=1, column=2, sticky="w", padx=5, pady=(6, 0))

        ctk.CTkButton(comp_right_frame, text="Save Composite", command=self._save_composite,
                       height=52, font=("Arial", 22)).pack(fill="x", side="bottom", pady=(8, 0))

    # ------------------------------------------------------------------
    # Slider helper
    # ------------------------------------------------------------------
    def _make_slider_row(self, parent, label, grid_row,
                         default_r=128, default_g=128, default_b=128):
        """Create a labelled row of R, G, B numeric inputs (0-255)."""
        ctk.CTkLabel(parent, text=label, width=50).grid(
            row=grid_row, column=0, sticky="w", pady=(26, 0))

        r_frame = ctk.CTkFrame(parent, fg_color="transparent")
        r_frame.grid(row=grid_row, column=1, padx=4)
        ctk.CTkLabel(r_frame, text="R", width=20).pack()
        r = NumericInput(r_frame, from_=0, to=255, default=default_r,
                         command=lambda val: self._on_top_slider_change())
        r.pack()

        g_frame = ctk.CTkFrame(parent, fg_color="transparent")
        g_frame.grid(row=grid_row, column=2, padx=4)
        ctk.CTkLabel(g_frame, text="G", width=20).pack()
        g = NumericInput(g_frame, from_=0, to=255, default=default_g,
                         command=lambda val: self._on_top_slider_change())
        g.pack()

        b_frame = ctk.CTkFrame(parent, fg_color="transparent")
        b_frame.grid(row=grid_row, column=3, padx=4)
        ctk.CTkLabel(b_frame, text="B", width=20).pack()
        b = NumericInput(b_frame, from_=0, to=255, default=default_b,
                         command=lambda val: self._on_top_slider_change())
        b.pack()

        def _pick_color(_r=r, _g=g, _b=b):
            initial = f"#{int(_r.get()):02x}{int(_g.get()):02x}{int(_b.get()):02x}"
            picker = AskColor(title=f"Pick {label} Color", initial_color=initial)
            color = picker.get()
            if color:
                # color is a hex string like "#ff00aa"
                hex_str = color.lstrip("#")
                _r.set(int(hex_str[0:2], 16))
                _g.set(int(hex_str[2:4], 16))
                _b.set(int(hex_str[4:6], 16))
                self._on_top_slider_change()

        pick_btn = ctk.CTkButton(parent, text="\u2630", width=30, height=28,
                                  command=_pick_color)
        pick_btn.grid(row=grid_row, column=4, padx=(4, 0), pady=(26, 0))

        return r, g, b

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def _cv2_to_tk(self, cv_img, max_width=DISPLAY_MAX_WIDTH):
        """Resize BGR image, convert to ImageTk.PhotoImage (for Canvas)."""
        h, w = cv_img.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(cv_img.shape) == 2:
            # greyscale -> 3-channel for display
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(cv_img)
        return ImageTk.PhotoImage(pil_img)

    def _cv2_to_ctk(self, cv_img, max_width=DISPLAY_MAX_WIDTH):
        """Resize BGR image, convert to CTkImage (for CTkLabel)."""
        h, w = cv_img.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(cv_img)
        return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

    def _display_on_canvas(self, canvas, cv_img):
        """Display a cv2 image on a Canvas widget."""
        tk_img = self._cv2_to_tk(cv_img)
        canvas.delete("all")
        canvas.config(width=tk_img.width(), height=tk_img.height())
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        return tk_img  # must keep reference

    def _display_on_label(self, label, cv_img):
        """Display a cv2 image on a CTkLabel widget."""
        ctk_img = self._cv2_to_ctk(cv_img)
        label.configure(image=ctk_img, text="")
        return ctk_img

    # ------------------------------------------------------------------
    # Load / Generate
    # ------------------------------------------------------------------
    def _load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("BMP", "*.bmp"), ("JPEG", "*.jpg *.jpeg"),
                       ("All Images", "*.bmp *.jpg *.jpeg *.png")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not load image:\n{path}")
            return

        self.original_image = img
        self.mode_var.set("Capture")

        # Run segmentation
        result = segment_image(img)
        if result is None:
            self._start_roi_fallback(img)
        else:
            self.segments = result
            self._detect_region_colors()
            self._cascade_all()

    def _generate_simulated_image(self):
        """Build a synthetic image from slider values and known coordinates."""
        bg_b, bg_g, bg_r = int(self.bg_b.get()), int(self.bg_g.get()), int(self.bg_r.get())
        img = np.full((SIM_HEIGHT, SIM_WIDTH, 3), (bg_b, bg_g, bg_r), dtype=np.uint8)

        # Master = rectangle (Yellow default)
        m_b, m_g, m_r = int(self.master_b.get()), int(self.master_g.get()), int(self.master_r.get())
        cv2.rectangle(img, SIM_RECT_TL, SIM_RECT_BR, (m_b, m_g, m_r), cv2.FILLED)

        # Slave = circle (Cyan default)
        s_b, s_g, s_r = int(self.slave_b.get()), int(self.slave_g.get()), int(self.slave_r.get())
        cv2.circle(img, SIM_CIRCLE_CENTER, SIM_CIRCLE_RADIUS, (s_b, s_g, s_r), cv2.FILLED)

        self.original_image = img
        self.segments = build_simulated_masks(SIM_HEIGHT, SIM_WIDTH)
        self._cascade_all()

    # ------------------------------------------------------------------
    # Segmentation helpers
    # ------------------------------------------------------------------
    def _detect_region_colors(self):
        """Set top 9 sliders to median RGB of each segmented region."""
        if self.original_image is None or self.segments is None:
            return
        img = self.original_image
        for region_key, r_slider, g_slider, b_slider in [
            ("master", self.master_r, self.master_g, self.master_b),
            ("slave", self.slave_r, self.slave_g, self.slave_b),
            ("background", self.bg_r, self.bg_g, self.bg_b),
        ]:
            mask = self.segments[region_key]["mask"]
            pixels = img[mask > 0]
            if pixels.size == 0:
                continue
            median_bgr = np.median(pixels, axis=0)
            b_slider.set(int(median_bgr[0]))
            g_slider.set(int(median_bgr[1]))
            r_slider.set(int(median_bgr[2]))

    # ------------------------------------------------------------------
    # ROI fallback
    # ------------------------------------------------------------------
    def _start_roi_fallback(self, image):
        """Prompt user to draw ROIs for missing shapes."""
        self.segments = {"master": None, "slave": None, "background": None}
        self.original_image = image
        self._tk_color = self._display_on_canvas(self.color_canvas, image)

        # Attempt to find what we can
        result = segment_image(image)
        found_master = result and result.get("master") is not None
        found_slave = result and result.get("slave") is not None

        if result:
            if found_master:
                self.segments["master"] = result["master"]
            if found_slave:
                self.segments["slave"] = result["slave"]

        self._roi_missing = []
        if not found_master:
            self._roi_missing.append("master")
        if not found_slave:
            self._roi_missing.append("slave")

        self._prompt_next_roi()

    def _prompt_next_roi(self):
        if not self._roi_missing:
            # All found - build background mask and finish
            h, w = self.original_image.shape[:2]
            union = np.zeros((h, w), dtype=np.uint8)
            for key in ("master", "slave"):
                if self.segments[key] is not None:
                    union = cv2.bitwise_or(union, self.segments[key]["mask"])
            self.segments["background"] = {"mask": cv2.bitwise_not(union)}
            self._detect_region_colors()
            self._cascade_all()
            return

        shape_name = self._roi_missing[0]
        messagebox.showinfo("ROI Selection",
                            f"Could not detect {shape_name.title()}.\n"
                            f"Draw a bounding rectangle around the {shape_name} region on the image.")
        self._roi_mode = True
        self._roi_callback = lambda roi: self._handle_roi(shape_name, roi)

    def _roi_press(self, event):
        if not self._roi_mode:
            return
        self._roi_start = (event.x, event.y)
        self._roi_rect_id = self.color_canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="lime", width=2)

    def _roi_motion(self, event):
        if not self._roi_mode or self._roi_start is None:
            return
        self.color_canvas.coords(self._roi_rect_id,
                                 self._roi_start[0], self._roi_start[1],
                                 event.x, event.y)

    def _roi_release(self, event):
        if not self._roi_mode or self._roi_start is None:
            return
        self._roi_mode = False
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        self._roi_start = None
        if self._roi_rect_id:
            self.color_canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

        # Map display coords back to original image coords
        h, w = self.original_image.shape[:2]
        disp_w = min(w, DISPLAY_MAX_WIDTH)
        scale = w / disp_w
        ox0, oy0 = int(min(x0, x1) * scale), int(min(y0, y1) * scale)
        ox1, oy1 = int(max(x0, x1) * scale), int(max(y0, y1) * scale)
        ox0, oy0 = max(0, ox0), max(0, oy0)
        ox1, oy1 = min(w, ox1), min(h, oy1)

        if self._roi_callback:
            self._roi_callback((ox0, oy0, ox1, oy1))

    def _handle_roi(self, shape_name, roi):
        """Re-run segmentation within the ROI crop. Fall back to ROI rect as mask."""
        x0, y0, x1, y1 = roi
        crop = self.original_image[y0:y1, x0:x1]
        result = segment_image(crop)

        h, w = self.original_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if result is not None:
            # Find the right shape in the crop
            key = "master" if shape_name == "master" else "slave"
            if result.get(key) is not None:
                crop_mask = result[key]["mask"]
                mask[y0:y1, x0:x1] = crop_mask
            else:
                # Use ROI rectangle as mask
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, cv2.FILLED)
        else:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, cv2.FILLED)

        self.segments[shape_name] = {"contour": None, "mask": mask}
        self._roi_missing.pop(0)
        self._prompt_next_roi()

    # ------------------------------------------------------------------
    # Update / cascade
    # ------------------------------------------------------------------
    def _cascade_all(self):
        """Full refresh: color -> greyscale -> composite. Also compute baseline contrast."""
        self._update_color_image()
        self._detect_half_mapping()
        self._compute_baseline_contrast()

    def _detect_half_mapping(self):
        """Determine which segment (Master/Slave) is in which half of the image
        and update composite slider + contrast labels accordingly."""
        if self.current_color_image is None or self.segments is None:
            return
        h, w = self.current_color_image.shape[:2]
        half = w // 2

        master_seg = self.segments.get("master")
        slave_seg = self.segments.get("slave")

        master_in_left = False
        slave_in_left = False

        if master_seg:
            m = master_seg["mask"]
            master_in_left = int(m[:, :half].sum()) > int(m[:, half:].sum())
        if slave_seg:
            m = slave_seg["mask"]
            slave_in_left = int(m[:, :half].sum()) > int(m[:, half:].sum())

        self._master_in_left = master_in_left

    def _compute_baseline_contrast(self):
        """Compute contrast at equal weights (33/33/33) and store as baseline."""
        if self.current_color_image is None or self.segments is None:
            return
        baseline_grey = color_to_gray(self.current_color_image, 33, 33, 33)
        master_seg = self.segments.get("master")
        slave_seg = self.segments.get("slave")
        bg_seg = self.segments.get("background")

        # Greyscale baseline
        if master_seg and bg_seg:
            self._baseline_grey_contrast["master"] = compute_contrast(
                baseline_grey, master_seg["mask"], bg_seg["mask"])
        if slave_seg and bg_seg:
            self._baseline_grey_contrast["slave"] = compute_contrast(
                baseline_grey, slave_seg["mask"], bg_seg["mask"])

        # Composite baseline (both halves at equal weights)
        h, w = baseline_grey.shape[:2]
        half = w // 2
        region_union = np.zeros_like(baseline_grey)
        if master_seg:
            region_union = cv2.bitwise_or(region_union, master_seg["mask"])
        if slave_seg:
            region_union = cv2.bitwise_or(region_union, slave_seg["mask"])

        left_region = np.zeros_like(baseline_grey)
        right_region = np.zeros_like(baseline_grey)
        left_bg = np.zeros_like(baseline_grey)
        right_bg = np.zeros_like(baseline_grey)
        left_region[:, :half] = region_union[:, :half]
        right_region[:, half:] = region_union[:, half:]
        if bg_seg:
            left_bg[:, :half] = bg_seg["mask"][:, :half]
            right_bg[:, half:] = bg_seg["mask"][:, half:]

        self._baseline_comp_contrast["left"] = compute_contrast(
            baseline_grey, left_region, left_bg)
        self._baseline_comp_contrast["right"] = compute_contrast(
            baseline_grey, right_region, right_bg)

        # Map left/right to Master/Slave
        if self._master_in_left:
            comp_master = self._baseline_comp_contrast["left"]
            comp_slave = self._baseline_comp_contrast["right"]
        else:
            comp_master = self._baseline_comp_contrast["right"]
            comp_slave = self._baseline_comp_contrast["left"]
        self._baseline_comp_contrast["master"] = comp_master
        self._baseline_comp_contrast["slave"] = comp_slave

        # Update static labels with baseline values
        self.contrast_master_label.configure(
            text=f"Master: {self._baseline_grey_contrast['master']:.1f}")
        self.contrast_slave_label.configure(
            text=f"Slave: {self._baseline_grey_contrast['slave']:.1f}")
        self.contrast_left_label.configure(
            text=f"Master: {comp_master:.1f}")
        self.contrast_right_label.configure(
            text=f"Slave: {comp_slave:.1f}")

    def _update_color_image(self):
        if self.original_image is None or self.segments is None:
            return

        if self.mode_var.get() == "Simulated":
            # Regenerate from sliders (already sets self.original_image + segments)
            bg_b, bg_g, bg_r = int(self.bg_b.get()), int(self.bg_g.get()), int(self.bg_r.get())
            img = np.full((SIM_HEIGHT, SIM_WIDTH, 3), (bg_b, bg_g, bg_r), dtype=np.uint8)
            m_b, m_g, m_r = int(self.master_b.get()), int(self.master_g.get()), int(self.master_r.get())
            cv2.rectangle(img, SIM_RECT_TL, SIM_RECT_BR, (m_b, m_g, m_r), cv2.FILLED)
            s_b, s_g, s_r = int(self.slave_b.get()), int(self.slave_g.get()), int(self.slave_r.get())
            cv2.circle(img, SIM_CIRCLE_CENTER, SIM_CIRCLE_RADIUS, (s_b, s_g, s_r), cv2.FILLED)
            self.original_image = img
            self.current_color_image = img.copy()
        else:
            # Capture mode: apply replace / blend per region
            img = self.original_image.copy()
            for region_key, r_s, g_s, b_s, blend_var in [
                ("master", self.master_r, self.master_g, self.master_b, self.blend_master_var),
                ("slave", self.slave_r, self.slave_g, self.slave_b, self.blend_slave_var),
                ("background", self.bg_r, self.bg_g, self.bg_b, self.blend_bg_var),
            ]:
                seg = self.segments.get(region_key)
                if seg is None:
                    continue
                mask = seg["mask"]
                color_bgr = np.array([int(b_s.get()), int(g_s.get()), int(r_s.get())], dtype=np.uint8)
                flat = np.full_like(img, color_bgr)

                if blend_var.get():
                    # Blend 50/50
                    blended = cv2.addWeighted(self.original_image, 0.5, flat, 0.5, 0)
                    img[mask > 0] = blended[mask > 0]
                else:
                    # Replace
                    img[mask > 0] = flat[mask > 0]

            self.current_color_image = img

        self._tk_color = self._display_on_canvas(self.color_canvas, self.current_color_image)
        self._detect_half_mapping()
        self._compute_baseline_contrast()
        self._update_greyscale_image()

    def _update_greyscale_image(self):
        if self.current_color_image is None:
            return
        r_w = int(self.grey_r.get())
        g_w = int(self.grey_g.get())
        b_w = int(self.grey_b.get())
        self.grey_image = color_to_gray(self.current_color_image, r_w, g_w, b_w)
        self._tk_grey = self._display_on_label(self.grey_label, self.grey_image)

        # \u0394C (delta from baseline at equal weights)
        if self.segments:
            master_seg = self.segments.get("master")
            slave_seg = self.segments.get("slave")
            bg_seg = self.segments.get("background")
            if master_seg and bg_seg:
                c = compute_contrast(self.grey_image, master_seg["mask"], bg_seg["mask"])
                delta = c - self._baseline_grey_contrast["master"]
                self.delta_master_label.configure(text=f"Master: {delta:+.1f}")
            if slave_seg and bg_seg:
                c = compute_contrast(self.grey_image, slave_seg["mask"], bg_seg["mask"])
                delta = c - self._baseline_grey_contrast["slave"]
                self.delta_slave_label.configure(text=f"Slave: {delta:+.1f}")

        self._update_composite_image()

    def _update_composite_image(self):
        if self.current_color_image is None:
            return

        # Row 0 sliders = Master weights, Row 1 sliders = Slave weights
        master_r = int(self.comp_left_r.get())
        master_g = int(self.comp_left_g.get())
        master_b = int(self.comp_left_b.get())
        slave_r = int(self.comp_right_r.get())
        slave_g = int(self.comp_right_g.get())
        slave_b = int(self.comp_right_b.get())

        # Map Master/Slave weights to left/right halves based on detection
        if self._master_in_left:
            grey_left = color_to_gray(self.current_color_image, master_r, master_g, master_b)
            grey_right = color_to_gray(self.current_color_image, slave_r, slave_g, slave_b)
        else:
            grey_left = color_to_gray(self.current_color_image, slave_r, slave_g, slave_b)
            grey_right = color_to_gray(self.current_color_image, master_r, master_g, master_b)

        h, w = grey_left.shape[:2]
        half = w // 2
        composite = np.zeros((h, w), dtype=np.uint8)
        composite[:, :half] = grey_left[:, :half]
        composite[:, half:] = grey_right[:, half:]
        self.composite_image = composite

        # Draw vertical divider at the centre
        display_comp = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
        cv2.line(display_comp, (half, 0), (half, h - 1), (0, 0, 255), 2)
        self._tk_comp = self._display_on_label(self.comp_label, display_comp)

        # Contrast per half -> report as Master / Slave
        if self.segments:
            bg_seg = self.segments.get("background")
            master_seg = self.segments.get("master")
            slave_seg = self.segments.get("slave")

            # Union of all region masks (master + slave)
            region_union = np.zeros_like(composite)
            if master_seg:
                region_union = cv2.bitwise_or(region_union, master_seg["mask"])
            if slave_seg:
                region_union = cv2.bitwise_or(region_union, slave_seg["mask"])

            # Build half-masks from the union
            left_region_mask = np.zeros_like(composite)
            right_region_mask = np.zeros_like(composite)
            left_bg_mask = np.zeros_like(composite)
            right_bg_mask = np.zeros_like(composite)

            left_region_mask[:, :half] = region_union[:, :half]
            right_region_mask[:, half:] = region_union[:, half:]
            if bg_seg:
                left_bg_mask[:, :half] = bg_seg["mask"][:, :half]
                right_bg_mask[:, half:] = bg_seg["mask"][:, half:]

            c_left = compute_contrast(composite, left_region_mask, left_bg_mask)
            c_right = compute_contrast(composite, right_region_mask, right_bg_mask)

            # Map left/right contrast to Master/Slave names
            if self._master_in_left:
                c_master, c_slave = c_left, c_right
            else:
                c_master, c_slave = c_right, c_left

            delta_master = c_master - self._baseline_comp_contrast["master"]
            delta_slave = c_slave - self._baseline_comp_contrast["slave"]
            self.delta_left_label.configure(text=f"Master: {delta_master:+.1f}")
            self.delta_right_label.configure(text=f"Slave: {delta_slave:+.1f}")

    # ------------------------------------------------------------------
    # Callbacks (debounced)
    # ------------------------------------------------------------------
    def _debounce(self, func, delay=50):
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(delay, func)

    def _on_top_slider_change(self, *_args):
        self._debounce(self._update_color_image)

    def _on_grey_slider_change(self, *_args):
        self._debounce(self._update_greyscale_image)

    def _on_comp_slider_change(self, *_args):
        self._debounce(self._update_composite_image)

    def _on_mode_change(self):
        if self.mode_var.get() == "Simulated":
            self._generate_simulated_image()
        else:
            # Capture mode - if we have an original loaded, re-cascade
            if self.original_image is not None:
                self._cascade_all()

    # ------------------------------------------------------------------
    # Reset / Optimize
    # ------------------------------------------------------------------
    def _reset_grey_weights(self):
        """Reset greyscale weights to equal distribution (33/33/33)."""
        self.grey_r.set(33)
        self.grey_g.set(33)
        self.grey_b.set(33)
        self._update_greyscale_image()

    def _optimize_weights(self):
        """Find the weight combination that maximises overall contrast.

        TODO: implement scoring logic (balance Master vs Slave contrast).
        """
        if self.current_color_image is None or self.segments is None:
            return

        # Show progress bar
        self.optimize_progress.set(0)
        self.optimize_progress.pack(fill="x", side="bottom", pady=(8, 0))
        self.update_idletasks()

        # TODO: real optimization loop — update self.optimize_progress.set(fraction)
        messagebox.showinfo("Optimize", "Optimization not yet implemented.\n"
                            "This will search for the weight set that maximises contrast.")

        # Hide progress bar
        self.optimize_progress.pack_forget()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _save_color(self):
        if self.current_color_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("BMP", "*.bmp"), ("All files", "*.*")])
        if path:
            cv2.imwrite(path, self.current_color_image)
            messagebox.showinfo("Saved", f"Color image saved to:\n{path}")

    def _save_grey(self):
        if self.grey_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("BMP", "*.bmp"), ("All files", "*.*")])
        if path:
            cv2.imwrite(path, self.grey_image)
            messagebox.showinfo("Saved", f"Mono image saved to:\n{path}")

    def _save_composite(self):
        if self.composite_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("BMP", "*.bmp"), ("All files", "*.*")])
        if path:
            cv2.imwrite(path, self.composite_image)
            messagebox.showinfo("Saved", f"Composite image saved to:\n{path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
