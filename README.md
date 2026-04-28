<div align="center">

# SEG_TEST

**Image Segmentation & Greyscale Contrast Analysis**

[Overview](#overview) · [Getting Started](#getting-started) · [GUI Layout](#gui-layout) · [Modes](#modes) · [Algorithms](#algorithms) · [Building](#building-standalone-executable)

</div>

A dark-themed [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) desktop application for segmenting images into regions, exploring weighted RGB-to-greyscale conversions, and measuring how channel weight changes affect contrast.

## Overview

SEG_TEST segments an image into three regions — **Master** (rectangle), **Slave** (circle), and **Background** — using HSV saturation-based contour detection. Three side-by-side panels let you:

1. **Color** — Adjust or replace each region's colour, with a color picker per row
2. **Mono** — Tune R/G/B greyscale weights and see contrast + ΔC from baseline
3. **Composite** — Compare two independent weight sets (one per segment half), split by a red centre divider

## Features

- **Dual mode** — Load real `.bmp`/`.jpeg` captures or generate a synthetic 2112×500 test pattern
- **HSV segmentation** — Saturation thresholding detects coloured shapes on grey backgrounds
- **ROI fallback** — Draw bounding rectangles when auto-detection fails
- **Color picker** — [CTkColorPicker](https://github.com/Akascape/CTkColorPicker) popup (☰ button) per region
- **Numeric inputs** — Direct R/G/B entry fields (no sliders), with clamping and live update
- **Baseline + ΔC** — Contrast measured at equal weights (33/33/33); delta shown live as weights change
- **Reset** — One-click return to equal weights (33/33/33)
- **Optimize** — Auto-search for the weight set that maximises overall contrast (WIP)
- **Dynamic half-detection** — Composite labels auto-map Master/Slave to the correct image half
- **Per-section save** — Individual Save buttons for Color, Mono, and Composite outputs
- **Blend mode** — 50/50 blend or flat replace per region (Capture mode)

## Getting Started

### Prerequisites

- Python 3.8+
- Tkinter (included with standard Python — used for `tk.Canvas` and file dialogs)

### Installation

```bash
pip install customtkinter CTkColorPicker opencv-python numpy Pillow
```

### Run

```bash
py SEG_TEST.py
```

The app launches in **Simulated** mode with a default test pattern (cyan circle, yellow rectangle, light grey background).

## GUI Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ [Load Image]  (o) Capture  (o) Simulated   ☐ Blend Master/Slave/BG        │
├──────────────────────────────────┬──────────────────────────────────────────┤
│                                  │  Color Image                            │
│         COLOR IMAGE              │  Master: [R] [G] [B] [☰]               │
│                                  │  Slave:  [R] [G] [B] [☰]               │
│                                  │  BG:     [R] [G] [B] [☰]               │
│                                  │              [Save Color]               │
├──────────────────────────────────┼──────────────────────────────────────────┤
│                                  │  Mono Image                             │
│         MONO IMAGE               │  Grey:  [R] [G] [B]                     │
│                                  │  Contrast: Master=XX   Slave=XX         │
│                                  │  ΔC:       Master=+X   Slave=+X         │
│                                  │    [Reset] [Optimize] [Save Mono]       │
├──────────────────────────────────┼──────────────────────────────────────────┤
│                                  │  Composite Image                        │
│   COMPOSITE  (Slave │ Master)    │  Master: [R] [G] [B]                    │
│           red divider ↑          │  Slave:  [R] [G] [B]                    │
│                                  │  Contrast: Master=XX   Slave=XX         │
│                                  │  ΔC:       Master=+X   Slave=+X         │
│                                  │           [Save Composite]               │
└──────────────────────────────────┴──────────────────────────────────────────┘
```

## Modes

### Capture

1. Click **Load Image** to open a `.bmp`, `.jpeg`, or `.png` file.
2. HSV saturation-based segmentation runs automatically.
3. If detection fails, you are prompted to draw ROI bounding rectangles.
4. Numeric inputs auto-populate with detected median RGB per region.
5. Toggle **Blend** checkboxes for 50/50 blend instead of flat colour replace.

### Simulated

- Generates a **2112×500** synthetic image with configurable shapes.
- Defaults: cyan circle (Slave), yellow rectangle (Master), light grey background `(240, 240, 240)`.
- Input changes regenerate the image immediately — no segmentation step needed.

## Algorithms

### Segmentation

1. Convert to HSV, threshold saturation (S > 50) to isolate coloured shapes
2. Morphological open + close (5×5 elliptical kernel) to clean noise
3. `findContours` with `RETR_EXTERNAL`
4. Classify: **Rectangle** = `approxPolyDP` → 4 vertices + circularity < 0.75; **Circle** = circularity > 0.6
5. Outermost circle (largest bounding radius) = Slave
6. Background = inverse of Master ∪ Slave masks

### Greyscale Conversion

$$\text{grey} = \frac{R \cdot w_R + G \cdot w_G + B \cdot w_B}{w_R + w_G + w_B}$$

Weights $w_R, w_G, w_B$ range 0–100, default 33 each (equal contribution).

### Contrast

$$C = |\text{median}(\text{region}) - \text{median}(\text{background})|$$

**Baseline** is computed once at equal weights. **ΔC** = current contrast − baseline, shown with `+`/`−` prefix.

> [!NOTE]
> Alternative formulas preserved for future use:
> - **Michelson**: $(I_{max} - I_{min}) / (I_{max} + I_{min})$
> - **Weber**: $(I_{target} - I_{bg}) / I_{bg}$

### Composite

The image is split at the midpoint with a red vertical divider. Each half applies its own R/G/B weight set. The app auto-detects which segment is in which half, so the Master/Slave labels always map correctly.

## Input Reference

| Section | Inputs | Range | Default | Purpose |
|---|---|---|---|---|
| Color | 9 numeric (R/G/B × Master/Slave/BG) + 3 color pickers | 0–255 | Detected or simulated | Region colour |
| Mono | 3 numeric (R/G/B weight) | 0–100 | 33 | Channel contribution to greyscale |
| Composite | 2×3 numeric (R/G/B weight × Master/Slave) | 0–100 | 33 | Per-half greyscale weights |

## File I/O

| Direction | Formats | Details |
|---|---|---|
| **Input** | `.bmp`, `.jpeg`, `.png` | Via Load Image dialog |
| **Save Color** | `.bmp` (default) | Colour image after region adjustments |
| **Save Mono** | `.bmp` (default) | Greyscale conversion |
| **Save Composite** | `.bmp` (default) | Composite image (without red divider) |

## Building Standalone Executable

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name SEG_TEST \
  --hidden-import customtkinter \
  --hidden-import CTkColorPicker \
  SEG_TEST.py
```

> [!TIP]
> Output lands in `dist/SEG_TEST.exe` (~200–400 MB). The `.exe` runs standalone without Python installed.

## Project Structure

```
PYTHON/
├── SEG_TEST.py        # Main GUI application
├── SEGMENTATION.py    # Original channel isolation experiments
├── README.md
└── PRE_TEST/          # Test images (.bmp, .png)
```
