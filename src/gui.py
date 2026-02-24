# ==========================================================
# pvBG - Private Background Removal
# Copyright (C) 2026 Saw it See had
# Licensed under the MIT License
# ==========================================================

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import threading
import os

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

from engine import Engine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PREVIEW_SIZE  = 380
MAX_HISTORY   = 20

def make_checkerboard(width: int, height: int, tile: int = 12, dark: bool = False) -> np.ndarray:
    """Generate an RGB checkerboard numpy array (fast, no Python loops)."""
    xs   = np.arange(width)  // tile
    ys   = np.arange(height) // tile
    mask = (xs[np.newaxis, :] + ys[:, np.newaxis]) % 2 == 0
    
    c1, c2 = (60, 40) if dark else (200, 155)
    
    arr  = np.where(mask[:, :, np.newaxis], c1, c2).astype(np.uint8)
    return np.repeat(arr, 3, axis=2)

def composite_np(rgba_np: np.ndarray, dark_bg: bool = False) -> np.ndarray:
    """
    Composite an RGBA numpy array onto a checkerboard.
    Accepts a `dark_bg` parameter for a bright image removal mode.
    """
    h, w    = rgba_np.shape[:2]
    checker = make_checkerboard(w, h, dark=dark_bg)
    alpha   = rgba_np[:, :, 3:4].astype(np.float32) / 255.0
    rgb     = rgba_np[:, :, :3].astype(np.float32)
    result  = (rgb * alpha + checker.astype(np.float32) * (1.0 - alpha))
    return result.astype(np.uint8)

def composite_repair_np(rgba_np: np.ndarray, orig_np: np.ndarray,
                        bg_opacity: float = 0.30) -> np.ndarray:
    """
    Create a composite for Repair Mode:
    Foreground = 100% of the original image.
    Background = 50% (or as per bg_opacity) of the original image.
    """
    alpha  = rgba_np[:, :, 3:4].astype(np.float32) / 255.0
    orig_float = orig_np.astype(np.float32)
    fg = orig_float
    bg = orig_float * bg_opacity
    result = fg * alpha + bg * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)

def rgb2lab_manual(rgb: np.ndarray) -> np.ndarray:
    """
    Manually convert an RGB numpy array image to CIELAB.
    Assumes RGB input is uint8 (0-255).
    """
    rgb_float = rgb.astype(np.float32) / 255.0
    
    gamma_mask = rgb_float > 0.04045
    rgb_linear = np.empty_like(rgb_float)
    
    rgb_linear[gamma_mask] = np.power((rgb_float[gamma_mask] + 0.055) / 1.055, 2.4)
    rgb_linear[~gamma_mask] = rgb_float[~gamma_mask] / 12.92
    
    xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    h, w, _ = rgb.shape
    xyz = np.dot(rgb_linear.reshape(-1, 3), xyz_matrix.T).reshape(h, w, 3)

    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 1] /= 1.00000
    xyz[:, :, 2] /= 1.08883

    xyz_mask = xyz > 0.008856
    xyz_conv = np.empty_like(xyz)
    xyz_conv[xyz_mask] = np.cbrt(xyz[xyz_mask])
    xyz_conv[~xyz_mask] = (903.3 * xyz[~xyz_mask] + 16) / 116
    
    lab = np.empty_like(xyz)
    lab[:, :, 0] = (116.0 * xyz_conv[:, :, 1]) - 16.0
    lab[:, :, 1] = 500.0 * (xyz_conv[:, :, 0] - xyz_conv[:, :, 1])
    lab[:, :, 2] = 200.0 * (xyz_conv[:, :, 1] - xyz_conv[:, :, 2])

    return lab

class RepairWindow(ctk.CTkToplevel):
    """
    A separate window for mask editing (repair mode).
    Receives the full-size result image (RGBA PIL) and original image (RGB numpy).
    On completion, it calls the callback with the modified result image.
    """
    def __init__(self, parent, result_image: Image.Image, original_np: np.ndarray, callback):
        super().__init__(parent)
        self.parent = parent
        self.callback = callback

        # Save data full size
        self.full_np = np.array(result_image, dtype=np.uint8)          # RGBA
        self.original_full_np = original_np.copy()                    # RGB

        self.title("pvBG - Repair Mask")
        self.geometry("900x700")
        self.minsize(1200, 300)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.lift()
        self.focus()

        # Repair state
        self.repair_mode = tk.StringVar(value="restore")
        self.magic_mode = tk.BooleanVar(value=False)
        self.magic_tol = tk.IntVar(value=10) 
        self.dark_bg_mode = tk.BooleanVar(value=False)
        self.brush_size = tk.IntVar(value=18)
        self.zoom_factor = tk.DoubleVar(value=1.0)
        self.zoom_factor.trace_add("write", self._on_zoom_change)
        self.zoom_offset = (0, 0)          
        self.zoom_disp_w = 0
        self.zoom_disp_h = 0
        self.last_mouse_x = None
        self.last_mouse_y = None
        self.history = []         
        self.redo_stack = []
        self.last_xy = None

        # Display data
        self.disp_np = None       
        self.disp_orig_np = None   
        self.disp_orig_lab = None  
        self.disp_w = 0
        self.disp_h = 0
        self.canvas_offset = (0, 0)
        self.canvas_photo = None
        self.cursor_oval = None

        # Icons
        self._load_icons()

        # UI
        self._build_ui()

        # Keyboard bindings
        self.bind("<Control-z>", lambda e: self._undo())
        self.bind("<Control-y>", lambda e: self._redo())

        # Initialize display cache
        self.after(100, self._rebuild_display_cache)

    def _load_icons(self):
        """Load icons from the assets folder (same as in App)."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, '..', 'assets')
        try:
            self.icon_undo = ctk.CTkImage(Image.open(os.path.join(assets_dir, 'undo.png')), size=(20, 20))
            self.icon_redo = ctk.CTkImage(Image.open(os.path.join(assets_dir, 'forward.png')), size=(20, 20))
        except Exception:
            self.icon_undo = self.icon_redo = None

    def _build_ui(self):
        """Build the toolbar, canvas, and Apply/Cancel buttons."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ctk.CTkFrame(self, fg_color="#1a1a30", corner_radius=8)
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        # Mode radio buttons
        ctk.CTkLabel(toolbar, text="Mode:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(10, 4))
        ctk.CTkRadioButton(toolbar, text="Restore", variable=self.repair_mode, value="restore",
                           font=ctk.CTkFont(size=12), fg_color="#1976d2",
                           command=self._update_zoom_display).pack(side="left", padx=6)
        ctk.CTkRadioButton(toolbar, text="Erase", variable=self.repair_mode, value="erase",
                           font=ctk.CTkFont(size=12), fg_color="#c62828",
                           command=self._update_zoom_display).pack(side="left", padx=6)

        # Dark BG switch
        self.bg_switch = ctk.CTkSwitch(toolbar, text="Dark BG", variable=self.dark_bg_mode,
                                        font=ctk.CTkFont(size=12),
                                        command=self._update_zoom_display, width=70)
        self.bg_switch.pack(side="left", padx=(15, 6))

        # Brush size
        ctk.CTkLabel(toolbar, text="  |  Size:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(10, 2))
        ctk.CTkSlider(toolbar, from_=4, to=80, variable=self.brush_size, width=110, height=16).pack(side="left", padx=4)
        self.lbl_brush_sz = ctk.CTkLabel(toolbar, text="18 px", font=ctk.CTkFont(size=11), width=42)
        self.lbl_brush_sz.pack(side="left", padx=(2, 8))
        self.brush_size.trace_add("write", lambda *_: self.lbl_brush_sz.configure(text=f"{self.brush_size.get()} px"))
        
        # Zoom controls
        ctk.CTkLabel(toolbar, text="Zoom:").pack(side="left", padx=(10,2))
        self.zoom_slider = ctk.CTkSlider(toolbar, from_=0.5, to=3.0, variable=self.zoom_factor, width=80)
        self.zoom_slider.pack(side="left", padx=2)
        self.zoom_label = ctk.CTkLabel(toolbar, text="100%", width=50)
        self.zoom_label.pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="+", width=30, command=lambda: self.zoom_factor.set(min(3.0, self.zoom_factor.get() + 0.1))).pack(side="left", padx=1)
        ctk.CTkButton(toolbar, text="-", width=30, command=lambda: self.zoom_factor.set(max(0.5, self.zoom_factor.get() - 0.1))).pack(side="left", padx=1)

        # Undo/Redo
        ctk.CTkButton(toolbar, text="", image=self.icon_undo, width=34, height=28,
                      fg_color="#37474f", hover_color="#455a64", command=self._undo).pack(side="left", padx=2)
        ctk.CTkButton(toolbar, text="", image=self.icon_redo, width=34, height=28,
                      fg_color="#37474f", hover_color="#455a64", command=self._redo).pack(side="left", padx=(2, 6))

        # Magic tools
        ctk.CTkLabel(toolbar, text=" | ", text_color="gray").pack(side="left", padx=2)
        ctk.CTkCheckBox(toolbar, text="Magic Tools", variable=self.magic_mode,
                        font=ctk.CTkFont(size=12, weight="bold"), width=60,
                        fg_color="#fbc02d", hover_color="#f9a825").pack(side="left", padx=(6, 2))
        
        # Tolerance slider adjusted for LAB
        ctk.CTkSlider(toolbar, from_=1, to=50, variable=self.magic_tol, width=80, height=16).pack(side="left", padx=(2, 6))

        # Canvas container
        self.canvas_container = tk.Frame(self, bg="#0d0d1a")
        self.canvas_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_container, bg="#0d0d1a", cursor="none", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Event bindings
        self.canvas.bind("<ButtonPress-1>", self._on_brush_press)
        self.canvas.bind("<B1-Motion>", self._on_brush_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_brush_release)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._hide_cursor)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # (Apply / Cancel)
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, pady=10)

        ctk.CTkButton(btn_frame, text="Apply", width=120, height=40,
                      fg_color="#2e7d32", hover_color="#388e3c",
                      command=self._apply).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Cancel", width=120, height=40,
                      fg_color="#b71c1c", hover_color="#c62828",
                      command=self._cancel).pack(side="left", padx=10)

    def _rebuild_display_cache(self, event=None):
        """Downscale the image to fit the canvas."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            self.after(50, self._rebuild_display_cache)
            return

        h, w = self.full_np.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        dw = max(1, int(w * scale))
        dh = max(1, int(h * scale))

        # Only process if the size has changed to avoid unnecessary work
        if dw != self.disp_w or dh != self.disp_h:
            # If a display cache already exists (contains edits), resize it.
            if self.disp_np is not None:
                disp_pil = Image.fromarray(self.disp_np, mode="RGBA").resize((dw, dh), Image.Resampling.BILINEAR)
                self.disp_np = np.array(disp_pil, dtype=np.uint8)
            # Otherwise, create it from scratch (on initialization)
            else:
                disp_pil = Image.fromarray(self.full_np, mode="RGBA").resize((dw, dh), Image.Resampling.BILINEAR)
                self.disp_np = np.array(disp_pil, dtype=np.uint8)

            # Always downscale the original image from the full-res source
            orig_pil = Image.fromarray(self.original_full_np, mode="RGB").resize((dw, dh), Image.Resampling.BILINEAR)
            self.disp_orig_np = np.array(orig_pil, dtype=np.uint8)
            
            # Pre-convert to LAB for Magic Tools using the manual function
            self.disp_orig_lab = rgb2lab_manual(self.disp_orig_np)

            self.disp_w = dw
            self.disp_h = dh

        # Update offset and refresh canvas
        ox = (cw - self.disp_w) // 2
        oy = (ch - self.disp_h) // 2
        self.canvas_offset = (ox, oy)
        self._update_zoom_display()

    def _refresh_canvas_from_cache(self):
        """Re-composite the display image and show it on the canvas."""
        if self.disp_np is None:
            return

        if self.repair_mode.get() == "restore":
            comp_np = composite_repair_np(self.disp_np, self.disp_orig_np, bg_opacity=0.5)
        else:
            comp_np = composite_np(self.disp_np, dark_bg=self.dark_bg_mode.get())

        img = Image.fromarray(comp_np, mode="RGB")
        ox, oy = self.canvas_offset
        self.canvas_photo = ImageTk.PhotoImage(img)
        self.canvas.delete("base")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.canvas_photo, tags="base")

        if self.cursor_oval is None:
            self.cursor_oval = self.canvas.create_oval(0, 0, 0, 0,
                                                        outline="#ffffff", width=1, dash=(3,3), tags="cursor")
        self.canvas.tag_raise("cursor")

    def _commit_display_to_fullres(self):
        """Upscale the editing result from display to full resolution."""
        if self.disp_np is None:
            return
        disp_pil = Image.fromarray(self.disp_np, mode="RGBA")
        h, w = self.full_np.shape[:2]
        self.full_np = np.array(disp_pil.resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)

    def _on_canvas_resize(self, event):
        self._rebuild_display_cache()
        
    def _on_zoom_change(self, *args):
        """Called when the zoom slider changes."""
        self._update_zoom_display()
        self._update_zoom_label()

    def _update_zoom_label(self, *args):
        self.zoom_label.configure(text=f"{int(self.zoom_factor.get()*100)}%")

    def _canvas_to_display(self, cx, cy):
        """Convert canvas coordinates (cx, cy) to original display coordinates (ix, iy)."""

        if self.zoom_disp_w == 0 or self.zoom_disp_h == 0:
            return None

        ox, oy = self.zoom_offset
        zf = self.zoom_factor.get()
        img_zx = cx - ox
        img_zy = cy - oy

        if 0 <= img_zx < self.zoom_disp_w and 0 <= img_zy < self.zoom_disp_h:
            ix = int(img_zx / zf)
            iy = int(img_zy / zf)
            ix = max(0, min(self.disp_w - 1, ix))
            iy = max(0, min(self.disp_h - 1, iy))
            return ix, iy
        else:
            return None

    def _update_zoom_display(self):
        """Update the canvas view with the zoomed image."""
        if self.disp_np is None:
            return
        zf = self.zoom_factor.get()
        new_w = max(1, int(self.disp_w * zf))
        new_h = max(1, int(self.disp_h * zf))
        self.zoom_disp_w = new_w
        self.zoom_disp_h = new_h

        rgba_pil = Image.fromarray(self.disp_np, mode="RGBA").resize((new_w, new_h), Image.Resampling.BILINEAR)
        zoom_rgba = np.array(rgba_pil, dtype=np.uint8)
        orig_pil = Image.fromarray(self.disp_orig_np, mode="RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
        zoom_orig = np.array(orig_pil, dtype=np.uint8)

        if self.repair_mode.get() == "restore":
            comp_np = composite_repair_np(zoom_rgba, zoom_orig, bg_opacity=0.5)
        else:
            comp_np = composite_np(zoom_rgba, dark_bg=self.dark_bg_mode.get())

        comp_img = Image.fromarray(comp_np, mode="RGB")
        self.canvas_photo = ImageTk.PhotoImage(comp_img)
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = 800, 600 
        ox = (cw - new_w) // 2
        oy = (ch - new_h) // 2
        self.zoom_offset = (ox, oy)

        self.canvas.delete("base")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.canvas_photo, tags="base")
        
        if self.cursor_oval is None:
            self.cursor_oval = self.canvas.create_oval(0, 0, 0, 0,
                                                        outline="#ffffff", width=1, dash=(3,3), tags="cursor")
        self.canvas.tag_raise("cursor")

        if self.last_mouse_x is not None:
            class DummyEvent:
                pass
            e = DummyEvent()
            e.x = self.last_mouse_x
            e.y = self.last_mouse_y
            self._on_mouse_move(e)

    def _push_history(self):
        """Save the display state to the undo stack."""
        if self.disp_np is not None:
            self.history.append(self.disp_np.copy())
            if len(self.history) > MAX_HISTORY:
                self.history.pop(0)

    def _undo(self):
        if not self.history:
            return
        self.redo_stack.append(self.disp_np.copy())
        self.disp_np = self.history.pop()
        self._update_zoom_display()

    def _redo(self):
        if not self.redo_stack:
            return
        self._push_history()
        self.disp_np = self.redo_stack.pop()
        self._update_zoom_display()

    def _paint_at(self, ix, iy):
        """Apply the brush at the original display coordinates (ix, iy)."""
        if self.disp_np is None:
            return

        dh, dw = self.disp_np.shape[:2]
        if not (0 <= ix < dw and 0 <= iy < dh):
            return

        r = max(1, self.brush_size.get() // 2)
        x0 = max(0, ix - r); x1 = min(dw, ix + r + 1)
        y0 = max(0, iy - r); y1 = min(dh, iy + r + 1)
        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1) - ix
        ys = np.arange(y0, y1) - iy
        mask = (xs[np.newaxis, :] ** 2 + ys[:, np.newaxis] ** 2) <= r * r

        if self.magic_mode.get() and self.disp_orig_lab is not None:
            # Use CIELAB for more perceptually accurate color difference
            ref_color = self.disp_orig_lab[iy, ix]
            roi = self.disp_orig_lab[y0:y1, x0:x1]
            # Delta E (CIE76) is the Euclidean distance in LAB space
            color_diff = np.linalg.norm(roi - ref_color, axis=2)
            mask = mask & (color_diff <= self.magic_tol.get())

        if self.repair_mode.get() == "restore":
            self.disp_np[y0:y1, x0:x1, :3][mask] = self.disp_orig_np[y0:y1, x0:x1][mask]
            self.disp_np[y0:y1, x0:x1, 3][mask] = 255
        else:
            # Erase: alpha=0
            self.disp_np[y0:y1, x0:x1, 3][mask] = 0

    def _on_brush_press(self, event):
        if self.disp_np is None:
            return
        coord = self._canvas_to_display(event.x, event.y)
        if coord:
            self._push_history()
            self.redo_stack.clear()
            self._paint_at(coord[0], coord[1])
            self._update_zoom_display()
            self.last_xy = (event.x, event.y)  

    def _on_brush_drag(self, event):
        if self.last_xy is None or self.disp_np is None:
            return
        x0, y0 = self.last_xy
        x1, y1 = event.x, event.y
        dist = int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        steps = max(1, dist)
        for i in range(1, steps + 1):
            t = i / steps
            cx = x0 + (x1 - x0) * t
            cy = y0 + (y1 - y0) * t
            coord = self._canvas_to_display(cx, cy)
            if coord:
                self._paint_at(coord[0], coord[1])
        self.last_xy = (event.x, event.y)
        self._update_zoom_display()
        self._on_mouse_move(event)

    def _on_brush_release(self, event):
        self.last_xy = None

    def _on_mouse_move(self, event):
        """Move the cursor circle (size adjusted for zoom)."""
        if self.cursor_oval is None:
            return
        if event is None:
            self.canvas.coords(self.cursor_oval, 0, 0, 0, 0)
            return
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        zf = self.zoom_factor.get()
        r = self.brush_size.get() * zf / 2
        self.canvas.coords(self.cursor_oval,
                        event.x - r, event.y - r,
                        event.x + r, event.y + r)
        self.canvas.tag_raise("cursor")

    def _hide_cursor(self, event):
        if self.cursor_oval is not None:
            self.canvas.coords(self.cursor_oval, 0, 0, 0, 0)

    def _apply(self):
        """Apply changes and close the window."""
        self._commit_display_to_fullres()
        result_img = Image.fromarray(self.full_np, mode="RGBA")
        if self.callback:
            self.callback(result_img)
        self.destroy()

    def _cancel(self):
        """Close the window without saving."""
        self.destroy()


class App(TkinterDnD.Tk if DND_AVAILABLE else ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("pvBG v1.3.2 — Private Background Removal (Offline)")
        self.geometry("1100x720")
        self.minsize(900, 620)

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.configure(bg="#12121f")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir  = os.path.join(current_dir, '..', 'assets')
        try:
            self.iconbitmap(os.path.join(assets_dir, 'icon.ico'))
        except Exception:
            pass

        # Load icons
        try:
            self.icon_save = ctk.CTkImage(Image.open(os.path.join(assets_dir, 'diskette.png')), size=(20, 20))
            self.icon_undo = ctk.CTkImage(Image.open(os.path.join(assets_dir, 'undo.png')), size=(20, 20))
            self.icon_play = ctk.CTkImage(Image.open(os.path.join(assets_dir, 'play-button.png')), size=(20, 20))
            self.icon_redo = ctk.CTkImage(Image.open(os.path.join(assets_dir, 'forward.png')), size=(20, 20))
            self.icon_support = ctk.CTkImage(Image.open(os.path.join(assets_dir, 'support.png')), size=(20, 20))
            self.icon_eraser = ctk.CTkImage(Image.open(os.path.join(assets_dir, 'eraser.png')), size=(20, 20))
        except Exception as e:
            print(f"Error loading icons: {e}")
            self.icon_save = self.icon_undo = self.icon_play = self.icon_redo = self.icon_support = self.icon_eraser = None

        model_path = os.path.join(current_dir, '..', 'models', 'pvBG_UNet_224_DiceBCE_v1.2.onnx')
        try:
            self.engine      = Engine(model_path)
            self.status_text = "OK: System ready. Select or drop an image to begin."
        except Exception as e:
            self.engine      = None
            self.status_text = f"ERROR: Engine error: {e}"

        # Core state
        self.current_result           = None   
        self._current_input_path      = None
        self._orig_photo_ref          = None
        self._original_rgb_np         = None   # full-res RGB numpy
        self.repair_window = None

        self._build_ui()

        if DND_AVAILABLE:
            self._setup_dnd()

    def _build_ui(self):
        """Build all UI widgets (without repair components)."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(16, 4))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="pvBG",
            font=ctk.CTkFont(family="Arial", size=30, weight="bold"),
            text_color="#4fc3f7"
        ).grid(row=0, column=0)

        ctk.CTkLabel(
            header,
            text="Private Background Removal — Offline & Private",
            font=ctk.CTkFont(size=12), text_color="gray"
        ).grid(row=1, column=0)

        # Left panel
        self.frame_left = self._make_panel("Original Image")
        self.frame_left.grid(row=1, column=0, padx=(14, 6), pady=6, sticky="nsew")

        self.lbl_orig = ctk.CTkLabel(
            self.frame_left,
            text="No image selected.\n\nDrop an image here\nor use the button below.",
            font=ctk.CTkFont(size=13), text_color="gray", wraplength=340
        )
        self.lbl_orig.pack(expand=True, fill="both", padx=8, pady=8)

        if DND_AVAILABLE:
            ctk.CTkLabel(
                self.frame_left, text="Drop image here",
                font=ctk.CTkFont(size=11), text_color="#444466"
            ).place(relx=0.5, rely=0.96, anchor="s")

        # Right panel
        self.frame_right = ctk.CTkFrame(self, corner_radius=12)
        self.frame_right.grid(row=1, column=1, padx=(6, 14), pady=6, sticky="nsew")
        self.frame_right.grid_columnconfigure(0, weight=1)
        self.frame_right.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            self.frame_right, text="Result",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#90caf9"
        ).grid(row=0, column=0, pady=(10, 0))

        ctk.CTkFrame(
            self.frame_right, height=1, fg_color="#2a2a4a"
        ).grid(row=1, column=0, sticky="ew", padx=12, pady=4)

        self.lbl_res = ctk.CTkLabel(
            self.frame_right,
            text="Result will appear here\nafter processing.",
            font=ctk.CTkFont(size=13), text_color="gray", wraplength=340,
            anchor="center"
        )
        self.lbl_res.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)

        # Controls
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 2))
        for i in range(5):
            ctrl.grid_columnconfigure(i, weight=1)

        self.btn_open = ctk.CTkButton(
            ctrl, text="Select Image", command=self.open_image,
            height=40, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10
        )
        self.btn_open.grid(row=0, column=0, padx=8, pady=6, sticky="ew")

        self.btn_process = ctk.CTkButton(
            ctrl, text="Remove Background", image=self.icon_play, command=self._trigger_process,
            height=40, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10,
            state="disabled", fg_color="#1565c0", hover_color="#1976d2"
        )
        self.btn_process.grid(row=0, column=1, padx=8, pady=6, sticky="ew")

        self.btn_repair = ctk.CTkButton(
            ctrl, text="Repair Mask", image=self.icon_support, command=self._toggle_repair,
            height=40, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10,
            state="disabled", fg_color="#6a1b9a", hover_color="#7b1fa2"
        )
        self.btn_repair.grid(row=0, column=2, padx=8, pady=6, sticky="ew")

        self.btn_save = ctk.CTkButton(
            ctrl, text="Save as PNG", image=self.icon_save, command=self.save_image,
            height=40, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10,
            state="disabled", fg_color="#2e7d32", hover_color="#388e3c"
        )
        self.btn_save.grid(row=0, column=3, padx=8, pady=6, sticky="ew")

        self.btn_clear = ctk.CTkButton(
            ctrl, text="Clear", image=self.icon_eraser, command=self.clear_all,
            height=40, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10,
            state="disabled", fg_color="#c62828", hover_color="#d32f2f"
        )
        self.btn_clear.grid(row=0, column=4, padx=8, pady=6, sticky="ew")

        self.lbl_info = ctk.CTkLabel(
            self, text=self.status_text,
            font=ctk.CTkFont(size=12), text_color="gray", wraplength=1000
        )
        self.lbl_info.grid(row=3, column=0, columnspan=2, pady=(0, 8))

    def _make_panel(self, title: str) -> ctk.CTkFrame:
        """Create a titled preview panel frame."""
        outer = ctk.CTkFrame(self, corner_radius=12)
        ctk.CTkLabel(
            outer, text=title,
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#90caf9"
        ).pack(pady=(10, 0))
        ctk.CTkFrame(outer, height=1, fg_color="#2a2a4a").pack(fill="x", padx=12, pady=4)
        return outer

    def _setup_dnd(self):
        """Register drag-and-drop targets."""
        for widget in (self, self.frame_left, self.lbl_orig):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        """Handle a file drop event."""
        if self._current_input_path is not None:
            messagebox.showwarning(
                "Workspace Not Empty",
                "Please clear the current image and result before loading a new one."
            )
            return

        path = event.data.strip().strip("{}").split("} {")[0]
        if os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXT):
            self._load_image(path)
        else:
            self._set_status("Warning: Unsupported file. Drop a JPG, PNG, WEBP, or BMP image.")

    def open_image(self):
        """Open file manager dialog and load the selected image."""
        if self._current_input_path is not None:
            messagebox.showwarning(
                "Workspace Not Empty",
                "Please clear the current image and result before loading a new one."
            )
            return

        if not self.engine:
            messagebox.showerror("Engine Error", "pvBG model is not loaded.")
            return
        
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")]
        )

        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        """
        Load an image from disk, show preview in Original panel,
        and reset the Result panel.
        """
        if self.repair_window and self.repair_window.winfo_exists():
            self.repair_window.destroy()
            self.repair_window = None

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            self._set_status(f"ERROR: Failed to open image: {e}")
            return

        self._current_input_path = path
        self._original_rgb_np    = np.array(img, dtype=np.uint8)  # keep full-res RGB numpy

        thumb = img.copy()
        thumb.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
        photo = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=thumb.size)
        self._orig_photo_ref = photo
        self.lbl_orig.configure(image=photo, text="")

        self.current_result = None
        self.lbl_res.configure(image=None, text="Ready to process.\nClick Remove Background.")
        self.btn_save.configure(state="disabled")
        self.btn_repair.configure(state="disabled")
        self.btn_process.configure(state="normal", text="Remove Background")
        self.btn_clear.configure(state="normal")

        w, h = img.size
        self._set_status(
            f"Loaded: {os.path.basename(path)}  ({w} × {h} px)"
            f" — Click Remove Background to process."
        )

    def _trigger_process(self):
        """Start background removal in a background thread."""
        if not self.engine or not self._current_input_path:
            return
        if self.repair_window and self.repair_window.winfo_exists():
            self.repair_window.destroy()
            self.repair_window = None

        self.btn_process.configure(state="disabled", text="Wait: Processing...")
        self.btn_open.configure(state="disabled")
        self.btn_repair.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.lbl_res.configure(image=None, text="Wait: Removing background...\nPlease wait.")
        self._set_status("Wait: pvBG is processing the image. Please wait...")

        threading.Thread(
            target=self._run_inference,
            args=(self._current_input_path,),
            daemon=True
        ).start()

    def _run_inference(self, path: str):
        """
        Run pvBG inference on a background thread, then schedule
        the UI update back on the main thread.
        """
        try:
            result = self.engine.remove_background(path)
            self.current_result = result
            self.after(0, self._show_result)
        except Exception as e:
            self.after(0, lambda: self._on_inference_error(str(e)))

    def _show_result(self):
        """Display the processed result and enable repair/save buttons."""
        if self.current_result is None:
            self._on_inference_error("Engine returned no result.")
            return

        self._display_result_label()
        self.btn_save.configure(state="normal")
        self.btn_repair.configure(state="normal")
        self.btn_process.configure(state="disabled", text="OK: Processed")
        self.btn_open.configure(state="normal")

        w, h = self.current_result.size
        self._set_status(
            f"OK: Done!  {w} × {h} px — "
            f"Repair Mask to fix edges, or Save as PNG to export."
        )

    def _display_result_label(self):
        """Render current_result into the standard CTkLabel preview."""
        rgba_np    = np.array(self.current_result, dtype=np.uint8)
        comp_np    = composite_np(rgba_np)
        comp_img   = Image.fromarray(comp_np, mode="RGB")
        thumb      = comp_img.copy()
        thumb.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
        photo = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=thumb.size)
        self._res_label_photo = photo
        self.lbl_res.configure(image=photo, text="")

    def _on_inference_error(self, message: str):
        """Handle inference errors and restore UI state."""
        self.lbl_res.configure(image=None, text="ERROR: Processing failed.")
        self.btn_process.configure(state="normal", text="Remove Background")
        self.btn_open.configure(state="normal")
        self._set_status(f"ERROR: {message}")

    def _toggle_repair(self):
        """Open the repair window or focus it if it already exists."""
        if self.current_result is None:
            return

        if self.repair_window and self.repair_window.winfo_exists():
            self.repair_window.lift()
            self.repair_window.focus()
        else:
            self.repair_window = RepairWindow(
                self,
                self.current_result,
                self._original_rgb_np,
                self._on_repair_applied
            )
            self.repair_window.lift()
            self.repair_window.focus()
            self.repair_window.after(10, self.repair_window.lift)

    def _on_repair_applied(self, new_result: Image.Image):
        """Callback from the repair window: update the result and display."""
        self.current_result = new_result
        self._display_result_label()
        self.repair_window = None

    def clear_all(self):
        """Reset the application state, clearing images and results."""
        if self.repair_window and self.repair_window.winfo_exists():
            self.repair_window.destroy()
            self.repair_window = None

        self._current_input_path = None
        self.current_result = None
        self._orig_photo_ref = None
        self._res_label_photo = None
        self._original_rgb_np = None

        self.lbl_orig.configure(
            image=None,
            text="No image selected.\n\nDrop an image here\nor use the button below."
        )
        if hasattr(self.lbl_orig, "_label"):
            self.lbl_orig._label.configure(image="")

        self.lbl_res.configure(
            image=None,
            text="Result will appear here\nafter processing."
        )
        if hasattr(self.lbl_res, "_label"):
            self.lbl_res._label.configure(image="")

        self.btn_process.configure(state="disabled", text="Remove Background")
        self.btn_repair.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.btn_clear.configure(state="disabled")
        self._set_status("OK: System ready. Select or drop an image to begin.")

    def save_image(self):
        """Save the result RGBA image as a PNG file."""
        if not self.current_result:
            return
        
        path = filedialog.asksaveasfilename(
            title="Save result as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )

        if path:
            try:
                self.current_result.save(path, format="PNG")
                messagebox.showinfo("Saved", f"Image saved!\n{path}")
                self._set_status(f"Saved to: {path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save:\n{e}")

    def _set_status(self, text: str):
        """Update the bottom status bar label."""
        self.lbl_info.configure(text=text)

if __name__ == "__main__":
    app = App()
    app.mainloop()