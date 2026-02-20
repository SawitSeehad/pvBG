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

def make_checkerboard(width: int, height: int, tile: int = 12) -> np.ndarray:
    """Generate an RGB checkerboard numpy array (fast, no Python loops)."""
    xs   = np.arange(width)  // tile
    ys   = np.arange(height) // tile
    mask = (xs[np.newaxis, :] + ys[:, np.newaxis]) % 2 == 0
    arr  = np.where(mask[:, :, np.newaxis], 200, 155).astype(np.uint8)
    return np.repeat(arr, 3, axis=2)

# Opacity of the dimmed original photo background shown in repair mode (0.0–1.0)
REPAIR_BG_OPACITY = 0.70


def composite_np(rgba_np: np.ndarray) -> np.ndarray:
    """
    Composite an RGBA numpy array onto a checkerboard.
    Used in normal (non-repair) preview mode.
    """
    h, w    = rgba_np.shape[:2]
    checker = make_checkerboard(w, h)
    alpha   = rgba_np[:, :, 3:4].astype(np.float32) / 255.0
    rgb     = rgba_np[:, :, :3].astype(np.float32)
    result  = (rgb * alpha + checker.astype(np.float32) * (1.0 - alpha))
    return result.astype(np.uint8)


def composite_repair_np(rgba_np: np.ndarray, orig_np: np.ndarray,
                        bg_opacity: float = REPAIR_BG_OPACITY) -> np.ndarray:
    """
    Composite the segmentation result over the dimmed original photo.
    Used exclusively in repair mode so the user can see the original
    image context while painting.

    Opaque foreground pixels render at full brightness.
    Transparent (erased) areas show the original photo at bg_opacity.

    Args:
        rgba_np    : H x W x 4 uint8 — current RGBA result at display size.
        orig_np    : H x W x 3 uint8 — original RGB photo at display size.
        bg_opacity : float in [0, 1] — brightness of the background photo.

    Returns:
        H x W x 3 uint8 RGB array ready for canvas display.
    """
    alpha  = rgba_np[:, :, 3:4].astype(np.float32) / 255.0
    fg     = rgba_np[:, :, :3].astype(np.float32)
    bg     = orig_np.astype(np.float32) * bg_opacity
    result = fg * alpha + bg * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)

class App(TkinterDnD.Tk if DND_AVAILABLE else ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("pvBG v1.0.0 — Private Background Removal (Offline)")
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

        model_path = os.path.join(current_dir, '..', 'models', 'pvBG.onnx')
        try:
            self.engine      = Engine(model_path)
            self.status_text = "✅  System ready. Select or drop an image to begin."
        except Exception as e:
            self.engine      = None
            self.status_text = f"❌  Engine error: {e}"

        # Core state
        self.current_result           = None   
        self._current_input_path      = None
        self._orig_photo_ref          = None

        # Repair state — all editing happens on display-size numpy arrays
        self._repair_active     = False
        self._repair_mode       = tk.StringVar(value="restore")
        self._brush_size        = tk.IntVar(value=18)
        self._history           = []           
        self._redo_stack        = []
        self._last_xy           = None

        self._disp_rgba_np      = None       
        self._disp_orig_np      = None         
        self._disp_w            = 0
        self._disp_h            = 0
        self._canvas_offset     = (0, 0)
        self._canvas_photo      = None
        self._cursor_oval       = None

        self._build_ui()

        self.bind("<Control-z>", lambda e: self._undo() if self._repair_active else None)
        self.bind("<Control-y>", lambda e: self._redo() if self._repair_active else None)

        if DND_AVAILABLE:
            self._setup_dnd()

    def _build_ui(self):
        """Build all UI widgets."""
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
                self.frame_left, text="⬇  Drop image here",
                font=ctk.CTkFont(size=11), text_color="#444466"
            ).place(relx=0.5, rely=0.96, anchor="s")

        # Right panel — grid-managed for repair toolbar swap
        self.frame_right = ctk.CTkFrame(self, corner_radius=12)
        self.frame_right.grid(row=1, column=1, padx=(6, 14), pady=6, sticky="nsew")
        self.frame_right.grid_columnconfigure(0, weight=1)
        self.frame_right.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            self.frame_right, text="Result",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#90caf9"
        ).grid(row=0, column=0, pady=(10, 0))

        ctk.CTkFrame(
            self.frame_right, height=1, fg_color="#2a2a4a"
        ).grid(row=1, column=0, sticky="ew", padx=12, pady=4)

        # Repair toolbar (hidden by default)
        self.repair_toolbar = ctk.CTkFrame(self.frame_right, fg_color="#1a1a30", corner_radius=8)
        self.repair_toolbar.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 2))
        self.repair_toolbar.grid_remove()
        self._build_repair_toolbar()

        # Canvas (hidden by default, shown in repair mode)
        self.canvas_container = tk.Frame(self.frame_right, bg="#0d0d1a")
        self.canvas_container.grid(row=3, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.canvas_container.grid_remove()

        self.result_canvas = tk.Canvas(
            self.canvas_container, bg="#0d0d1a",
            cursor="none", highlightthickness=0
        )
        self.result_canvas.pack(fill="both", expand=True)
        self.result_canvas.bind("<ButtonPress-1>",   self._on_brush_press)
        self.result_canvas.bind("<B1-Motion>",       self._on_brush_drag)
        self.result_canvas.bind("<ButtonRelease-1>", self._on_brush_release)
        self.result_canvas.bind("<Motion>",          self._on_mouse_move)
        self.result_canvas.bind("<Leave>",           self._hide_cursor)
        self.result_canvas.bind("<Configure>",       self._on_canvas_resize)

        # Result label (shown when NOT in repair mode)
        self.lbl_res = ctk.CTkLabel(
            self.frame_right,
            text="Result will appear here\nafter processing.",
            font=ctk.CTkFont(size=13), text_color="gray", wraplength=340
        )
        self.lbl_res.grid(row=3, column=0, sticky="nsew", padx=8, pady=8)

        # Controls
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 2))
        for i in range(5):
            ctrl.grid_columnconfigure(i, weight=1)

        self.btn_open = ctk.CTkButton(
            ctrl, text="📂  Select Image", command=self.open_image,
            height=40, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10
        )
        self.btn_open.grid(row=0, column=0, padx=8, pady=6, sticky="ew")

        self.btn_process = ctk.CTkButton(
            ctrl, text="▶  Remove Background", command=self._trigger_process,
            height=40, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10,
            state="disabled", fg_color="#1565c0", hover_color="#1976d2"
        )
        self.btn_process.grid(row=0, column=1, padx=8, pady=6, sticky="ew")

        self.btn_repair = ctk.CTkButton(
            ctrl, text="🖌  Repair Mask", command=self._toggle_repair,
            height=40, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10,
            state="disabled", fg_color="#6a1b9a", hover_color="#7b1fa2"
        )
        self.btn_repair.grid(row=0, column=2, padx=8, pady=6, sticky="ew")

        self.btn_save = ctk.CTkButton(
            ctrl, text="💾  Save as PNG", command=self.save_image,
            height=40, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10,
            state="disabled", fg_color="#2e7d32", hover_color="#388e3c"
        )
        self.btn_save.grid(row=0, column=3, padx=8, pady=6, sticky="ew")

        self.btn_clear = ctk.CTkButton(
            ctrl, text="🗑  Clear", command=self.clear_all,
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

    def _build_repair_toolbar(self):
        """Build the inline repair toolbar widgets."""
        ctk.CTkLabel(
            self.repair_toolbar, text="Mode:", font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(10, 4), pady=6)

        ctk.CTkRadioButton(
            self.repair_toolbar, text="✏️ Restore",
            variable=self._repair_mode, value="restore",
            font=ctk.CTkFont(size=12), fg_color="#1976d2",
            command=self._refresh_canvas_from_cache
        ).pack(side="left", padx=6)

        ctk.CTkRadioButton(
            self.repair_toolbar, text="🧹 Erase",
            variable=self._repair_mode, value="erase",
            font=ctk.CTkFont(size=12), fg_color="#c62828",
            command=self._refresh_canvas_from_cache
        ).pack(side="left", padx=6)

        ctk.CTkLabel(
            self.repair_toolbar, text="  |  Size:", font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(10, 2))

        ctk.CTkSlider(
            self.repair_toolbar, from_=4, to=80,
            variable=self._brush_size, width=110, height=16
        ).pack(side="left", padx=4)

        self._lbl_brush_sz = ctk.CTkLabel(
            self.repair_toolbar, text="18 px",
            font=ctk.CTkFont(size=11), width=42
        )
        self._lbl_brush_sz.pack(side="left", padx=(2, 8))
        self._brush_size.trace_add(
            "write",
            lambda *_: self._lbl_brush_sz.configure(text=f"{self._brush_size.get()} px")
        )

        ctk.CTkButton(
            self.repair_toolbar, text="↩", width=34, height=28,
            font=ctk.CTkFont(size=14), fg_color="#37474f",
            hover_color="#455a64", command=self._undo
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            self.repair_toolbar, text="↪", width=34, height=28,
            font=ctk.CTkFont(size=14), fg_color="#37474f",
            hover_color="#455a64", command=self._redo
        ).pack(side="left", padx=(2, 6))

        ctk.CTkButton(
            self.repair_toolbar, text="✖ Cancel", width=90, height=28,
            font=ctk.CTkFont(size=11), fg_color="#b71c1c",
            hover_color="#c62828", command=lambda: self._exit_repair(save=False)
        ).pack(side="right", padx=8)

    def _setup_dnd(self):
        """Register drag-and-drop targets."""
        for widget in (self, self.frame_left, self.lbl_orig):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        """Handle a file drop event."""
        path = event.data.strip().strip("{}").split("} {")[0]
        if os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXT):
            self._load_image(path)
        else:
            self._set_status("⚠️  Unsupported file. Drop a JPG, PNG, WEBP, or BMP image.")

    def open_image(self):
        """Open file manager dialog and load the selected image."""
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

        Args:
            path (str): Absolute path to the image file.
        """
        if self._repair_active:
            self._exit_repair(save=False)

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            self._set_status(f"❌  Failed to open image: {e}")
            return

        self._current_input_path = path
        self._original_rgb_np    = np.array(img, dtype=np.uint8)  # keep full-res RGB numpy

        thumb = img.copy()
        thumb.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
        photo = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=thumb.size)
        self._orig_photo_ref = photo
        self.lbl_orig.configure(image=photo, text="")

        self.current_result = None
        self.lbl_res.configure(image=None, text="Ready to process.\nClick ▶ Remove Background.")
        self.btn_save.configure(state="disabled")
        self.btn_repair.configure(state="disabled")
        self.btn_process.configure(state="normal")
        self.btn_clear.configure(state="normal")

        w, h = img.size
        self._set_status(
            f"📂  Loaded: {os.path.basename(path)}  ({w} × {h} px)"
            f"  —  Click ▶ Remove Background to process."
        )

    def _trigger_process(self):
        """Start background removal in a background thread."""
        if not self.engine or not self._current_input_path:
            return
        if self._repair_active:
            self._exit_repair(save=False)

        self.btn_process.configure(state="disabled", text="⏳  Processing...")
        self.btn_open.configure(state="disabled")
        self.btn_repair.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.lbl_res.configure(image=None, text="⏳  Removing background...\nPlease wait.")
        self._set_status("⏳  pvBG is processing the image. Please wait...")

        threading.Thread(
            target=self._run_inference,
            args=(self._current_input_path,),
            daemon=True
        ).start()

    def _run_inference(self, path: str):
        """
        Run pvBG inference on a background thread, then schedule
        the UI update back on the main thread.

        Args:
            path (str): Path to the input image.
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
        self.btn_process.configure(state="normal", text="▶  Remove Background")
        self.btn_open.configure(state="normal")

        w, h = self.current_result.size
        self._set_status(
            f"✅  Done!  {w} × {h} px  —  "
            f"🖌 Repair Mask to fix edges,  💾 Save as PNG to export."
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
        self.lbl_res.configure(image=None, text="❌  Processing failed.")
        self.btn_process.configure(state="normal", text="▶  Remove Background")
        self.btn_open.configure(state="normal")
        self._set_status(f"❌  Error: {message}")

    def _toggle_repair(self):
        """Toggle repair mode on/off."""
        if self._repair_active:
            self._exit_repair()
        else:
            self._enter_repair()

    def _enter_repair(self):
        """
        Switch to repair mode.

        Downscales both the result and original image to display resolution
        once. All painting happens on these small display-size arrays only.
        The full-resolution image is only updated on mouse release.
        """
        if self.current_result is None:
            return

        self._repair_active = True
        self._history.clear()
        self._redo_stack.clear()

        self.lbl_res.grid_remove()
        self.repair_toolbar.grid()
        self.canvas_container.grid()

        self.btn_repair.configure(
            text="✅  Done Repairing", fg_color="#00695c", hover_color="#00796b"
        )
        self.btn_process.configure(state="disabled")
        self.btn_open.configure(state="disabled")

        self._set_status(
            "🖌  Repair mode  —  ✏️ Restore or 🧹 Erase.  "
            "Ctrl+Z = Undo  |  Ctrl+Y = Redo  |  Click ✅ Done when finished."
        )

        self.after(50, self._rebuild_display_cache)

    def _exit_repair(self, save: bool = True):
        """
        Exit repair mode.
        If save=True, apply display-res edits back to full-res.
        If save=False, discard edits.
        """
        if self._repair_active and self._disp_rgba_np is not None:
            if save:
                self._commit_display_to_fullres()

        self._repair_active = False
        self._disp_rgba_np  = None
        self._disp_orig_np  = None

        self.canvas_container.grid_remove()
        self.repair_toolbar.grid_remove()
        self.lbl_res.grid(row=3, column=0, sticky="nsew", padx=8, pady=8)

        self._display_result_label()

        self.btn_repair.configure(text="🖌  Repair Mask", fg_color="#6a1b9a", hover_color="#7b1fa2")
        self.btn_process.configure(state="normal")
        self.btn_open.configure(state="normal")

        if save:
            self._set_status("🖌  Repair applied.  💾 Save as PNG to export.")
        else:
            self._set_status("❎  Repair cancelled. Changes discarded.")

    def _rebuild_display_cache(self):
        """
        Downscale current_result and original RGB to canvas display size.
        This runs once on enter_repair, once on undo/redo.
        All brush painting operates on these small arrays — very fast.
        """
        if self.current_result is None:
            return

        cw = self.result_canvas.winfo_width()
        ch = self.result_canvas.winfo_height()

        if cw < 2 or ch < 2:
            self.after(60, self._rebuild_display_cache)
            return

        iw, ih = self.current_result.size
        scale  = min(cw / iw, ch / ih, 1.0)
        dw     = max(1, int(iw * scale))
        dh     = max(1, int(ih * scale))
        ox     = (cw - dw) // 2
        oy     = (ch - dh) // 2

        self._canvas_offset = (ox, oy)
        self._disp_w        = dw
        self._disp_h        = dh

        # Downscale RGBA result to display size
        disp_rgba = self.current_result.resize((dw, dh), Image.Resampling.BILINEAR)
        self._disp_rgba_np = np.array(disp_rgba, dtype=np.uint8)

        # Downscale original RGB to display size
        orig_pil = Image.fromarray(self._original_rgb_np, mode="RGB")
        disp_orig = orig_pil.resize((dw, dh), Image.Resampling.BILINEAR)
        self._disp_orig_np = np.array(disp_orig, dtype=np.uint8)

        self._refresh_canvas_from_cache()

    def _refresh_canvas_from_cache(self):
        """
        Composite self._disp_rgba_np onto background based on mode and push to canvas.
        Operates entirely on the small display-size array — instant.
        """
        if self._disp_rgba_np is None:
            return

        if self._repair_mode.get() == "restore":
            comp_np = composite_repair_np(self._disp_rgba_np, self._disp_orig_np)
        else:
            comp_np = composite_np(self._disp_rgba_np)

        photo_img  = Image.fromarray(comp_np, mode="RGB")
        ox, oy     = self._canvas_offset

        self._canvas_photo = ImageTk.PhotoImage(photo_img)
        self.result_canvas.delete("base")
        self.result_canvas.create_image(
            ox, oy, anchor="nw", image=self._canvas_photo, tags="base"
        )

        if self._cursor_oval is None:
            self._cursor_oval = self.result_canvas.create_oval(
                0, 0, 0, 0, outline="#ffffff", width=1, dash=(3, 3), tags="cursor"
            )
        self.result_canvas.tag_raise("cursor")

    def _commit_display_to_fullres(self):
        """
        Upscale the edited display-size RGBA back to full resolution and
        update self.current_result. Called once on exit_repair.
        """
        if self._disp_rgba_np is None:
            return
        disp_pil = Image.fromarray(self._disp_rgba_np, mode="RGBA")
        iw, ih   = self.current_result.size
        fullres  = disp_pil.resize((iw, ih), Image.Resampling.BILINEAR)
        self.current_result = fullres

    def _on_canvas_resize(self, event):
        """Rebuild display cache when panel size changes."""
        if self._repair_active:
            self._rebuild_display_cache()

    def _paint_at(self, cx: float, cy: float):
        """
        Paint one brush stamp directly onto self._disp_rgba_np (display size).
        Pure numpy — no PIL, no full-res access, no loop.
        Runs on every mouse move event — must be instant.
        """
        if self._disp_rgba_np is None:
            return

        ox, oy = self._canvas_offset
        ix = int(cx - ox)
        iy = int(cy - oy)
        r  = max(1, self._brush_size.get() // 2)
        dh, dw = self._disp_rgba_np.shape[:2]

        # Clamp bounding box
        x0 = max(0, ix - r);  x1 = min(dw, ix + r + 1)
        y0 = max(0, iy - r);  y1 = min(dh, iy + r + 1)

        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1) - ix
        ys = np.arange(y0, y1) - iy
        mask = (xs[np.newaxis, :] ** 2 + ys[:, np.newaxis] ** 2) <= r * r

        if self._repair_mode.get() == "restore":
            self._disp_rgba_np[y0:y1, x0:x1, :3][mask] = \
                self._disp_orig_np[y0:y1, x0:x1][mask]
            self._disp_rgba_np[y0:y1, x0:x1,  3][mask] = 255
        else:
            self._disp_rgba_np[y0:y1, x0:x1, 3][mask] = 0

    def _on_brush_press(self, event):
        """Handle mouse press — save undo state and paint first stamp."""
        if not self._repair_active:
            return
        self._push_history()
        self._redo_stack.clear()
        self._paint_at(event.x, event.y)
        self._refresh_canvas_from_cache()
        self._last_xy = (event.x, event.y)

    def _on_brush_drag(self, event):
        """
        Handle mouse drag — paint along the stroke path and refresh canvas.
        Interpolates between last and current position for smooth lines.
        All work is numpy on display-size array — stays fast.
        """
        if not self._repair_active:
            return

        if self._last_xy:
            x0, y0 = self._last_xy
            x1, y1 = event.x, event.y
            dist   = int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
            steps  = max(1, dist)
            for i in range(1, steps + 1):
                t = i / steps
                self._paint_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

        self._last_xy = (event.x, event.y)
        self._refresh_canvas_from_cache()
        self._on_mouse_move(event)

    def _on_brush_release(self, event):
        """Handle mouse release — end of stroke."""
        self._last_xy = None

    def _on_mouse_move(self, event):
        """Update cursor circle preview on mouse move (no paint)."""
        if not self._repair_active or self._cursor_oval is None:
            return
        r = self._brush_size.get() / 2
        self.result_canvas.coords(
            self._cursor_oval,
            event.x - r, event.y - r,
            event.x + r, event.y + r
        )
        self.result_canvas.tag_raise("cursor")

    def _hide_cursor(self, event):
        """Hide the cursor circle when mouse leaves the canvas."""
        if self._cursor_oval is not None:
            self.result_canvas.coords(self._cursor_oval, 0, 0, 0, 0)

    def _push_history(self):
        """Save current display-size RGBA into the undo history."""
        if self._disp_rgba_np is not None:
            self._history.append(self._disp_rgba_np.copy())
            if len(self._history) > MAX_HISTORY:
                self._history.pop(0)

    def _undo(self):
        """Revert to the previous state."""
        if not self._repair_active or not self._history:
            return
        self._redo_stack.append(self._disp_rgba_np.copy())
        self._disp_rgba_np = self._history.pop()
        self._refresh_canvas_from_cache()

    def _redo(self):
        """Re-apply the last undone state."""
        if not self._repair_active or not self._redo_stack:
            return
        self._push_history()
        self._disp_rgba_np = self._redo_stack.pop()
        self._refresh_canvas_from_cache()

    def clear_all(self):
        """Reset the application state, clearing images and results."""
        if self._repair_active:
            # Manually reset repair state to avoid triggering result display
            self._repair_active = False
            self.canvas_container.grid_remove()
            self.repair_toolbar.grid_remove()
            self.lbl_res.grid(row=3, column=0, sticky="nsew", padx=8, pady=8)
            self._disp_rgba_np = None
            self._disp_orig_np = None
            self._history.clear()
            self._redo_stack.clear()

        self._current_input_path = None
        self.current_result = None
        self._orig_photo_ref = None
        self._res_label_photo = None
        self._original_rgb_np = None

        self.lbl_orig.configure(
            image=None,
            text="No image selected.\n\nDrop an image here\nor use the button below."
        )
        # Fix: Force underlying tkinter label to forget the image to prevent TclError
        if hasattr(self.lbl_orig, "_label"):
            self.lbl_orig._label.configure(image="")

        self.lbl_res.configure(
            image=None,
            text="Result will appear here\nafter processing."
        )
        if hasattr(self.lbl_res, "_label"):
            self.lbl_res._label.configure(image="")

        self.btn_process.configure(state="disabled", text="▶  Remove Background")
        self.btn_repair.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.btn_clear.configure(state="disabled")
        self._set_status("✅  System ready. Select or drop an image to begin.")

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
                self._set_status(f"💾  Saved to: {path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save:\n{e}")

    def _set_status(self, text: str):
        """Update the bottom status bar label."""
        self.lbl_info.configure(text=text)

if __name__ == "__main__":
    app = App()
    app.mainloop()