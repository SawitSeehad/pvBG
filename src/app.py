# ==========================================================
# AI Background Remover
# Copyright (C) 2026 Saw it See had
# Licensed under the MIT License
# ==========================================================

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import threading
import os
import sys

from app import Engine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("reMBG v1.0.0 (Offline)")
        self.geometry("900x600")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, '..', 'assets')

        try:
            icon_path = os.path.join(assets_dir, 'icon.ico')
            self.iconbitmap(icon_path)
        except:
            pass 

        try:
            icon_png = os.path.join(assets_dir, 'icon.png')
            img = Image.open(icon_png)
            photo = ImageTk.PhotoImage(img)
            self.wm_iconphoto(True, photo)
        except:
            pass

        # Setup Model Path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'models', 'segmentasi_manusia.onnx')

        # Initialize Engine
        try:
            self.engine = Engine(model_path)
            self.status_text = "System Ready."
        except Exception as e:
            self.engine = None
            self.status_text = f"Error: {str(e)}"

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        self.lbl_title = ctk.CTkLabel(self, text="reMBG", font=("Arial", 28, "bold"))
        self.lbl_title.grid(row=0, column=0, columnspan=2, pady=20)

        # Preview Area
        self.frame_left = ctk.CTkFrame(self)
        self.frame_left.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.lbl_orig = ctk.CTkLabel(self.frame_left, text="Original Image")
        self.lbl_orig.pack(expand=True)

        self.frame_right = ctk.CTkFrame(self)
        self.frame_right.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.lbl_res = ctk.CTkLabel(self.frame_right, text="Result")
        self.lbl_res.pack(expand=True)

        # Controls
        self.btn_open = ctk.CTkButton(self, text="Select Image", command=self.open_image, height=40)
        self.btn_open.grid(row=2, column=0, padx=20, pady=20)

        self.btn_save = ctk.CTkButton(self, text="Save as PNG", command=self.save_image, state="disabled", fg_color="green", height=40)
        self.btn_save.grid(row=2, column=1, padx=20, pady=20)

        self.lbl_info = ctk.CTkLabel(self, text=self.status_text, text_color="gray")
        self.lbl_info.grid(row=3, column=0, columnspan=2, pady=5)

        self.current_result = None

    def open_image(self):
        if not self.engine:
            messagebox.showerror("Error", "Model not found or failed to load.")
            return

        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
        if path:
            img = Image.open(path)
            img.thumbnail((350, 350))
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.lbl_orig.configure(image=photo, text="")
            self.lbl_res.configure(image=None, text="Processing...")
            self.btn_save.configure(state="disabled")
            self.lbl_info.configure(text="⏳ Processing AI...")

            threading.Thread(target=self.run_process, args=(path,)).start()

    def run_process(self, path):
        try:
            result = self.engine.remove_background(path)
            self.current_result = result

            self.after(0, self.show_result)
        except Exception as e:
            self.lbl_info.configure(text=f"Error: {e}")

    def show_result(self):
        thumb = self.current_result.copy()
        thumb.thumbnail((350, 350))
        photo = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=thumb.size)
        
        self.lbl_res.configure(image=photo, text="")
        self.lbl_info.configure(text="✅ Done!")
        self.btn_save.configure(state="normal")

    def save_image(self):
        if self.current_result:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if path:
                self.current_result.save(path)
                messagebox.showinfo("Success", "Image saved successfully!")

if __name__ == "__main__":
    app = App()
    app.mainloop()