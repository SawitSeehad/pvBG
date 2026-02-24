# pvBG

![pvBG Logo](assets/icon.png)

![Version](https://img.shields.io/badge/version-v1.3.1-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

**pvBG** is a specialized, privacy-first desktop application designed to **remove backgrounds specifically from human portraits** (Selfies, ID photos, Group photos).

Unlike general-purpose tools, pvBG is optimized for **Human Segmentation**. It runs **100% offline** on your computer, ensuring that your personal photos are processed locally and **never uploaded to the cloud**.

---

## ✨ Key Features

- 👤 **Human-Centric AI:** Fine-tuned to detect human hair, poses, and silhouettes with high precision.
    > *Note: This model is specialized for people. It may not perform well on inanimate objects (cars, furniture, products).*
- 🔒 **Maximum Privacy:** Your photos never leave your device. No API keys, no internet connection required, no data collection.
- ⚡ **Lightweight & Fast:** Powered by ONNX Runtime, optimized for standard CPUs (No expensive GPU needed).
- 🚀 **Native Experience:** Installs as a standalone Desktop App with a custom icon.
- 🖥️ **Cross-Platform:** Works seamlessly on Windows and Linux.

---

## 🆕 What's New in v1.3.2
- **Advanced Editing:** Introduced **Magic Tools** for smarter and faster mask refinement.
- **Workspace Control:** Added **Zoom In / Zoom Out** functionality for pixel-perfect precision on the canvas.
- **UI Enhancements:** Added a **Dark Background** option to help users contrast and inspect extracted subjects better.
- **Stability Fix:** Resolved a critical **Segmentation Fault** issue that caused unexpected crashes by replacing system emojis with stable `.png` icons.

---

## 🧠 Technical Deep Dive

To achieve high-quality results on local hardware, pvBG utilizes a custom-trained deep learning pipeline:

* **Architecture:** **Custom U-Net** with a 32-base channel configuration, optimized for a balance between speed and edge detail.
* **Input Resolution:** $224 \times 224$ pixels, providing optimal inference speed on standard CPUs.
* **Training Dataset:** Trained on **P3M-10k** (Privacy-Preserving Portrait Matting), ensuring high-fidelity segmentation for diverse human poses.
* **Loss Function:** Optimized using a hybrid **Dice-BCE Loss** to handle class imbalance and ensure sharp boundary masks.
* **Engine:** Exported to **ONNX** format for high-performance, cross-platform execution without the need for a heavy deep learning framework.

---

## 📋 Prerequisites

Before running this application, please ensure you have **Python 3.10+** installed.
- **Windows:** Download from [python.org](https://www.python.org/downloads/) or Microsoft Store.
- **Linux:** `sudo apt install python3-full` (Ubuntu/Debian).

---

## 🚀 Installation (One-Click Setup)

We provide an automated installer that handles dependencies and creates a Desktop Shortcut for you.

### 🪟 For Windows Users

1.  Download and extract the folder.
2.  Double-click **`SETUP_WINDOWS.bat`**.
3.  Wait for the installation to finish.
4.  🎉 **Success!** A shortcut named **pvBG** will appear on your Desktop.

### 🐧 For Linux Users

1.  Open terminal in the project folder.
2.  Run the setup script: `bash SETUP_LINUX.sh`
3.  🎉 **Success!** A launcher named **pvBG** will appear on your Desktop.

---

## 📂 Project Structure

```text
pvBG/
│
├── assets/                                 # UI Icons and Graphics
├── models/
│   └── pvBG_UNet_224_DiceBCE_v1.2.onnx     # The AI Brain (U-Net Model)
├── src/
│   ├── app.py                              # Backend Logic & Image Processing
│   └── gui.py                              # Frontend UI (Tkinter/PyQt)
├── requirements.txt                        # Project Dependencies
├── LICENSE                                 # MIT License
└── README.md                               # Documentation

```

---

## ⚖️ License & Copyright

This project is protected by a **Dual License** structure:

### 1. Application Code (Source Code)

Licensed under the **MIT License**. Free to use and modify for personal or commercial use.

### 2. AI Model (`pvBG_UNet_224_DiceBCE_v1.2.onnx`)

Licensed under **CC BY-NC-SA 4.0** (Creative Commons).

* ✅ Free for research and personal projects.
* 🚫 **Commercial use is strictly prohibited.**
* 👤 Attribution to **SawitSeehad** team is required.

---

**Copyright © 2026 SawitSeehad. All Rights Reserved.**