# pvBG

![pvBG Logo](assets/icon.png)

![Version](https://img.shields.io/badge/version-v1.2.0-blue.svg)
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


## 🆕 What's New in v1.2.0
- **Enhanced Accuracy:** Improved human segmentation for distant subjects.
- **Manual Editing:** Added **Restore** and **Eraser** tools for fine-tuning results.
- **Improved UX:** Fixed brush delays and added a **Clear** button for faster workflow.

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
4.  🎉 **Success!** A shortcut named **BG** will appear on your Desktop.
5.  Click the Desktop icon to start the app.

### 🐧 For Linux Users

1.  Open terminal in the project folder.
2.  Run the setup script:
    ```bash
    bash SETUP_LINUX.sh
    ```
3.  🎉 **Success!** A launcher named **BG** will appear on your Desktop.
4.  *Note:* You might need to right-click the icon and select **"Allow Launching"**.

---

## 📂 Project Structure

```text
pvBG/
│
├── assets/
│   ├── icon.ico          # Windows Icon
│   └── icon.png          # App Icon
│
├── models/
│   └── pvBG.onnx   # The AI Brain
│
├── src/
│   ├── app.py            # Backend Logic
│   └── gui.py            # Frontend UI
│
├── requirements.txt      # Dependencies
├── SETUP_WINDOWS.bat     # Windows Installer
├── SETUP_LINUX.sh        # Linux Installer
├── LICENSE               # MIT License
└── README.md             # Documentation

```

---

## ⚖️ License & Copyright

This project is protected by a **Dual License** structure:

### 1. Application Code (Source Code)

The source code (Python scripts, installers, GUI) is licensed under the **MIT License**.
You are free to use, modify, and distribute the code, provided you include the original copyright notice.

### 2. AI Model (`pvBG.onnx`)

The trained AI model provided in this repository is licensed under **CC BY-NC-SA 4.0** (Creative Commons).

* ✅ You are free to use it for research and personal projects.
* 🚫 **Commercial use of the model file is strictly prohibited.**
* 👤 Attribution to **Saw it See had** team is required.

---

**Copyright © 2026 Saw it See had. All Rights Reserved.**
