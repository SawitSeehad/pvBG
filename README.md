
# reMBG (Offline Edition)

![reMBG Logo](assets/icon.png)

**reMBG** is a professional, privacy-focused desktop application to remove person image backgrounds automatically using AI. It runs 100% offline on your machine.

**Key Features:**
- 🔒 **100% Offline & Private:** Your images never leave your computer. No API keys, no cloud uploads.
- ⚡ **Lightweight:** Optimized for CPU inference (no expensive GPU required).
- 🚀 **Native Experience:** Installs directly as a Desktop Application with a custom icon.
- 🖥️ **Cross-Platform:** Works seamlessly on Windows and Linux.

---

## 📋 Prerequisites

Before running this application, please ensure you have **Python** installed on your system.
- **Windows:** Download from Microsoft Store or python.org.
- **Linux:** `sudo apt install python3-full` (Ubuntu/Debian) or equivalent.

---

## 🚀 Installation & Setup

You don't need to manually install libraries. We provide a **One-Click Setup** script that handles everything and creates a Desktop Shortcut for you.

### 🪟 For Windows Users

1.  Open the folder.
2.  Double-click **`SETUP_WINDOWS.bat`**.
3.  Wait for the installation to finish.
4.  🎉 **Success!** A shortcut named **reMBG** will appear on your Desktop.
5.  Click the Desktop icon to start the app.

### 🐧 For Linux Users

1.  Open terminal in the project folder.
2.  Run the setup script:
    ```bash
    bash SETUP_LINUX.sh
    ```
3.  🎉 **Success!** A launcher named **reMBG** will appear on your Desktop.
4.  *Note:* You might need to right-click the icon and select **"Allow Launching"**.

---

## 📂 Project Structure

```text
reMBG/
│
├── assets/
│   ├── icon.ico          # Windows Icon
│   └── icon.png          # Linux/App Icon
│
├── models/
│   └── segmentasi_manusia.onnx   # The AI Brain (Protected Model)
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

The source code (Python scripts, installers) is licensed under the **MIT License**.
You are free to use, modify, and distribute the code, provided you include the original copyright notice.

### 2. AI Model (`segmentasi_manusia.onnx`)

The trained AI model provided in this repository is licensed under **CC BY-NC-SA 4.0** (Creative Commons).

* ✅ You are free to use it for research and personal projects.
* 🚫 **Commercial use of the model file is strictly prohibited.**
* 👤 Attribution to **Saw it See had** team is required.

---

**Copyright © 2026 Saw it See had. All Rights Reserved.**

