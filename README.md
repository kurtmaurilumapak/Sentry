<p align="center">
  <img src="src/assets/logo.png" alt="Sentry Logo" width="120" height="120">
</p>

<h1 align="center"><strong>Sentry</strong></h1>

<p align="center">
  <strong>Real-time AI Object Detection & Surveillance System</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Vue.js-3.5-4FC08D?style=for-the-badge&logo=vue.js">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi">
  <img src="https://img.shields.io/badge/YOLO-v8%20%7C%20v11-00FFFF?style=for-the-badge">
  <img src="https://img.shields.io/badge/Electron-39.x-47848F?style=for-the-badge&logo=electron">
</p>

---

# 📚 **Table of Contents**
- [📖 Overview](#-overview)
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Run as Desktop App](#run-as-desktop-app)
- [🎮 Usage](#-usage)
- [📦 Tech Stack](#-tech-stack)

---

## 📖 **Overview**
Sentry is a real-time object detection and tracking system powered by YOLO models. It supports live video streaming, GPU acceleration, YouTube analysis, and cross-platform deployment through Electron.

---

## ✨ **Features**
- 🎯 **Real-time Object Detection** — YOLO11s & YOLOv8s  
- 📹 **Multiple Input Sources** — Files, images, YouTube URLs  
- 🔴 **WebSocket Streaming** — 60 FPS annotated frames  
- 🖥️ **Cross-Platform** — Browser + Electron desktop app  
- 🚀 **GPU Acceleration** — CUDA, MPS, DirectML  
- 🔄 **Hot Model Switching** — Swap models instantly  
- 🎨 **Modern UI** — Vuetify 3, dark theme  

---

## 🚀 **Quick Start**

### **Prerequisites**
- Node.js **20.19+** or **22.12+**  
- Python **3.10+**  
- YOLO model file placed in:


### **Installation**
```bash
# Clone repository
git clone https://github.com/kurtmaurilumapak/sentry.git
cd sentry

# Install frontend dependencies
npm install

# Run dev mode (backend + frontend)
npm run dev
```
### **Run as Desktop App**
```bash
# Development (Electron + Vite)
npm run electron-dev

# Build production desktop app
npm run electron-build
```
##
### 🎮 **Usage**
- **Drag & drop video/image files**
- **Paste YouTube URLs for online analysis**
- **WebSocket Streaming**
- **Start real-time tracking with one click**
- **GPU Acceleration**
- **Switch between YOLO11s and YOLOv8s**

##
### 📦 **Tech Stack**
- Frontend
- Vue.js 3.5 (Composition API)
- Vuetify 3
- Vite 7
- Backend
- FastAPI + Uvicorn
- Ultralytics YOLO
- OpenCV + Pillow
- yt-dlp
- Desktop
- Electron 39

<p align="center"> Made with ❤️ using YOLO + Vue.js + FastAPI </p>

