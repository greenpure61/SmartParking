# 🚗 AI-Powered Parking Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-green.svg)](https://docs.ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Proprietary-orange.svg)](#-license)

---

## 🧠 Overview

**AI-Powered Parking Monitoring System** is an intelligent computer-vision application designed to automatically detect and count occupied and free parking spaces in real time using live video streams.

It uses **Ultralytics YOLO models** for object detection and an **IoU (Intersection-over-Union)** based rule to determine whether each parking spot is occupied.  
This project aims to support smart parking systems for private companies, municipalities, and public parking operators.

---

## ✨ Key Features

- **Real-Time Detection:** Continuously monitors parking lots through live video streams.  
- **IoU-Based Occupancy:** Determines parking spot usage based on overlap between vehicle detections and predefined parking zones.  
- **Color-Coded Visualization:** Green boxes = free spots, Red boxes = occupied spots.  
- **Privacy Mode:** Blurs faces and license plates automatically (GDPR compliant).  
- **Dashboard Statistics:** Displays total, occupied, and available spots in real time.  
- **Multi-Camera Support:** Can handle multiple parking zones simultaneously.

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Programming Language** | Python |
| **AI Model** | YOLOv8 / YOLOv11 (Ultralytics) |
| **Computer Vision** | OpenCV |
| **Data Handling** | NumPy |
| **Web Dashboard (optional)** | Flask |
| **Database (optional)** | SQLite / MS SQL / PostgreSQL |
| **Privacy Layer** | OpenCV face and plate blurring |

---

## ⚙️ How It Works

1. Define parking spot coordinates in a configuration file.  
2. Load your trained YOLO model.  
3. The system captures frames from a live camera feed or video file.  
4. Vehicles are detected using YOLO.  
5. Each detection is compared with predefined parking regions using IoU logic.  
6. The occupancy status is updated and displayed in real time.

---

## 🧰 Installation

### Prerequisites
- Python 3.10 or higher  
- `pip` package manager  
- A webcam or RTSP-compatible camera (for live mode)

### Install Dependencies
```bash
pip install ultralytics opencv-python numpy
```

### Clone the Repository
```bash
git clone https://github.com/greenpure61/SmartParking.git
cd SmartParking
```

### Run the App
```bash
python main.py
```

---

## 🎯 Example Output

When running, you’ll see:
- Real-time video with colored boxes per parking spot  
- Console logs like:
  ```
  Total spots: 50 | Occupied: 37 | Free: 13
  ```

---

## 🚀 Future Enhancements

- Cloud-based dashboard with statistics & analytics  
- AI-driven parking prediction using time-series data  
- Mobile companion app for real-time status  
- Integration with ANPR (Automatic Number Plate Recognition)

---

## 💡 Use Cases

- Smart city parking solutions  
- Corporate and university parking management  
- Shopping malls and private parking facilities  

---

## 🛡️ License

**Proprietary — All Rights Reserved © 2025 Kasper Mikkelsen**

This software and its source code are the intellectual property of Kasper Mikkelsen.  
Unauthorized copying, modification, or commercial use is prohibited without written permission.  

For licensing or business inquiries, please contact: **Kaspermikkelsen21@gmail.com**

---

## 🌐 References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Parking Management Guide (Ultralytics)](https://docs.ultralytics.com/guides/parking-management/)
