# 📈 Stock Market Prediction App

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**A full-stack deep learning web app** that predicts future stock prices using a trained neural network model.  
Built with **FastAPI** (backend), **Streamlit** (frontend), and **Docker Compose** for seamless deployment.

---

## 🚀 Overview

This project demonstrates how to integrate a **machine learning model** into a **modern, scalable web application** using a microservices architecture.

- 🧠 **FastAPI** serves as the backend REST API for ML inference.  
- 🎨 **Streamlit** provides an interactive frontend for visualization and exploration.  
- 🐳 **Docker Compose** ties both together for easy setup, scaling, and deployment.  

You can fetch real-time market data, make predictions for multiple future days, and visualize results with interactive Plotly charts.

---

## 🧩 Architecture

```
                   ┌────────────────────┐
                   │     User Browser   │
                   └──────────┬─────────┘
                              │
                              ▼
                   ┌────────────────────┐
                   │   Streamlit UI     │  ← (Frontend)
                   │  Port: 8501        │
                   └──────────┬─────────┘
                              │ REST API Calls (HTTP)
                              ▼
                   ┌────────────────────┐
                   │     FastAPI API    │  ← (Backend)
                   │  Port: 8000        │
                   └──────────┬─────────┘
                              │
                              ▼
                   ┌────────────────────┐
                   │  ML Model (.h5)    │
                   │  + Scalers (.pck)  │
                   └────────────────────┘
```

---

## ✨ Features

✅ **Deep Learning Predictions** — Uses your trained `.h5` model for stock forecasting  
✅ **7-Day (or more) Forecast** — Iterative multi-day predictions with LSTM or similar models  
✅ **Interactive UI** — Explore predictions and visualize with Plotly charts  
✅ **FastAPI Endpoints** — JSON-based ML inference API  
✅ **Dockerized** — Single-command setup using Docker Compose  
✅ **Scalable** — Clean separation between frontend and backend  
✅ **Realtime Data Fetching** — Uses `yfinance` to load historical data  

---

## 🏗️ Project Structure

```
stock-prediction/
│
├── backend/                     # FastAPI backend service
│   ├── api.py                   # API logic for predictions
│   ├── requirements.txt         # Backend dependencies
│   ├── Dockerfile               # Backend Dockerfile
│
├── frontend/                    # Streamlit frontend service
│   ├── app.py                   # Streamlit UI app
│   ├── requirements.txt         # Frontend dependencies
│   ├── Dockerfile               # Frontend Dockerfile
│
├── docker-compose.yml           # Compose file to run both services
├── Model.h5                     # Trained Keras model (not included)
├── Feature_Scaler.pck           # Feature scaler (not included)
└── Target_Scaler.pck            # Target scaler (not included)
```

---

## 🧰 Tech Stack

| Layer | Technology | Description |
|-------|-------------|-------------|
| 🧠 Machine Learning | TensorFlow / Keras | Deep learning model for stock prediction |
| ⚙️ Backend | FastAPI | High-performance Python API framework |
| 💡 Frontend | Streamlit + Plotly | Data visualization and interactive dashboard |
| 🐳 Deployment | Docker Compose | Containerized architecture |
| 💾 Data | yfinance | Fetches live stock data |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/stock-prediction.git
cd stock-prediction
```

### 2️⃣ Add Your Model Files
Place the following files in the **backend** folder:
```
Model.h5
Feature_Scaler.pck
Target_Scaler.pck
```

These are your trained model and scaler files exported from your Jupyter notebook.

### 3️⃣ Build and Run with Docker Compose
```bash
docker compose up --build
```

### 4️⃣ Access the Apps
| Service | URL | Description |
|----------|-----|-------------|
| **Frontend (Streamlit)** | [http://localhost:8501](http://localhost:8501) | Interactive UI |
| **Backend (FastAPI)** | [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger API documentation |

---

## 🧠 API Reference

### `POST /predict/multi-day`

Predict multiple future stock prices.

**Request Body:**
```json
{
  "ticker": "AAPL",
  "start_date": "2020-01-01",
  "end_date": "2024-01-01",
  "previous_date": "2023-12-30",
  "days": 7,
  "feature_length": 32
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "start_date": "2020-01-01T00:00:00",
  "previous_date": "2023-12-30T00:00:00",
  "days": 7,
  "forecast": [
    {"date": "2023-12-31T00:00:00", "predicted_price": 182.34},
    {"date": "2024-01-01T00:00:00", "predicted_price": 183.21}
  ]
}
```

**Try it out:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🎨 Streamlit Frontend

The Streamlit dashboard allows you to:

- Enter stock ticker, date range, and prediction window  
- Fetch historical prices via `yfinance`  
- Visualize predicted vs. actual prices  
- Download forecast data as CSV  

---

## 🐳 Docker Cheat Sheet

| Command | Description |
|----------|-------------|
| `docker compose up` | Run both services |
| `docker compose up --build` | Rebuild images |
| `docker compose down` | Stop and remove containers |
| `docker logs stock-frontend` | View Streamlit logs |
| `docker logs stock-backend` | View FastAPI logs |

---

## 🔐 Environment Variables

You can configure frontend ↔ backend communication using:
```
API_URL=http://backend:8000
```
Defined in `docker-compose.yml` so Streamlit can call the backend service internally.

---

## 🧱 Future Improvements

- Add authentication (JWT or OAuth2)
- Integrate a database for historical predictions
- Deploy to AWS / Azure with HTTPS (Nginx + SSL)
- Add model retraining pipeline
- Include uncertainty/confidence bands on predictions

---

## 💡 Key Learnings

- How to serve ML models via FastAPI  
- How to build a data visualization dashboard with Streamlit  
- How to containerize multi-service apps with Docker Compose  
- How to architect clean frontend-backend ML systems  

---

## 🧑‍💻 Author

**[Abiola Olatunbosun]**  
AI Engineer • Full Stack Developer  
🌐 [LinkedIn](https://linkedin.com/in/abiola-olatunbosun/) | [GitHub](https://github.com/bosunolaide)

---

## 🪪 License

This project is licensed under the **MIT License** — you’re free to use, modify, and distribute it.

---

## ⭐ Acknowledgments

- [FastAPI Documentation](https://fastapi.tiangolo.com/)  
- [Streamlit](https://streamlit.io/)  
- [TensorFlow / Keras](https://www.tensorflow.org/)  
- [yfinance](https://pypi.org/project/yfinance/)  

---

### 🧭 TL;DR

```bash
# Quick Start
docker compose up --build
# Then open
# Streamlit → http://localhost:8501
# FastAPI → http://localhost:8000/docs
```

Enjoy predicting the market 📊🚀
