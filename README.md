<div align="center">

# Manobal-AI — Welfare Signals, Human-Led Support

---

### An AI-assisted welfare platform that helps authorized officers review personnel welfare signals and provide human-led support.

*AI should assist. Humans should decide.*

![React](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/Language-TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Build-Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![LightGBM](https://img.shields.io/badge/ML-LightGBM-2E8B57?style=for-the-badge)

• [GitHub Repository](https://github.com/imkrishna07/ManobalAi)

---

</div>

## About

Manobal-AI is a prototype welfare decision-support platform for uniformed personnel. It turns voluntary check-ins and related records into contextual welfare signals for authorized human review.

The platform includes personnel, welfare officer, commander, and administrator workspaces. Its model provides a Low, Moderate, or High signal with supporting context; it is not a medical diagnosis or a substitute for human judgment.

> **Prototype note:** The included demonstration data is synthetic. The model and workflow have not been validated for operational use.

## Features

- Role-based workspaces for personnel, welfare officers, commanders, and administrators
- Welfare assessments using a 44-feature machine-learning pipeline
- Historical context, data-quality indicators, explanations, and recommendations
- Human-reviewed intervention records
- Synthetic demo data for local presentation

## Tech Stack

- **Frontend:** React, TypeScript, Vite
- **Backend:** Python, FastAPI
- **Database:** MongoDB
- **Machine learning:** LightGBM, scikit-learn

## Run Locally

You’ll need Docker Desktop and Node.js with Yarn installed.

1. Start Docker Desktop.
2. From the project root, start MongoDB and the backend:

   ```bash
   docker compose up --build backend
