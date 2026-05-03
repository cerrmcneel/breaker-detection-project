# PanelSafe: Electrical Breaker Detection Beta

PanelSafe is a computer vision data ingestion and analysis platform designed to automate the auditing of residential and industrial electrical panels. Developed during an intensive bootcamp, this project bridges the gap between traditional electrical expertise and modern AI-driven safety inspections.

## Project Evolution & Portfolio Milestones
This repository has evolved through distinct professional phases, reflecting my growth from Data Analysis to Machine Learning Engineering:

*   **[v1.0: Data Analysis Phase](https://github.com/cerrmcneel/breaker-detection-project/releases/tag/v1.0-data-analysis)**: Focus on EDA, data cleaning, and initial algorithmic logic. (Frozen for portfolio review).
*   **[Current: Machine Learning & Edge Phase]**: Implementation of YOLOv10 for real-time breaker detection and synthetic dataset generation.

---

## Project Vision
The primary goal of PanelSafe is to provide a seamless way for users to upload images of circuit breakers from the field. This data is currently being used to train a custom YOLO (You Only Look Once) model to identify:

- **Breaker Types:** (MCBs, RCCBs, RCBOs).
- **Safety Compliance:** Identification of missing labels or outdated components.
- **Regional Variations:** Specifically targeting the Spanish market (REBT regulations).

##  Features
- **Guided Viewfinder:** Real-time client-side analysis of image brightness, blurriness, and crop variance to guide users into taking the perfect dataset image.
- **Multi-Language Support:** Instant English/Spanish translation toggles for field workers across different regions.
- **Smart Deduplication:** Backend SHA-256 fingerprinting that silently ignores multiple uploads of the exact same image to save NAS storage.
- **Hidden Batch Upload:** Admin-only, password-protected batch upload utility to mass-ingest existing breaker datasets smoothly.
- **Live Progress Tracking:** Dynamic goal-oriented visual badges matching current database ingest sizes against target milestones in real time.

##  The Tech Stack
The infrastructure is built to be resilient, secure, and "home-lab" hosted:

- **Frontend:** HTML5/CSS3/JS utilizing internal Web APIs (`getUserMedia`) for native browser integration.
- **Backend:** FastAPI (Python) handling asynchronous image processing, file hashing, and metadata logging.
- **Virtualization:** Hosted on Proxmox VE (VM 101).
- **Containerization:** Fully Dockerized with Docker Compose.
- **Storage:** Direct mount to a TrueNAS core via NFS/SMB for long-term data persistence.
- **Networking:** Secured via Cloudflare Tunnels (`panelsafe.cv`), providing end-to-end encryption without opening home router ports.

##  Project Structure
```plaintext
.
├── app/
│   ├── main.py            # FastAPI Logic, NAS Ingestion & Batch Deduplication
│   └── frontend/
│       ├── index.html     # SPA containing Guided Camera UI, i18n logic, and CSS styles
│       └── assets/        # Stored media (banners, etc.)
├── data/                  # Symlinked to TrueNAS Volume
│   ├── images/            # Raw .jpg/.png uploads
│   └── upload_log.json    # JSON Metadata (Timestamp, Country, SHA-256 Hash)
├── .env                   # Local secrets file for Admin passcodes
├── requirements.txt       # Python dependencies
├── Dockerfile             # Multi-stage Python build
└── docker-compose.yml     # App + Cloudflare Tunnel orchestration
```

## Deployment & Development
To replicate this environment:

**Clone and Configure:**
```bash
git clone https://github.com/cerrmcneel/breaker-detection-project.git
cd breaker-detection-project
```

**Set Environment Variables:**
1. Ensure your `TUNNEL_TOKEN` and `UPLOAD_DIR` are configured in the `docker-compose.yml`.
2. Create a `.env` file in the root directory to set your admin passcode for batch tools:
```env
ADMIN_PASSWORD=your_secure_password
```

**Launch Stack:**
```bash
docker-compose up -d --build
```

## 📈 Current Milestone: Data Collection
We are currently in the active data collection phase.

- **Target:** 300+ unique electrical panel images.
- **Deadline:** Wednesday, March 4th.
- **Live Portal:** [https://panelsafe.cv](https://panelsafe.cv)

## 👨‍💻 About the Developer
With a professional background as an Electrician and ESL teacher, I am transitioning into Data Science to build tools that solve real-world problems in the electrical industry. This project combines my domain knowledge of circuit circuitry with full-stack engineering and machine learning.