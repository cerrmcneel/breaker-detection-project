# PanelSafe: AI-Powered Electrical Safety Auditing
**Final Project Summary | May 16, 2026**

## 🎯 Project Vision
PanelSafe bridges the gap between traditional electrical expertise and modern AI-driven safety inspections. It transforms a simple photo of an electrical panel into a structured, validated data schema, reducing the auditing burden for professionals and providing clarity for homeowners.

---

## 🚀 Key Achievements

### 1. Hybrid MLOps Infrastructure
Built a "Home-Lab Hybrid" cluster to provide high-performance AI inference without cloud GPU costs.
- **Orchestration**: Kubernetes (K3s) on Proxmox.
- **Hardware**: NVIDIA RTX 3060 worker node via WSL2 GPU passthrough.
- **Security**: Cloudflare Tunnels providing secure, encrypted edge access.
- **Result**: Low-latency, edge-accessible YOLO inference.

### 2. Specialized Computer Vision Pipeline
Transitioned from generic object detection to a domain-specific vision engine.
- **Model**: **YOLO26-Nano (NMS-free)** architecture, optimized for dense object arrays.
- **Training**: Completed **100 epochs** at **1280px resolution** to capture tiny amperage text.
- **Performance**: Achieved a stellar **mAP50 of 0.974** across 6 electrical classes.
- **Inference Strategy**: Implemented **SAHI (Slicing Aided Hyper Inference)** to maintain detail on 4K high-res panel photos.

### 3. Human-in-the-Loop (HITL) Workspace
Acknowledging that AI is a tool, not a replacement, we built a professional validation environment.
- **Interactive Canvas**: High-performance pan/zoom interface for navigating dense electrical boards.
- **OCR Cleaning Engine**: Regex-based cleaning that scrubs visual noise to extract precise ratings (e.g., `C16`, `30mA`).
- **Spatial Heuristics**: A Python logic layer that "heals" detection errors using REBT (Spanish Electrical Code) standards, such as identifying the Main Breaker based on physical positioning.

### 4. Full-Stack Evolution
The project reflects a complete engineering journey:
- **Phase 1**: Data Analysis & EDA in Tableau/Jupyter.
- **Phase 2**: Synthetic Data Generation via "Electrical Grammar" logic.
- **Phase 3**: MLOps Deployment and HITL UI development.

---

## 📈 Technical Impact
| Feature | Implementation | Benefit |
| :--- | :--- | :--- |
| **Detection** | YOLO26-Nano + SAHI | Accurate detection in dense, "non-island" arrays. |
| **Validation** | HITL Canvas | 100% data integrity via human-verified corrections. |
| **Scale** | K3s Cluster | Resilient, hardware-accelerated serving. |
| **Compliance** | Spatial Heuristics | Automatic classification of RCD classes and MCB roles. |

---

## 🔮 Future Roadmap
- **Esquema Unifilar Generation**: Auto-generating professional single-line diagrams from HITL-verified JSON.
- **Mobile Edge Inference**: Optimized TFLite/ONNX exports for completely offline "basement-mode" inspections.
- **Active Learning**: Fully automating the "Data Flywheel" to retrain models on HITL correction exports.

---
**Developed by:** [Your Name/GitHub Handle]
*Ironhack Data Science & Machine Learning Bootcamp 2026*
