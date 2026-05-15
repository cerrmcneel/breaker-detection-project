# Storytelling Context: PanelSafe Project
**Target Audience:** Non-technical bootcamp judges / Product Pitch (7 Minutes)
**Goal:** Create a compelling narrative arc that bridges data science with real-world home safety.

---

## 1. The Origin Story (The "Why")
- **The Persona**: An Electrician and ESL Teacher who spent years looking at dangerous electrical panels in residential basements.
- **The "Inciting Incident"**: Realizing that while the world is obsessing over Generative AI for images and text, 41.9% of home fires in Spain are caused by the "silent heart" of the home—the electrical board—which most people haven't checked in a decade.
- **The Mission**: Demystify the "magic box" in the hallway. Put a professional inspector's brain into a smartphone.

## 2. The Problem (The Stakes)
- **The "Silent Killer"**: Old RCD (Residual Current Devices) are "deaf" to modern electronics (LEDs, Laptops, EVs). They are Type AC in a Type A world.
- **The Friction**: Getting an electrician to visit just for an audit is expensive and slow.
- **The Data Gap**: There is no public dataset for Spanish electrical panels. To build a solution, we had to build the data first.

## 3. The Road of Trials (The Technical Struggle)
- **The "Basement" Constraint**: The app must work where there is no signal. This forced a "Push Once Connected" asynchronous logic and a highly efficient model.
- **The GPU Battle**: Connecting a local RTX 3060 gaming PC to a Proxmox server via Cloudflare Tunnels to create a "Home-Lab Hybrid" that rivals cloud performance.
- **The AI Hurdles**: 
    - Breakers aren't "islands"; they are dense, overlapping arrays.
    - Standard YOLO failed at reading tiny amperage text (C16, C32).
    - **Therefore**, I implemented **SAHI** (Slicing Aided Hyper Inference) to "zoom in" on the details without losing context.

## 4. The Revelation (The Pivot)
- **The "Aha!" Moment**: Realizing AI doesn't need to be 100% perfect to be useful. It needs a **Human-in-the-Loop**.
- **The Solution**: The **PanelSafe HITL Workspace**. An interactive canvas where the AI "drafts" the audit, and the human "signs off".
- **The Spatial Heuristics**: Teaching the computer "Electrical Grammar"—knowing that the biggest breaker is usually the Main Breaker simply by where it sits.

## 5. The Results (The "Payout")
- **The Pipeline**: A production-ready Beta that handles 4K images, cleans up OCR noise with Regex, and identifies RCD classes (A vs AC).
- **The "Flywheel"**: Every correction made in the HITL dashboard becomes high-quality training data for the next version.
- **The Transformation**: A project that started as a "Data Analysis" exercise in Tableau evolved into a full-stack MLOps ecosystem.

## 6. Key Data Points for the Pitch
- **41.9%**: Spanish home fires caused by electrical faults.
- **50+ Real Field Photos**: Crowdsourced to validate the initial model.
- **1280px**: The resolution needed to see what the human eye often misses.
- **7 Minutes**: The time it takes to go from a photo to a validated electrical schema.

---

## 7. Tone & Style for the Script Doctor
- **Tone**: Professional yet urgent. Grounded in safety but optimistic about AI.
- **Focus**: The transition from "Maintenance" to "Intelligence".
- **The Ending**: PanelSafe isn't just an app; it's the new standard for the "Silent Heart" of your home.
