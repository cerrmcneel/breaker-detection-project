# How I Accidentally Built an MLOps Cluster (And Lost a Fight With a GPU)

> **TL;DR:** I connected a Proxmox Kubernetes server to my Windows gaming PC as a GPU worker node. The GPU part took 2 hours of brutal debugging. Here's every mistake I made so you don't have to.

---

## The Goal

I'm building **PanelSafe**, a YOLO26-Nano model for detecting electrical breaker panels. The plan was ambitious:

- A **Proxmox server** running K3s as the Kubernetes master
- My **Windows PC (RTX 3060)** as a GPU-accelerated worker node via WSL2
- A distributed, edge-accessible AI inference cluster — for a *portfolio project*

Yes, I know. Overkill. That's the point.

---

## Phase 1: Joining the Cluster (The Easy Part)

Joining my PC to the cluster was supposed to be one command:

```bash
curl -sfL https://get.k3s.io | K3S_URL=https://192.168.1.152:6443 K3S_TOKEN=<TOKEN> sh -
```

It wasn't one command. It was a lesson in **token formatting**.

### Mistake #1: The Token Had a Secret
The K3s node token looks like this:

```
K1092f97ea65...::server:a739832328cf54d11c7008e3688aa85
```

I copied everything up to `::server:` and thought the rest was a label, not part of the token. It is. **The entire string including `::server:...` at the end is the password.** 45 minutes lost.

### Mistake #2: The "Ghost Token" Cache
Even after fixing the token, the agent kept saying `not authorized`. The fix was a full uninstall and clean reinstall:

```bash
/usr/local/bin/k3s-agent-uninstall.sh
sudo rm -rf /etc/rancher/k3s
sudo rm -rf /var/lib/rancher/k3s
# Re-run with QUOTED token to prevent shell interpolation
curl -sfL https://get.k3s.io | K3S_URL=https://... K3S_TOKEN="<TOKEN>" sh -
```

**Lesson:** Always quote tokens in shell commands. Colons (`:`) have special meaning in bash.

---

## Phase 2: The GPU Battle (The Hard Part)

With both nodes showing `Ready` in `kubectl get nodes`, I installed the NVIDIA Device Plugin:

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
```

And then began the **two-hour war**.

### Error #1: `libnvidia-ml.so.1: cannot open shared object file`
The plugin pod couldn't find the NVIDIA libraries. In WSL2, these live in `/usr/lib/wsl/lib/`, not the standard Linux paths.

**Fix attempt:** Mount the WSL library folder into the pod.

```bash
kubectl patch ds nvidia-device-plugin-daemonset -n kube-system --type='json' -p='[
  {"op": "add", "path": "/spec/template/spec/containers/0/volumeMounts/-",
   "value": {"name": "wsl-lib", "mountPath": "/usr/lib/wsl/lib"}},
  ...
]'
```

**Result:** Progressed to the next error.

---

### Error #2: `ERROR_DRIVER_NOT_LOADED`
The plugin found the libraries but couldn't initialize the driver. This is the classic WSL2 "trap" — the libraries are present as stubs, but the actual driver communication goes through `/dev/dxg`, a special DirectX kernel device unique to WSL2.

**Fix attempt:** Mount `/dev/dxg` into the pod.

```bash
{"op": "add", "path": "/spec/template/spec/volumes/-",
 "value": {"name": "dxg", "hostPath": {"path": "/dev/dxg"}}}
```

**Result:** Still `ERROR_DRIVER_NOT_LOADED`. The hardware door was open, but the pod still couldn't walk through it.

---

### Error #3: The Missing Symlink
Running `ls /usr/lib/wsl/lib` inside the pod revealed the file list had `libnvidia-ml.so.1` but NOT `libnvidia-ml.so` (without the `.1`). Many plugin versions are hardcoded to look for the unversioned name.

**Fix:**
```bash
sudo ln -s /usr/lib/wsl/lib/libnvidia-ml.so.1 /usr/lib/wsl/lib/libnvidia-ml.so
```

**Result:** Still failing. The symlink helped but wasn't the root cause.

---

### The Root Cause: Wrong Container Runtime

After exhausting every patch, the real issue emerged: **the plugin pod was running under the default `runc` container runtime**, not the NVIDIA runtime. That meant no matter how many files we mounted, the pod was never given a true kernel-level connection to the GPU.

The fix required a **custom YAML** with one critical line:

```yaml
spec:
  runtimeClassName: nvidia   # ← This was the entire solution
```

This tells K3s: *"Run this pod under the NVIDIA container runtime from the very start."* Everything else — the mounted volumes, the symlinks, the environment variables — only works *after* this is set.

**Final working DaemonSet structure:**
```yaml
spec:
  runtimeClassName: nvidia
  containers:
  - name: nvidia-device-plugin-ctr
    securityContext:
      privileged: true
    env:
    - name: LD_LIBRARY_PATH
      value: /usr/lib/wsl/lib
    volumeMounts:
    - mountPath: /usr/lib/wsl/lib   # WSL driver stubs
    - mountPath: /dev/dxg            # DirectX GPU bridge
```

**Result:**
```
Starting GRPC server for 'nvidia.com/gpu'
Registered device plugin for 'nvidia.com/gpu' with Kubelet
```

```bash
$ kubectl describe node datainmind | grep nvidia
  nvidia.com/gpu:     1
```

**The RTX 3060 was online.**

---

## What I Learned

| Problem | Root Cause | Fix |
|---|---|---|
| `not authorized` | Incomplete/malformed token | Use full token with `::server:...`, quote it in shell |
| `libnvidia-ml not found` | WSL2 uses non-standard driver paths | Mount `/usr/lib/wsl/lib` |
| `ERROR_DRIVER_NOT_LOADED` | Missing `/dev/dxg` device | Mount `/dev/dxg` hostPath |
| Still failing after mounts | Wrong container runtime | `runtimeClassName: nvidia` in pod spec |

---

## Is This Blog-Worthy Despite Using AI?

**Yes. Absolutely.**

Here's the honest truth: I used an AI assistant to help debug every step of this. Does that make the achievement less real? I'd argue the opposite.

Every experienced engineer uses tools — Stack Overflow, documentation, colleagues, and yes, now AI. What matters is:

1. **You understood each step.** You weren't blindly pasting commands. You recognized when the token was wrong. You identified the `::server:` issue yourself.
2. **The infrastructure is real.** That cluster is running. That GPU is registered. No AI did that for you — you typed the commands, debugged the errors, and kept going when it failed 10 times in a row.
3. **The *judgment* was yours.** Deciding which approach to try next, recognizing patterns in error messages, deciding when to wipe and start clean — that's engineering judgment, not AI.

The most valuable thing a blog post like this does is show **the messy reality of building infrastructure** — not the polished "I ran one command and it worked" tutorial. The struggle *is* the portfolio.

---

---

## Phase 3: The Application Layer (Bridging AI and Reality)

Once the GPU was online, the real challenge began: making the AI actually *useful* for an electrician. A raw YOLO detection is just a bunch of boxes; an electrician needs a **diagram**.

### The "Human-in-the-Loop" (HITL) Solution
AI isn't perfect, especially in the messy, high-shadow environment of an industrial panel. Instead of fighting for 100% model accuracy (which is impossible), we built a **Human-in-the-Loop Workspace**:

- **Unified Pipeline:** A FastAPI backend that orchestrates YOLOv8 (detection), Tesseract (OCR), and a custom **Spatial Heuristic Engine** (electrical logic).
- **Interactive Canvas:** A custom-built web editor that allows the user to:
    - **Pan & Zoom:** Navigate 4000px+ high-res board photos smoothly.
    - **Edit Boxes:** Drag-and-drop handles to manually resize AI detections.
    - **Add Missed Breakers:** A "Manual Draw" tool to capture items the AI missed.
    - **RCD Specific Schema:** Logic that swaps inputs dynamically for RCD sensitivity (30mA/300mA) vs. MCB ratings.

### Solving the "Scaling Paradox"
We hit a major roadblock where high-resolution images caused the browser's CSS to collapse, making the UI microscopic. We solved this by abandoning standard Flexbox scaling for a hybrid **Absolute/Flex row** layout, ensuring the sidebar remains legible regardless of the image resolution.

### The Vision & Logic Pipeline (Fighting Noise)
Raw OCR on grainy plastic engravings is a nightmare. We built a multi-stage vision pipeline to extract data where standard tools failed:
- **Pre-processing:** Implemented **Lanczos Interpolation** (upscaling) and **Otsu’s Binarization** to isolate black text from grey plastic backgrounds.
- **Regex Cleaning:** Created a backend logic layer that scrubs OCR noise. It identifies specific electrical patterns like `C16`, `C32`, or the `SI` marker for RCDs, ignoring the surrounding visual artifacts.
- **Heuristic Engine:** Built a spatial logic engine that "heals" detection errors, such as merging fragmented bounding boxes for multi-module devices like Oversurge protectors.

### The Data Flywheel (Active Learning)
This isn't just a UI—it's a **labeling factory**. Every manual correction the user makes is saved to `localStorage` and can be exported as "Gold Standard" training data. This closes the MLOps loop, allowing us to retrain the model on the very errors it made yesterday.

---

## What's Next

- `[x]` Deploy YOLO26-Nano inference pipeline.
- `[x]` Build interactive HITL verification workspace.
- `[ ]` **Phase 4: Diagram Generation:** Implement the engine to transform corrected JSON schemas into a professional **Esquema Unifilar** (PDF/SVG).
- `[ ]` **MLOps Automation:** Build a one-click pipeline to move exported HITL data into the YOLO training folder.
- `[ ]` Expose the endpoint via Cloudflare Tunnel for edge-accessible AI.

The infrastructure is alive. The UI is functional. The diagram is the final boss.

---


## Technical Insights

### The "Island" Object Problem vs. Dense Arrays
Most foundation models like YOLO are trained on datasets (like COCO) where objects are isolated "islands" within a broader context (e.g., a person standing in a field, a dog on a couch). 
Breaker boards present a fundamentally different and harder domain: they are dense arrays of highly similar, tightly packed objects of varying classes (MCB next to RCD next to OVERSURGE). The objects don't exist as islands; their boundaries bleed into one another.
This makes synthetic data generation techniques like Poisson Image Editing (Seamless Cloning) significantly more difficult, as the overlapping gradients of tightly packed objects can cause severe texture distortion when blending into the background.
