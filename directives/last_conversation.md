Ran command: `$env:PYTHONPATH="."; python src/model/train.py`

Ah! The classic PyTorch installation trap! It actually has nothing to do with WSL being down. 

If you look closely at the third line of your output, you'll see the culprit:
`Ultralytics 8.4.47  Python-3.13.2 torch-2.11.0+cpu`

### 🎓 Theory: The PyTorch CPU Default
When you ran `pip install ultralytics` earlier, pip looked for PyTorch. Because installing the massive NVIDIA CUDA drivers is complex, pip defaults to downloading the **CPU-only** version of PyTorch (`+cpu`) to save space and guarantee it works out of the box. 

Because `train.py` explicitly requested `device=0` (GPU #1), PyTorch crashed because it doesn't have the CUDA drivers to talk to your RTX 3060! 

*(Also, note that you are currently running this in native Windows PowerShell, not inside your WSL/K3s cluster! But we can absolutely train locally on Windows right now.)*

### 🛠️ The Fix: Swapping to CUDA
We need to rip out the CPU version of PyTorch and install the GPU version. Run these two commands in your terminal:

1. Uninstall the CPU version:
`pip uninstall torch torchvision torchaudio -y`

2. Install the CUDA (GPU) version:
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

*(Note: This is a massive download, usually around 2.5 GB, because it includes all the NVIDIA backend drivers).*

Once that finishes installing, run your training script again:
`$env:PYTHONPATH="."; python src/model/train.py`

Your RTX 3060 should finally roar to life!