# Installation Guide - Local Multimodal AI

## Step-by-Step Installation

### Step 1: Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- ~15GB free disk space
- 8GB+ RAM (16GB+ recommended)
- Internet connection (for initial model download)

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Basic Dependencies

```bash
pip install -r requirements_local.txt
```

This will install:
- PyTorch (CPU version)
- Transformers
- Diffusers
- Pyttsx3
- Pillow
- And other dependencies

### Step 4: (Optional) Install GPU Support

**For NVIDIA GPU users only:**

```bash
# Uninstall CPU version first
pip uninstall torch torchvision torchaudio

# Install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU installation:**
```python
import torch
print(torch.cuda.is_available())  # Should print True
```

### Step 5: Verify Installation

```bash
python -c "import torch; import transformers; import diffusers; import pyttsx3; print('All packages installed successfully!')"
```

### Step 6: Run the Application

```bash
python local_main.py
```

On first run, models will be downloaded automatically. This may take 10-30 minutes depending on your internet connection.

## Troubleshooting Installation

### Problem: pip install fails

**Solution:**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Try again
pip install -r requirements_local.txt
```

### Problem: PyTorch installation fails

**Solution:**
```bash
# Install PyTorch separately
pip install torch torchvision torchaudio
```

### Problem: Out of memory during installation

**Solution:**
- Close other applications
- Install packages one by one
- Use CPU version (smaller)

### Problem: Models won't download

**Solution:**
- Check internet connection
- Verify disk space (need ~15GB)
- Check Hugging Face is accessible
- Try downloading manually from Hugging Face

### Problem: CUDA not found (for GPU users)

**Solution:**
1. Verify NVIDIA GPU is installed
2. Install CUDA toolkit from NVIDIA
3. Install matching PyTorch CUDA version
4. Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

## System-Specific Notes

### Windows

- TTS works out of the box (uses SAPI)
- GPU support requires CUDA toolkit
- Use PowerShell or Command Prompt

### macOS

- TTS works out of the box (uses NSSpeechSynthesizer)
- GPU support limited (M1/M2 chips have different setup)
- Use Terminal

### Linux

- May need to install TTS: `sudo apt-get install espeak`
- GPU support requires CUDA toolkit
- Use terminal/bash

## Verification Checklist

After installation, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All packages installed
- [ ] Models can be downloaded
- [ ] Application runs without errors
- [ ] (Optional) GPU detected if available

## Next Steps

1. Read `QUICKSTART_LOCAL.md` for usage
2. Read `README_LOCAL.md` for detailed documentation
3. Run `python local_main.py` to start using the AI

## Uninstallation

To remove everything:

```bash
# Deactivate virtual environment
deactivate

# Delete virtual environment folder
rm -rf venv  # macOS/Linux
# or
rmdir /s venv  # Windows

# Delete models cache (optional, frees ~15GB)
rm -rf ~/.cache/huggingface  # macOS/Linux
# or
rmdir /s C:\Users\<username>\.cache\huggingface  # Windows
```

## Getting Help

If you encounter issues:

1. Check error messages carefully
2. Verify all prerequisites are met
3. Check disk space and memory
4. Try reinstalling packages
5. Check internet connection for model downloads

---

**Happy installing! 🚀**

