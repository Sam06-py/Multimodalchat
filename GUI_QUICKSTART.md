# GUI Quick Start Guide

## 🚀 Fastest Way to Get Started

### Option 1: Gradio Web Interface (Recommended - 2 minutes)

```bash
# 1. Install Gradio
pip install gradio

# 2. Run the interface
python gui_gradio.py

# 3. Interface opens automatically in your browser!
```

**That's it!** The interface will be at `http://127.0.0.1:7860`

### Option 2: Tkinter Desktop GUI (1 minute)

```bash
# 1. Run (tkinter usually pre-installed)
python gui_tkinter.py

# 2. Desktop window opens!
```

**Note:** If you get "tkinter not found" on Linux:
```bash
sudo apt-get install python3-tk
```

## 📋 What You Can Do

### Text Generation
1. Go to "Text Generation" tab
2. Type a prompt (e.g., "The future of AI is")
3. Click "Generate Text"
4. Read the result!

### Image Generation
1. Go to "Image Generation" tab
2. Describe an image (e.g., "a beautiful sunset over mountains")
3. Click "Generate Image"
4. Wait 30-60 seconds (GPU) or 2-5 minutes (CPU)
5. See your generated image!

### Audio Generation
1. Go to "Audio Generation" tab
2. Type text (e.g., "Hello, this is a test")
3. Click "Generate Audio"
4. Audio file is created and played!

### Image Understanding
1. Go to "Image Understanding" tab
2. Upload an image
3. Optionally ask a question
4. Click "Analyze Image"
5. Get description/answer!

### Combined Generation
1. Go to "Combined" tab
2. Enter text and/or image prompts
3. Optionally enable audio
4. Click "Generate All"
5. Get everything at once!

## 🎯 Which Interface Should I Use?

**Use Gradio if:**
- ✅ You want a modern, beautiful interface
- ✅ You want to share with others
- ✅ You prefer web-based apps

**Use Tkinter if:**
- ✅ You want a desktop app
- ✅ You need completely offline
- ✅ You prefer traditional GUIs

## ⚡ Tips

1. **First Run:** Models load on first use (takes ~1 minute)
2. **Image Generation:** Be patient - it takes time but worth it!
3. **Parameters:** Adjust sliders for different results
4. **Save Files:** Generated files can be saved to your computer

## 🐛 Troubleshooting

**Gradio won't start?**
```bash
pip install --upgrade gradio
```

**Tkinter not found?**
```bash
# Linux only
sudo apt-get install python3-tk
```

**Interface is slow?**
- Normal for image generation
- Use GPU for faster generation
- Reduce image generation steps

## 📚 More Info

- Full GUI guide: `GUI_README.md`
- Full documentation: `README_LOCAL.md`
- Installation: `INSTALL_LOCAL.md`

---

**Enjoy your GUI! 🎨**

