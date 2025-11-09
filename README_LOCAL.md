# Local Multimodal AI System

A completely local multimodal AI system that can **generate text, audio, and images** without using any external APIs. All processing happens on your local machine.

## Features

- ✅ **Text Generation**: Generate text using local language models (GPT-2, etc.)
- ✅ **Image Generation**: Generate images from text prompts using Stable Diffusion
- ✅ **Audio Generation**: Convert text to speech using offline TTS (pyttsx3)
- ✅ **Image Understanding**: Analyze and describe images using local vision models
- ✅ **No API Keys Required**: Everything runs locally
- ✅ **Privacy**: All data stays on your machine

## Requirements

- Python 3.8 or higher
- **Disk Space**: ~10-15GB for models (downloaded automatically on first run)
- **RAM**: 8GB+ recommended (16GB+ for image generation)
- **GPU** (Optional): NVIDIA GPU with CUDA for faster processing
- Microphone (optional, for voice input)

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Activate it:**
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### 2. Install Dependencies

```bash
pip install -r requirements_local.txt
```

### 3. (Optional) Install PyTorch with CUDA for GPU Acceleration

If you have an NVIDIA GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only systems, the regular PyTorch installation will work (slower).

## Usage

### Running the CLI Interface

```bash
python local_main.py
```

The interactive CLI will present you with a menu:

```
==================================================
🤖 Local Multimodal AI Interface
==================================================
1. Generate Text
2. Generate Image
3. Generate Audio (Text-to-Speech)
4. Understand Image (Image Captioning)
5. Combined: Text + Image Generation
6. Combined: Text + Image + Audio
7. Exit
==================================================
```

### Option 1: Generate Text
Enter a text prompt and the AI will generate a continuation.

### Option 2: Generate Image
Enter a description and the AI will generate an image (takes 1-2 minutes).

### Option 3: Generate Audio
Convert any text to speech and save as an audio file.

### Option 4: Understand Image
Upload an image and get a description or ask questions about it.

### Option 5: Combined Text + Image
Generate both text and an image from prompts.

### Option 6: Combined All
Generate text, image, and audio all at once.

## Programmatic Usage

You can also use the `LocalMultimodalAI` class directly:

```python
from local_multimodal_ai import LocalMultimodalAI

# Initialize (downloads models on first run)
ai = LocalMultimodalAI()

# Generate text
text = ai.generate_text("The future of AI is", max_length=100)
print(text)

# Generate image
image = ai.generate_image("a beautiful sunset over mountains")
ai.save_image(image, "sunset.png")

# Generate audio
ai.generate_audio("Hello, this is a test.", "output.mp3")

# Understand image
caption = ai.understand_image("photo.jpg")
print(caption)

# Combined
results = ai.process_multimodal(
    text_prompt="Write a short story about space",
    generate_image=True,
    image_prompt="astronaut in space",
    generate_audio=True,
    audio_text="This is the generated story"
)
```

## Model Information

### Text Generation
- **Default Model**: GPT-2 (124M parameters)
- **Location**: Downloaded from Hugging Face (~500MB)
- **Speed**: Fast on CPU, very fast on GPU

### Image Generation
- **Default Model**: Stable Diffusion v1.5
- **Location**: Downloaded from Hugging Face (~4GB)
- **Speed**: 30-60 seconds on GPU, 2-5 minutes on CPU
- **Quality**: High-quality 512x512 images

### Image Understanding
- **Default Model**: BLIP (Base)
- **Location**: Downloaded from Hugging Face (~1GB)
- **Capabilities**: Image captioning and visual question answering

### Text-to-Speech
- **Engine**: pyttsx3 (offline, uses system TTS)
- **Requirements**: System TTS (Windows SAPI, macOS NSSpeechSynthesizer, Linux espeak)
- **Speed**: Real-time

## Performance Tips

### For Faster Image Generation:
1. Use GPU if available (CUDA)
2. Reduce `num_inference_steps` (default 50, try 20-30 for faster generation)
3. Use smaller models if available

### For Better Text Quality:
1. Use larger models (change `text_model` parameter)
2. Adjust `temperature` (lower = more focused, higher = more creative)
3. Increase `max_length` for longer outputs

### For Better Audio:
1. Configure system TTS voices
2. Adjust rate and volume in code
3. Use different TTS engines if needed

## Troubleshooting

### Models Not Downloading
- Check internet connection (needed only for first download)
- Verify disk space (need ~10-15GB)
- Check Hugging Face access

### Out of Memory Errors
- Reduce image generation steps
- Use CPU mode (slower but less memory)
- Close other applications
- Use smaller models

### Image Generation Too Slow
- Install CUDA-enabled PyTorch for GPU acceleration
- Reduce `num_inference_steps`
- Use CPU mode if GPU not available (will be slower)

### TTS Not Working
- Windows: Should work out of the box
- macOS: Requires system TTS
- Linux: Install `espeak`: `sudo apt-get install espeak`

### Audio File Not Playing
- Check file format (saves as .mp3 or .wav)
- Verify audio player supports the format
- Check file permissions

## Model Storage

Models are cached in:
- **Windows**: `C:\Users\<username>\.cache\huggingface\`
- **macOS/Linux**: `~/.cache/huggingface/`

You can delete this folder to free up space, but models will re-download on next run.

## Privacy & Security

- ✅ **100% Local**: No data sent to external servers
- ✅ **No API Keys**: No registration or API keys needed
- ✅ **Offline Capable**: Once models are downloaded, works offline
- ✅ **Private**: All your prompts and generations stay on your machine

## Limitations

- **First Run**: Models need to be downloaded (~10-15GB)
- **Speed**: Slower than cloud APIs (but private and free)
- **Quality**: GPT-2 is smaller than GPT-4, but still useful
- **Image Size**: Generated images are 512x512 by default
- **TTS Quality**: System TTS may not sound as natural as cloud TTS

## Advanced Usage

### Using Different Models

```python
# Use a larger text model
ai = LocalMultimodalAI(text_model="gpt2-medium")  # or gpt2-large, gpt2-xl

# Use a different image model
ai = LocalMultimodalAI(image_gen_model="runwayml/stable-diffusion-v1-5")

# Force CPU mode
ai = LocalMultimodalAI(device="cpu")

# Force GPU mode
ai = LocalMultimodalAI(device="cuda")
```

### Custom TTS Settings

```python
# After initialization, modify TTS engine
ai.tts_engine.setProperty('rate', 200)  # Faster speech
ai.tts_engine.setProperty('volume', 1.0)  # Maximum volume
```

## Comparison: API vs Local

| Feature | API Version | Local Version |
|---------|------------|---------------|
| **Setup** | Easy (just API key) | Moderate (download models) |
| **Speed** | Fast | Slower (but improving) |
| **Cost** | Pay per use | Free after setup |
| **Privacy** | Data sent to API | 100% local |
| **Offline** | No | Yes |
| **Quality** | Very high | Good to very good |
| **Internet** | Required | Only for initial download |

## License

This project is open source and available under the MIT License.

## Credits

- **Text Models**: Hugging Face Transformers
- **Image Models**: Stability AI (Stable Diffusion)
- **Vision Models**: Salesforce BLIP
- **TTS**: pyttsx3

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Check model downloads completed successfully
4. Review error messages for specific issues

---

**Enjoy your local, private, multimodal AI! 🚀**

