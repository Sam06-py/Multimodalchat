"""
Local Multimodal AI System
Generates text, audio, and images without using external APIs
Uses local models: transformers, pyttsx3, Stable Diffusion
"""

import os
import torch
from typing import Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np
import io

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BlipProcessor, 
        BlipForConditionalGeneration,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Install with: pip install pyttsx3")

try:
    from diffusers import StableDiffusionPipeline
    import torch
    IMAGE_GEN_AVAILABLE = True
except ImportError:
    IMAGE_GEN_AVAILABLE = False
    print("Warning: diffusers not available. Install with: pip install diffusers accelerate")


class LocalMultimodalAI:
    """Local Multimodal AI that generates text, audio, and images without APIs"""
    
    def __init__(self, 
                 text_model: str = "gpt2",
                 image_gen_model: str = "runwayml/stable-diffusion-v1-5",
                 device: Optional[str] = None):
        """
        Initialize the Local Multimodal AI
        
        Args:
            text_model: Hugging Face model name for text generation (default: gpt2)
            image_gen_model: Hugging Face model name for image generation
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize text generation
        self.text_tokenizer = None
        self.text_model = None
        self._init_text_generation(text_model)
        
        # Initialize image generation
        self.image_pipeline = None
        self._init_image_generation(image_gen_model)
        
        # Initialize image understanding
        self.image_processor = None
        self.image_caption_model = None
        self._init_image_understanding()
        
        # Initialize TTS
        self.tts_engine = None
        self._init_tts()
        
        print("✓ Local Multimodal AI initialized successfully!")
    
    def _init_text_generation(self, model_name: str):
        """Initialize text generation model"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠ Text generation not available (transformers not installed)")
            return
        
        try:
            print(f"Loading text generation model: {model_name}...")
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Set padding token if not present
            if self.text_tokenizer.pad_token is None:
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            
            self.text_model.eval()
            print("✓ Text generation model loaded")
        except Exception as e:
            print(f"⚠ Error loading text generation model: {e}")
            print("  You can use a smaller model or install required dependencies")
            self.text_tokenizer = None
            self.text_model = None
    
    def _init_image_generation(self, model_name: str):
        """Initialize image generation model"""
        if not IMAGE_GEN_AVAILABLE:
            print("⚠ Image generation not available (diffusers not installed)")
            return
        
        try:
            print(f"Loading image generation model: {model_name}...")
            self.image_pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Optimize for memory if on CPU
            if self.device == "cpu":
                self.image_pipeline.enable_attention_slicing()
            
            print("✓ Image generation model loaded")
        except Exception as e:
            print(f"⚠ Error loading image generation model: {e}")
            print("  This model is large (~4GB). Make sure you have enough disk space and memory.")
            self.image_pipeline = None
    
    def _init_image_understanding(self):
        """Initialize image understanding/captioning model"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠ Image understanding not available (transformers not installed)")
            return
        
        try:
            print("Loading image understanding model...")
            model_name = "Salesforce/blip-image-captioning-base"
            self.image_processor = BlipProcessor.from_pretrained(model_name)
            self.image_caption_model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.image_caption_model.eval()
            print("✓ Image understanding model loaded")
        except Exception as e:
            print(f"⚠ Error loading image understanding model: {e}")
            self.image_processor = None
            self.image_caption_model = None
    
    def _init_tts(self):
        """Initialize text-to-speech engine"""
        if not TTS_AVAILABLE:
            print("⚠ Text-to-speech not available (pyttsx3 not installed)")
            return
        
        try:
            self.tts_engine = pyttsx3.init()
            # Configure TTS properties
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to use a more natural voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            print("✓ Text-to-speech engine initialized")
        except Exception as e:
            print(f"⚠ Error initializing TTS: {e}")
            self.tts_engine = None
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            
        Returns:
            Generated text
        """
        if self.text_model is None or self.text_tokenizer is None:
            return "Error: Text generation model not available. Please install transformers and torch."
        
        try:
            # Tokenize input
            inputs = self.text_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.text_model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.text_tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode output
            generated_text = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        except Exception as e:
            return f"Error generating text: {str(e)}"
    
    def generate_image(self, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5) -> Optional[Image.Image]:
        """
        Generate an image from a text prompt
        
        Args:
            prompt: Text description of the image
            num_inference_steps: Number of denoising steps (more = better quality, slower)
            guidance_scale: How closely to follow the prompt (higher = more adherence)
            
        Returns:
            Generated PIL Image or None if error
        """
        if self.image_pipeline is None:
            print("Error: Image generation model not available.")
            return None
        
        try:
            print(f"Generating image: '{prompt}'...")
            with torch.no_grad():
                image = self.image_pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            print("✓ Image generated successfully")
            return image
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None
    
    def save_image(self, image: Image.Image, filepath: Union[str, Path]) -> bool:
        """
        Save generated image to file
        
        Args:
            image: PIL Image to save
            filepath: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image.save(filepath)
            print(f"✓ Image saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False
    
    def understand_image(self, image_path: Union[str, Path], question: Optional[str] = None) -> str:
        """
        Understand and describe an image
        
        Args:
            image_path: Path to the image file
            question: Optional question about the image
            
        Returns:
            Description or answer about the image
        """
        if self.image_processor is None or self.image_caption_model is None:
            return "Error: Image understanding model not available."
        
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return f"Error: Image file not found at {image_path}"
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if question:
                # Answer question about image
                inputs = self.image_processor(image, question, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.image_caption_model.generate(**inputs, max_length=50)
                answer = self.image_processor.decode(out[0], skip_special_tokens=True)
                return answer
            else:
                # Generate caption
                inputs = self.image_processor(image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.image_caption_model.generate(**inputs, max_length=50)
                caption = self.image_processor.decode(out[0], skip_special_tokens=True)
                return caption
        except Exception as e:
            return f"Error understanding image: {str(e)}"
    
    def generate_audio(self, text: str, output_path: Optional[Union[str, Path]] = None, save_to_file: bool = True) -> Optional[str]:
        """
        Generate audio (speech) from text
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file (optional, will use .wav extension)
            save_to_file: Whether to save to file (if False, just plays audio)
            
        Returns:
            Path to saved audio file or None
        """
        if self.tts_engine is None:
            return "Error: Text-to-speech engine not available. Install pyttsx3."
        
        try:
            if output_path and save_to_file:
                output_path = Path(output_path)
                # pyttsx3 saves as WAV, so change extension if needed
                if output_path.suffix.lower() not in ['.wav', '.mp3']:
                    output_path = output_path.with_suffix('.wav')
                elif output_path.suffix.lower() == '.mp3':
                    # pyttsx3 doesn't support MP3 directly, use WAV instead
                    output_path = output_path.with_suffix('.wav')
                
                # Save to file
                self.tts_engine.save_to_file(text, str(output_path))
                self.tts_engine.runAndWait()
                print(f"✓ Audio saved to {output_path}")
                return str(output_path)
            else:
                # Just speak (don't save)
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return None
        except Exception as e:
            return f"Error generating audio: {str(e)}"
    
    def process_multimodal(self, 
                          text_prompt: Optional[str] = None,
                          generate_image: bool = False,
                          image_prompt: Optional[str] = None,
                          generate_audio: bool = False,
                          audio_text: Optional[str] = None,
                          image_path: Optional[Union[str, Path]] = None) -> dict:
        """
        Process multiple modalities at once
        
        Args:
            text_prompt: Text prompt for text generation
            generate_image: Whether to generate an image
            image_prompt: Prompt for image generation
            generate_audio: Whether to generate audio
            audio_text: Text to convert to audio
            image_path: Path to image for understanding
            
        Returns:
            Dictionary with results
        """
        results = {}
        
        # Generate text
        if text_prompt:
            results['text'] = self.generate_text(text_prompt)
        
        # Generate image
        if generate_image and image_prompt:
            image = self.generate_image(image_prompt)
            if image:
                results['image'] = image
        
        # Understand image
        if image_path:
            results['image_understanding'] = self.understand_image(image_path)
        
        # Generate audio
        if generate_audio and audio_text:
            audio_path = self.generate_audio(audio_text, f"output_{len(results)}.mp3")
            results['audio'] = audio_path
        
        return results


if __name__ == "__main__":
    # Example usage
    print("Initializing Local Multimodal AI...")
    ai = LocalMultimodalAI()
    
    # Test text generation
    print("\n=== Text Generation Test ===")
    response = ai.generate_text("The future of AI is", max_length=50)
    print(f"Generated: {response}")
    
    # Test image generation (commented out by default as it's slow)
    # print("\n=== Image Generation Test ===")
    # image = ai.generate_image("a beautiful sunset over mountains")
    # if image:
    #     ai.save_image(image, "generated_image.png")
    
    # Test TTS
    print("\n=== Audio Generation Test ===")
    ai.generate_audio("Hello, this is a test of text to speech generation.")
    print("Audio generated (should have played)")
    
    # Test image understanding (if you have an image)
    # print("\n=== Image Understanding Test ===")
    # if Path("test_image.jpg").exists():
    #     caption = ai.understand_image("test_image.jpg")
    #     print(f"Caption: {caption}")

