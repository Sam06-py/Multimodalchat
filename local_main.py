"""
Local Multimodal AI Interface
Interactive CLI for local text, audio, and image generation
"""

import os
import sys
from pathlib import Path
from local_multimodal_ai import LocalMultimodalAI


class LocalMultimodalAICLI:
    """Command-line interface for Local Multimodal AI"""
    
    def __init__(self):
        """Initialize the CLI"""
        try:
            print("Initializing Local Multimodal AI...")
            print("This may take a few minutes to download models on first run...")
            self.ai = LocalMultimodalAI()
            print("✓ Local Multimodal AI initialized successfully!")
        except Exception as e:
            print(f"✗ Error initializing AI: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you installed: pip install -r requirements_local.txt")
            print("2. For image generation, you need at least 4GB free disk space")
            print("3. For GPU acceleration, install CUDA-enabled PyTorch")
            sys.exit(1)
    
    def print_menu(self):
        """Print the main menu"""
        print("\n" + "="*50)
        print("🤖 Local Multimodal AI Interface")
        print("="*50)
        print("1. Generate Text")
        print("2. Generate Image")
        print("3. Generate Audio (Text-to-Speech)")
        print("4. Understand Image (Image Captioning)")
        print("5. Combined: Text + Image Generation")
        print("6. Combined: Text + Image + Audio")
        print("7. Exit")
        print("="*50)
    
    def handle_text_generation(self):
        """Handle text generation"""
        print("\n--- Text Generation Mode ---")
        prompt = input("Enter your text prompt (or 'back' to return): ").strip()
        if prompt.lower() == 'back':
            return
        
        if not prompt:
            print("No prompt provided.")
            return
        
        try:
            max_length = int(input("Max length (default 100): ").strip() or "100")
            temperature = float(input("Temperature 0.1-1.0 (default 0.7): ").strip() or "0.7")
        except ValueError:
            max_length = 100
            temperature = 0.7
        
        print("\n🤔 Generating text...")
        response = self.ai.generate_text(prompt, max_length=max_length, temperature=temperature)
        print(f"\n💬 Generated Text:\n{response}\n")
        
        # Option to generate audio from the text
        generate_audio = input("Generate audio from this text? (y/n): ").strip().lower()
        if generate_audio == 'y':
            output_path = input("Audio file path (default: output_audio.wav): ").strip() or "output_audio.wav"
            self.ai.generate_audio(response, output_path)
    
    def handle_image_generation(self):
        """Handle image generation"""
        print("\n--- Image Generation Mode ---")
        prompt = input("Enter image description (or 'back' to return): ").strip()
        if prompt.lower() == 'back':
            return
        
        if not prompt:
            print("No prompt provided.")
            return
        
        output_path = input("Output file path (default: generated_image.png): ").strip() or "generated_image.png"
        
        try:
            steps = int(input("Quality steps 20-100 (default 50, higher=slower): ").strip() or "50")
        except ValueError:
            steps = 50
        
        print("\n🎨 Generating image... (This may take a minute or two)")
        image = self.ai.generate_image(prompt, num_inference_steps=steps)
        
        if image:
            self.ai.save_image(image, output_path)
            print(f"\n✓ Image saved to {output_path}")
            
            # Option to open image
            try:
                image.show()
            except:
                pass
        else:
            print("\n✗ Failed to generate image")
    
    def handle_audio_generation(self):
        """Handle audio generation"""
        print("\n--- Audio Generation Mode ---")
        text = input("Enter text to convert to speech (or 'back' to return): ").strip()
        if text.lower() == 'back':
            return
        
        if not text:
            print("No text provided.")
            return
        
        output_path = input("Output file path (default: output_audio.wav): ").strip() or "output_audio.wav"
        
        print("\n🔊 Generating audio...")
        result = self.ai.generate_audio(text, output_path, save_to_file=True)
        
        if result and not result.startswith("Error"):
            print(f"\n✓ Audio saved to {result}")
            play = input("Play audio? (y/n): ").strip().lower()
            if play == 'y':
                self.ai.generate_audio(text, save_to_file=False)
        else:
            print(f"\n✗ {result}")
    
    def handle_image_understanding(self):
        """Handle image understanding"""
        print("\n--- Image Understanding Mode ---")
        image_path = input("Enter image file path (or 'back' to return): ").strip()
        if image_path.lower() == 'back':
            return
        
        if not Path(image_path).exists():
            print(f"Error: Image file not found at {image_path}")
            return
        
        question = input("Ask a question about the image (press Enter for general description): ").strip()
        if not question:
            question = None
        
        print("\n👁️ Analyzing image...")
        result = self.ai.understand_image(image_path, question)
        print(f"\n💬 Result:\n{result}\n")
    
    def handle_combined_text_image(self):
        """Handle combined text and image generation"""
        print("\n--- Combined: Text + Image Generation ---")
        text_prompt = input("Enter text generation prompt: ").strip()
        image_prompt = input("Enter image generation prompt: ").strip()
        
        if not text_prompt or not image_prompt:
            print("Both prompts are required.")
            return
        
        output_dir = input("Output directory (default: ./output): ").strip() or "./output"
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n🤔 Generating text...")
        text_result = self.ai.generate_text(text_prompt)
        print(f"\n💬 Generated Text:\n{text_result}\n")
        
        print("\n🎨 Generating image... (This may take a minute or two)")
        image = self.ai.generate_image(image_prompt)
        
        if image:
            image_path = Path(output_dir) / "generated_image.png"
            self.ai.save_image(image, image_path)
            print(f"\n✓ Image saved to {image_path}")
        
        # Save text to file
        text_path = Path(output_dir) / "generated_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_result)
        print(f"✓ Text saved to {text_path}")
    
    def handle_combined_all(self):
        """Handle combined text, image, and audio generation"""
        print("\n--- Combined: Text + Image + Audio Generation ---")
        text_prompt = input("Enter text generation prompt: ").strip()
        image_prompt = input("Enter image generation prompt: ").strip()
        
        if not text_prompt or not image_prompt:
            print("Both prompts are required.")
            return
        
        output_dir = input("Output directory (default: ./output): ").strip() or "./output"
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n🤔 Generating text...")
        text_result = self.ai.generate_text(text_prompt)
        print(f"\n💬 Generated Text:\n{text_result}\n")
        
        print("\n🎨 Generating image... (This may take a minute or two)")
        image = self.ai.generate_image(image_prompt)
        
        if image:
            image_path = Path(output_dir) / "generated_image.png"
            self.ai.save_image(image, image_path)
            print(f"✓ Image saved to {image_path}")
        
        print("\n🔊 Generating audio from text...")
        audio_path = Path(output_dir) / "generated_audio.wav"
        result = self.ai.generate_audio(text_result, audio_path, save_to_file=True)
        
        if result and not result.startswith("Error"):
            print(f"✓ Audio saved to {result}")
        
        # Save text to file
        text_path = Path(output_dir) / "generated_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_result)
        print(f"✓ Text saved to {text_path}")
        
        print(f"\n✓ All outputs saved to {output_dir}/")
    
    def run(self):
        """Run the main CLI loop"""
        print("\nWelcome to Local Multimodal AI!")
        print("This AI can generate text, images, and audio locally without any APIs.")
        print("\nNote: First run will download models (~5-10GB). Subsequent runs are faster.")
        
        while True:
            self.print_menu()
            choice = input("\nSelect an option (1-7): ").strip()
            
            if choice == "1":
                self.handle_text_generation()
            elif choice == "2":
                self.handle_image_generation()
            elif choice == "3":
                self.handle_audio_generation()
            elif choice == "4":
                self.handle_image_understanding()
            elif choice == "5":
                self.handle_combined_text_image()
            elif choice == "6":
                self.handle_combined_all()
            elif choice == "7":
                print("\n👋 Goodbye!")
                break
            else:
                print("Invalid option. Please try again.")
            
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    cli = LocalMultimodalAICLI()
    cli.run()

