"""
Gradio Web Interface for Local Multimodal AI
Beautiful, modern web UI for text, image, and audio generation
"""

import gradio as gr
from pathlib import Path
from local_multimodal_ai import LocalMultimodalAI
import tempfile
import os


class GradioMultimodalUI:
    """Gradio web interface for Local Multimodal AI"""
    
    def __init__(self):
        """Initialize the UI"""
        print("Initializing Local Multimodal AI...")
        try:
            self.ai = LocalMultimodalAI()
            print("✓ AI initialized successfully!")
        except Exception as e:
            print(f"✗ Error initializing AI: {e}")
            raise
    
    def generate_text(self, prompt, max_length, temperature):
        """Generate text from prompt"""
        if not prompt.strip():
            return "Please enter a text prompt."
        
        try:
            result = self.ai.generate_text(
                prompt, 
                max_length=int(max_length),
                temperature=float(temperature)
            )
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_image(self, prompt, steps, guidance):
        """Generate image from prompt"""
        if not prompt.strip():
            return None, "Please enter an image description."
        
        try:
            image = self.ai.generate_image(
                prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance)
            )
            if image:
                return image, "✓ Image generated successfully!"
            else:
                return None, "Failed to generate image."
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def generate_audio(self, text, play_audio):
        """Generate audio from text"""
        if not text.strip():
            return None, "Please enter text to convert to speech."
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                output_path = tmp_file.name
            
            result = self.ai.generate_audio(text, output_path, save_to_file=True)
            
            if result and not result.startswith("Error"):
                if play_audio:
                    # Play audio
                    self.ai.generate_audio(text, save_to_file=False)
                return result, "✓ Audio generated successfully!"
            else:
                return None, result if result else "Failed to generate audio."
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def understand_image(self, image, question):
        """Understand and describe image"""
        if image is None:
            return "Please upload an image."
        
        try:
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image.save(tmp_file.name)
                image_path = tmp_file.name
            
            result = self.ai.understand_image(
                image_path,
                question if question.strip() else None
            )
            
            # Clean up
            try:
                os.unlink(image_path)
            except:
                pass
            
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def combined_generation(self, text_prompt, image_prompt, generate_audio, audio_text):
        """Combined text, image, and audio generation"""
        results = []
        image = None
        audio_path = None
        
        # Generate text
        if text_prompt.strip():
            try:
                text_result = self.ai.generate_text(text_prompt, max_length=150)
                results.append(f"**Generated Text:**\n{text_result}\n")
            except Exception as e:
                results.append(f"Text generation error: {str(e)}\n")
        else:
            results.append("No text prompt provided.\n")
        
        # Generate image
        if image_prompt.strip():
            try:
                image = self.ai.generate_image(image_prompt, num_inference_steps=50)
                if image:
                    results.append("✓ Image generated successfully!\n")
                else:
                    results.append("Failed to generate image.\n")
            except Exception as e:
                results.append(f"Image generation error: {str(e)}\n")
        else:
            results.append("No image prompt provided.\n")
        
        # Generate audio
        if generate_audio and audio_text.strip():
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    audio_path = tmp_file.name
                result = self.ai.generate_audio(audio_text, audio_path, save_to_file=True)
                if result and not result.startswith("Error"):
                    results.append("✓ Audio generated successfully!\n")
                else:
                    results.append("Failed to generate audio.\n")
            except Exception as e:
                results.append(f"Audio generation error: {str(e)}\n")
        
        return "\n".join(results), image, audio_path
    
    def create_interface(self):
        """Create and launch the Gradio interface"""
        
        with gr.Blocks(title="Local Multimodal AI", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🤖 Local Multimodal AI
                Generate text, images, and audio completely offline with no APIs required!
                
                **Features:**
                - 📝 Text Generation (GPT-2)
                - 🎨 Image Generation (Stable Diffusion)
                - 🔊 Audio Generation (Text-to-Speech)
                - 👁️ Image Understanding (BLIP)
                """
            )
            
            with gr.Tabs():
                # Tab 1: Text Generation
                with gr.Tab("📝 Text Generation"):
                    gr.Markdown("### Generate text from a prompt")
                    with gr.Row():
                        with gr.Column():
                            text_prompt = gr.Textbox(
                                label="Text Prompt",
                                placeholder="Enter your text prompt here...",
                                lines=3
                            )
                            with gr.Row():
                                max_length = gr.Slider(
                                    minimum=20,
                                    maximum=500,
                                    value=100,
                                    step=10,
                                    label="Max Length"
                                )
                                temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=2.0,
                                    value=0.7,
                                    step=0.1,
                                    label="Temperature (creativity)"
                                )
                            text_btn = gr.Button("Generate Text", variant="primary")
                        with gr.Column():
                            text_output = gr.Textbox(
                                label="Generated Text",
                                lines=10,
                                interactive=False
                            )
                    text_btn.click(
                        self.generate_text,
                        inputs=[text_prompt, max_length, temperature],
                        outputs=text_output
                    )
                
                # Tab 2: Image Generation
                with gr.Tab("🎨 Image Generation"):
                    gr.Markdown("### Generate images from text descriptions")
                    with gr.Row():
                        with gr.Column():
                            image_prompt = gr.Textbox(
                                label="Image Description",
                                placeholder="Describe the image you want to generate...",
                                lines=3
                            )
                            with gr.Row():
                                steps = gr.Slider(
                                    minimum=20,
                                    maximum=100,
                                    value=50,
                                    step=5,
                                    label="Quality Steps"
                                )
                                guidance = gr.Slider(
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=7.5,
                                    step=0.5,
                                    label="Guidance Scale"
                                )
                            image_btn = gr.Button("Generate Image", variant="primary")
                            image_status = gr.Textbox(label="Status", interactive=False)
                        with gr.Column():
                            image_output = gr.Image(label="Generated Image", type="pil")
                    image_btn.click(
                        self.generate_image,
                        inputs=[image_prompt, steps, guidance],
                        outputs=[image_output, image_status]
                    )
                
                # Tab 3: Audio Generation
                with gr.Tab("🔊 Audio Generation"):
                    gr.Markdown("### Convert text to speech")
                    with gr.Row():
                        with gr.Column():
                            audio_text = gr.Textbox(
                                label="Text to Speak",
                                placeholder="Enter text to convert to speech...",
                                lines=5
                            )
                            play_audio = gr.Checkbox(
                                label="Play audio after generation",
                                value=True
                            )
                            audio_btn = gr.Button("Generate Audio", variant="primary")
                            audio_status = gr.Textbox(label="Status", interactive=False)
                        with gr.Column():
                            audio_output = gr.Audio(label="Generated Audio", type="filepath")
                    audio_btn.click(
                        self.generate_audio,
                        inputs=[audio_text, play_audio],
                        outputs=[audio_output, audio_status]
                    )
                
                # Tab 4: Image Understanding
                with gr.Tab("👁️ Image Understanding"):
                    gr.Markdown("### Analyze and describe images")
                    with gr.Row():
                        with gr.Column():
                            upload_image = gr.Image(
                                label="Upload Image",
                                type="pil"
                            )
                            image_question = gr.Textbox(
                                label="Question (optional)",
                                placeholder="Ask a question about the image, or leave blank for general description...",
                                lines=2
                            )
                            understand_btn = gr.Button("Analyze Image", variant="primary")
                        with gr.Column():
                            understanding_output = gr.Textbox(
                                label="Analysis Result",
                                lines=10,
                                interactive=False
                            )
                    understand_btn.click(
                        self.understand_image,
                        inputs=[upload_image, image_question],
                        outputs=understanding_output
                    )
                
                # Tab 5: Combined Generation
                with gr.Tab("🚀 Combined Generation"):
                    gr.Markdown("### Generate text, image, and audio all at once")
                    with gr.Row():
                        with gr.Column():
                            combined_text = gr.Textbox(
                                label="Text Generation Prompt",
                                placeholder="Prompt for text generation...",
                                lines=2
                            )
                            combined_image = gr.Textbox(
                                label="Image Generation Prompt",
                                placeholder="Description for image generation...",
                                lines=2
                            )
                            generate_audio_check = gr.Checkbox(
                                label="Generate Audio",
                                value=False
                            )
                            combined_audio_text = gr.Textbox(
                                label="Text for Audio",
                                placeholder="Text to convert to speech...",
                                lines=2,
                                visible=False
                            )
                            combined_btn = gr.Button("Generate All", variant="primary", size="lg")
                            
                            generate_audio_check.change(
                                lambda x: gr.update(visible=x),
                                inputs=generate_audio_check,
                                outputs=combined_audio_text
                            )
                        with gr.Column():
                            combined_output = gr.Markdown(label="Results")
                            combined_image_output = gr.Image(label="Generated Image", type="pil")
                            combined_audio_output = gr.Audio(label="Generated Audio", type="filepath")
                    
                    combined_btn.click(
                        self.combined_generation,
                        inputs=[combined_text, combined_image, generate_audio_check, combined_audio_text],
                        outputs=[combined_output, combined_image_output, combined_audio_output]
                    )
            
            gr.Markdown(
                """
                ---
                **Note:** First generation may take longer as models load into memory.
                Image generation typically takes 30-60 seconds on GPU, 2-5 minutes on CPU.
                """
            )
        
        return demo
    
    def launch(self, share=False, server_name="127.0.0.1", server_port=7860):
        """Launch the Gradio interface"""
        demo = self.create_interface()
        demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True
        )


if __name__ == "__main__":
    print("Starting Gradio Web Interface...")
    print("This will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        ui = GradioMultimodalUI()
        ui.launch(share=False)  # Set share=True to create a public link
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("pip install -r requirements_local.txt")
        print("pip install gradio")

