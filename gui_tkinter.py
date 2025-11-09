"""
Tkinter Desktop GUI for Local Multimodal AI
Offline desktop interface for text, image, and audio generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from local_multimodal_ai import LocalMultimodalAI
from PIL import Image, ImageTk
import threading
import os


class TkinterMultimodalUI:
    """Tkinter desktop GUI for Local Multimodal AI"""
    
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Local Multimodal AI")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize AI (in a thread to avoid blocking)
        self.ai = None
        self.ai_ready = False
        self.init_ai()
        
        # Create UI
        self.create_widgets()
    
    def init_ai(self):
        """Initialize AI in a separate thread"""
        def load_ai():
            try:
                status_label = self.root.nametowidget("status_label")
                status_label.config(text="Loading AI models... This may take a minute.")
                self.ai = LocalMultimodalAI()
                self.ai_ready = True
                status_label.config(text="✓ AI Ready!", fg='green')
                messagebox.showinfo("Success", "AI initialized successfully!")
            except Exception as e:
                status_label.config(text=f"Error: {str(e)}", fg='red')
                messagebox.showerror("Error", f"Failed to initialize AI:\n{str(e)}")
        
        threading.Thread(target=load_ai, daemon=True).start()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🤖 Local Multimodal AI",
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#ecf0f1', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        status_label = tk.Label(
            status_frame,
            text="Initializing...",
            font=('Arial', 10),
            bg='#ecf0f1',
            fg='#34495e',
            anchor='w'
        )
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
        status_label.name = "status_label"
        
        # Notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Text Generation
        self.create_text_tab(notebook)
        
        # Tab 2: Image Generation
        self.create_image_tab(notebook)
        
        # Tab 3: Audio Generation
        self.create_audio_tab(notebook)
        
        # Tab 4: Image Understanding
        self.create_understanding_tab(notebook)
        
        # Tab 5: Combined
        self.create_combined_tab(notebook)
    
    def create_text_tab(self, notebook):
        """Create text generation tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="📝 Text Generation")
        
        # Input
        input_frame = ttk.LabelFrame(frame, text="Input", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        prompt_label = ttk.Label(input_frame, text="Text Prompt:")
        prompt_label.pack(anchor='w', pady=(0, 5))
        
        self.text_prompt = scrolledtext.ScrolledText(input_frame, height=5, wrap=tk.WORD)
        self.text_prompt.pack(fill=tk.BOTH, expand=True)
        
        # Parameters
        params_frame = ttk.Frame(input_frame)
        params_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(params_frame, text="Max Length:").grid(row=0, column=0, padx=5, sticky='w')
        self.max_length = tk.IntVar(value=100)
        max_length_spin = ttk.Spinbox(params_frame, from_=20, to=500, textvariable=self.max_length, width=10)
        max_length_spin.grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="Temperature:").grid(row=0, column=2, padx=5, sticky='w')
        self.temperature = tk.DoubleVar(value=0.7)
        temp_spin = ttk.Spinbox(params_frame, from_=0.1, to=2.0, textvariable=self.temperature, width=10, increment=0.1)
        temp_spin.grid(row=0, column=3, padx=5)
        
        # Button
        text_btn = ttk.Button(input_frame, text="Generate Text", command=self.on_generate_text)
        text_btn.pack(pady=10)
        
        # Output
        output_frame = ttk.LabelFrame(frame, text="Generated Text", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.text_output = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.text_output.pack(fill=tk.BOTH, expand=True)
    
    def create_image_tab(self, notebook):
        """Create image generation tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🎨 Image Generation")
        
        # Input
        input_frame = ttk.LabelFrame(frame, text="Input", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        prompt_label = ttk.Label(input_frame, text="Image Description:")
        prompt_label.pack(anchor='w', pady=(0, 5))
        
        self.image_prompt = scrolledtext.ScrolledText(input_frame, height=3, wrap=tk.WORD)
        self.image_prompt.pack(fill=tk.X)
        
        # Parameters
        params_frame = ttk.Frame(input_frame)
        params_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(params_frame, text="Steps:").grid(row=0, column=0, padx=5, sticky='w')
        self.steps = tk.IntVar(value=50)
        steps_spin = ttk.Spinbox(params_frame, from_=20, to=100, textvariable=self.steps, width=10)
        steps_spin.grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="Guidance:").grid(row=0, column=2, padx=5, sticky='w')
        self.guidance = tk.DoubleVar(value=7.5)
        guidance_spin = ttk.Spinbox(params_frame, from_=1.0, to=20.0, textvariable=self.guidance, width=10, increment=0.5)
        guidance_spin.grid(row=0, column=3, padx=5)
        
        # Button
        image_btn = ttk.Button(input_frame, text="Generate Image", command=self.on_generate_image)
        image_btn.pack(pady=10)
        
        # Output
        output_frame = ttk.LabelFrame(frame, text="Generated Image", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_output = tk.Label(output_frame, text="No image generated yet", bg='white', relief=tk.SUNKEN)
        self.image_output.pack(fill=tk.BOTH, expand=True)
    
    def create_audio_tab(self, notebook):
        """Create audio generation tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🔊 Audio Generation")
        
        # Input
        input_frame = ttk.LabelFrame(frame, text="Input", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        prompt_label = ttk.Label(input_frame, text="Text to Convert to Speech:")
        prompt_label.pack(anchor='w', pady=(0, 5))
        
        self.audio_text = scrolledtext.ScrolledText(input_frame, height=5, wrap=tk.WORD)
        self.audio_text.pack(fill=tk.BOTH, expand=True)
        
        # Button
        audio_btn = ttk.Button(input_frame, text="Generate Audio", command=self.on_generate_audio)
        audio_btn.pack(pady=10)
        
        # Status
        self.audio_status = ttk.Label(input_frame, text="")
        self.audio_status.pack()
    
    def create_understanding_tab(self, notebook):
        """Create image understanding tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="👁️ Image Understanding")
        
        # Input
        input_frame = ttk.LabelFrame(frame, text="Input", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        upload_btn = ttk.Button(input_frame, text="Upload Image", command=self.on_upload_image)
        upload_btn.pack(pady=10)
        
        self.uploaded_image_path = None
        self.upload_status = ttk.Label(input_frame, text="No image uploaded")
        self.upload_status.pack()
        
        question_label = ttk.Label(input_frame, text="Question (optional):")
        question_label.pack(anchor='w', pady=(10, 5))
        
        self.image_question = scrolledtext.ScrolledText(input_frame, height=2, wrap=tk.WORD)
        self.image_question.pack(fill=tk.X)
        
        understand_btn = ttk.Button(input_frame, text="Analyze Image", command=self.on_understand_image)
        understand_btn.pack(pady=10)
        
        # Output
        output_frame = ttk.LabelFrame(frame, text="Analysis Result", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.understanding_output = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.understanding_output.pack(fill=tk.BOTH, expand=True)
    
    def create_combined_tab(self, notebook):
        """Create combined generation tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🚀 Combined")
        
        # Inputs
        input_frame = ttk.LabelFrame(frame, text="Inputs", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(input_frame, text="Text Prompt:").pack(anchor='w', pady=(0, 5))
        self.combined_text = scrolledtext.ScrolledText(input_frame, height=2, wrap=tk.WORD)
        self.combined_text.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Image Prompt:").pack(anchor='w', pady=(0, 5))
        self.combined_image = scrolledtext.ScrolledText(input_frame, height=2, wrap=tk.WORD)
        self.combined_image.pack(fill=tk.X, pady=(0, 10))
        
        self.generate_audio_var = tk.BooleanVar()
        audio_check = ttk.Checkbutton(input_frame, text="Generate Audio", variable=self.generate_audio_var)
        audio_check.pack(anchor='w', pady=5)
        
        ttk.Label(input_frame, text="Audio Text:").pack(anchor='w', pady=(10, 5))
        self.combined_audio_text = scrolledtext.ScrolledText(input_frame, height=2, wrap=tk.WORD)
        self.combined_audio_text.pack(fill=tk.X)
        
        # Button
        combined_btn = ttk.Button(input_frame, text="Generate All", command=self.on_combined_generation)
        combined_btn.pack(pady=10)
        
        # Output
        output_frame = ttk.LabelFrame(frame, text="Results", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.combined_output = scrolledtext.ScrolledText(output_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.combined_output.pack(fill=tk.BOTH, expand=True)
    
    def check_ai_ready(self):
        """Check if AI is ready"""
        if not self.ai_ready:
            messagebox.showwarning("Not Ready", "AI is still initializing. Please wait...")
            return False
        return True
    
    def on_generate_text(self):
        """Handle text generation"""
        if not self.check_ai_ready():
            return
        
        prompt = self.text_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Input Required", "Please enter a text prompt.")
            return
        
        def generate():
            try:
                result = self.ai.generate_text(
                    prompt,
                    max_length=self.max_length.get(),
                    temperature=self.temperature.get()
                )
                self.text_output.config(state=tk.NORMAL)
                self.text_output.delete("1.0", tk.END)
                self.text_output.insert("1.0", result)
                self.text_output.config(state=tk.DISABLED)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate text:\n{str(e)}")
        
        threading.Thread(target=generate, daemon=True).start()
    
    def on_generate_image(self):
        """Handle image generation"""
        if not self.check_ai_ready():
            return
        
        prompt = self.image_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Input Required", "Please enter an image description.")
            return
        
        def generate():
            try:
                image = self.ai.generate_image(
                    prompt,
                    num_inference_steps=self.steps.get(),
                    guidance_scale=self.guidance.get()
                )
                if image:
                    # Resize for display
                    display_image = image.copy()
                    display_image.thumbnail((600, 600), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(display_image)
                    self.image_output.config(image=photo, text="")
                    self.image_output.image = photo  # Keep a reference
                    
                    # Ask to save
                    if messagebox.askyesno("Save Image", "Image generated! Save to file?"):
                        filename = filedialog.asksaveasfilename(
                            defaultextension=".png",
                            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
                        )
                        if filename:
                            self.ai.save_image(image, filename)
                            messagebox.showinfo("Success", f"Image saved to {filename}")
                else:
                    messagebox.showerror("Error", "Failed to generate image.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate image:\n{str(e)}")
        
        threading.Thread(target=generate, daemon=True).start()
    
    def on_generate_audio(self):
        """Handle audio generation"""
        if not self.check_ai_ready():
            return
        
        text = self.audio_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Required", "Please enter text to convert to speech.")
            return
        
        def generate():
            try:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".wav",
                    filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
                )
                if filename:
                    result = self.ai.generate_audio(text, filename, save_to_file=True)
                    if result and not result.startswith("Error"):
                        self.audio_status.config(text=f"✓ Audio saved to {filename}")
                        # Play audio
                        self.ai.generate_audio(text, save_to_file=False)
                    else:
                        self.audio_status.config(text=result if result else "Failed to generate audio")
                        messagebox.showerror("Error", result if result else "Failed to generate audio")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate audio:\n{str(e)}")
        
        threading.Thread(target=generate, daemon=True).start()
    
    def on_upload_image(self):
        """Handle image upload"""
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")]
        )
        if filename:
            self.uploaded_image_path = filename
            self.upload_status.config(text=f"Uploaded: {Path(filename).name}")
    
    def on_understand_image(self):
        """Handle image understanding"""
        if not self.check_ai_ready():
            return
        
        if not self.uploaded_image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        
        question = self.image_question.get("1.0", tk.END).strip()
        
        def understand():
            try:
                result = self.ai.understand_image(
                    self.uploaded_image_path,
                    question if question else None
                )
                self.understanding_output.config(state=tk.NORMAL)
                self.understanding_output.delete("1.0", tk.END)
                self.understanding_output.insert("1.0", result)
                self.understanding_output.config(state=tk.DISABLED)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to analyze image:\n{str(e)}")
        
        threading.Thread(target=understand, daemon=True).start()
    
    def on_combined_generation(self):
        """Handle combined generation"""
        if not self.check_ai_ready():
            return
        
        text_prompt = self.combined_text.get("1.0", tk.END).strip()
        image_prompt = self.combined_image.get("1.0", tk.END).strip()
        audio_text = self.combined_audio_text.get("1.0", tk.END).strip() if self.generate_audio_var.get() else ""
        
        if not text_prompt and not image_prompt:
            messagebox.showwarning("Input Required", "Please enter at least one prompt.")
            return
        
        def generate():
            try:
                results = []
                
                # Generate text
                if text_prompt:
                    text_result = self.ai.generate_text(text_prompt, max_length=150)
                    results.append(f"Generated Text:\n{text_result}\n\n")
                
                # Generate image
                image = None
                if image_prompt:
                    image = self.ai.generate_image(image_prompt, num_inference_steps=50)
                    if image:
                        results.append("✓ Image generated successfully!\n\n")
                
                # Generate audio
                if self.generate_audio_var.get() and audio_text:
                    filename = "combined_audio.wav"
                    result = self.ai.generate_audio(audio_text, filename, save_to_file=True)
                    if result and not result.startswith("Error"):
                        results.append(f"✓ Audio saved to {filename}\n")
                
                # Update output
                self.combined_output.config(state=tk.NORMAL)
                self.combined_output.delete("1.0", tk.END)
                self.combined_output.insert("1.0", "".join(results))
                self.combined_output.config(state=tk.DISABLED)
                
                messagebox.showinfo("Success", "Generation complete!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate:\n{str(e)}")
        
        threading.Thread(target=generate, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = TkinterMultimodalUI(root)
    root.mainloop()

