"""
Simple GUI for FFT Image Compression
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from fft_compressor import FFTCompressor
from evaluator import ImageEvaluator

class FFTCompressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FFT Image Compression System")
        self.root.geometry("1200x700")
        
        # Variables
        self.image_path = None
        self.original_image = None
        self.compressed_image = None
        self.metrics = {}
        
        # Initialize components
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Image selection
        ttk.Label(control_frame, text="Select Image:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.image_path_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.image_path_var, width=30).grid(row=1, column=0, columnspan=2, pady=(0, 10))
        ttk.Button(control_frame, text="Browse", command=self.browse_image).grid(row=1, column=2, padx=(5, 0))
        
        # Filter type
        ttk.Label(control_frame, text="Filter Type:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        self.filter_var = tk.StringVar(value="gaussian")
        filter_combo = ttk.Combobox(control_frame, textvariable=self.filter_var, 
                                   values=["ideal", "gaussian", "butterworth"], state="readonly", width=15)
        filter_combo.grid(row=3, column=0, columnspan=2, pady=(0, 10))
        
        # Cutoff ratio
        ttk.Label(control_frame, text="Cutoff Ratio:").grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        self.cutoff_var = tk.DoubleVar(value=0.3)
        cutoff_scale = ttk.Scale(control_frame, from_=0.1, to=0.9, variable=self.cutoff_var, 
                                orient=tk.HORIZONTAL, length=150)
        cutoff_scale.grid(row=5, column=0, columnspan=2, pady=(0, 5))
        ttk.Label(control_frame, textvariable=self.cutoff_var).grid(row=5, column=2, padx=(5, 0))
        
        # JPEG quality
        ttk.Label(control_frame, text="JPEG Quality:").grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        self.quality_var = tk.IntVar(value=85)
        quality_scale = ttk.Scale(control_frame, from_=1, to=100, variable=self.quality_var, 
                                 orient=tk.HORIZONTAL, length=150)
        quality_scale.grid(row=7, column=0, columnspan=2, pady=(0, 5))
        ttk.Label(control_frame, textvariable=self.quality_var).grid(row=7, column=2, padx=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=8, column=0, columnspan=3, pady=(20, 10))
        
        ttk.Button(button_frame, text="Compress", command=self.compress_image, 
                  width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Save Results", command=self.save_results, 
                  width=15).pack(side=tk.LEFT)
        
        # Metrics display
        metrics_frame = ttk.LabelFrame(control_frame, text="Compression Metrics", padding="10")
        metrics_frame.grid(row=9, column=0, columnspan=3, pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.metrics_text = tk.Text(metrics_frame, height=8, width=30)
        self.metrics_text.grid(row=0, column=0)
        
        # Right panel - Images
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image
        ttk.Label(image_frame, text="Original Image").grid(row=0, column=0, padx=(0, 10))
        self.original_canvas = tk.Canvas(image_frame, width=300, height=300, bg="gray")
        self.original_canvas.grid(row=1, column=0, padx=(0, 10))
        
        # Compressed image
        ttk.Label(image_frame, text="Compressed Image").grid(row=0, column=1)
        self.compressed_canvas = tk.Canvas(image_frame, width=300, height=300, bg="gray")
        self.compressed_canvas.grid(row=1, column=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.image_path_var.set(file_path)
            self.display_image(file_path, self.original_canvas)
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
    
    def display_image(self, image_path, canvas):
        try:
            # Load and resize image
            img = Image.open(image_path)
            img.thumbnail((300, 300))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update canvas
            canvas.image = photo  # Keep reference
            canvas.delete("all")
            canvas.create_image(150, 150, image=photo, anchor=tk.CENTER)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def compress_image(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            self.status_var.set("Compressing...")
            self.root.update()
            
            # Load image
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            
            # Initialize compressor
            compressor = FFTCompressor()
            evaluator = ImageEvaluator()
            
            # Get parameters
            cutoff = self.cutoff_var.get()
            filter_type = self.filter_var.get()
            
            # Compress image
            self.compressed_image, _, _ = compressor.compress(
                img, 
                cutoff_ratio=cutoff, 
                filter_type=filter_type
            )
            
            # Calculate metrics
            self.metrics = evaluator.calculate_metrics(img, self.compressed_image)
            
            # Display compressed image
            self.display_compressed_image()
            
            # Update metrics display
            self.update_metrics_display()
            
            self.status_var.set("Compression complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Compression failed: {str(e)}")
            self.status_var.set("Error")
    
    def display_compressed_image(self):
        if self.compressed_image is not None:
            # Convert to PIL Image
            img_pil = Image.fromarray(self.compressed_image)
            img_pil.thumbnail((300, 300))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img_pil)
            
            # Update canvas
            self.compressed_canvas.image = photo
            self.compressed_canvas.delete("all")
            self.compressed_canvas.create_image(150, 150, image=photo, anchor=tk.CENTER)
    
    def update_metrics_display(self):
        self.metrics_text.delete(1.0, tk.END)
        
        if self.metrics:
            text = f"Compression Results:\n"
            text += "="*20 + "\n\n"
            text += f"PSNR: {self.metrics.get('psnr', 0):.2f} dB\n"
            text += f"MSE: {self.metrics.get('mse', 0):.2f}\n"
            text += f"SSIM: {self.metrics.get('ssim', 0):.4f}\n"
            text += f"Histogram Correlation: {self.metrics.get('histogram_correlation', 0):.4f}\n"
            text += f"Edge Preservation: {self.metrics.get('edge_preservation', 0):.4f}\n"
            
            self.metrics_text.insert(1.0, text)
    
    def save_results(self):
        if self.compressed_image is None:
            messagebox.showwarning("Warning", "No compressed image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.compressed_image)
                messagebox.showinfo("Success", f"Image saved to:\n{file_path}")
                self.status_var.set(f"Saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

def main():
    root = tk.Tk()
    app = FFTCompressionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()