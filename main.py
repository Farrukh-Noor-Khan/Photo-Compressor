"""
Main script for Frequency Domain Image Compression Using Fourier Transform
Authors: Hashir Imran, Zafran Ajab, Taimoor Noor Khan
Institute of Space Technology, KICSIT
"""

import os
import cv2
import argparse
from fft_compressor import FFTCompressor
from evaluator import ImageEvaluator
from visualizer import ResultVisualizer
from utils import create_directory

def main():
    parser = argparse.ArgumentParser(description='FFT-based Image Compression System')
    parser.add_argument('--image', type=str, default='images/a.jpg', help='Input image path')
    parser.add_argument('--cutoff', type=float, default=0.3, help='Cutoff ratio (0.1-0.9)')
    parser.add_argument('--filter', type=str, default='gaussian', 
                        choices=['ideal', 'gaussian', 'butterworth'], help='Filter type')
    parser.add_argument('--jpeg_quality', type=int, default=85, help='JPEG quality (1-100)')
    parser.add_argument('--batch', action='store_true', help='Process all images in folder')
    parser.add_argument('--visualize', action='store_true', help='Show visualizations')
    
    args = parser.parse_args()
    
    # Create output directories
    create_directory('output')
    create_directory('output/compressed')
    create_directory('output/plots')
    
    if args.batch:
        process_batch(args)
    else:
        process_single(args)

def process_single(args):
    """Process a single image"""
    print(f"\n{'='*60}")
    print("FFT-BASED IMAGE COMPRESSION SYSTEM")
    print(f"{'='*60}")
    
    # Load image
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    print(f"\n[1] Loading image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print("Error: Unable to read image")
        return
    
    # Initialize compressor
    compressor = FFTCompressor()
    evaluator = ImageEvaluator()
    visualizer = ResultVisualizer()
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        print("[2] Converting to grayscale")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    print(f"[3] Original image size: {img_gray.shape[1]}x{img_gray.shape[0]}")
    
    # Perform compression
    print(f"[4] Applying FFT compression with {args.filter} filter (cutoff={args.cutoff})")
    compressed_img, fft_original, fft_compressed = compressor.compress(
        img_gray, 
        cutoff_ratio=args.cutoff, 
        filter_type=args.filter
    )
    
    # Save compressed image
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    compressed_path = f"output/compressed/{base_name}_compressed.jpg"
    
    # Save with JPEG compression
    cv2.imwrite(compressed_path, compressed_img, 
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    
    # Load saved image to get actual compressed size
    saved_img = cv2.imread(compressed_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate metrics
    print("[5] Calculating compression metrics...")
    metrics = evaluator.calculate_metrics(img_gray, saved_img)
    
    # File size comparison
    original_size = os.path.getsize(args.image)
    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    # Display results
    print(f"\n{'='*60}")
    print("COMPRESSION RESULTS")
    print(f"{'='*60}")
    print(f"Original file size: {original_size / 1024:.2f} KB")
    print(f"Compressed file size: {compressed_size / 1024:.2f} KB")
    print(f"Compression ratio: {compression_ratio:.2f}%")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"Compressed image saved to: {compressed_path}")
    
    # Save metrics to file
    with open(f"output/{base_name}_metrics.txt", 'w') as f:
        f.write("FFT Image Compression Results\n")
        f.write("="*40 + "\n")
        f.write(f"Original image: {args.image}\n")
        f.write(f"Filter type: {args.filter}\n")
        f.write(f"Cutoff ratio: {args.cutoff}\n")
        f.write(f"JPEG quality: {args.jpeg_quality}\n")
        f.write(f"Original size: {original_size / 1024:.2f} KB\n")
        f.write(f"Compressed size: {compressed_size / 1024:.2f} KB\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}%\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"MSE: {metrics['mse']:.2f}\n")
        f.write(f"SSIM: {metrics['ssim']:.4f}\n")
    
    # Visualize results
    if args.visualize:
        visualizer.plot_all_results(
            img_gray, 
            compressed_img, 
            fft_original, 
            fft_compressed,
            metrics,
            save_path=f"output/plots/{base_name}_results.png"
        )
    
    print(f"\n{'='*60}")
    print("Compression completed successfully!")
    print(f"{'='*60}")

def process_batch(args):
    """Process all images in a folder"""
    image_folder = args.image
    if not os.path.isdir(image_folder):
        print(f"Error: {image_folder} is not a directory")
        return
    
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"\nProcessing {len(image_files)} images in batch mode...")
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file}")
        args.image = os.path.join(image_folder, img_file)
        process_single(args)

if __name__ == "__main__":
    main()