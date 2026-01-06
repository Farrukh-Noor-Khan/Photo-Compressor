
"""
Test script for FFT image compression
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from fft_compressor import FFTCompressor
from evaluator import ImageEvaluator
from visualizer import ResultVisualizer

def test_basic_compression():
    """Test basic compression functionality"""
    print("Testing basic compression...")
    
    # Create a simple test image
    test_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_image, (128, 128), 50, 255, -1)
    cv2.rectangle(test_image, (50, 50), (100, 100), 150, -1)
    
    # Initialize compressor
    compressor = FFTCompressor()
    evaluator = ImageEvaluator()
    
    # Test different cutoff ratios
    cutoff_ratios = [0.1, 0.3, 0.5, 0.7]
    
    for cutoff in cutoff_ratios:
        print(f"\nCutoff ratio: {cutoff}")
        
        # Compress image
        compressed, _, _ = compressor.compress(test_image, cutoff_ratio=cutoff)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(test_image, compressed)
        
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  MSE: {metrics['mse']:.2f}")
        print(f"  SSIM: {metrics['ssim']:.4f}")
    
    print("\nBasic compression test completed!")

def test_filter_types():
    """Compare different filter types"""
    print("\nTesting different filter types...")
    
    # Load sample image
    test_image = cv2.imread('images/sample.jpg', cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        # Create test pattern if no image
        test_image = np.zeros((256, 256), dtype=np.uint8)
        for i in range(0, 256, 32):
            test_image[i:i+16, :] = 255
    
    compressor = FFTCompressor()
    evaluator = ImageEvaluator()
    
    filter_types = ['ideal', 'gaussian', 'butterworth']
    
    results = []
    for filter_type in filter_types:
        print(f"\nFilter type: {filter_type}")
        
        # Compress with current filter
        compressed, _, _ = compressor.compress(test_image, 
                                               cutoff_ratio=0.3, 
                                               filter_type=filter_type)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(test_image, compressed)
        results.append((filter_type, compressed, metrics))
        
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, len(filter_types) + 1, figsize=(15, 5))
    
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i, (filter_type, compressed, metrics) in enumerate(results, 1):
        axes[i].imshow(compressed, cmap='gray')
        axes[i].set_title(f'{filter_type}\nPSNR: {metrics["psnr"]:.1f} dB')
        axes[i].axis('off')
    
    plt.suptitle('Comparison of Different Filter Types', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nFilter type comparison completed!")

def test_visualization():
    """Test visualization functions"""
    print("\nTesting visualization functions...")
    
    # Create test image
    test_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_image, (128, 128), 80, 200, -1)
    cv2.rectangle(test_image, (30, 30), (100, 100), 150, -1)
    
    # Add some texture
    noise = np.random.normal(0, 25, test_image.shape).astype(np.uint8)
    test_image = cv2.add(test_image, noise)
    
    # Compress image
    compressor = FFTCompressor()
    compressed, fft_orig, fft_comp = compressor.compress(test_image, cutoff_ratio=0.4)
    
    # Calculate metrics
    evaluator = ImageEvaluator()
    metrics = evaluator.calculate_metrics(test_image, compressed)
    
    # Visualize results
    visualizer = ResultVisualizer()
    visualizer.plot_all_results(test_image, compressed, fft_orig, fft_comp, metrics)
    
    print("Visualization test completed!")

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("FFT IMAGE COMPRESSION SYSTEM - TEST SUITE")
    print("="*60)
    
    test_basic_compression()
    test_filter_types()
    test_visualization()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()