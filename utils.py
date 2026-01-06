"""
Utility Functions Module
"""

import os
import numpy as np
import cv2
import json
from datetime import datetime
import hashlib

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_image(image_path, grayscale=True):
    """
    Load image from path
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    grayscale : bool
        Whether to convert to grayscale
    
    Returns:
    --------
    image : numpy.ndarray
        Loaded image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    return image

def save_image(image, output_path, jpeg_quality=85):
    """
    Save image to file
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image to save
    output_path : str
        Output file path
    jpeg_quality : int
        JPEG quality (1-100)
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Determine format from extension
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    elif ext == '.png':
        cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(output_path, image)
    
    print(f"Image saved to: {output_path}")

def calculate_file_sizes(original_path, compressed_path):
    """
    Calculate file sizes and compression ratio
    
    Returns:
    --------
    sizes : dict
        Dictionary with size information
    """
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    
    compression_ratio = (1 - compressed_size / original_size) * 100
    size_reduction = original_size - compressed_size
    
    sizes = {
        'original_bytes': original_size,
        'compressed_bytes': compressed_size,
        'original_kb': original_size / 1024,
        'compressed_kb': compressed_size / 1024,
        'compression_ratio': compression_ratio,
        'size_reduction_bytes': size_reduction,
        'size_reduction_kb': size_reduction / 1024
    }
    
    return sizes

def generate_report(original_path, compressed_path, metrics, parameters):
    """
    Generate a comprehensive report
    
    Parameters:
    -----------
    original_path : str
        Path to original image
    compressed_path : str
        Path to compressed image
    metrics : dict
        Quality metrics
    parameters : dict
        Compression parameters
    
    Returns:
    --------
    report : dict
        Complete report
    """
    sizes = calculate_file_sizes(original_path, compressed_path)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'original_image': original_path,
        'compressed_image': compressed_path,
        'parameters': parameters,
        'file_sizes': sizes,
        'quality_metrics': metrics,
        'summary': {
            'compression_achieved': f"{sizes['compression_ratio']:.2f}%",
            'quality_assessment': "Good" if metrics.get('psnr', 0) > 30 else "Acceptable",
            'processing_time': parameters.get('processing_time', 'N/A')
        }
    }
    
    return report

def save_report(report, output_path):
    """Save report to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {output_path}")

def calculate_image_entropy(image):
    """
    Calculate image entropy (measure of information content)
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    
    Returns:
    --------
    entropy : float
        Image entropy
    """
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Normalize histogram to get probabilities
    hist = hist / hist.sum()
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return entropy

def get_image_info(image_path):
    """
    Get detailed information about an image
    
    Returns:
    --------
    info : dict
        Image information
    """
    image = cv2.imread(image_path)
    
    if image is None:
        return None
    
    info = {
        'path': image_path,
        'dimensions': f"{image.shape[1]}x{image.shape[0]}",
        'channels': image.shape[2] if len(image.shape) == 3 else 1,
        'dtype': str(image.dtype),
        'size_bytes': os.path.getsize(image_path),
        'format': os.path.splitext(image_path)[1],
        'entropy': calculate_image_entropy(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) 
                   if len(image.shape) == 3 else calculate_image_entropy(image)
    }
    
    return info

def batch_process_images(input_folder, output_folder, cutoff_ratio=0.3, filter_type='gaussian'):
    """
    Process all images in a folder
    
    Parameters:
    -----------
    input_folder : str
        Input folder path
    output_folder : str
        Output folder path
    cutoff_ratio : float
        Cutoff ratio for filter
    filter_type : str
        Type of filter to use
    """
    from fft_compressor import FFTCompressor
    from evaluator import ImageEvaluator
    
    compressor = FFTCompressor()
    evaluator = ImageEvaluator()
    
    create_directory(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    results = []
    
    for img_file in image_files:
        print(f"Processing: {img_file}")
        
        try:
            # Load image
            img_path = os.path.join(input_folder, img_file)
            img = load_image(img_path)
            
            # Compress image
            compressed, _, _ = compressor.compress(img, cutoff_ratio, filter_type)
            
            # Save compressed image
            output_path = os.path.join(output_folder, f"compressed_{img_file}")
            save_image(compressed, output_path)
            
            # Calculate metrics
            metrics = evaluator.calculate_metrics(img, compressed)
            
            # Calculate file sizes
            sizes = calculate_file_sizes(img_path, output_path)
            
            # Store results
            result = {
                'filename': img_file,
                'original_size': sizes['original_bytes'],
                'compressed_size': sizes['compressed_bytes'],
                'compression_ratio': sizes['compression_ratio'],
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim']
            }
            
            results.append(result)
            
            print(f"  ✓ Compression: {sizes['compression_ratio']:.2f}%")
            print(f"  ✓ PSNR: {metrics['psnr']:.2f} dB")
            
        except Exception as e:
            print(f"  ✗ Error processing {img_file}: {str(e)}")
    
    # Save batch results
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_folder, f"batch_results_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBatch processing complete. Results saved to: {results_path}")
    
    return results