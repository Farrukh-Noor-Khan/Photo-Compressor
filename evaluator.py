"""
Image Quality Evaluation Module
"""

import numpy as np
import cv2
from scipy import ndimage
import math

class ImageEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, original, compressed):
        """
        Calculate all quality metrics between original and compressed images
        
        Parameters:
        -----------
        original : numpy.ndarray
            Original image
        compressed : numpy.ndarray
            Compressed image
        
        Returns:
        --------
        metrics : dict
            Dictionary containing all metrics
        """
        
        # Ensure images have same dimensions
        if original.shape != compressed.shape:
            compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
        
        # Calculate MSE
        mse = self.calculate_mse(original, compressed)
        
        # Calculate PSNR
        psnr = self.calculate_psnr(original, compressed)
        
        # Calculate SSIM
        ssim = self.calculate_ssim(original, compressed)
        
        # Calculate Compression Ratio (needs file sizes, calculated separately)
        
        # Calculate Histogram Correlation
        hist_corr = self.calculate_histogram_correlation(original, compressed)
        
        # Calculate Edge Preservation
        edge_preservation = self.calculate_edge_preservation(original, compressed)
        
        metrics = {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'histogram_correlation': hist_corr,
            'edge_preservation': edge_preservation
        }
        
        self.metrics = metrics
        return metrics
    
    def calculate_mse(self, original, compressed):
        """Calculate Mean Squared Error"""
        mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
        return mse
    
    def calculate_psnr(self, original, compressed):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = self.calculate_mse(original, compressed)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, original, compressed):
        """Calculate Structural Similarity Index"""
        # Constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        original = original.astype(np.float64)
        compressed = compressed.astype(np.float64)
        
        # Calculate means
        mu_x = np.mean(original)
        mu_y = np.mean(compressed)
        
        # Calculate variances
        sigma_x = np.var(original)
        sigma_y = np.var(compressed)
        
        # Calculate covariance
        sigma_xy = np.cov(original.flatten(), compressed.flatten())[0, 1]
        
        # Calculate SSIM
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        ssim = numerator / denominator
        return ssim
    
    def calculate_histogram_correlation(self, original, compressed):
        """Calculate histogram correlation"""
        # Calculate histograms
        hist_original = cv2.calcHist([original], [0], None, [256], [0, 256])
        hist_compressed = cv2.calcHist([compressed], [0], None, [256], [0, 256])
        
        # Normalize histograms
        hist_original = hist_original / np.sum(hist_original)
        hist_compressed = hist_compressed / np.sum(hist_compressed)
        
        # Calculate correlation
        correlation = np.corrcoef(hist_original.flatten(), hist_compressed.flatten())[0, 1]
        return correlation
    
    def calculate_edge_preservation(self, original, compressed):
        """Calculate edge preservation ratio"""
        # Apply Sobel edge detection
        sobel_original = cv2.Sobel(original, cv2.CV_64F, 1, 1, ksize=3)
        sobel_compressed = cv2.Sobel(compressed, cv2.CV_64F, 1, 1, ksize=3)
        
        # Calculate edge preservation
        original_edges = np.abs(sobel_original)
        compressed_edges = np.abs(sobel_compressed)
        
        # Normalize
        original_edges = original_edges / np.max(original_edges)
        compressed_edges = compressed_edges / np.max(compressed_edges)
        
        # Calculate correlation of edge maps
        edge_correlation = np.corrcoef(original_edges.flatten(), 
                                      compressed_edges.flatten())[0, 1]
        return edge_correlation
    
    def get_quality_assessment(self, psnr_value):
        """Provide qualitative assessment based on PSNR"""
        if psnr_value > 40:
            return "Excellent quality"
        elif psnr_value > 30:
            return "Good quality"
        elif psnr_value > 20:
            return "Acceptable quality"
        else:
            return "Poor quality"
    
    def print_metrics(self):
        """Print all calculated metrics"""
        print("\n" + "="*60)
        print("IMAGE QUALITY METRICS")
        print("="*60)
        
        for metric, value in self.metrics.items():
            if metric == 'psnr':
                quality = self.get_quality_assessment(value)
                print(f"{metric.upper():25s}: {value:8.2f} dB ({quality})")
            elif metric == 'ssim':
                print(f"{metric.upper():25s}: {value:8.4f}")
            elif metric in ['histogram_correlation', 'edge_preservation']:
                print(f"{metric.replace('_', ' ').title():25s}: {value:8.4f}")
            else:
                print(f"{metric.upper():25s}: {value:8.2f}")