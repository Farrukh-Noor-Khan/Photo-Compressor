"""
Result Visualization Module
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import gridspec

class ResultVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        self.figsize = (15, 10)
    
    def plot_all_results(self, original, compressed, fft_original, fft_compressed, 
                        metrics=None, save_path=None):
        """
        Create comprehensive visualization of all results
        
        Parameters:
        -----------
        original : numpy.ndarray
            Original image
        compressed : numpy.ndarray
            Compressed image
        fft_original : numpy.ndarray
            Original FFT magnitude
        fft_compressed : numpy.ndarray
            Compressed FFT magnitude
        metrics : dict
            Quality metrics
        save_path : str
            Path to save the figure
        """
        
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        # Plot 1: Original Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original, cmap='gray')
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Plot 2: Compressed Image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(compressed, cmap='gray')
        ax2.set_title('Compressed Image', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Plot 3: Difference Image
        ax3 = fig.add_subplot(gs[0, 2])
        difference = cv2.absdiff(original, compressed)
        im3 = ax3.imshow(difference, cmap='hot')
        ax3.set_title('Difference (Absolute)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Plot 4: Histogram Comparison
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.hist(original.flatten(), bins=50, alpha=0.5, label='Original', color='blue')
        ax4.hist(compressed.flatten(), bins=50, alpha=0.5, label='Compressed', color='red')
        ax4.set_title('Histogram Comparison', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Pixel Intensity')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Original FFT Spectrum
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(fft_original, cmap='gray')
        ax5.set_title('Original FFT Spectrum', fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        # Plot 6: Compressed FFT Spectrum
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(fft_compressed, cmap='gray')
        ax6.set_title('Compressed FFT Spectrum', fontsize=12, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
        # Plot 7: FFT Difference
        ax7 = fig.add_subplot(gs[1, 2])
        fft_diff = np.abs(fft_original - fft_compressed)
        im7 = ax7.imshow(fft_diff, cmap='hot')
        ax7.set_title('FFT Spectrum Difference', fontsize=12, fontweight='bold')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        
        # Plot 8: 3D Frequency Plot (simplified)
        ax8 = fig.add_subplot(gs[1, 3], projection='3d')
        rows, cols = fft_original.shape
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        Z = fft_original
        # Downsample for performance
        stride = max(1, rows//50)
        ax8.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], 
                        Z[::stride, ::stride], cmap='viridis', alpha=0.7)
        ax8.set_title('3D Frequency Spectrum', fontsize=12, fontweight='bold')
        ax8.set_xlabel('X')
        ax8.set_ylabel('Y')
        ax8.set_zlabel('Magnitude')
        
        # Plot 9: Quality Metrics Bar Chart
        if metrics:
            ax9 = fig.add_subplot(gs[2, 0:2])
            metric_names = ['PSNR (dB)', 'SSIM', 'Hist Corr', 'Edge Pres']
            metric_values = [metrics.get('psnr', 0), 
                           metrics.get('ssim', 0), 
                           metrics.get('histogram_correlation', 0),
                           metrics.get('edge_preservation', 0)]
            
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            bars = ax9.bar(metric_names, metric_values, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax9.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax9.set_title('Quality Metrics Comparison', fontsize=12, fontweight='bold')
            ax9.set_ylabel('Value')
            ax9.set_ylim([0, max(metric_values) * 1.2])
            ax9.grid(True, alpha=0.3, axis='y')
        
        # Plot 10: Pixel Value Scatter Plot
        ax10 = fig.add_subplot(gs[2, 2])
        sample_size = min(1000, original.size)
        indices = np.random.choice(original.size, sample_size, replace=False)
        original_samples = original.flatten()[indices]
        compressed_samples = compressed.flatten()[indices]
        
        ax10.scatter(original_samples, compressed_samples, alpha=0.5, s=10, color='green')
        ax10.plot([0, 255], [0, 255], 'r--', alpha=0.7, label='Ideal (y=x)')
        ax10.set_title('Pixel Value Correlation', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Original Pixel Value')
        ax10.set_ylabel('Compressed Pixel Value')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # Plot 11: Info Text
        ax11 = fig.add_subplot(gs[2, 3])
        ax11.axis('off')
        
        info_text = "FFT Compression Results\n"
        info_text += "="*30 + "\n"
        info_text += f"Original Size: {original.shape[1]}x{original.shape[0]}\n"
        info_text += f"Bits per pixel: 8\n"
        
        if metrics:
            info_text += f"\nQuality Metrics:\n"
            info_text += f"PSNR: {metrics.get('psnr', 0):.2f} dB\n"
            info_text += f"MSE: {metrics.get('mse', 0):.2f}\n"
            info_text += f"SSIM: {metrics.get('ssim', 0):.4f}\n"
            info_text += f"Hist Corr: {metrics.get('histogram_correlation', 0):.4f}\n"
        
        info_text += f"\nMethod:\n"
        info_text += "1. FFT to frequency domain\n"
        info_text += "2. Apply low-pass filter\n"
        info_text += "3. Inverse FFT\n"
        info_text += "4. Quantization & encoding"
        
        ax11.text(0.1, 0.95, info_text, transform=ax11.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('FFT-Based Image Compression Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def plot_filter_comparison(self, image, cutoff_ratios=[0.1, 0.3, 0.5, 0.7]):
        """
        Compare different filter cutoff ratios
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        cutoff_ratios : list
            List of cutoff ratios to compare
        """
        
        from fft_compressor import FFTCompressor
        
        compressor = FFTCompressor()
        
        fig, axes = plt.subplots(2, len(cutoff_ratios) + 1, figsize=(15, 8))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original', fontsize=10)
        axes[0, 0].axis('off')
        
        axes[1, 0].hist(image.flatten(), bins=50, color='blue', alpha=0.7)
        axes[1, 0].set_title('Original Histogram', fontsize=10)
        
        for i, cutoff in enumerate(cutoff_ratios, 1):
            compressed, _, _ = compressor.compress(image, cutoff_ratio=cutoff)
            
            axes[0, i].imshow(compressed, cmap='gray')
            axes[0, i].set_title(f'Cutoff={cutoff}', fontsize=10)
            axes[0, i].axis('off')
            
            axes[1, i].hist(compressed.flatten(), bins=50, color='red', alpha=0.7)
            axes[1, i].set_title(f'Histogram (cutoff={cutoff})', fontsize=10)
        
        plt.suptitle('Effect of Different Cutoff Ratios', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_frequency_response(self, rows=256, cols=256):
        """
        Plot frequency response of different filters
        
        Parameters:
        -----------
        rows, cols : int
            Image dimensions for filter
        """
        
        from fft_compressor import FFTCompressor
        
        compressor = FFTCompressor()
        cutoff = 0.3
        
        filters = ['ideal', 'gaussian', 'butterworth']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, filter_type in enumerate(filters):
            mask = compressor._create_filter_mask(rows, cols, cutoff, filter_type)
            
            im = axes[i].imshow(mask, cmap='viridis')
            axes[i].set_title(f'{filter_type.capitalize()} Filter\nCutoff={cutoff}', 
                            fontsize=12, fontweight='bold')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.suptitle('Frequency Response of Different Low-Pass Filters', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()