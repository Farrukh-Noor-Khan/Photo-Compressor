"""
FFT-based Image Compressor Module
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class FFTCompressor:
    def __init__(self):
        self.filter_types = ['ideal', 'gaussian', 'butterworth']
    
    def compress(self, image, cutoff_ratio=0.3, filter_type='gaussian', order=2):
        """
        Compress image using FFT-based frequency domain filtering
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input grayscale image
        cutoff_ratio : float
            Ratio of frequencies to keep (0.1-0.9)
        filter_type : str
            Type of low-pass filter: 'ideal', 'gaussian', 'butterworth'
        order : int
            Order for Butterworth filter
        
        Returns:
        --------
        compressed_image : numpy.ndarray
            Compressed image
        fft_original : numpy.ndarray
            Original FFT magnitude spectrum
        fft_compressed : numpy.ndarray
            Compressed FFT magnitude spectrum
        """
        
        # Validate input
        if len(image.shape) != 2:
            raise ValueError("Input must be a grayscale image")
        
        # Step 1: Apply FFT
        fft_original = np.fft.fft2(image.astype(float))
        fft_shifted = np.fft.fftshift(fft_original)
        
        # Get magnitude spectrum for visualization
        magnitude_original = np.log(np.abs(fft_shifted) + 1)
        
        # Step 2: Create filter mask
        rows, cols = image.shape
        mask = self._create_filter_mask(rows, cols, cutoff_ratio, filter_type, order)
        
        # Step 3: Apply filter in frequency domain
        fft_filtered = fft_shifted * mask
        
        # Get filtered magnitude spectrum
        magnitude_filtered = np.log(np.abs(fft_filtered) + 1)
        
        # Step 4: Inverse FFT
        fft_ishift = np.fft.ifftshift(fft_filtered)
        img_back = np.fft.ifft2(fft_ishift)
        img_back = np.abs(img_back)
        
        # Step 5: Normalize to 0-255 range
        img_back = np.clip(img_back, 0, 255)
        compressed_image = img_back.astype(np.uint8)
        
        return compressed_image, magnitude_original, magnitude_filtered
    
    def _create_filter_mask(self, rows, cols, cutoff_ratio, filter_type, order=2):
        """
        Create low-pass filter mask
        
        Parameters:
        -----------
        rows, cols : int
            Image dimensions
        cutoff_ratio : float
            Ratio of frequencies to keep
        filter_type : str
            Type of filter
        order : int
            Order for Butterworth filter
        
        Returns:
        --------
        mask : numpy.ndarray
            Filter mask
        """
        
        # Create coordinate grid
        center_row, center_col = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        
        # Distance from center
        distance = np.sqrt((x - center_col)**2 + (y - center_row)**2)
        
        # Maximum distance
        max_distance = np.sqrt(center_row**2 + center_col**2)
        cutoff_distance = cutoff_ratio * max_distance
        
        if filter_type == 'ideal':
            # Ideal low-pass filter
            mask = np.zeros((rows, cols))
            mask[distance <= cutoff_distance] = 1
        
        elif filter_type == 'gaussian':
            # Gaussian low-pass filter
            sigma = cutoff_distance / 3  # 99.7% of values within 3 sigma
            mask = np.exp(-(distance**2) / (2 * sigma**2))
        
        elif filter_type == 'butterworth':
            # Butterworth low-pass filter
            mask = 1 / (1 + (distance / cutoff_distance)**(2 * order))
        
        else:
            raise ValueError(f"Unknown filter type: {filter_type}. Choose from {self.filter_types}")
        
        return mask
    
    def compress_color(self, image, cutoff_ratio=0.3, filter_type='gaussian'):
        """
        Compress color image by processing each channel separately
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input color image (BGR format)
        cutoff_ratio : float
            Ratio of frequencies to keep
        filter_type : str
            Type of low-pass filter
        
        Returns:
        --------
        compressed_image : numpy.ndarray
            Compressed color image
        """
        
        if len(image.shape) != 3:
            raise ValueError("Input must be a color image")
        
        channels = cv2.split(image)
        compressed_channels = []
        
        for channel in channels:
            compressed_channel, _, _ = self.compress(channel, cutoff_ratio, filter_type)
            compressed_channels.append(compressed_channel)
        
        compressed_image = cv2.merge(compressed_channels)
        return compressed_image
    
    def get_filter_info(self, rows, cols, cutoff_ratio, filter_type='gaussian'):
        """
        Get information about the filter
        
        Returns:
        --------
        info : dict
            Filter information
        """
        mask = self._create_filter_mask(rows, cols, cutoff_ratio, filter_type)
        
        info = {
            'rows': rows,
            'cols': cols,
            'cutoff_ratio': cutoff_ratio,
            'filter_type': filter_type,
            'mask_sum': np.sum(mask),
            'mask_mean': np.mean(mask),
            'frequencies_kept': (np.sum(mask > 0.5) / (rows * cols)) * 100
        }
        
        return info