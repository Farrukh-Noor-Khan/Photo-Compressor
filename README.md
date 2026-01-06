# Frequency Domain Image Compression Using Fourier Transform

> *When pixels step into the frequency domain, redundancy fades and clarity remains.*

---

## 📋 Project Overview
This project presents an **FFT-based image compression system** that reduces image size by filtering out high-frequency components in the frequency domain while preserving essential visual information.

By transforming images from the spatial domain into the frequency domain using the **Fast Fourier Transform (FFT)**, applying controlled low-pass filtering, and reconstructing them using the inverse FFT, the system achieves efficient compression with minimal perceptual quality loss.

---

## 🎯 Key Features
- **Multiple Filter Types**
  - Ideal Low-Pass Filter
  - Gaussian Low-Pass Filter
  - Butterworth Low-Pass Filter
- **Adjustable Compression**
  - Control compression level using cutoff frequency ratio
- **Comprehensive Quality Metrics**
  - PSNR, MSE, SSIM
  - Histogram correlation
  - Edge preservation analysis
- **Visual Analysis**
  - Side-by-side image comparison
  - FFT magnitude spectrum visualization
  - Difference maps
- **Batch Processing**
  - Compress multiple images in one run
- **Optional GUI Interface**
  - Simple and intuitive graphical user interface

---

## 📁 Project Structure
```

image_compression_project/
├── main.py                 # Main execution script
├── fft_compressor.py       # FFT-based compression logic
├── evaluator.py            # Image quality evaluation metrics
├── visualizer.py           # Visualization utilities
├── utils.py                # Helper functions
├── simple_gui.py           # Optional GUI interface
├── test_compression.py     # Automated test suite
├── images/                 # Input images
├── output/                 # Generated results
│   ├── compressed/
│   ├── plots/
│   └── reports/
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

````

---

## 🚀 Installation

### Step 1: Project Setup
- Create a new project folder
- Copy all Python scripts into it
- Create two subfolders:
  - `images/`
  - `output/`
- Place test images inside the `images/` folder

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
````

---

## 💻 Usage

### Method 1: Command Line (Recommended)

```bash
# Basic compression
python main.py --image images/sample.jpg

# Custom parameters
python main.py --image images/sample.jpg --cutoff 0.4 --filter gaussian --jpeg_quality 90

# Batch processing
python main.py --image images/ --batch

# Enable visualization
python main.py --image images/sample.jpg --visualize
```

---

### Method 2: GUI Interface

```bash
python simple_gui.py
```

---

### Method 3: Import as a Module

```python
from fft_compressor import FFTCompressor
from evaluator import ImageEvaluator

compressor = FFTCompressor()
evaluator = ImageEvaluator()

compressed, fft_orig, fft_comp = compressor.compress(image, cutoff_ratio=0.3)
metrics = evaluator.calculate_metrics(image, compressed)
```

---

## ⚙️ Parameters

| Parameter    | Range                          | Default  | Description                   |
| ------------ | ------------------------------ | -------- | ----------------------------- |
| cutoff_ratio | 0.1 – 0.9                      | 0.3      | Ratio of frequencies retained |
| filter_type  | ideal / gaussian / butterworth | gaussian | Low-pass filter type          |
| jpeg_quality | 1 – 100                        | 85       | JPEG compression quality      |
| order        | 1 – 10                         | 2        | Butterworth filter order      |

---

## 📊 Output Files

The system generates:

* **Compressed images** → `output/compressed/`
* **Quality metrics** → `output/*_metrics.txt`
* **Visualization plots** → `output/plots/`
* **Batch reports** → JSON format

---

## 📈 Results Interpretation

| PSNR Range | Quality Assessment |
| ---------- | ------------------ |
| > 40 dB    | Excellent quality  |
| 30–40 dB   | Good quality       |
| 20–30 dB   | Acceptable         |
| < 20 dB    | Poor quality       |

---

## 🔧 Technical Details

### Compression Pipeline

1. **FFT Transformation**
   Convert image from spatial domain to frequency domain
2. **Frequency Filtering**
   Apply low-pass filter to remove high-frequency components
3. **Inverse FFT**
   Reconstruct image back to spatial domain
4. **Quantization**
   Reduce bit depth for additional compression

---

### Mathematical Foundation

**2D Fourier Transform**
[
F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) e^{-j2\pi(ux/M + vy/N)}
]

**Mean Squared Error (MSE)**
[
MSE = \frac{1}{MN} \sum_{i,j} [I(i,j) - K(i,j)]^2
]

**Peak Signal-to-Noise Ratio (PSNR)**
[
PSNR = 10 \cdot \log_{10} \left( \frac{MAX_I^2}{MSE} \right)
]

---

## 🧪 Test Suite

Run all tests using:

```bash
python test_compression.py
```

Test coverage includes:

* Cutoff ratio variation
* Filter type comparison
* Visualization validation
* Metric consistency checks

---

## ⚠️ Limitations

* Currently optimized for **grayscale images**
* Compression artifacts increase at aggressive cutoff ratios
* Computationally intensive for very large images

---

## 🔮 Future Enhancements

* Color image compression
* Real-time compression preview
* Parallel FFT processing
* Deep learning–based adaptive filter selection

---

## 👥 Project Team

* **Hashir Imran** (242101025)
* **Zafran Ajab** (242101002)
* **Taimoor Noor Khan** (242101027)

**Instructor:** Sir Engr. Muhammad Waqas
**Institute:** Institute of Space Technology, KICSIT Kahuta Campus

---

## 📚 References

* Gonzalez, R. C., & Woods, R. E. *Digital Image Processing*
* Oppenheim, A. V., & Schafer, R. W. *Discrete-Time Signal Processing*
* JPEG Standard (ISO/IEC 10918-1)

---

> *Compression is not about loss—it’s about knowing what truly matters.*

```

---

```
