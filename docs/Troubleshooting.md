# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Problem: `pip install -r requirements.txt` fails
**Symptoms:**
- Package installation errors
- Dependency conflicts
- Build failures

**Solutions:**
```bash
# 1. Update pip and setuptools
pip install --upgrade pip setuptools wheel

# 2. Use specific Python version (3.8-3.11 recommended)
python3.9 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# 3. Install dependencies individually if bulk install fails
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit
pip install mne scipy numpy matplotlib seaborn plotly pandas pillow tqdm

# 4. Clear pip cache if corrupted
pip cache purge
```

#### Problem: PyTorch CUDA installation issues
**Symptoms:**
- CUDA runtime errors
- GPU not detected
- Version mismatches

**Solutions:**
```bash
# Check CUDA version
nvidia-smi

# Install correct PyTorch version for your CUDA
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### Problem: MNE-Python installation fails
**Symptoms:**
- Missing system dependencies
- Compilation errors

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential python3-dev

# macOS
brew install gcc
xcode-select --install

# Windows (use Anaconda)
conda install -c conda-forge mne

# Alternative: Use pip with pre-compiled wheels
pip install --only-binary=all mne
```

### Application Runtime Issues

#### Problem: Streamlit app won't start
**Symptoms:**
- `streamlit run` command fails
- Port already in use
- Module import errors

**Solutions:**
```bash
# 1. Check if port 8501 is in use
lsof -i :8501  # Linux/Mac
netstat -ano | findstr :8501  # Windows

# 2. Kill existing Streamlit processes
pkill -f streamlit  # Linux/Mac
taskkill /f /im streamlit.exe  # Windows

# 3. Try different port
streamlit run src/app.py --server.port 8502

# 4. Check Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/neuromind-eeg-classifier/src"

# 5. Verbose logging for debugging
streamlit run src/app.py --logger.level=debug
```

#### Problem: Model files not found
**Symptoms:**
- "FileNotFoundError: Model not found"
- Application crashes on model loading

**Solutions:**
```bash
# 1. Check model directory structure
ls -la models/
# Should contain: best_resnet18.pth, best_efficientnet_b0.pth, etc.

# 2. Download models if missing
python scripts/download_data.py --models-only

# 3. Check file permissions
chmod 644 models/*.pth

# 4. Verify model file integrity
python -c "
import torch
model = torch.load('models/best_resnet18.pth', map_location='cpu')
print('Model loaded successfully')
"
```

#### Problem: Out of memory errors
**Symptoms:**
- "CUDA out of memory"
- "RuntimeError: DefaultCPUAllocator"
- Application crashes during inference

**Solutions:**
```bash
# 1. Reduce batch size in configuration
# Edit src/app.py, change BATCH_SIZE = 1

# 2. Use CPU instead of GPU
# Set device = 'cpu' in model loading

# 3. Clear GPU memory cache
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleared')
"

# 4. Monitor memory usage
nvidia-smi  # For GPU
htop        # For system memory
```

### Data Processing Issues

#### Problem: EEG file upload fails
**Symptoms:**
- "Unsupported file format"
- Upload stuck or crashes
- Corrupted file errors

**Solutions:**
```bash
# 1. Check supported formats
# Supported: .edf, .bdf, .gdf, .set (EEGLAB)

# 2. Validate file integrity
python -c "
import mne
raw = mne.io.read_raw_edf('your_file.edf', preload=False)
print(f'Channels: {len(raw.ch_names)}')
print(f'Duration: {raw.times[-1]:.1f}s')
print(f'Sampling rate: {raw.info[\"sfreq\"]}Hz')
"

# 3. Convert unsupported formats
# For .cnt files (Neuroscan)
python -c "
import mne
raw = mne.io.read_raw_cnt('file.cnt', preload=True)
raw.save('file.fif')  # Convert to MNE format
"

# 4. File size limits (adjust in Streamlit config)
# Create .streamlit/config.toml:
[server]
maxUploadSize = 200  # MB
```

#### Problem: Preprocessing fails
**Symptoms:**
- "Filtering failed"
- "No events found"
- "Epoch extraction error"

**Solutions:**
```python
# 1. Check EEG data properties
import mne
raw = mne.io.read_raw_edf('file.edf')
print(f"Channels: {raw.ch_names}")
print(f"Sample rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} seconds")

# 2. Manual preprocessing with error handling
try:
    raw.filter(4, 45, fir_design='firwin')
    print("Filtering successful")
except Exception as e:
    print(f"Filtering failed: {e}")
    # Try alternative filter
    raw.filter(4, 45, method='iir')

# 3. Check for bad channels
raw.plot()  # Visual inspection
raw.info['bads'] = ['Ch1', 'Ch5']  # Mark bad channels
```

### Model Performance Issues

#### Problem: Low classification accuracy
**Symptoms:**
- Predictions seem random
- Confidence scores very low
- Inconsistent results

**Solutions:**
```python
# 1. Verify model is loaded correctly
import torch
model = torch.load('models/best_resnet18.pth')
model.eval()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# 2. Check input preprocessing
# Ensure spectrograms are in correct format (224x224x3)
# Verify normalization matches training

# 3. Test with known good data
python scripts/evaluate_models.py --test-sample

# 4. Re-download models if corrupted
python scripts/download_data.py --force-redownload
```

#### Problem: Grad-CAM visualization not working
**Symptoms:**
- Empty heatmaps
- Visualization crashes
- No attention maps generated

**Solutions:**
```python
# 1. Check target layer compatibility
from src.utils.gradcam import GradCAM
# Ensure target layer has gradients enabled

# 2. Verify model architecture
model = build_model('resnet18')
print(model)  # Check layer names

# 3. Test with simplified input
import torch
test_input = torch.randn(1, 3, 224, 224)
gradcam = GradCAM(model, model.layer4[-1])
heatmap = gradcam.generate_heatmap(test_input, class_idx=0)
```

### Docker Issues

#### Problem: Docker build fails
**Symptoms:**
- Package installation errors in container
- Layer build failures
- Permission denied errors

**Solutions:**
```bash
# 1. Build with verbose output
docker build --no-cache --progress=plain -t neuromind .

# 2. Check Docker daemon and permissions
sudo usermod -aG docker $USER
sudo systemctl restart docker

# 3. Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory (min 4GB)

# 4. Clean Docker cache
docker system prune -a
```

#### Problem: Container runs but app inaccessible
**Symptoms:**
- Container starts successfully
- Cannot access http://localhost:8501
- Connection refused errors

**Solutions:**
```bash
# 1. Check port mapping
docker ps
# Should show: 0.0.0.0:8501->8501/tcp

# 2. Check container logs
docker logs <container_name>

# 3. Test container connectivity
docker exec -it <container_name> curl http://localhost:8501

# 4. Verify firewall settings
sudo ufw allow 8501  # Linux
# Check Windows Firewall settings
```

### Performance Optimization

#### Problem: Slow inference times
**Symptoms:**
- Long wait times for predictions
- UI becomes unresponsive
- Timeout errors

**Solutions:**
```python
# 1. Profile model inference
import time
import torch

model.eval()
with torch.no_grad():
    start_time = time.time()
    output = model(input_tensor)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.3f}s")

# 2. Optimize model for inference
# Use torch.jit.script for faster execution
scripted_model = torch.jit.script(model)

# 3. Enable optimization flags
torch.backends.cudnn.benchmark = True  # For consistent input sizes
torch.backends.cudnn.deterministic = False

# 4. Use smaller models for real-time applications
# Switch to EfficientNet-B0 instead of Ensemble
```

## Environment-Specific Issues

### Windows Issues

#### Problem: Path separator issues
```python
# Use pathlib for cross-platform paths
from pathlib import Path
model_path = Path("models") / "best_model.pth"
```

#### Problem: PowerShell execution policy
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS Issues

#### Problem: SSL certificate issues
```bash
# Update certificates
/Applications/Python\ 3.x/Install\ Certificates.command

# Or install certificates manually
pip install --upgrade certifi
```

### Linux Issues

#### Problem: Display backend errors
```bash
# Set matplotlib backend for headless systems
export MPLBACKEND=Agg

# Or in Python
import matplotlib
matplotlib.use('Agg')
```

## Debugging Tools and Commands

### Streamlit Debugging
```bash
# Run with debug logging
streamlit run src/app.py --logger.level=debug

# Check Streamlit configuration
streamlit config show

# Clear Streamlit cache
streamlit cache clear
```

### Python Debugging
```python
# Add to problematic code sections
import pdb; pdb.set_trace()  # Interactive debugger

# Or use logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### System Monitoring
```bash
# Monitor resource usage
htop          # System overview
nvidia-smi    # GPU monitoring
iostat -x 1   # Disk I/O
netstat -tulpn # Network connections
```

## Getting Help

### Log Collection
Before reporting issues, collect relevant logs:

```bash
# Application logs
streamlit run src/app.py 2>&1 | tee streamlit.log

# Docker logs
docker logs neuromind-app > docker.log

# System information
python -c "
import sys, torch, streamlit, mne
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Streamlit: {streamlit.__version__}')
print(f'MNE: {mne.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Support Channels
1. **GitHub Issues**: Report bugs and feature requests
2. **Documentation**: Check docs/ folder for detailed guides
3. **Community**: Streamlit and PyTorch communities
4. **Stack Overflow**: Tag with `streamlit`, `pytorch`, `eeg`

### Contributing Fixes
Found a solution? Consider contributing:
1. Fork the repository
2. Create a fix branch
3. Test thoroughly
4. Submit pull request with clear description

This troubleshooting guide covers the most common issues. For specific problems not covered here, please check the GitHub issues or create a new issue with detailed error logs and system information.