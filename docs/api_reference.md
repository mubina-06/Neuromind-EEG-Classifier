# API Reference

## Core Modules

### Data Processing (`src.data`)

#### `preprocessing.py`
- **`preprocess_raw(raw, filter_params)`**: Apply bandpass and notch filtering to raw EEG data
- **`epoch_raw(raw, task_labels, epoch_length)`**: Segment continuous EEG into epochs
- **`epoch_to_spectrogram(epoch, fs, method)`**: Convert EEG epoch to STFT spectrogram
- **`load_physionet_data(subject_ids, tasks)`**: Download and load PhysioNet EEG data

#### `dataset.py`
- **`EEGSpectrogramDataset`**: PyTorch dataset for EEG spectrograms with augmentation
- **`get_data_loaders(data_path, batch_size, split)`**: Create train/val/test data loaders

### Models (`src.models`)

#### `model.py`
- **`build_model(model_name, num_classes, pretrained)`**: Build ResNet18 or EfficientNet-B0
- **`build_ensemble(models, weights)`**: Create ensemble model from multiple models
- **`get_last_conv_layer(model)`**: Get final convolutional layer for Grad-CAM

### Training (`src.training`)

#### `trainer.py`
- **`train_model(model, train_loader, val_loader, epochs)`**: Complete training pipeline
- **`validate_model(model, val_loader)`**: Validation loop with metrics

#### `evaluate.py`
- **`evaluate_model(model, test_loader)`**: Full model evaluation with plots
- **`generate_confusion_matrix(y_true, y_pred)`**: Create confusion matrix visualization
- **`plot_roc_curves(y_true, y_pred_proba)`**: Generate ROC curves for all classes

#### `calibration.py`
- **`temperature_scaling(model, val_loader)`**: Calibrate model confidence scores
- **`reliability_diagram(y_true, y_pred_proba)`**: Plot calibration reliability diagram

### Utilities (`src.utils`)

#### `gradcam.py`
- **`GradCAM(model, target_layer)`**: Gradient-weighted Class Activation Mapping
- **`generate_heatmap(input_tensor, class_idx)`**: Generate attention heatmap
- **`overlay_heatmap(image, heatmap, alpha)`**: Overlay heatmap on original image

## Usage Examples

### Basic Model Training
```python
from src.models.model import build_model
from src.training.trainer import train_model
from src.data.dataset import get_data_loaders

# Load data
train_loader, val_loader, test_loader = get_data_loaders(
    data_path="data/spectrograms",
    batch_size=32,
    split=[0.7, 0.15, 0.15]
)

# Build model
model = build_model("resnet18", num_classes=3, pretrained=True)

# Train
best_accuracy = train_model(model, train_loader, val_loader, epochs=50)
```

### Grad-CAM Visualization
```python
from src.utils.gradcam import GradCAM
from src.models.model import build_model, get_last_conv_layer

# Load model
model = build_model("resnet18", num_classes=3)
model.load_state_dict(torch.load("models/best_resnet18.pth"))

# Create Grad-CAM
target_layer = get_last_conv_layer(model)
gradcam = GradCAM(model, target_layer)

# Generate heatmap
heatmap = gradcam.generate_heatmap(input_tensor, class_idx=0)
```

### Model Evaluation
```python
from src.training.evaluate import evaluate_model

# Evaluate model
accuracy, f1_score, classification_report = evaluate_model(
    model_path="models/best_resnet18.pth",
    data_path="data/spectrograms",
    output_dir="results"
)
```