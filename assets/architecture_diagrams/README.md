# Architecture Diagrams

This directory contains system architecture and technical diagrams for the NeuroMind EEG Classifier project.

## 📊 Available Diagrams

### 1. **System Architecture** (`system_architecture.png`)
- **Overview**: Complete system architecture from data input to classification output
- **Components**: PhysioNet database, preprocessing pipeline, CNN models, explainable AI, web interface
- **Data Flow**: Shows how EEG signals flow through the entire system
- **Technologies**: All major components and their relationships

### 2. **Data Processing Pipeline** (`data_flow_pipeline.png`) 
- **EEG Processing**: Step-by-step signal processing workflow
- **Technical Details**: Filtering, epoching, STFT transformation, CNN classification
- **Visualizations**: Sample EEG signals, spectrograms, and classification outputs
- **Parameters**: Specific technical parameters for each processing stage

### 3. **Model Architecture Comparison** (`model_architecture_comparison.png`)
- **Three CNN Models**: ResNet18, EfficientNet-B0, and Ensemble comparison
- **Performance Metrics**: Accuracy, speed, memory usage, parameter count
- **Use Case Recommendations**: Which model to use for different scenarios
- **Trade-off Analysis**: Performance vs efficiency considerations

## 🎯 Usage Guidelines

### **In Documentation**
- Use `system_architecture.png` in README.md for overall project overview
- Include `data_flow_pipeline.png` in technical reports and user guides  
- Add `model_architecture_comparison.png` to performance analysis sections

### **In Presentations**
- **Academic**: All diagrams suitable for research presentations
- **Industry**: Focus on system architecture and model comparison
- **Technical**: Data flow pipeline for detailed technical discussions

### **In Portfolio**
- Demonstrate system design and architecture skills
- Show understanding of end-to-end ML pipeline development
- Highlight technical depth and professional diagram creation

## 🎨 Design Principles

### **Visual Consistency**
- **Dark Theme**: Professional dark background matching application UI
- **Color Coding**: Consistent colors for different system components
- **Typography**: Clean, readable fonts with proper hierarchy
- **Branding**: NeuroMind pink accent color (#FF6B9D) for titles

### **Technical Accuracy**
- **Real Specifications**: All numbers and parameters match actual implementation
- **Scientific Notation**: Proper EEG terminology and measurement units
- **Flow Accuracy**: Diagrams accurately represent actual data and control flow
- **Component Relationships**: Correct representation of system dependencies

## 📐 Technical Specifications

### **Image Properties**
- **Format**: PNG with transparent backgrounds where applicable
- **Resolution**: 300 DPI for print quality
- **Dimensions**: Optimized for both web and print display
- **Color Space**: RGB for digital display

### **Component Color Scheme**
```
Data Processing: #4ECDC4 (Teal)
Signal Processing: #45B7D1 (Blue)
Machine Learning: #96CEB4 (Green) 
User Interface: #FFEAA7 (Yellow)
Output/Results: #DDA0DD (Purple)
Accent/Branding: #FF6B9D (Pink)
```

## 🔄 Updating Diagrams

To regenerate or update diagrams:

```bash
# Run the diagram generation script
python scripts/generate_diagrams.py

# This will create/update all diagrams in this directory
```

### **When to Update**
- Architecture changes in the system
- Performance metrics updates
- New model implementations
- UI/UX modifications
- Additional feature implementations

## 📚 Integration with Documentation

These diagrams are referenced in:

- **README.md**: System architecture overview
- **Technical Report**: Detailed methodology and architecture
- **User Guide**: System overview for new users
- **API Documentation**: Component relationships and data flow
- **Presentation Materials**: Conference and demo presentations

## 🎓 Educational Value

### **For Students/Researchers**
- Complete ML pipeline visualization
- EEG signal processing workflow
- CNN architecture comparison methodology
- System design best practices

### **For Developers**
- Microservices architecture example
- Data processing pipeline design
- Model deployment architecture
- Web application integration patterns

### **For Recruiters**
- System design capabilities
- Technical documentation skills
- End-to-end project architecture
- Professional diagram creation abilities

## 🔗 External Tools Used

- **Python matplotlib**: Core diagram generation
- **Scientific visualization**: EEG signal and spectrogram representations
- **Architecture patterns**: Standard software architecture diagram conventions
- **Color theory**: Professional color schemes for technical diagrams