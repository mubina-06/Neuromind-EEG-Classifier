#!/usr/bin/env python3
"""
Sample Image Dataset Extractor
==============================
Extract sample spectrogram images for testing the EEG classifier
"""

import os
import shutil
from pathlib import Path
import random

def create_sample_dataset():
    """Create a sample dataset with a few images from each class"""
    
    # Source directories
    source_dir = Path("data/spectrograms")
    
    # Create sample directory
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Clear existing samples
    for file in sample_dir.glob("*"):
        if file.is_file():
            file.unlink()
    
    print("🧠 EEG Spectrogram Sample Dataset Creator")
    print("=" * 50)
    
    # Copy samples from each class
    classes = ["focused", "relaxed", "stressed"]
    samples_per_class = 5
    
    total_copied = 0
    
    for class_name in classes:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            print(f"❌ {class_name} directory not found")
            continue
            
        # Get all images in this class
        images = list(class_dir.glob("*.png"))
        
        if len(images) == 0:
            print(f"❌ No images found in {class_name}")
            continue
            
        # Randomly select sample images
        sample_images = random.sample(images, min(samples_per_class, len(images)))
        
        print(f"📁 {class_name.upper()}: {len(images)} total images")
        
        # Copy sample images
        for i, img_path in enumerate(sample_images):
            new_name = f"{class_name}_{i+1:02d}_{img_path.name}"
            dest_path = sample_dir / new_name
            shutil.copy2(img_path, dest_path)
            print(f"   ✅ Copied: {new_name}")
            total_copied += 1
    
    print("=" * 50)
    print(f"✅ Created sample dataset with {total_copied} images")
    print(f"📂 Location: {sample_dir.absolute()}")
    print("\n💡 Usage:")
    print("   1. Use these images to test the Batch Processing tab")
    print("   2. Upload them in the Streamlit app")
    print("   3. See how the AI classifies different mental states")
    
    return sample_dir

def show_dataset_info():
    """Show information about the complete dataset"""
    
    source_dir = Path("data/spectrograms")
    
    print("🧠 Complete EEG Spectrogram Dataset")
    print("=" * 50)
    
    total_images = 0
    
    for class_name in ["focused", "relaxed", "stressed"]:
        class_dir = source_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.png"))
            count = len(images)
            total_images += count
            
            # Show sample filenames
            sample_files = images[:3] if images else []
            sample_names = [f.name for f in sample_files]
            
            print(f"📁 {class_name.upper()}: {count} images")
            if sample_names:
                print(f"   Examples: {', '.join(sample_names)}")
        else:
            print(f"❌ {class_name} directory not found")
    
    print("=" * 50)
    print(f"📊 Total Images: {total_images}")
    print(f"📂 Location: {source_dir.absolute()}")
    
    # Show file structure
    print("\n📋 Dataset Structure:")
    print("data/spectrograms/")
    print("├── focused/     (🎯 Motor imagery - concentration)")
    print("├── relaxed/     (😌 Eyes open rest - relaxed)")
    print("└── stressed/    (😰 Motor execution - stressed)")

def main():
    """Main function"""
    
    print("Choose an option:")
    print("1. Show complete dataset info")
    print("2. Create sample dataset for testing")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice in ["1", "3"]:
            show_dataset_info()
            print()
        
        if choice in ["2", "3"]:
            create_sample_dataset()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()