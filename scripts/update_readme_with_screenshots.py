#!/usr/bin/env python3
"""
Script to update README.md with screenshot gallery after screenshots are added.

Run this script after adding screenshots to assets/demo_screenshots/
"""

import os
from pathlib import Path

def check_screenshots_exist():
    """Check if all expected screenshots exist."""
    screenshot_dir = Path("assets/demo_screenshots")
    
    expected_screenshots = [
        "01_model_performance_dashboard.png",
        "02_training_progress_analysis.png", 
        "03_system_information_panel.png",
        "04_advanced_model_analysis.png",
        "05_feature_importance_analysis.png",
        "06_model_selection_recommendations.png",
        "07_main_application_interface.png",
        "08_ai_brain_state_analysis.png",
        "09_gradcam_explainable_ai.png"
    ]
    
    existing = []
    missing = []
    
    for screenshot in expected_screenshots:
        screenshot_path = screenshot_dir / screenshot
        if screenshot_path.exists():
            existing.append(screenshot)
        else:
            missing.append(screenshot)
    
    return existing, missing

def generate_screenshot_section():
    """Generate the screenshot gallery section for README.md"""
    
    existing, missing = check_screenshots_exist()
    
    if not existing:
        print("❌ No screenshots found. Please add screenshots first using the SCREENSHOT_GUIDE.md")
        return None
    
    if missing:
        print(f"⚠️  Missing screenshots: {', '.join(missing)}")
        print("Some screenshots will not be displayed.")
    
    print(f"✅ Found {len(existing)} screenshots")
    
    # Generate markdown section
    screenshot_section = """
## 🖥️ Application Screenshots

### Main Application Interface
*Professional EEG analysis platform with real-time signal processing*

![NeuroMind Main Interface](assets/demo_screenshots/07_main_application_interface.png)

### AI Brain State Classification  
*Real-time mental state detection with confidence scoring*

![Brain State Analysis](assets/demo_screenshots/08_ai_brain_state_analysis.png)

### Model Performance Dashboard
*Comprehensive model metrics and comparison analytics*

![Performance Dashboard](assets/demo_screenshots/01_model_performance_dashboard.png)

### Explainable AI Visualization
*Grad-CAM attention maps showing AI decision-making process*

![Grad-CAM Visualization](assets/demo_screenshots/09_gradcam_explainable_ai.png)

### Training Progress Analysis
*Real-time training curves and detailed performance metrics*

<details>
<summary>📊 Click to view detailed analytics</summary>

![Training Progress](assets/demo_screenshots/02_training_progress_analysis.png)

</details>

### Advanced Model Analysis
*Architecture comparison and performance trade-off analysis*

<details>
<summary>🔬 Click to view advanced analysis</summary>

![Advanced Analysis](assets/demo_screenshots/04_advanced_model_analysis.png)

</details>

### Feature Importance & EEG Analysis
*Scientific analysis of EEG frequency bands and electrode importance*

<details>
<summary>🧠 Click to view feature analysis</summary>

![Feature Analysis](assets/demo_screenshots/05_feature_importance_analysis.png)

</details>

### Smart Model Recommendations
*AI-powered model selection based on use case requirements*

<details>
<summary>💡 Click to view model recommendations</summary>

![Model Recommendations](assets/demo_screenshots/06_model_selection_recommendations.png)

</details>

"""
    
    return screenshot_section

def update_readme():
    """Update README.md with screenshot gallery."""
    
    screenshot_section = generate_screenshot_section()
    if not screenshot_section:
        return False
    
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        print("❌ README.md not found!")
        return False
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find insertion point (after the "Technical Architecture" section)
    insertion_markers = [
        "## 🏗️ Technical Architecture",
        "## 📊 Dataset & Performance", 
        "## 📁 Project Structure"
    ]
    
    insertion_point = None
    for marker in insertion_markers:
        if marker in content:
            insertion_point = content.find(marker)
            break
    
    if insertion_point is None:
        # If no markers found, add before "Installation" section
        if "## 🚀 Quick Start" in content:
            insertion_point = content.find("## 🚀 Quick Start")
        else:
            print("⚠️  Could not find suitable insertion point. Adding to end of file.")
            insertion_point = len(content)
    
    # Insert screenshot section
    if insertion_point < len(content):
        updated_content = (
            content[:insertion_point] + 
            screenshot_section + 
            "\n" + 
            content[insertion_point:]
        )
    else:
        updated_content = content + "\n" + screenshot_section
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✅ Successfully updated README.md with screenshot gallery!")
    print(f"📸 Added {len(check_screenshots_exist()[0])} screenshots to documentation")
    
    return True

def main():
    """Main function to update README with screenshots."""
    
    print("🧠 NeuroMind Screenshot Integration")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("README.md").exists() or not Path("assets/demo_screenshots").exists():
        print("❌ Please run this script from the project root directory")
        print("   (where README.md and assets/ folder are located)")
        return 1
    
    # Check screenshots and update README
    existing, missing = check_screenshots_exist()
    
    print(f"📊 Screenshot Status:")
    print(f"   ✅ Found: {len(existing)} screenshots")
    if missing:
        print(f"   ❌ Missing: {len(missing)} screenshots")
        for screenshot in missing[:3]:  # Show first 3 missing
            print(f"      - {screenshot}")
        if len(missing) > 3:
            print(f"      - ... and {len(missing)-3} more")
    
    if not existing:
        print("\n📋 Next Steps:")
        print("1. Add screenshots to assets/demo_screenshots/ using SCREENSHOT_GUIDE.md")
        print("2. Run this script again to update README.md")
        return 0
    
    # Update README
    success = update_readme()
    
    if success:
        print("\n🎉 README.md successfully updated!")
        print("\n📋 Next Steps:")
        print("1. Review the updated README.md") 
        print("2. Commit changes to git")
        print("3. Push to GitHub to see screenshots in repository")
        print("4. Your repository is now 100% portfolio-ready! 🚀")
    
    return 0

if __name__ == "__main__":
    exit(main())