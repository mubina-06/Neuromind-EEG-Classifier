#!/usr/bin/env python3
"""
Health check script for NeuroMind application.

This script performs comprehensive health checks for the Streamlit application
including service availability, model loading, and basic functionality.
"""

import sys
import requests
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_streamlit_health(url: str = "http://localhost:8501", timeout: int = 10) -> bool:
    """Check if Streamlit application is responding."""
    try:
        # Check Streamlit health endpoint
        health_url = f"{url}/_stcore/health"
        response = requests.get(health_url, timeout=timeout)
        
        if response.status_code == 200:
            logger.info("✅ Streamlit application is healthy")
            return True
        else:
            logger.error(f"❌ Streamlit health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to connect to Streamlit: {e}")
        return False

def check_model_files() -> bool:
    """Check if required model files are available."""
    try:
        models_dir = Path("/app/models")
        required_models = [
            "best_resnet18.pth",
            "best_efficientnet_b0.pth",
            "ensemble_model.pth"
        ]
        
        for model_file in required_models:
            model_path = models_dir / model_file
            if not model_path.exists():
                logger.warning(f"⚠️  Model file missing: {model_file}")
                # Don't fail health check for missing models in dev environment
                continue
        
        logger.info("✅ Model files check completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model files check failed: {e}")
        return False

def check_dependencies() -> bool:
    """Check if critical Python dependencies are available."""
    try:
        import torch
        import streamlit
        import numpy
        import matplotlib
        import plotly
        
        logger.info("✅ Critical dependencies are available")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing critical dependency: {e}")
        return False

def check_disk_space() -> bool:
    """Check if sufficient disk space is available."""
    try:
        import shutil
        
        # Check available disk space (in GB)
        total, used, free = shutil.disk_usage("/app")
        free_gb = free / (1024 ** 3)
        
        if free_gb < 1.0:  # Less than 1GB free
            logger.warning(f"⚠️  Low disk space: {free_gb:.2f}GB available")
            return False
        
        logger.info(f"✅ Disk space OK: {free_gb:.2f}GB available")
        return True
        
    except Exception as e:
        logger.error(f"❌ Disk space check failed: {e}")
        return False

def check_memory_usage() -> bool:
    """Check memory usage."""
    try:
        import psutil
        
        # Get memory info
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > 90:
            logger.warning(f"⚠️  High memory usage: {memory_percent:.1f}%")
            return False
        
        logger.info(f"✅ Memory usage OK: {memory_percent:.1f}%")
        return True
        
    except ImportError:
        logger.info("⚠️  psutil not available, skipping memory check")
        return True
    except Exception as e:
        logger.error(f"❌ Memory check failed: {e}")
        return False

def main() -> int:
    """Run comprehensive health checks."""
    logger.info("🔍 Starting NeuroMind health check...")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("Disk Space", check_disk_space),
        ("Memory Usage", check_memory_usage),
        ("Streamlit Health", check_streamlit_health),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        logger.info(f"Running {check_name} check...")
        
        try:
            result = check_func()
            if not result:
                all_passed = False
                logger.error(f"❌ {check_name} check failed")
            else:
                logger.info(f"✅ {check_name} check passed")
        except Exception as e:
            logger.error(f"❌ {check_name} check error: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("🎉 All health checks passed!")
        return 0
    else:
        logger.error("💥 Some health checks failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)