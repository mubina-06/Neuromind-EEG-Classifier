#!/usr/bin/env python3
"""
Performance Monitoring Script for NeuroMind.

Monitors model performance, system resources, and provides alerts for production deployment.
"""

import time
import psutil
import torch
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model import build_model, load_checkpoint
from src.data.preprocessing import epoch_to_spectrogram

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor NeuroMind performance metrics in production."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize performance monitor."""
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_history = []
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load model for performance testing."""
        try:
            if "resnet" in model_path.lower():
                arch = "resnet18"
            elif "efficient" in model_path.lower():
                arch = "efficientnet_b0"
            else:
                arch = "resnet18"  # default
            
            self.model = load_checkpoint(arch, device=self.device)
            logger.info(f"Loaded model: {arch} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to building new model
            self.model = build_model(arch="resnet18", pretrained=False)
            self.model.to(self.device)
    
    def measure_inference_time(self, batch_size: int = 1, num_trials: int = 10) -> Dict[str, float]:
        """Measure model inference time."""
        if not self.model:
            logger.error("No model loaded")
            return {}
        
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy_input)
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_trials):
                start_time = time.perf_counter()
                _ = self.model(dummy_input)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        return {
            'mean_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'batch_size': batch_size,
            'device': str(self.device)
        }
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure system and GPU memory usage."""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info.update({
            'system_memory_total_gb': system_memory.total / (1024**3),
            'system_memory_used_gb': system_memory.used / (1024**3),
            'system_memory_percent': system_memory.percent
        })
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_memory_cached_gb': torch.cuda.memory_reserved() / (1024**3),
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        
        return memory_info
    
    def measure_throughput(self, duration_seconds: int = 60) -> Dict[str, float]:
        """Measure model throughput (samples per second)."""
        if not self.model:
            logger.error("No model loaded")
            return {}
        
        self.model.eval()
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        
        start_time = time.time()
        count = 0
        
        with torch.no_grad():
            while time.time() - start_time < duration_seconds:
                _ = self.model(dummy_input)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                count += 1
        
        elapsed_time = time.time() - start_time
        throughput = count / elapsed_time
        
        return {
            'throughput_samples_per_second': throughput,
            'total_samples': count,
            'duration_seconds': elapsed_time
        }
    
    def measure_preprocessing_performance(self) -> Dict[str, float]:
        """Measure preprocessing performance."""
        # Create synthetic EEG epoch
        np.random.seed(42)
        epoch_data = np.random.randn(64, 640)  # 64 channels, 4 seconds at 160Hz
        sfreq = 160.0
        
        # Measure spectrogram generation time
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            _ = epoch_to_spectrogram(epoch_data, sfreq)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'preprocessing_mean_time': np.mean(times),
            'preprocessing_std_time': np.std(times),
            'preprocessing_min_time': np.min(times),
            'preprocessing_max_time': np.max(times)
        }
    
    def check_system_health(self) -> Dict[str, any]:
        """Check overall system health."""
        health = {}
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        health['cpu_usage_percent'] = cpu_percent
        health['cpu_count'] = psutil.cpu_count()
        
        # Memory usage
        memory = psutil.virtual_memory()
        health['memory_usage_percent'] = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        health['disk_usage_percent'] = (disk.used / disk.total) * 100
        
        # Temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                health['temperature_celsius'] = temps.get('coretemp', [{}])[0].get('current', None)
        except:
            health['temperature_celsius'] = None
        
        # Load average (Unix systems)
        try:
            health['load_average'] = psutil.getloadavg()
        except:
            health['load_average'] = None
        
        return health
    
    def run_comprehensive_benchmark(self) -> Dict[str, any]:
        """Run comprehensive performance benchmark."""
        logger.info("Starting comprehensive benchmark...")
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            benchmark_results['cuda_version'] = torch.version.cuda
            benchmark_results['gpu_name'] = torch.cuda.get_device_name()
        
        # System health
        logger.info("Checking system health...")
        benchmark_results['system_health'] = self.check_system_health()
        
        # Memory usage
        logger.info("Measuring memory usage...")
        benchmark_results['memory_usage'] = self.measure_memory_usage()
        
        # Preprocessing performance
        logger.info("Measuring preprocessing performance...")
        benchmark_results['preprocessing_performance'] = self.measure_preprocessing_performance()
        
        if self.model:
            # Inference time (different batch sizes)
            logger.info("Measuring inference times...")
            benchmark_results['inference_performance'] = {}
            for batch_size in [1, 4, 8, 16]:
                batch_results = self.measure_inference_time(batch_size=batch_size)
                benchmark_results['inference_performance'][f'batch_{batch_size}'] = batch_results
            
            # Throughput
            logger.info("Measuring throughput...")
            benchmark_results['throughput'] = self.measure_throughput(duration_seconds=30)
        
        # Store results
        self.metrics_history.append(benchmark_results)
        
        logger.info("Benchmark completed!")
        return benchmark_results
    
    def save_results(self, filepath: str) -> None:
        """Save benchmark results to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def generate_report(self, results: Dict[str, any]) -> str:
        """Generate human-readable performance report."""
        report = []
        report.append("=" * 60)
        report.append("NeuroMind Performance Benchmark Report")
        report.append("=" * 60)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Device: {results['device']}")
        report.append(f"PyTorch Version: {results['pytorch_version']}")
        
        if results['cuda_available']:
            report.append(f"CUDA Version: {results.get('cuda_version', 'N/A')}")
            report.append(f"GPU: {results.get('gpu_name', 'N/A')}")
        
        report.append("\n" + "-" * 40)
        report.append("SYSTEM HEALTH")
        report.append("-" * 40)
        health = results['system_health']
        report.append(f"CPU Usage: {health['cpu_usage_percent']:.1f}%")
        report.append(f"Memory Usage: {health['memory_usage_percent']:.1f}%")
        report.append(f"Disk Usage: {health['disk_usage_percent']:.1f}%")
        
        if health.get('temperature_celsius'):
            report.append(f"CPU Temperature: {health['temperature_celsius']:.1f}°C")
        
        report.append("\n" + "-" * 40)
        report.append("MEMORY USAGE")
        report.append("-" * 40)
        memory = results['memory_usage']
        report.append(f"System Memory: {memory['system_memory_used_gb']:.2f}/{memory['system_memory_total_gb']:.2f} GB")
        
        if 'gpu_memory_allocated_gb' in memory:
            report.append(f"GPU Memory: {memory['gpu_memory_allocated_gb']:.2f}/{memory['gpu_memory_total_gb']:.2f} GB")
        
        report.append("\n" + "-" * 40)
        report.append("PREPROCESSING PERFORMANCE")
        report.append("-" * 40)
        prep = results['preprocessing_performance']
        report.append(f"Mean Time: {prep['preprocessing_mean_time']*1000:.2f}ms")
        report.append(f"Std Dev: {prep['preprocessing_std_time']*1000:.2f}ms")
        
        if 'inference_performance' in results:
            report.append("\n" + "-" * 40)
            report.append("MODEL INFERENCE PERFORMANCE")
            report.append("-" * 40)
            
            for batch_key, batch_data in results['inference_performance'].items():
                batch_size = batch_data['batch_size']
                mean_time = batch_data['mean_inference_time'] * 1000
                report.append(f"Batch Size {batch_size}: {mean_time:.2f}ms ± {batch_data['std_inference_time']*1000:.2f}ms")
            
            if 'throughput' in results:
                throughput = results['throughput']
                report.append(f"\nThroughput: {throughput['throughput_samples_per_second']:.2f} samples/second")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def main():
    """Main function to run performance monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroMind Performance Monitor")
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="logs/benchmark_results.json", 
                       help="Output file for results")
    parser.add_argument("--duration", type=int, default=30,
                       help="Throughput test duration in seconds")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=300,
                       help="Interval between measurements in seconds (continuous mode)")
    
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize monitor
    monitor = PerformanceMonitor(args.model_path)
    
    if args.continuous:
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        
        try:
            while True:
                results = monitor.run_comprehensive_benchmark()
                
                # Generate and display report
                report = monitor.generate_report(results)
                print(report)
                
                # Save results
                monitor.save_results(args.output)
                
                # Wait for next interval
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    
    else:
        # Single benchmark run
        results = monitor.run_comprehensive_benchmark()
        
        # Generate and display report
        report = monitor.generate_report(results)
        print(report)
        
        # Save results
        monitor.save_results(args.output)


if __name__ == "__main__":
    main()