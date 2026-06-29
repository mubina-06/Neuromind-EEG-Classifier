# Deployment Guide

This guide covers various deployment options for the NeuroMind EEG Classifier.

## Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- Git
- 8GB RAM (16GB recommended)
- NVIDIA GPU (optional, for faster training)

## Local Development

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/neuromind-eeg-classifier.git
cd neuromind-eeg-classifier

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data and Models

```bash
# Run the complete setup pipeline
python scripts/download_data.py

# Or run individual components
python scripts/download_data.py --verify-only  # Check installation
python scripts/download_data.py --subjects 5   # Fewer subjects for testing
```

### 3. Run the Application

```bash
# Start the Streamlit web application
streamlit run src/app.py

# The app will be available at http://localhost:8501
```

## Docker Deployment

### 1. Build and Run Container

```bash
# Build the Docker image
docker build -t neuromind-eeg-classifier .

# Run the container
docker run -p 8501:8501 neuromind-eeg-classifier
```

### 2. Docker Compose (Recommended)

```bash
# Start with docker-compose
docker-compose up -d

# For production with nginx
docker-compose --profile production up -d

# Stop the services
docker-compose down
```

## Cloud Deployment

### Streamlit Cloud

1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Connect to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
3. **Deploy**: Point to your repository and `src/app.py`
4. **Configure**: Set environment variables if needed

**Limitations**:
- Repository size limit (may need Git LFS for models)
- Resource constraints for large models

### Heroku

1. **Create Heroku app**:
```bash
heroku create your-neuromind-app
```

2. **Add buildpacks**:
```bash
heroku buildpacks:add heroku/python
```

3. **Create Procfile**:
```bash
echo "web: streamlit run src/app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
```

4. **Deploy**:
```bash
git push heroku main
```

### AWS EC2

1. **Launch EC2 instance** (t3.medium or larger recommended)
2. **Install Docker**:
```bash
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user
```

3. **Deploy with Docker**:
```bash
git clone your-repo-url
cd neuromind-eeg-classifier
sudo docker build -t neuromind .
sudo docker run -d -p 80:8501 neuromind
```

4. **Configure security groups** to allow HTTP traffic on port 80

### Google Cloud Platform

1. **Enable APIs**: Cloud Run, Container Registry
2. **Build and push image**:
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/neuromind
```

3. **Deploy to Cloud Run**:
```bash
gcloud run deploy --image gcr.io/PROJECT-ID/neuromind --platform managed
```

### Digital Ocean App Platform

1. **Create App**: Use the web interface or doctl
2. **Connect repository**: Point to your GitHub repo
3. **Configure**: Set build command and run command
4. **Deploy**: Automatic deployment on git push

## Production Considerations

### Security

- **Environment Variables**: Store sensitive data in environment variables
- **HTTPS**: Use SSL certificates (Let's Encrypt recommended)
- **Authentication**: Add user authentication if needed
- **Input Validation**: Validate all user inputs

### Performance

- **Model Optimization**: Use quantized or pruned models for faster inference
- **Caching**: Implement result caching for repeated requests
- **CDN**: Use CDN for static assets
- **Load Balancing**: Use multiple instances for high traffic

### Monitoring

- **Health Checks**: Implement health check endpoints
- **Logging**: Use structured logging (JSON format)
- **Metrics**: Monitor response times, error rates, resource usage
- **Alerts**: Set up alerts for service degradation

### Backup and Recovery

- **Data Backup**: Regular backups of user data and models
- **Configuration**: Version control all configuration files
- **Disaster Recovery**: Document recovery procedures

## Environment Configuration

### Required Environment Variables

```bash
# Optional: Custom data paths
export MNE_DATA=/path/to/mne/data
export NEUROMIND_DATA_PATH=/path/to/spectrograms

# Optional: Model configuration
export NEUROMIND_MODEL_PATH=/path/to/models
export NEUROMIND_BATCH_SIZE=32

# Optional: Streamlit configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Development vs Production

**Development**:
```bash
export NEUROMIND_DEBUG=true
export NEUROMIND_LOG_LEVEL=DEBUG
```

**Production**:
```bash
export NEUROMIND_DEBUG=false
export NEUROMIND_LOG_LEVEL=INFO
export NEUROMIND_CACHE_ENABLED=true
```

## Troubleshooting

### Common Issues

1. **Memory Issues**:
   - Reduce batch size
   - Use model quantization
   - Increase instance memory

2. **Slow Loading**:
   - Implement model caching
   - Use smaller model variants
   - Optimize image loading

3. **Port Conflicts**:
   - Change Streamlit port: `--server.port=8502`
   - Check for running services: `netstat -tulpn`

4. **Dependencies Issues**:
   - Use virtual environments
   - Pin dependency versions
   - Check Python version compatibility

### Performance Tuning

1. **Model Inference**:
```python
# Use CPU for small models
device = "cpu"

# Enable torch optimizations
torch.backends.cudnn.benchmark = True
```

2. **Streamlit Configuration**:
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
enableCORS = false

[browser]
gatherUsageStats = false
```

### Logs and Debugging

```bash
# View container logs
docker logs container-name

# Monitor resource usage
docker stats

# Debug Streamlit issues
streamlit run src/app.py --logger.level=debug
```

## Scaling

### Horizontal Scaling

- **Load Balancer**: Distribute traffic across multiple instances
- **Container Orchestration**: Use Kubernetes for automatic scaling
- **Database**: Use external database for session storage

### Vertical Scaling

- **CPU**: Increase CPU cores for faster processing
- **Memory**: More RAM for larger models and batch sizes
- **GPU**: NVIDIA GPU for faster inference

### Auto-scaling Configuration

**Kubernetes HPA**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neuromind-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neuromind-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Cost Optimization

- **Right-sizing**: Use appropriate instance sizes
- **Spot Instances**: Use spot instances for development
- **Auto-shutdown**: Shut down development environments when not in use
- **Model Optimization**: Use smaller models when accuracy permits
- **Caching**: Cache results to reduce computation

## Support and Maintenance

- **Regular Updates**: Keep dependencies updated
- **Security Patches**: Apply security updates promptly
- **Performance Monitoring**: Regular performance reviews
- **Backup Testing**: Test backup and recovery procedures
- **Documentation**: Keep deployment documentation updated