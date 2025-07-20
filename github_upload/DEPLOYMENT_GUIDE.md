# 🚀 Deployment Guide - Real-time Classroom Engagement Analyzer

## 📋 Pre-Deployment Checklist

### System Requirements
- [ ] Python 3.8+ installed
- [ ] Camera/webcam available
- [ ] Microphone available
- [ ] 4GB+ RAM available
- [ ] Internet connection (for model downloads)

### Hardware Verification
```bash
# Check camera
python -c "import cv2; cap=cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL'); cap.release()"

# Check audio
python -c "import pyaudio; p=pyaudio.PyAudio(); print(f'Audio devices: {p.get_device_count()}'); p.terminate()"

# Check memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available/1024/1024/1024:.1f}GB')"
```

## 🛠️ Installation Steps

### 1. Quick Setup (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd piper

# Run automated setup
python setup.py

# Activate environment
# Windows:
venv\Scripts\activate
# Unix/Linux/MacOS:
source venv/bin/activate
```

### 2. Manual Setup (If needed)
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Unix/Linux/MacOS
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env
```

## 🧪 Testing & Validation

### 1. Run System Tests
```bash
# Full test suite
python tests/test_pipeline.py

# Expected output:
# ✅ All performance tests passed!
# ✅ All functionality tests passed!
```

### 2. Performance Benchmark
```bash
# Quick performance check
python -c "
from tests.test_pipeline import run_performance_benchmark
run_performance_benchmark()
"
```

### 3. Demo Mode
```bash
# Interactive demo with checks
python run_demo.py
```

## 🎯 Deployment Options

### Option 1: Standalone Demo
```bash
# Start with built-in mock backend
python run_demo.py
```

### Option 2: Production with Backend
```bash
# Set backend URLs
export WEBSOCKET_URL="ws://your-backend:3000/engagement"
export API_BASE_URL="http://your-backend:3000/api"

# Start analyzer
python src/main.py
```

### Option 3: Headless Mode
```bash
# Disable display for server deployment
python -c "
from src.main import EngagementAnalyzer
analyzer = EngagementAnalyzer()
analyzer.show_display = False
analyzer.start()
"
```

## 📊 Performance Optimization

### For Low-End Hardware
```python
# Edit config/config.py
video.frame_width = 320
video.frame_height = 240
video.fps = 15
video.max_faces = 5
audio.sample_rate = 8000
```

### For High-End Hardware
```python
# Edit config/config.py
video.frame_width = 1280
video.frame_height = 720
video.fps = 30
system.num_threads = 8
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Camera Issues
```bash
# Problem: Camera not detected
# Solution: Check camera index
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        print(f'Camera {i}: Not available')
"

# Update config with correct index
export CAMERA_INDEX=1  # Use detected camera index
```

#### 2. Audio Issues
```bash
# Problem: No audio input
# Solution: List audio devices
python -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'Input Device {i}: {info[\"name\"]}')
p.terminate()
"
```

#### 3. Performance Issues
```bash
# Problem: Low FPS
# Solutions:
# 1. Reduce resolution
# 2. Lower FPS target
# 3. Disable display mode
# 4. Reduce max faces

# Check current performance
python -c "
from src.main import EngagementAnalyzer
analyzer = EngagementAnalyzer()
analyzer.initialize()
# Monitor FPS in logs
"
```

#### 4. Memory Issues
```bash
# Problem: High memory usage
# Solutions:
# 1. Reduce buffer sizes
# 2. Lower history window
# 3. Disable audio processing (if not needed)

# Monitor memory
python -c "
import psutil
import time
process = psutil.Process()
while True:
    print(f'Memory: {process.memory_info().rss/1024/1024:.1f}MB')
    time.sleep(5)
"
```

#### 5. Backend Connection Issues
```bash
# Problem: WebSocket connection failed
# Solutions:
# 1. Check backend is running
curl http://localhost:3000/api/health

# 2. Test WebSocket manually
python -c "
import websocket
try:
    ws = websocket.create_connection('ws://localhost:3000/engagement')
    print('WebSocket connection successful')
    ws.close()
except Exception as e:
    print(f'WebSocket connection failed: {e}')
"

# 3. Use API fallback
export WEBSOCKET_URL=""  # Disable WebSocket
```

## 📈 Monitoring & Maintenance

### Performance Monitoring
```bash
# Real-time performance log
tail -f logs/engagement_analyzer.log | grep PERFORMANCE

# Memory monitoring
python -c "
import psutil
import time
while True:
    mem = psutil.virtual_memory()
    print(f'Memory: {mem.percent}% used, {mem.available/1024/1024/1024:.1f}GB available')
    time.sleep(10)
"
```

### Log Analysis
```bash
# Check for errors
grep ERROR logs/engagement_analyzer.log

# Check performance metrics
grep PERFORMANCE logs/engagement_analyzer.log | tail -20

# Check engagement events
grep ENGAGEMENT logs/engagement_analyzer.log | tail -10
```

## 🎯 Production Deployment

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "src/main.py"]
```

### Systemd Service (Linux)
```ini
[Unit]
Description=Engagement Analyzer
After=network.target

[Service]
Type=simple
User=engagement
WorkingDirectory=/opt/engagement-analyzer
ExecStart=/opt/engagement-analyzer/venv/bin/python src/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Windows Service
```bash
# Install as Windows service using NSSM
nssm install EngagementAnalyzer "C:\path\to\venv\Scripts\python.exe" "C:\path\to\src\main.py"
nssm start EngagementAnalyzer
```

## 🔒 Security Considerations

### Privacy Settings
```python
# Ensure privacy compliance
config.system.save_faces = False
config.system.anonymize_data = True
config.system.data_retention_hours = 24
```

### Network Security
```bash
# Use HTTPS/WSS in production
export WEBSOCKET_URL="wss://secure-backend:443/engagement"
export API_BASE_URL="https://secure-backend:443/api"
```

## 📞 Support & Maintenance

### Health Checks
```bash
# System health check
python -c "
from src.main import EngagementAnalyzer
analyzer = EngagementAnalyzer()
print('✅ System healthy' if analyzer.initialize() else '❌ System unhealthy')
"
```

### Backup & Recovery
```bash
# Backup configuration
cp config/config.py config/config.py.backup
cp .env .env.backup

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

### Updates
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update models (if needed)
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads latest
"
```

## 🎉 Success Metrics

### Performance Targets
- ✅ **FPS**: >24 FPS sustained
- ✅ **Latency**: <100ms processing time
- ✅ **Memory**: <2GB total usage
- ✅ **Accuracy**: >80% engagement detection

### Functional Validation
- ✅ Face detection working
- ✅ Pose estimation accurate
- ✅ Gesture recognition responsive
- ✅ Audio processing functional
- ✅ Backend communication stable

---

**🚀 Ready for Hackathon Demo!**

For support during the hackathon, check logs and use the troubleshooting guide above.
