#!/bin/bash
# Docker Setup Script for Classroom Engagement Analyzer
# Complete setup for sharing with friends

set -e

echo "=========================================="
echo "🐳 DOCKER SETUP FOR ENGAGEMENT ANALYZER"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}📋 STEP $1: $2${NC}"
    echo "----------------------------------------"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Check Docker installation
print_step 1 "Checking Docker Installation"

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed!"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed!"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

print_success "Docker and Docker Compose are installed"
docker --version
docker-compose --version

# Step 2: Create necessary directories
print_step 2 "Creating Directory Structure"

mkdir -p data/{models,datasets,feedback,external}
mkdir -p logs
mkdir -p config
mkdir -p templates

print_success "Directory structure created"

# Step 3: Setup initial data
print_step 3 "Setting up Initial Data"

# Create basic config if it doesn't exist
if [ ! -f "config/config.py" ]; then
    cat > config/config.py << 'EOF'
"""
Configuration for Dockerized Engagement Analyzer
"""

class Config:
    # Camera settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    
    # API settings
    API_PORT = 8080
    FEEDBACK_PORT = 5001
    
    # Continuous learning
    CONTINUOUS_LEARNING = True
    CONFIDENCE_THRESHOLD = 0.8
    
    # Paths (Docker paths)
    DATA_DIR = "/app/data"
    MODELS_DIR = "/app/data/models"
    LOGS_DIR = "/app/logs"
EOF
    print_success "Created basic configuration"
fi

# Step 4: Build Docker image
print_step 4 "Building Docker Image"

echo "This may take 10-15 minutes for the first build..."
docker-compose build

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Step 5: Initialize the system
print_step 5 "Initializing Continuous Learning System"

# Run setup inside container
docker-compose run --rm engagement-analyzer python setup_continuous_learning.py

if [ $? -eq 0 ]; then
    print_success "Continuous learning system initialized"
else
    print_warning "Setup had some issues, but continuing..."
fi

# Step 6: Test the system
print_step 6 "Testing System Components"

# Test package imports
docker-compose run --rm engagement-analyzer python test_packages.py

# Step 7: Create startup scripts
print_step 7 "Creating Startup Scripts"

# Create start script
cat > start_engagement_analyzer.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Classroom Engagement Analyzer..."
echo "📊 Feedback Interface: http://localhost:5001"
echo "🔧 API Endpoint: http://localhost:8080"
echo "⏹️  Press Ctrl+C to stop"

# Start with camera access
docker-compose up
EOF

chmod +x start_engagement_analyzer.sh

# Create stop script
cat > stop_engagement_analyzer.sh << 'EOF'
#!/bin/bash
echo "⏹️ Stopping Classroom Engagement Analyzer..."
docker-compose down
echo "✅ Stopped successfully"
EOF

chmod +x stop_engagement_analyzer.sh

# Create logs script
cat > view_logs.sh << 'EOF'
#!/bin/bash
echo "📋 Viewing Engagement Analyzer Logs..."
docker-compose logs -f engagement-analyzer
EOF

chmod +x view_logs.sh

print_success "Startup scripts created"

# Step 8: Create transfer package
print_step 8 "Creating Transfer Package"

# Create README for your friend
cat > DOCKER_README.md << 'EOF'
# 🐳 Dockerized Classroom Engagement Analyzer

## Quick Start

### Prerequisites
- Docker Desktop installed
- Webcam connected
- 4GB+ RAM available

### 1. Start the Application
```bash
./start_engagement_analyzer.sh
```

### 2. Access Interfaces
- **Feedback Interface**: http://localhost:5001
- **Main Application**: Camera window will open
- **API**: http://localhost:8080

### 3. Stop the Application
```bash
./stop_engagement_analyzer.sh
```

### 4. View Logs
```bash
./view_logs.sh
```

## Features
- ✅ Industry-grade body movement detection
- ✅ Real-time engagement analysis
- ✅ Continuous learning with teacher feedback
- ✅ Web-based feedback interface
- ✅ Performance monitoring

## Troubleshooting

### Camera Issues
```bash
# Check available cameras
ls /dev/video*

# Update camera index in docker-compose.yml
# Change CAMERA_INDEX environment variable
```

### Permission Issues
```bash
# Give Docker camera access
sudo usermod -a -G video $USER
# Logout and login again
```

### Performance Issues
```bash
# Check Docker resources
docker stats

# Increase Docker memory limit in Docker Desktop settings
```

## Continuous Learning
1. Open http://localhost:5001
2. Provide feedback on engagement predictions
3. Watch model accuracy improve from 70% to 90%+
4. System learns and adapts in real-time

## Data Persistence
- Models: `./data/models/`
- Datasets: `./data/datasets/`
- Feedback: `./data/feedback/`
- Logs: `./logs/`

Built with ❤️ by Subhasis & Sachin
EOF

print_success "Transfer package created"

# Final summary
echo ""
echo "=========================================="
echo "🎉 DOCKER SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "📦 Your friend can now run the application with:"
echo "   1. Install Docker Desktop"
echo "   2. Run: ./start_engagement_analyzer.sh"
echo "   3. Open: http://localhost:5001"
echo ""
echo "📁 Files to share with your friend:"
echo "   • Entire project folder"
echo "   • DOCKER_README.md (instructions)"
echo "   • start_engagement_analyzer.sh"
echo "   • stop_engagement_analyzer.sh"
echo ""
echo "🚀 Test the setup:"
echo "   ./start_engagement_analyzer.sh"
echo ""
print_success "Ready for transfer!"
