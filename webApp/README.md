# 🎓 Real-time Classroom Engagement Analyzer

A cutting-edge AI-powered system that provides real-time insights into classroom engagement, attendance, and participation through computer vision and audio analysis.

## 🏆 Hackathon Project

**Team:** Sachin (Full-stack) + Subhasis (AI/ML)  
**Deadline:** MVP by 2am tonight  
**Tech Stack:** Node.js, React, TypeScript, Socket.io, PostgreSQL, Redis, Python AI Pipeline

## 🚀 Features

### Real-time Analytics
- **Attendance Tracking**: Automatic face detection and counting
- **Engagement Monitoring**: Multi-zone classroom engagement analysis
- **Attention Scoring**: Head pose estimation for attention tracking
- **Participation Detection**: Hand gesture recognition for participation
- **Audio Analysis**: Speaker diarization and noise level monitoring

### Live Dashboard
- **Real-time Metrics**: <5s latency from detection to dashboard
- **Zone-based Heatmaps**: Visual representation of classroom engagement
- **Alert System**: Instant notifications for disengagement or issues
- **Mobile Responsive**: Works on all devices
- **Export Functionality**: Session reports for LMS integration

### Privacy-First Design
- **Edge Processing**: All AI processing on local hardware
- **No Face Storage**: Only engagement metrics, no personal identification
- **Anonymized Data**: Privacy-compliant analytics
- **Opt-in Consent**: Transparent data usage

## 🏗️ Architecture

```
[Laptop Camera + Mic] → [AI Pipeline] → [Real-time Backend] → [Live Dashboard]
                                    ↓
                              [PostgreSQL + Redis]
```

## 🛠️ Quick Start

### Prerequisites
- Node.js 20+
- PostgreSQL database
- Redis server
- Python 3.8+ (for AI pipeline)

### Backend Setup
```bash
# Install dependencies
npm install

# Set up environment
cp .env.example .env
# Edit .env with your database credentials

# Set up database
npx prisma generate
npx prisma db push

# Start backend server
npm run dev:server
```

### Frontend Setup
```bash
# Install frontend dependencies
cd client
npm install

# Start frontend development server
npm run dev
```

### Full Development
```bash
# Run both backend and frontend
npm run dev
```

## 📊 API Endpoints

### Sessions
- `GET /api/sessions` - List all sessions
- `POST /api/sessions` - Create new session
- `GET /api/sessions/:id` - Get session details
- `PUT /api/sessions/:id` - Update session
- `DELETE /api/sessions/:id` - Delete session

### Metrics
- `GET /api/metrics/session/:id` - Get session metrics
- `GET /api/metrics/session/:id/latest` - Get latest metrics
- `GET /api/metrics/session/:id/stats` - Get statistical summary

### Alerts
- `GET /api/alerts/session/:id` - Get session alerts
- `PUT /api/alerts/:id/acknowledge` - Acknowledge alert
- `GET /api/alerts/session/:id/summary` - Get alert summary

### AI Integration
- `POST /api/ai/metrics` - Receive metrics from AI pipeline
- `POST /api/ai/batch-metrics` - Receive batch metrics
- `GET /api/ai/session/:id/status` - Get AI processing status

## 🔌 WebSocket Events

### Client → Server
- `join-session` - Join a session room
- `leave-session` - Leave session room
- `acknowledge-alert` - Acknowledge an alert
- `session-control` - Control session (pause/resume/stop)

### Server → Client
- `session-joined` - Confirmation of joining session
- `metrics-updated` - Real-time metrics update
- `alert-triggered` - New alert notification
- `session-status-changed` - Session status update

## 🧠 AI Pipeline Integration

The system expects JSON data from the AI pipeline in this format:

```json
{
  "sessionId": "session-uuid",
  "timestamp": "2025-01-19T20:30:00Z",
  "attendance": {
    "total_detected": 8,
    "confidence_avg": 0.92
  },
  "engagement": {
    "overall_score": 0.75,
    "attention_score": 0.80,
    "participation_score": 0.60,
    "zones": {
      "front": 0.85,
      "middle": 0.70,
      "back": 0.55
    }
  },
  "audio": {
    "noise_level": 0.3,
    "speaker_activity": 0.8
  }
}
```

Send to: `POST http://localhost:3001/api/ai/metrics`

## 📱 Demo Setup

For hackathon demo using laptop:
1. Start the backend server
2. Start the frontend dashboard
3. Create a new session
4. Use laptop camera as "classroom camera"
5. Run AI pipeline pointing to laptop camera
6. View real-time analytics on dashboard

## 🎯 Business Impact

- **Market Size**: $2B+ TAM (4,000+ universities globally)
- **ROI**: 15-25% improvement in learning outcomes
- **Cost Savings**: 80% reduction in infrastructure costs vs cloud
- **Privacy Compliance**: GDPR/FERPA compliant edge processing

## 🏅 Competitive Advantages

1. **Real-time Processing**: <5s latency requirement
2. **Privacy-First**: On-premise edge computing
3. **Multi-modal AI**: Vision + Audio + Gesture recognition
4. **Scalable Architecture**: Edge computing reduces costs
5. **Mobile-First Dashboard**: Accessible anywhere

## 🔧 Development Scripts

```bash
npm run dev              # Run both backend and frontend
npm run dev:server       # Run backend only
npm run dev:client       # Run frontend only
npm run build           # Build for production
npm run start           # Start production server
npm run db:migrate      # Run database migrations
npm run db:studio       # Open Prisma Studio
```

## 📈 Monitoring

- **Health Check**: `GET /health`
- **Logs**: Check `logs/` directory
- **Database**: Use `npm run db:studio`
- **Real-time**: Monitor WebSocket connections

## 🚨 Alerts Configuration

Configure alert thresholds in `.env`:
```
ENGAGEMENT_ALERT_THRESHOLD=0.6
ATTENDANCE_ALERT_THRESHOLD=0.8
NOISE_ALERT_THRESHOLD=0.7
```

## 🤝 Team

- **Sachin**: Full-stack development, real-time systems, dashboard
- **Subhasis**: AI/ML pipeline, computer vision, audio processing

---

**Built for Hackathon 2025** 🚀  
*Real-time classroom engagement analyzer with AI-powered insights*
