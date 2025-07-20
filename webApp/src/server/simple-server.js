const express = require('express');
const { createServer } = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const { PrismaClient } = require('@prisma/client');
const aiService = require('./services/aiService');

const app = express();
const server = createServer(app);
const prisma = new PrismaClient();

// Initialize Socket.IO with CORS
const io = new Server(server, {
  cors: {
    origin: "http://localhost:5173",
    methods: ["GET", "POST"],
    credentials: true
  },
  transports: ['websocket', 'polling']
});

// Middleware
app.use(cors({
  origin: "http://localhost:5173",
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));

// In-memory storage for real-time data (replacing Redis)
const realTimeData = {
  sessions: new Map(),
  metrics: new Map(),
  alerts: new Map(),
  connections: new Map()
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Create demo session summaries for last 5 days
app.post('/api/create-demo-summaries', async (req, res) => {
  try {
    const demoSessions = [];
    const today = new Date();

    // Create sessions for last 5 days
    for (let i = 1; i <= 5; i++) {
      const sessionDate = new Date(today);
      sessionDate.setDate(today.getDate() - i);
      sessionDate.setHours(10, 0, 0, 0); // 10 AM

      const session = await prisma.session.create({
        data: {
          name: `Biology Class - Day ${i}`,
          description: `Interactive biology session covering cellular structures and functions`,
          startTime: sessionDate,
          endTime: new Date(sessionDate.getTime() + 60 * 60 * 1000), // 1 hour later
          status: 'COMPLETED',
          totalStudents: 25
        }
      });

      // Create AI summary for this session
      const aiSummary = await aiService.generateDemoSummary(
        session.name,
        sessionDate.toLocaleDateString()
      );

      await prisma.sessionSummary.create({
        data: {
          sessionId: session.id,
          summary: aiSummary,
          topics: ['Cell Biology', 'Mitochondria', 'Cellular Respiration', 'Lab Work']
        }
      });

      demoSessions.push(session);
    }

    res.json({
      success: true,
      message: `Created ${demoSessions.length} demo sessions with AI summaries`,
      data: { sessions: demoSessions }
    });
  } catch (error) {
    console.error('Error creating demo summaries:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to create demo summaries'
    });
  }
});

// Get session summaries for calendar
app.get('/api/session-summaries', async (req, res) => {
  try {
    const { startDate, endDate } = req.query;

    const summaries = await prisma.sessionSummary.findMany({
      where: {
        createdAt: {
          gte: startDate ? new Date(startDate) : new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
          lte: endDate ? new Date(endDate) : new Date()
        }
      },
      include: {
        session: {
          select: {
            id: true,
            name: true,
            description: true,
            startTime: true,
            endTime: true
          }
        }
      },
      orderBy: {
        createdAt: 'desc'
      }
    });

    res.json({
      success: true,
      data: { summaries }
    });
  } catch (error) {
    console.error('Error fetching session summaries:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch session summaries'
    });
  }
});

// Get specific session summary
app.get('/api/session-summaries/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;

    let summary = await prisma.sessionSummary.findUnique({
      where: { sessionId },
      include: {
        session: true
      }
    });

    // If no summary exists, generate one for demo
    if (!summary) {
      const session = await prisma.session.findUnique({
        where: { id: sessionId }
      });

      if (session) {
        const aiSummary = await aiService.generateDemoSummary(
          session.name,
          session.startTime.toLocaleDateString()
        );

        summary = await prisma.sessionSummary.create({
          data: {
            sessionId,
            summary: aiSummary,
            topics: ['Introduction', 'Key Concepts', 'Practical Applications']
          },
          include: {
            session: true
          }
        });
      }
    }

    res.json({
      success: true,
      data: { summary }
    });
  } catch (error) {
    console.error('Error fetching session summary:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch session summary'
    });
  }
});

// Sessions API
app.get('/api/sessions', async (req, res) => {
  try {
    const sessions = await prisma.session.findMany({
      include: {
        _count: {
          select: {
            metrics: true,
            alerts: true
          }
        }
      },
      orderBy: { createdAt: 'desc' },
      take: 50
    });

    res.json({
      success: true,
      data: {
        sessions,
        pagination: {
          total: sessions.length,
          limit: 50,
          offset: 0,
          hasMore: false
        }
      }
    });
  } catch (error) {
    console.error('Error fetching sessions:', error);
    res.status(500).json({ success: false, error: { message: 'Failed to fetch sessions' } });
  }
});

app.post('/api/sessions', async (req, res) => {
  try {
    const { name, description, totalStudents } = req.body;

    const session = await prisma.session.create({
      data: {
        name,
        description,
        totalStudents: totalStudents ? parseInt(totalStudents) : null,
        status: 'ACTIVE'
      }
    });

    // Initialize session in memory
    realTimeData.sessions.set(session.id, {
      status: 'ACTIVE',
      startTime: session.startTime.toISOString(),
      totalStudents: session.totalStudents
    });

    res.status(201).json({
      success: true,
      data: { session }
    });
  } catch (error) {
    console.error('Error creating session:', error);
    res.status(500).json({ success: false, error: { message: 'Failed to create session' } });
  }
});

app.get('/api/sessions/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const session = await prisma.session.findUnique({
      where: { id },
      include: {
        metrics: {
          orderBy: { timestamp: 'desc' },
          take: 10
        },
        alerts: {
          orderBy: { timestamp: 'desc' },
          take: 10
        }
      }
    });

    if (!session) {
      return res.status(404).json({ success: false, error: { message: 'Session not found' } });
    }

    const sessionState = realTimeData.sessions.get(id) || {};
    const latestMetrics = realTimeData.metrics.get(id) || null;
    const connectionCount = realTimeData.connections.get(id)?.size || 0;

    res.json({
      success: true,
      data: {
        session,
        realTime: {
          sessionState,
          latestMetrics,
          connectionCount
        }
      }
    });
  } catch (error) {
    console.error('Error fetching session:', error);
    res.status(500).json({ success: false, error: { message: 'Failed to fetch session' } });
  }
});

// AI Metrics endpoint
app.post('/api/ai/metrics', async (req, res) => {
  try {
    const { sessionId, metrics, timestamp } = req.body;

    if (!sessionId || !metrics) {
      return res.status(400).json({ success: false, error: { message: 'Session ID and metrics are required' } });
    }

    // Store in database
    const dbMetric = await prisma.sessionMetric.create({
      data: {
        sessionId,
        timestamp: timestamp ? new Date(timestamp) : new Date(),
        attendanceCount: metrics.attendance?.total_detected || 0,
        attendanceConfidence: metrics.attendance?.confidence_avg || 0,
        overallEngagement: metrics.engagement?.overall_score || 0,
        attentionScore: metrics.engagement?.attention_score || 0,
        participationScore: metrics.engagement?.participation_score || 0,
        frontZoneEngagement: metrics.engagement?.zones?.front || null,
        middleZoneEngagement: metrics.engagement?.zones?.middle || null,
        backZoneEngagement: metrics.engagement?.zones?.back || null,
        noiseLevel: metrics.audio?.noise_level || 0,
        speakerActivity: metrics.audio?.speaker_activity || 0,
        rawData: metrics
      }
    });

    // Store in memory for real-time access
    realTimeData.metrics.set(sessionId, {
      ...metrics,
      timestamp: timestamp || new Date().toISOString(),
      sessionId
    });

    // Broadcast to all clients in the session
    io.to(`session:${sessionId}`).emit('metrics-updated', {
      sessionId,
      metrics: {
        ...metrics,
        timestamp: timestamp || new Date().toISOString()
      },
      dbId: dbMetric.id
    });

    // Check for alerts
    await checkAndTriggerAlerts(sessionId, metrics);

    res.json({
      success: true,
      data: {
        metricId: dbMetric.id,
        timestamp: dbMetric.timestamp
      }
    });
  } catch (error) {
    console.error('Error processing metrics:', error);
    res.status(500).json({ success: false, error: { message: 'Failed to process metrics' } });
  }
});

// Metrics endpoints for Dashboard
app.get('/api/metrics/session/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { limit = 50, interval = '1m' } = req.query;

    const metrics = await prisma.sessionMetric.findMany({
      where: { sessionId },
      orderBy: { timestamp: 'desc' },
      take: parseInt(limit)
    });

    res.json({ success: true, data: { metrics } });
  } catch (error) {
    console.error('Error fetching metrics:', error);
    res.status(500).json({ success: false, error: { message: 'Failed to fetch metrics' } });
  }
});

app.get('/api/metrics/session/:sessionId/stats', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { timeRange = '1h' } = req.query;

    // Return mock stats for now
    const stats = {
      attendance: { current: 0, trend: 'stable', average: 0 },
      engagement: { current: 0, trend: 'stable', average: 0 },
      attention: { current: 0, trend: 'stable', average: 0 },
      participation: { current: 0, trend: 'stable', average: 0 },
      audio: { currentNoise: 0, currentActivity: 0 }
    };

    res.json({ success: true, data: { stats } });
  } catch (error) {
    console.error('Error fetching stats:', error);
    res.status(500).json({ success: false, error: { message: 'Failed to fetch stats' } });
  }
});

// Quiz/Poll endpoints
app.post('/api/sessions/:sessionId/quiz', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { question, options, type, duration } = req.body;

    const quiz = await prisma.quiz.create({
      data: {
        sessionId,
        question,
        options: JSON.stringify(options),
        type, // 'quiz' or 'poll'
        duration: duration || 60,
        status: 'ACTIVE',
        createdAt: new Date()
      }
    });

    const quizData = {
      quizId: quiz.id,
      sessionId: quiz.sessionId,
      question: quiz.question,
      options: JSON.parse(quiz.options),
      correctAnswer: req.body.correctAnswer || 0,
      type: quiz.type,
      duration: quiz.duration,
      startTime: quiz.createdAt
    };

    // Broadcast to all students in session
    console.log(`🎯 SERVER: Broadcasting quiz to session:${sessionId}`);
    io.to(`session:${sessionId}`).emit('quiz-started', quizData);

    // Also broadcast globally as backup
    console.log('🎯 SERVER: Broadcasting quiz globally');
    io.emit('global-quiz', quizData);

    // Send test event to verify connection
    io.emit('test-event', { message: 'Quiz broadcast test', timestamp: new Date() });

    res.json({ success: true, data: { quiz } });
  } catch (error) {
    console.error('Error creating quiz:', error);
    res.status(500).json({ success: false, error: { message: 'Failed to create quiz' } });
  }
});

app.post('/api/quiz/:quizId/response', async (req, res) => {
  try {
    const { quizId } = req.params;
    const { studentId, answer, sessionId } = req.body;

    const response = await prisma.quizResponse.create({
      data: {
        quizId,
        studentId: studentId || `student_${Date.now()}`,
        answer,
        submittedAt: new Date()
      }
    });

    // Get current results
    const results = await prisma.quizResponse.groupBy({
      by: ['answer'],
      where: { quizId },
      _count: { answer: true }
    });

    // Broadcast updated results to teachers
    io.to(`session:${sessionId}`).emit('quiz-results-updated', {
      quizId,
      results,
      totalResponses: results.reduce((sum, r) => sum + r._count.answer, 0)
    });

    res.json({ success: true, data: { response } });
  } catch (error) {
    console.error('Error submitting response:', error);
    res.status(500).json({ success: false, error: { message: 'Failed to submit response' } });
  }
});

app.get('/api/quiz/:quizId/results', async (req, res) => {
  try {
    const { quizId } = req.params;

    const results = await prisma.quizResponse.groupBy({
      by: ['answer'],
      where: { quizId },
      _count: { answer: true }
    });

    const quiz = await prisma.quiz.findUnique({
      where: { id: quizId }
    });

    res.json({
      success: true,
      data: {
        quiz,
        results,
        totalResponses: results.reduce((sum, r) => sum + r._count.answer, 0)
      }
    });
  } catch (error) {
    console.error('Error fetching results:', error);
    res.status(500).json({ success: false, error: { message: 'Failed to fetch results' } });
  }
});

// Alert checking function
async function checkAndTriggerAlerts(sessionId, metrics) {
  try {
    const alerts = [];

    // Check engagement thresholds
    const engagementThreshold = 0.6;
    if (metrics.engagement?.overall_score < engagementThreshold) {
      alerts.push({
        type: 'DISENGAGEMENT',
        severity: metrics.engagement.overall_score < 0.4 ? 'HIGH' : 'MEDIUM',
        message: `Overall engagement dropped to ${(metrics.engagement.overall_score * 100).toFixed(1)}%`,
        triggerValue: metrics.engagement.overall_score,
        threshold: engagementThreshold
      });
    }

    // Check noise levels
    const noiseThreshold = 0.7;
    if (metrics.audio?.noise_level > noiseThreshold) {
      alerts.push({
        type: 'HIGH_NOISE',
        severity: metrics.audio.noise_level > 0.9 ? 'HIGH' : 'MEDIUM',
        message: `High noise level detected: ${(metrics.audio.noise_level * 100).toFixed(1)}%`,
        triggerValue: metrics.audio.noise_level,
        threshold: noiseThreshold
      });
    }

    // Store and broadcast alerts
    for (const alertData of alerts) {
      const alert = await prisma.alert.create({
        data: {
          sessionId,
          type: alertData.type,
          severity: alertData.severity,
          message: alertData.message,
          triggerValue: alertData.triggerValue,
          threshold: alertData.threshold
        }
      });

      // Broadcast to clients
      io.to(`session:${sessionId}`).emit('alert-triggered', {
        alert: {
          ...alert,
          timestamp: alert.timestamp.toISOString()
        },
        source: 'auto-detection'
      });
    }
  } catch (error) {
    console.error('Error checking alerts:', error);
  }
}

// Socket.IO handlers
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('join-session', async (data) => {
    try {
      const { sessionId } = data;
      
      // Join session room
      socket.join(`session:${sessionId}`);
      
      // Track connection
      if (!realTimeData.connections.has(sessionId)) {
        realTimeData.connections.set(sessionId, new Set());
      }
      realTimeData.connections.get(sessionId).add(socket.id);

      // Send current data
      const sessionState = realTimeData.sessions.get(sessionId) || {};
      const latestMetrics = realTimeData.metrics.get(sessionId) || null;
      const connectionCount = realTimeData.connections.get(sessionId).size;

      socket.emit('session-joined', {
        sessionId,
        sessionState,
        latestMetrics,
        activeAlerts: [],
        connectionCount
      });

      console.log(`Client ${socket.id} joined session ${sessionId}`);
    } catch (error) {
      console.error('Error joining session:', error);
      socket.emit('error', { message: 'Failed to join session' });
    }
  });

  // Handle quiz responses
  socket.on('quiz-response', (response) => {
    console.log('🎯 SERVER: Quiz response received:', response);
    // Forward to all clients in the session (teachers will receive it)
    if (response.sessionId) {
      io.to(`session:${response.sessionId}`).emit('quiz-response', response);
      console.log(`🎯 SERVER: Forwarded response to session:${response.sessionId}`);
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
    
    // Remove from all session connections
    for (const [sessionId, connections] of realTimeData.connections.entries()) {
      if (connections.has(socket.id)) {
        connections.delete(socket.id);
        socket.to(`session:${sessionId}`).emit('user-left', {
          socketId: socket.id,
          connectionCount: connections.size
        });
      }
    }
  });
});

// Start server
const PORT = process.env.PORT || 3001;

server.listen(PORT, async () => {
  try {
    await prisma.$connect();
    console.log(`🚀 Server running on port ${PORT}`);
    console.log(`🌐 Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`📊 Dashboard: http://localhost:5173`);
    console.log(`🔌 WebSocket: ws://localhost:${PORT}`);
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
});

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('SIGINT received, shutting down gracefully');
  await prisma.$disconnect();
  server.close(() => {
    console.log('Process terminated');
  });
});
