import { Router } from 'express';
import { prisma } from '../config/database';
import { asyncHandler, CustomError } from '../middleware/errorHandler';
import { logger } from '../utils/logger';
import { RealTimeCache } from '../config/redis';

const router = Router();
// Cache will be initialized when needed

// GET /api/sessions - Get all sessions
router.get('/', asyncHandler(async (req, res) => {
  const { status, limit = 20, offset = 0 } = req.query;
  
  const where = status ? { status: status as any } : {};
  
  const sessions = await prisma.session.findMany({
    where,
    include: {
      _count: {
        select: {
          metrics: true,
          alerts: true
        }
      }
    },
    orderBy: { createdAt: 'desc' },
    take: parseInt(limit as string),
    skip: parseInt(offset as string)
  });

  const total = await prisma.session.count({ where });

  res.json({
    success: true,
    data: {
      sessions,
      pagination: {
        total,
        limit: parseInt(limit as string),
        offset: parseInt(offset as string),
        hasMore: total > parseInt(offset as string) + parseInt(limit as string)
      }
    }
  });
}));

// GET /api/sessions/:id - Get session by ID
router.get('/:id', asyncHandler(async (req, res) => {
  const { id } = req.params;
  
  const session = await prisma.session.findUnique({
    where: { id },
    include: {
      metrics: {
        orderBy: { timestamp: 'desc' },
        take: 100 // Last 100 metrics
      },
      alerts: {
        orderBy: { timestamp: 'desc' },
        take: 50 // Last 50 alerts
      },
      _count: {
        select: {
          metrics: true,
          alerts: true
        }
      }
    }
  });

  if (!session) {
    throw new CustomError('Session not found', 404);
  }

  // Get real-time data from cache
  let sessionState = null;
  let latestMetrics = null;
  let connectionCount = 0;

  try {
    const cache = RealTimeCache.getInstance();
    sessionState = await cache.getSessionState(id);
    latestMetrics = await cache.getLatestMetrics(id);
    connectionCount = await cache.getConnectionCount(id);
  } catch (error) {
    console.warn('Redis not available, skipping cache data');
  }

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
}));

// POST /api/sessions - Create new session
router.post('/', asyncHandler(async (req, res) => {
  const { name, description, totalStudents, classroomLayout } = req.body;

  if (!name) {
    throw new CustomError('Session name is required', 400);
  }

  const session = await prisma.session.create({
    data: {
      name,
      description,
      totalStudents: totalStudents ? parseInt(totalStudents) : null,
      classroomLayout: classroomLayout || null,
      status: 'ACTIVE'
    }
  });

  // Initialize session state in cache
  try {
    const cache = RealTimeCache.getInstance();
    await cache.setSessionState(session.id, {
      status: 'ACTIVE',
      startTime: session.startTime.toISOString(),
      totalStudents: session.totalStudents,
      classroomLayout: session.classroomLayout
    });
  } catch (error) {
    console.warn('Redis not available, skipping cache initialization');
  }

  logger.info(`New session created: ${session.id} - ${session.name}`);

  res.status(201).json({
    success: true,
    data: { session }
  });
}));

// PUT /api/sessions/:id - Update session
router.put('/:id', asyncHandler(async (req, res) => {
  const { id } = req.params;
  const { name, description, status, totalStudents, classroomLayout } = req.body;

  const existingSession = await prisma.session.findUnique({
    where: { id }
  });

  if (!existingSession) {
    throw new CustomError('Session not found', 404);
  }

  const updateData: any = {};
  if (name) updateData.name = name;
  if (description !== undefined) updateData.description = description;
  if (status) updateData.status = status;
  if (totalStudents !== undefined) updateData.totalStudents = totalStudents ? parseInt(totalStudents) : null;
  if (classroomLayout !== undefined) updateData.classroomLayout = classroomLayout;

  // If stopping the session, set end time
  if (status === 'COMPLETED' || status === 'CANCELLED') {
    updateData.endTime = new Date();
  }

  const session = await prisma.session.update({
    where: { id },
    data: updateData
  });

  // Update cache
  try {
    const cache = RealTimeCache.getInstance();
    const sessionState = await cache.getSessionState(id) || {};
    Object.assign(sessionState, updateData);
    await cache.setSessionState(id, sessionState);
  } catch (error) {
    console.warn('Redis not available, skipping cache update');
  }

  logger.info(`Session updated: ${id} - Status: ${status || 'unchanged'}`);

  res.json({
    success: true,
    data: { session }
  });
}));

// DELETE /api/sessions/:id - Delete session
router.delete('/:id', asyncHandler(async (req, res) => {
  const { id } = req.params;

  const session = await prisma.session.findUnique({
    where: { id }
  });

  if (!session) {
    throw new CustomError('Session not found', 404);
  }

  // Delete session (cascade will handle related records)
  await prisma.session.delete({
    where: { id }
  });

  logger.info(`Session deleted: ${id}`);

  res.json({
    success: true,
    message: 'Session deleted successfully'
  });
}));

// GET /api/sessions/:id/summary - Get session summary
router.get('/:id/summary', asyncHandler(async (req, res) => {
  const { id } = req.params;

  const session = await prisma.session.findUnique({
    where: { id },
    include: {
      metrics: {
        orderBy: { timestamp: 'asc' }
      },
      alerts: {
        orderBy: { timestamp: 'asc' }
      }
    }
  });

  if (!session) {
    throw new CustomError('Session not found', 404);
  }

  // Calculate summary statistics
  const metrics = session.metrics;
  const alerts = session.alerts;

  const summary = {
    session: {
      id: session.id,
      name: session.name,
      description: session.description,
      startTime: session.startTime,
      endTime: session.endTime,
      duration: session.endTime 
        ? Math.round((session.endTime.getTime() - session.startTime.getTime()) / 1000 / 60) // minutes
        : null,
      status: session.status
    },
    attendance: {
      average: metrics.length > 0 
        ? Math.round(metrics.reduce((sum, m) => sum + m.attendanceCount, 0) / metrics.length)
        : 0,
      peak: metrics.length > 0 
        ? Math.max(...metrics.map(m => m.attendanceCount))
        : 0,
      confidence: metrics.length > 0
        ? Math.round((metrics.reduce((sum, m) => sum + m.attendanceConfidence, 0) / metrics.length) * 100) / 100
        : 0
    },
    engagement: {
      average: metrics.length > 0
        ? Math.round((metrics.reduce((sum, m) => sum + m.overallEngagement, 0) / metrics.length) * 100) / 100
        : 0,
      lowest: metrics.length > 0
        ? Math.min(...metrics.map(m => m.overallEngagement))
        : 0,
      highest: metrics.length > 0
        ? Math.max(...metrics.map(m => m.overallEngagement))
        : 0
    },
    alerts: {
      total: alerts.length,
      byType: alerts.reduce((acc: any, alert) => {
        acc[alert.type] = (acc[alert.type] || 0) + 1;
        return acc;
      }, {}),
      bySeverity: alerts.reduce((acc: any, alert) => {
        acc[alert.severity] = (acc[alert.severity] || 0) + 1;
        return acc;
      }, {}),
      acknowledged: alerts.filter(a => a.acknowledged).length
    },
    timeline: metrics.map(m => ({
      timestamp: m.timestamp,
      attendance: m.attendanceCount,
      engagement: m.overallEngagement,
      attention: m.attentionScore,
      participation: m.participationScore
    }))
  };

  res.json({
    success: true,
    data: { summary }
  });
}));

// POST /api/sessions/:id/quiz - Send quiz to session
router.post('/:id/quiz', asyncHandler(async (req, res) => {
  const { id } = req.params;
  const { question, options, correctAnswer, duration = 60 } = req.body;

  // Validate session exists
  const session = await prisma.session.findUnique({
    where: { id }
  });

  if (!session) {
    throw new CustomError('Session not found', 404);
  }

  // Create quiz data
  const quiz = {
    sessionId: id,
    quizId: `quiz_${Date.now()}`,
    question,
    options,
    correctAnswer,
    duration,
    startTime: new Date().toISOString(),
    type: 'quiz'
  };

  // Here you would typically save to database and emit via socket
  // For now, just return success
  res.json({
    success: true,
    data: { quiz },
    message: 'Quiz sent successfully'
  });
}));

export default router;
