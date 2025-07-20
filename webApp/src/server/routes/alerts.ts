import { Router } from 'express';
import { prisma } from '../config/database';
import { asyncHandler, CustomError } from '../middleware/errorHandler';
import { RealTimeCache } from '../config/redis';

const router = Router();
const cache = new RealTimeCache();

// GET /api/alerts/session/:sessionId - Get alerts for a session
router.get('/session/:sessionId', asyncHandler(async (req, res) => {
  const { sessionId } = req.params;
  const { 
    limit = 50, 
    offset = 0, 
    type, 
    severity, 
    acknowledged,
    startTime,
    endTime
  } = req.query;

  // Validate session exists
  const session = await prisma.session.findUnique({
    where: { id: sessionId }
  });

  if (!session) {
    throw new CustomError('Session not found', 404);
  }

  // Build where clause
  const where: any = { sessionId };
  
  if (type) where.type = type;
  if (severity) where.severity = severity;
  if (acknowledged !== undefined) where.acknowledged = acknowledged === 'true';
  
  if (startTime || endTime) {
    where.timestamp = {};
    if (startTime) where.timestamp.gte = new Date(startTime as string);
    if (endTime) where.timestamp.lte = new Date(endTime as string);
  }

  const alerts = await prisma.alert.findMany({
    where,
    orderBy: { timestamp: 'desc' },
    take: parseInt(limit as string),
    skip: parseInt(offset as string)
  });

  const total = await prisma.alert.count({ where });

  // Get active alerts from cache
  let activeAlerts = [];
  try {
    const cache = RealTimeCache.getInstance();
    activeAlerts = await cache.getActiveAlerts(sessionId);
  } catch (error) {
    console.warn('Redis not available, skipping cache data');
  }

  res.json({
    success: true,
    data: {
      alerts,
      activeAlerts,
      pagination: {
        total,
        limit: parseInt(limit as string),
        offset: parseInt(offset as string),
        hasMore: total > parseInt(offset as string) + parseInt(limit as string)
      }
    }
  });
}));

// GET /api/alerts/:id - Get specific alert
router.get('/:id', asyncHandler(async (req, res) => {
  const { id } = req.params;

  const alert = await prisma.alert.findUnique({
    where: { id },
    include: {
      session: {
        select: {
          id: true,
          name: true,
          status: true
        }
      }
    }
  });

  if (!alert) {
    throw new CustomError('Alert not found', 404);
  }

  res.json({
    success: true,
    data: { alert }
  });
}));

// PUT /api/alerts/:id/acknowledge - Acknowledge an alert
router.put('/:id/acknowledge', asyncHandler(async (req, res) => {
  const { id } = req.params;

  const alert = await prisma.alert.findUnique({
    where: { id }
  });

  if (!alert) {
    throw new CustomError('Alert not found', 404);
  }

  if (alert.acknowledged) {
    throw new CustomError('Alert already acknowledged', 400);
  }

  const updatedAlert = await prisma.alert.update({
    where: { id },
    data: {
      acknowledged: true,
      acknowledgedAt: new Date()
    }
  });

  res.json({
    success: true,
    data: { alert: updatedAlert },
    message: 'Alert acknowledged successfully'
  });
}));

// PUT /api/alerts/session/:sessionId/acknowledge-all - Acknowledge all alerts for a session
router.put('/session/:sessionId/acknowledge-all', asyncHandler(async (req, res) => {
  const { sessionId } = req.params;
  const { type, severity } = req.body;

  // Build where clause
  const where: any = { 
    sessionId,
    acknowledged: false
  };
  
  if (type) where.type = type;
  if (severity) where.severity = severity;

  const result = await prisma.alert.updateMany({
    where,
    data: {
      acknowledged: true,
      acknowledgedAt: new Date()
    }
  });

  res.json({
    success: true,
    data: { 
      acknowledgedCount: result.count 
    },
    message: `${result.count} alerts acknowledged successfully`
  });
}));

// GET /api/alerts/session/:sessionId/summary - Get alert summary for a session
router.get('/session/:sessionId/summary', asyncHandler(async (req, res) => {
  const { sessionId } = req.params;
  const { timeRange = '1h' } = req.query;

  // Calculate time range
  let startTime: Date | undefined;
  if (timeRange !== 'all') {
    const now = new Date();
    const minutes = {
      '15m': 15,
      '1h': 60,
      '4h': 240,
      '24h': 1440
    }[timeRange as string] || 60;
    
    startTime = new Date(now.getTime() - minutes * 60 * 1000);
  }

  const where: any = { sessionId };
  if (startTime) {
    where.timestamp = { gte: startTime };
  }

  // Get all alerts for the time range
  const alerts = await prisma.alert.findMany({
    where,
    orderBy: { timestamp: 'desc' }
  });

  // Calculate summary statistics
  const summary = {
    total: alerts.length,
    acknowledged: alerts.filter(a => a.acknowledged).length,
    unacknowledged: alerts.filter(a => !a.acknowledged).length,
    byType: alerts.reduce((acc: any, alert) => {
      acc[alert.type] = (acc[alert.type] || 0) + 1;
      return acc;
    }, {}),
    bySeverity: alerts.reduce((acc: any, alert) => {
      acc[alert.severity] = (acc[alert.severity] || 0) + 1;
      return acc;
    }, {}),
    byZone: alerts.reduce((acc: any, alert) => {
      if (alert.zone) {
        acc[alert.zone] = (acc[alert.zone] || 0) + 1;
      }
      return acc;
    }, {}),
    timeline: alerts.map(alert => ({
      id: alert.id,
      timestamp: alert.timestamp,
      type: alert.type,
      severity: alert.severity,
      zone: alert.zone,
      acknowledged: alert.acknowledged
    })),
    recentAlerts: alerts.slice(0, 10), // Last 10 alerts
    criticalAlerts: alerts.filter(a => a.severity === 'CRITICAL' && !a.acknowledged)
  };

  res.json({
    success: true,
    data: { summary }
  });
}));

// DELETE /api/alerts/:id - Delete an alert (admin only)
router.delete('/:id', asyncHandler(async (req, res) => {
  const { id } = req.params;

  const alert = await prisma.alert.findUnique({
    where: { id }
  });

  if (!alert) {
    throw new CustomError('Alert not found', 404);
  }

  await prisma.alert.delete({
    where: { id }
  });

  res.json({
    success: true,
    message: 'Alert deleted successfully'
  });
}));

// POST /api/alerts/test - Create test alert (development only)
router.post('/test', asyncHandler(async (req, res) => {
  if (process.env.NODE_ENV === 'production') {
    throw new CustomError('Test alerts not allowed in production', 403);
  }

  const { sessionId, type = 'DISENGAGEMENT', severity = 'MEDIUM', message, zone } = req.body;

  if (!sessionId) {
    throw new CustomError('Session ID is required', 400);
  }

  // Validate session exists
  const session = await prisma.session.findUnique({
    where: { id: sessionId }
  });

  if (!session) {
    throw new CustomError('Session not found', 404);
  }

  const alert = await prisma.alert.create({
    data: {
      sessionId,
      type: type as any,
      severity: severity as any,
      zone: zone || null,
      message: message || `Test ${type.toLowerCase()} alert`,
      triggerValue: Math.random(),
      threshold: 0.6
    }
  });

  // Add to cache
  await cache.addAlert(sessionId, alert);

  res.status(201).json({
    success: true,
    data: { alert },
    message: 'Test alert created successfully'
  });
}));

export default router;
