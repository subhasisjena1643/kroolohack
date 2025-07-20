import { Router } from 'express';
import { prisma } from '../config/database';
import { asyncHandler, CustomError } from '../middleware/errorHandler';
import { strictRateLimiter } from '../middleware/rateLimiter';
import { logger } from '../utils/logger';
import { io } from '../index';
import { RealTimeCache } from '../config/redis';

const router = Router();
// Cache will be initialized when needed

// Apply strict rate limiting for AI endpoints
router.use(strictRateLimiter);

// POST /api/ai/metrics - Receive metrics from AI pipeline
router.post('/metrics', asyncHandler(async (req, res) => {
  const { sessionId, metrics, timestamp } = req.body;

  if (!sessionId || !metrics) {
    throw new CustomError('Session ID and metrics are required', 400);
  }

  // Validate session exists and is active
  const session = await prisma.session.findUnique({
    where: { id: sessionId }
  });

  if (!session) {
    throw new CustomError('Session not found', 404);
  }

  if (session.status !== 'ACTIVE') {
    throw new CustomError('Session is not active', 400);
  }

  try {
    // Store metrics in cache for real-time access
    const metricsWithTimestamp = {
      ...metrics,
      timestamp: timestamp || new Date().toISOString(),
      sessionId
    };

    await cache.setLatestMetrics(sessionId, metricsWithTimestamp);

    // Store in database for persistence
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

    // Broadcast to all clients in the session via WebSocket
    io.to(`session:${sessionId}`).emit('metrics-updated', {
      sessionId,
      metrics: metricsWithTimestamp,
      dbId: dbMetric.id
    });

    // Check for alerts
    await checkAndTriggerAlerts(sessionId, metrics);

    logger.debug(`AI metrics received for session ${sessionId}`);

    res.json({
      success: true,
      data: {
        metricId: dbMetric.id,
        timestamp: dbMetric.timestamp
      },
      message: 'Metrics processed successfully'
    });

  } catch (error) {
    logger.error('Error processing AI metrics:', error);
    throw new CustomError('Failed to process metrics', 500);
  }
}));

// POST /api/ai/batch-metrics - Receive batch metrics from AI pipeline
router.post('/batch-metrics', asyncHandler(async (req, res) => {
  const { sessionId, metricsArray } = req.body;

  if (!sessionId || !Array.isArray(metricsArray) || metricsArray.length === 0) {
    throw new CustomError('Session ID and metrics array are required', 400);
  }

  // Validate session
  const session = await prisma.session.findUnique({
    where: { id: sessionId }
  });

  if (!session) {
    throw new CustomError('Session not found', 404);
  }

  try {
    const dbMetrics = [];
    
    // Process each metric in the batch
    for (const metrics of metricsArray) {
      const timestamp = metrics.timestamp ? new Date(metrics.timestamp) : new Date();
      
      const dbMetric = await prisma.sessionMetric.create({
        data: {
          sessionId,
          timestamp,
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
      
      dbMetrics.push(dbMetric);
    }

    // Update cache with latest metrics
    const latestMetrics = metricsArray[metricsArray.length - 1];
    await cache.setLatestMetrics(sessionId, {
      ...latestMetrics,
      timestamp: latestMetrics.timestamp || new Date().toISOString(),
      sessionId
    });

    // Broadcast latest metrics
    io.to(`session:${sessionId}`).emit('batch-metrics-updated', {
      sessionId,
      metricsCount: metricsArray.length,
      latestMetrics: {
        ...latestMetrics,
        timestamp: latestMetrics.timestamp || new Date().toISOString()
      }
    });

    logger.info(`Batch metrics processed for session ${sessionId}: ${metricsArray.length} records`);

    res.json({
      success: true,
      data: {
        processedCount: dbMetrics.length,
        metricIds: dbMetrics.map(m => m.id)
      },
      message: 'Batch metrics processed successfully'
    });

  } catch (error) {
    logger.error('Error processing batch metrics:', error);
    throw new CustomError('Failed to process batch metrics', 500);
  }
}));

// POST /api/ai/alert - Trigger alert from AI pipeline
router.post('/alert', asyncHandler(async (req, res) => {
  const { sessionId, alertData } = req.body;

  if (!sessionId || !alertData) {
    throw new CustomError('Session ID and alert data are required', 400);
  }

  // Validate session
  const session = await prisma.session.findUnique({
    where: { id: sessionId }
  });

  if (!session) {
    throw new CustomError('Session not found', 404);
  }

  try {
    const alert = await prisma.alert.create({
      data: {
        sessionId,
        type: alertData.type || 'TECHNICAL_ISSUE',
        severity: alertData.severity || 'MEDIUM',
        zone: alertData.zone || null,
        message: alertData.message || 'Alert from AI pipeline',
        triggerValue: alertData.triggerValue || null,
        threshold: alertData.threshold || null
      }
    });

    // Cache the alert
    await cache.addAlert(sessionId, alert);

    // Broadcast to clients
    io.to(`session:${sessionId}`).emit('alert-triggered', {
      alert: {
        ...alert,
        timestamp: alert.timestamp.toISOString()
      },
      source: 'ai-pipeline'
    });

    logger.info(`AI alert triggered for session ${sessionId}: ${alertData.type}`);

    res.status(201).json({
      success: true,
      data: { alert },
      message: 'Alert triggered successfully'
    });

  } catch (error) {
    logger.error('Error triggering AI alert:', error);
    throw new CustomError('Failed to trigger alert', 500);
  }
}));

// GET /api/ai/session/:sessionId/status - Get AI processing status
router.get('/session/:sessionId/status', asyncHandler(async (req, res) => {
  const { sessionId } = req.params;

  // Get latest metrics to determine AI status
  const latestMetrics = await cache.getLatestMetrics(sessionId);
  const lastDbMetric = await prisma.sessionMetric.findFirst({
    where: { sessionId },
    orderBy: { timestamp: 'desc' }
  });

  const now = new Date();
  const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000);

  const isActive = latestMetrics && new Date(latestMetrics.timestamp) > fiveMinutesAgo;
  const lastActivity = lastDbMetric ? lastDbMetric.timestamp : null;

  const status = {
    isActive,
    lastActivity,
    latestMetrics: latestMetrics ? {
      timestamp: latestMetrics.timestamp,
      attendance: latestMetrics.attendance?.total_detected || 0,
      engagement: latestMetrics.engagement?.overall_score || 0
    } : null,
    healthCheck: {
      cacheConnected: true, // We got here, so cache is working
      databaseConnected: true, // We got here, so DB is working
      lastHeartbeat: now.toISOString()
    }
  };

  res.json({
    success: true,
    data: { status }
  });
}));

// POST /api/ai/heartbeat - AI pipeline heartbeat
router.post('/heartbeat', asyncHandler(async (req, res) => {
  const { sessionId, status, metadata } = req.body;

  if (!sessionId) {
    throw new CustomError('Session ID is required', 400);
  }

  // Update session state in cache
  const sessionState = await cache.getSessionState(sessionId) || {};
  sessionState.aiStatus = {
    status: status || 'active',
    lastHeartbeat: new Date().toISOString(),
    metadata: metadata || {}
  };
  await cache.setSessionState(sessionId, sessionState);

  res.json({
    success: true,
    message: 'Heartbeat received',
    timestamp: new Date().toISOString()
  });
}));

// Helper function to check and trigger alerts
async function checkAndTriggerAlerts(sessionId: string, metrics: any): Promise<void> {
  try {
    const alerts = [];

    // Check engagement thresholds
    const engagementThreshold = parseFloat(process.env.ENGAGEMENT_ALERT_THRESHOLD || '0.6');
    if (metrics.engagement?.overall_score < engagementThreshold) {
      alerts.push({
        type: 'DISENGAGEMENT',
        severity: metrics.engagement.overall_score < 0.4 ? 'HIGH' : 'MEDIUM',
        message: `Overall engagement dropped to ${(metrics.engagement.overall_score * 100).toFixed(1)}%`,
        triggerValue: metrics.engagement.overall_score,
        threshold: engagementThreshold
      });
    }

    // Check zone-specific engagement
    if (metrics.engagement?.zones) {
      Object.entries(metrics.engagement.zones).forEach(([zone, score]: [string, any]) => {
        if (score < engagementThreshold) {
          alerts.push({
            type: 'DISENGAGEMENT',
            severity: score < 0.4 ? 'HIGH' : 'MEDIUM',
            zone: zone,
            message: `${zone} zone engagement dropped to ${(score * 100).toFixed(1)}%`,
            triggerValue: score,
            threshold: engagementThreshold
          });
        }
      });
    }

    // Check noise levels
    const noiseThreshold = parseFloat(process.env.NOISE_ALERT_THRESHOLD || '0.7');
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
          type: alertData.type as any,
          severity: alertData.severity as any,
          zone: alertData.zone || null,
          message: alertData.message,
          triggerValue: alertData.triggerValue,
          threshold: alertData.threshold
        }
      });

      // Cache the alert
      await cache.addAlert(sessionId, alert);

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
    logger.error('Error checking alerts:', error);
  }
}

export default router;
