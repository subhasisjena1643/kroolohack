import { Server as SocketIOServer, Socket } from 'socket.io';
import { logger } from '../utils/logger';
import { RealTimeCache } from '../config/redis';
import { prisma } from '../config/database';

interface SocketData {
  sessionId?: string;
  userId?: string;
  role?: string;
}

export function initializeSocketHandlers(io: SocketIOServer): void {
  const cache = RealTimeCache.getInstance();

  io.on('connection', (socket: Socket) => {
    logger.info(`Client connected: ${socket.id}`);

    // Handle session join
    socket.on('join-session', async (data: { sessionId: string; userId?: string; role?: string }) => {
      try {
        const { sessionId, userId, role } = data;
        
        // Validate session exists
        const session = await prisma.session.findUnique({
          where: { id: sessionId }
        });

        if (!session) {
          socket.emit('error', { message: 'Session not found' });
          return;
        }

        // Join session room
        socket.join(`session:${sessionId}`);
        socket.data = { sessionId, userId, role } as SocketData;

        // Track connection in Redis
        await cache.addConnection(sessionId, socket.id);

        // Send current session state
        const sessionState = await cache.getSessionState(sessionId);
        const latestMetrics = await cache.getLatestMetrics(sessionId);
        const activeAlerts = await cache.getActiveAlerts(sessionId);

        socket.emit('session-joined', {
          sessionId,
          sessionState,
          latestMetrics,
          activeAlerts,
          connectionCount: await cache.getConnectionCount(sessionId)
        });

        // Notify others in the session
        socket.to(`session:${sessionId}`).emit('user-joined', {
          socketId: socket.id,
          userId,
          role,
          connectionCount: await cache.getConnectionCount(sessionId)
        });

        logger.info(`Client ${socket.id} joined session ${sessionId}`);
      } catch (error) {
        logger.error('Error joining session:', error);
        socket.emit('error', { message: 'Failed to join session' });
      }
    });

    // Handle session leave
    socket.on('leave-session', async () => {
      const socketData = socket.data as SocketData;
      if (socketData?.sessionId) {
        await handleSessionLeave(socket, socketData.sessionId, cache);
      }
    });

    // Handle real-time metrics update (from AI pipeline)
    socket.on('metrics-update', async (data: {
      sessionId: string;
      metrics: any;
      timestamp?: string;
    }) => {
      try {
        const { sessionId, metrics, timestamp } = data;
        
        // Validate session
        const session = await prisma.session.findUnique({
          where: { id: sessionId }
        });

        if (!session) {
          socket.emit('error', { message: 'Session not found' });
          return;
        }

        // Store metrics in cache for real-time access
        await cache.setLatestMetrics(sessionId, {
          ...metrics,
          timestamp: timestamp || new Date().toISOString()
        });

        // Store in database for persistence
        await prisma.sessionMetric.create({
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

        // Broadcast to all clients in the session
        io.to(`session:${sessionId}`).emit('metrics-updated', {
          sessionId,
          metrics: {
            ...metrics,
            timestamp: timestamp || new Date().toISOString()
          }
        });

        // Check for alerts
        await checkAndTriggerAlerts(sessionId, metrics, io, cache);

        logger.debug(`Metrics updated for session ${sessionId}`);
      } catch (error) {
        logger.error('Error updating metrics:', error);
        socket.emit('error', { message: 'Failed to update metrics' });
      }
    });

    // Handle alert acknowledgment
    socket.on('acknowledge-alert', async (data: { alertId: string }) => {
      try {
        const { alertId } = data;
        
        await prisma.alert.update({
          where: { id: alertId },
          data: {
            acknowledged: true,
            acknowledgedAt: new Date()
          }
        });

        const alert = await prisma.alert.findUnique({
          where: { id: alertId }
        });

        if (alert) {
          io.to(`session:${alert.sessionId}`).emit('alert-acknowledged', {
            alertId,
            acknowledgedAt: new Date().toISOString()
          });
        }

        logger.info(`Alert ${alertId} acknowledged by ${socket.id}`);
      } catch (error) {
        logger.error('Error acknowledging alert:', error);
        socket.emit('error', { message: 'Failed to acknowledge alert' });
      }
    });

    // Handle session control (pause/resume)
    socket.on('session-control', async (data: { sessionId: string; action: 'pause' | 'resume' | 'stop' }) => {
      try {
        const { sessionId, action } = data;
        const socketData = socket.data as SocketData;

        // Only instructors/admins can control sessions
        if (socketData?.role !== 'INSTRUCTOR' && socketData?.role !== 'ADMIN') {
          socket.emit('error', { message: 'Insufficient permissions' });
          return;
        }

        let status;
        switch (action) {
          case 'pause':
            status = 'PAUSED';
            break;
          case 'resume':
            status = 'ACTIVE';
            break;
          case 'stop':
            status = 'COMPLETED';
            break;
          default:
            socket.emit('error', { message: 'Invalid action' });
            return;
        }

        await prisma.session.update({
          where: { id: sessionId },
          data: { 
            status: status as any,
            ...(action === 'stop' && { endTime: new Date() })
          }
        });

        // Update session state in cache
        const sessionState = await cache.getSessionState(sessionId) || {};
        sessionState.status = status;
        await cache.setSessionState(sessionId, sessionState);

        // Broadcast to all clients
        io.to(`session:${sessionId}`).emit('session-status-changed', {
          sessionId,
          status,
          timestamp: new Date().toISOString()
        });

        logger.info(`Session ${sessionId} ${action} by ${socket.id}`);
      } catch (error) {
        logger.error('Error controlling session:', error);
        socket.emit('error', { message: 'Failed to control session' });
      }
    });

    // Handle disconnect
    socket.on('disconnect', async () => {
      const socketData = socket.data as SocketData;
      if (socketData?.sessionId) {
        await handleSessionLeave(socket, socketData.sessionId, cache);
      }
      logger.info(`Client disconnected: ${socket.id}`);
    });
  });
}

async function handleSessionLeave(socket: Socket, sessionId: string, cache: RealTimeCache): Promise<void> {
  try {
    // Remove from session room
    socket.leave(`session:${sessionId}`);
    
    // Remove connection tracking
    await cache.removeConnection(sessionId, socket.id);
    
    // Notify others
    socket.to(`session:${sessionId}`).emit('user-left', {
      socketId: socket.id,
      connectionCount: await cache.getConnectionCount(sessionId)
    });
    
    logger.info(`Client ${socket.id} left session ${sessionId}`);
  } catch (error) {
    logger.error('Error handling session leave:', error);
  }
}

async function checkAndTriggerAlerts(
  sessionId: string, 
  metrics: any, 
  io: SocketIOServer, 
  cache: RealTimeCache
): Promise<void> {
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
        }
      });
    }
  } catch (error) {
    logger.error('Error checking alerts:', error);
  }
}
