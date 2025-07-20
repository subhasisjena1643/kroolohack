import { Router } from 'express';
import { prisma } from '../config/database';
import { asyncHandler, CustomError } from '../middleware/errorHandler';
import { RealTimeCache } from '../config/redis';
import { lenientRateLimiter } from '../middleware/rateLimiter';

const router = Router();
// Cache will be initialized when needed

// Apply lenient rate limiting for metrics endpoints
router.use(lenientRateLimiter);

// GET /api/metrics/session/:sessionId - Get metrics for a session
router.get('/session/:sessionId', asyncHandler(async (req, res) => {
  const { sessionId } = req.params;
  const { 
    limit = 100, 
    offset = 0, 
    startTime, 
    endTime,
    interval = '1m' // 1m, 5m, 15m, 1h
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
  
  if (startTime || endTime) {
    where.timestamp = {};
    if (startTime) where.timestamp.gte = new Date(startTime as string);
    if (endTime) where.timestamp.lte = new Date(endTime as string);
  }

  // Get metrics
  const metrics = await prisma.sessionMetric.findMany({
    where,
    orderBy: { timestamp: 'desc' },
    take: parseInt(limit as string),
    skip: parseInt(offset as string)
  });

  // Get aggregated data based on interval
  let aggregatedData = null;
  if (interval !== 'raw') {
    aggregatedData = await getAggregatedMetrics(sessionId, interval as string, startTime as string, endTime as string);
  }

  // Get latest real-time metrics
  let latestMetrics = null;
  try {
    const cache = RealTimeCache.getInstance();
    latestMetrics = await cache.getLatestMetrics(sessionId);
  } catch (error) {
    console.warn('Redis not available, skipping cache data');
  }

  res.json({
    success: true,
    data: {
      metrics,
      aggregated: aggregatedData,
      latest: latestMetrics,
      pagination: {
        limit: parseInt(limit as string),
        offset: parseInt(offset as string),
        hasMore: metrics.length === parseInt(limit as string)
      }
    }
  });
}));

// GET /api/metrics/session/:sessionId/latest - Get latest metrics
router.get('/session/:sessionId/latest', asyncHandler(async (req, res) => {
  const { sessionId } = req.params;

  // Get from cache first (fastest)
  let cachedMetrics = null;
  try {
    const cache = RealTimeCache.getInstance();
    cachedMetrics = await cache.getLatestMetrics(sessionId);
  } catch (error) {
    console.warn('Redis not available, falling back to database');
  }
  
  if (cachedMetrics) {
    res.json({
      success: true,
      data: { metrics: cachedMetrics },
      source: 'cache'
    });
    return;
  }

  // Fallback to database
  const latestMetric = await prisma.sessionMetric.findFirst({
    where: { sessionId },
    orderBy: { timestamp: 'desc' }
  });

  if (!latestMetric) {
    throw new CustomError('No metrics found for this session', 404);
  }

  res.json({
    success: true,
    data: { metrics: latestMetric },
    source: 'database'
  });
}));

// GET /api/metrics/session/:sessionId/stats - Get statistical summary
router.get('/session/:sessionId/stats', asyncHandler(async (req, res) => {
  const { sessionId } = req.params;
  const { timeRange = '1h' } = req.query; // 15m, 1h, 4h, 24h, all

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

  // Get metrics for the time range
  const metrics = await prisma.sessionMetric.findMany({
    where,
    orderBy: { timestamp: 'asc' }
  });

  if (metrics.length === 0) {
    res.json({
      success: true,
      data: {
        stats: null,
        message: 'No metrics found for the specified time range'
      }
    });
    return;
  }

  // Calculate statistics
  const stats = {
    timeRange: {
      start: metrics[0].timestamp,
      end: metrics[metrics.length - 1].timestamp,
      duration: Math.round((metrics[metrics.length - 1].timestamp.getTime() - metrics[0].timestamp.getTime()) / 1000 / 60), // minutes
      dataPoints: metrics.length
    },
    attendance: {
      current: metrics[metrics.length - 1].attendanceCount,
      average: Math.round(metrics.reduce((sum, m) => sum + m.attendanceCount, 0) / metrics.length),
      peak: Math.max(...metrics.map(m => m.attendanceCount)),
      low: Math.min(...metrics.map(m => m.attendanceCount)),
      confidence: Math.round((metrics.reduce((sum, m) => sum + m.attendanceConfidence, 0) / metrics.length) * 100) / 100
    },
    engagement: {
      current: Math.round(metrics[metrics.length - 1].overallEngagement * 100) / 100,
      average: Math.round((metrics.reduce((sum, m) => sum + m.overallEngagement, 0) / metrics.length) * 100) / 100,
      peak: Math.round(Math.max(...metrics.map(m => m.overallEngagement)) * 100) / 100,
      low: Math.round(Math.min(...metrics.map(m => m.overallEngagement)) * 100) / 100,
      trend: calculateTrend(metrics.map(m => m.overallEngagement))
    },
    attention: {
      current: Math.round(metrics[metrics.length - 1].attentionScore * 100) / 100,
      average: Math.round((metrics.reduce((sum, m) => sum + m.attentionScore, 0) / metrics.length) * 100) / 100,
      trend: calculateTrend(metrics.map(m => m.attentionScore))
    },
    participation: {
      current: Math.round(metrics[metrics.length - 1].participationScore * 100) / 100,
      average: Math.round((metrics.reduce((sum, m) => sum + m.participationScore, 0) / metrics.length) * 100) / 100,
      trend: calculateTrend(metrics.map(m => m.participationScore))
    },
    audio: {
      currentNoise: Math.round(metrics[metrics.length - 1].noiseLevel * 100) / 100,
      averageNoise: Math.round((metrics.reduce((sum, m) => sum + m.noiseLevel, 0) / metrics.length) * 100) / 100,
      currentActivity: Math.round(metrics[metrics.length - 1].speakerActivity * 100) / 100,
      averageActivity: Math.round((metrics.reduce((sum, m) => sum + m.speakerActivity, 0) / metrics.length) * 100) / 100
    },
    zones: calculateZoneStats(metrics)
  };

  res.json({
    success: true,
    data: { stats }
  });
}));

// Helper function to get aggregated metrics
async function getAggregatedMetrics(sessionId: string, interval: string, startTime?: string, endTime?: string) {
  // This would typically use a time-series database or SQL window functions
  // For now, we'll do basic aggregation
  
  const intervalMinutes = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '1h': 60
  }[interval] || 5;

  const where: any = { sessionId };
  if (startTime || endTime) {
    where.timestamp = {};
    if (startTime) where.timestamp.gte = new Date(startTime);
    if (endTime) where.timestamp.lte = new Date(endTime);
  }

  const metrics = await prisma.sessionMetric.findMany({
    where,
    orderBy: { timestamp: 'asc' }
  });

  // Group by time intervals
  const grouped = metrics.reduce((acc: any, metric) => {
    const intervalStart = new Date(
      Math.floor(metric.timestamp.getTime() / (intervalMinutes * 60 * 1000)) * (intervalMinutes * 60 * 1000)
    );
    const key = intervalStart.toISOString();
    
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(metric);
    return acc;
  }, {});

  // Calculate averages for each interval
  return Object.entries(grouped).map(([timestamp, intervalMetrics]: [string, any]) => ({
    timestamp,
    attendanceCount: Math.round(intervalMetrics.reduce((sum: number, m: any) => sum + m.attendanceCount, 0) / intervalMetrics.length),
    attendanceConfidence: Math.round((intervalMetrics.reduce((sum: number, m: any) => sum + m.attendanceConfidence, 0) / intervalMetrics.length) * 100) / 100,
    overallEngagement: Math.round((intervalMetrics.reduce((sum: number, m: any) => sum + m.overallEngagement, 0) / intervalMetrics.length) * 100) / 100,
    attentionScore: Math.round((intervalMetrics.reduce((sum: number, m: any) => sum + m.attentionScore, 0) / intervalMetrics.length) * 100) / 100,
    participationScore: Math.round((intervalMetrics.reduce((sum: number, m: any) => sum + m.participationScore, 0) / intervalMetrics.length) * 100) / 100,
    noiseLevel: Math.round((intervalMetrics.reduce((sum: number, m: any) => sum + m.noiseLevel, 0) / intervalMetrics.length) * 100) / 100,
    speakerActivity: Math.round((intervalMetrics.reduce((sum: number, m: any) => sum + m.speakerActivity, 0) / intervalMetrics.length) * 100) / 100,
    dataPoints: intervalMetrics.length
  }));
}

// Helper function to calculate trend
function calculateTrend(values: number[]): 'increasing' | 'decreasing' | 'stable' {
  if (values.length < 2) return 'stable';
  
  const recent = values.slice(-Math.min(10, Math.floor(values.length / 2))); // Last 10 or half of values
  const earlier = values.slice(0, Math.min(10, Math.floor(values.length / 2))); // First 10 or half of values
  
  const recentAvg = recent.reduce((sum, val) => sum + val, 0) / recent.length;
  const earlierAvg = earlier.reduce((sum, val) => sum + val, 0) / earlier.length;
  
  const difference = recentAvg - earlierAvg;
  const threshold = 0.05; // 5% threshold
  
  if (difference > threshold) return 'increasing';
  if (difference < -threshold) return 'decreasing';
  return 'stable';
}

// Helper function to calculate zone statistics
function calculateZoneStats(metrics: any[]) {
  const zones = ['front', 'middle', 'back'];
  const zoneStats: any = {};
  
  zones.forEach(zone => {
    const zoneKey = `${zone}ZoneEngagement`;
    const zoneValues = metrics
      .map(m => m[zoneKey])
      .filter(val => val !== null && val !== undefined);
    
    if (zoneValues.length > 0) {
      zoneStats[zone] = {
        current: Math.round(zoneValues[zoneValues.length - 1] * 100) / 100,
        average: Math.round((zoneValues.reduce((sum, val) => sum + val, 0) / zoneValues.length) * 100) / 100,
        peak: Math.round(Math.max(...zoneValues) * 100) / 100,
        low: Math.round(Math.min(...zoneValues) * 100) / 100,
        trend: calculateTrend(zoneValues)
      };
    }
  });
  
  return zoneStats;
}

export default router;
