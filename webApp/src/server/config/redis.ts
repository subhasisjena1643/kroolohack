import { createClient, RedisClientType } from 'redis';
import { logger } from '../utils/logger';

let redisClient: RedisClientType;

export async function connectRedis(): Promise<RedisClientType> {
  try {
    redisClient = createClient({
      url: process.env.REDIS_URL || 'redis://localhost:6379',
      socket: {
        reconnectStrategy: (retries) => Math.min(retries * 50, 1000)
      }
    });

    redisClient.on('error', (err) => {
      logger.error('Redis Client Error:', err);
    });

    redisClient.on('connect', () => {
      logger.info('Redis Client Connected');
    });

    redisClient.on('ready', () => {
      logger.info('Redis Client Ready');
    });

    redisClient.on('end', () => {
      logger.info('Redis Client Disconnected');
    });

    await redisClient.connect();
    
    // Test the connection
    await redisClient.ping();
    logger.info('Redis connection established successfully');
    
    return redisClient;
  } catch (error) {
    logger.error('Failed to connect to Redis:', error);
    throw error;
  }
}

export function getRedisClient(): RedisClientType {
  if (!redisClient) {
    throw new Error('Redis client not initialized. Call connectRedis() first.');
  }
  return redisClient;
}

// Real-time data operations
export class RealTimeCache {
  private static instance: RealTimeCache;
  private client: RedisClientType;

  private constructor(client: RedisClientType) {
    this.client = client;
  }

  public static getInstance(): RealTimeCache {
    if (!RealTimeCache.instance) {
      RealTimeCache.instance = new RealTimeCache(getRedisClient());
    }
    return RealTimeCache.instance;
  }

  // Session state management
  async setSessionState(sessionId: string, state: any): Promise<void> {
    const key = `session:${sessionId}:state`;
    await this.client.setEx(key, 3600, JSON.stringify(state)); // 1 hour TTL
  }

  async getSessionState(sessionId: string): Promise<any | null> {
    const key = `session:${sessionId}:state`;
    const data = await this.client.get(key);
    return data ? JSON.parse(data) : null;
  }

  // Real-time metrics
  async setLatestMetrics(sessionId: string, metrics: any): Promise<void> {
    const key = `session:${sessionId}:latest_metrics`;
    await this.client.setEx(key, 300, JSON.stringify(metrics)); // 5 minutes TTL
  }

  async getLatestMetrics(sessionId: string): Promise<any | null> {
    const key = `session:${sessionId}:latest_metrics`;
    const data = await this.client.get(key);
    return data ? JSON.parse(data) : null;
  }

  // Active alerts
  async addAlert(sessionId: string, alert: any): Promise<void> {
    const key = `session:${sessionId}:active_alerts`;
    await this.client.lPush(key, JSON.stringify(alert));
    await this.client.expire(key, 1800); // 30 minutes TTL
    
    // Keep only last 50 alerts
    await this.client.lTrim(key, 0, 49);
  }

  async getActiveAlerts(sessionId: string): Promise<any[]> {
    const key = `session:${sessionId}:active_alerts`;
    const alerts = await this.client.lRange(key, 0, -1);
    return alerts.map(alert => JSON.parse(alert));
  }

  // Connection tracking
  async addConnection(sessionId: string, socketId: string): Promise<void> {
    const key = `session:${sessionId}:connections`;
    await this.client.sAdd(key, socketId);
    await this.client.expire(key, 3600); // 1 hour TTL
  }

  async removeConnection(sessionId: string, socketId: string): Promise<void> {
    const key = `session:${sessionId}:connections`;
    await this.client.sRem(key, socketId);
  }

  async getConnectionCount(sessionId: string): Promise<number> {
    const key = `session:${sessionId}:connections`;
    return await this.client.sCard(key);
  }
}

export async function disconnectRedis(): Promise<void> {
  try {
    if (redisClient) {
      await redisClient.quit();
      logger.info('Redis disconnected successfully');
    }
  } catch (error) {
    logger.error('Error disconnecting from Redis:', error);
  }
}
