import { Request, Response, NextFunction } from 'express';
import { getRedisClient } from '../config/redis';
import { logger } from '../utils/logger';

interface RateLimitOptions {
  windowMs: number;
  maxRequests: number;
  message?: string;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
}

class RateLimiter {
  private windowMs: number;
  private maxRequests: number;
  private message: string;
  private skipSuccessfulRequests: boolean;
  private skipFailedRequests: boolean;

  constructor(options: RateLimitOptions) {
    this.windowMs = options.windowMs;
    this.maxRequests = options.maxRequests;
    this.message = options.message || 'Too many requests, please try again later.';
    this.skipSuccessfulRequests = options.skipSuccessfulRequests || false;
    this.skipFailedRequests = options.skipFailedRequests || false;
  }

  async middleware(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const redis = getRedisClient();
      const key = `rate_limit:${req.ip}`;
      const now = Date.now();
      const window = Math.floor(now / this.windowMs);
      const windowKey = `${key}:${window}`;

      // Get current count
      const current = await redis.get(windowKey);
      const count = current ? parseInt(current) : 0;

      if (count >= this.maxRequests) {
        logger.warn(`Rate limit exceeded for IP: ${req.ip}`);
        res.status(429).json({
          success: false,
          error: {
            message: this.message,
            retryAfter: Math.ceil(this.windowMs / 1000),
          },
          timestamp: new Date().toISOString(),
        });
        return;
      }

      // Increment counter
      await redis.incr(windowKey);
      await redis.expire(windowKey, Math.ceil(this.windowMs / 1000));

      // Add headers
      res.set({
        'X-RateLimit-Limit': this.maxRequests.toString(),
        'X-RateLimit-Remaining': (this.maxRequests - count - 1).toString(),
        'X-RateLimit-Reset': (now + this.windowMs).toString(),
      });

      next();
    } catch (error) {
      logger.error('Rate limiter error:', error);
      // If Redis is down, allow the request to proceed
      next();
    }
  }
}

// Default rate limiter - 100 requests per 15 minutes
export const rateLimiter = new RateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  maxRequests: 100,
  message: 'Too many requests from this IP, please try again later.',
}).middleware.bind(new RateLimiter({
  windowMs: 15 * 60 * 1000,
  maxRequests: 100,
  message: 'Too many requests from this IP, please try again later.',
}));

// Strict rate limiter for AI endpoints - 30 requests per minute
export const strictRateLimiter = new RateLimiter({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 30,
  message: 'Too many AI requests, please slow down.',
}).middleware.bind(new RateLimiter({
  windowMs: 60 * 1000,
  maxRequests: 30,
  message: 'Too many AI requests, please slow down.',
}));

// Lenient rate limiter for real-time endpoints - 1000 requests per minute
export const lenientRateLimiter = new RateLimiter({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 1000,
  message: 'Too many real-time requests.',
}).middleware.bind(new RateLimiter({
  windowMs: 60 * 1000,
  maxRequests: 1000,
  message: 'Too many real-time requests.',
}));
