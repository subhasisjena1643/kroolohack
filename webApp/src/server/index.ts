import express from 'express';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import dotenv from 'dotenv';

import { logger } from './utils/logger';
import { connectRedis } from './config/redis';
import { prisma } from './config/database';
import { errorHandler } from './middleware/errorHandler';
import { rateLimiter } from './middleware/rateLimiter';

// Routes
import sessionRoutes from './routes/sessions';
import metricsRoutes from './routes/metrics';
import alertRoutes from './routes/alerts';
import aiRoutes from './routes/ai';

// Socket handlers
import { initializeSocketHandlers } from './sockets/socketHandlers';

// Load environment variables
dotenv.config();

const app = express();
const server = createServer(app);

// Initialize Socket.IO with CORS
const io = new SocketIOServer(server, {
  cors: {
    origin: process.env.SOCKET_IO_CORS_ORIGIN || "http://localhost:5173",
    methods: ["GET", "POST"],
    credentials: true
  },
  transports: ['websocket', 'polling']
});

// Middleware
app.use(helmet({
  crossOriginEmbedderPolicy: false,
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
}));

app.use(compression());
app.use(cors({
  origin: process.env.CORS_ORIGIN || "http://localhost:5173",
  credentials: true
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting
app.use(rateLimiter);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// API Routes
app.use('/api/sessions', sessionRoutes);
app.use('/api/metrics', metricsRoutes);
app.use('/api/alerts', alertRoutes);
app.use('/api/ai', aiRoutes);

// Error handling
app.use(errorHandler);

// Initialize services
async function initializeServices() {
  try {
    // Connect to Redis
    await connectRedis();
    logger.info('Redis connected successfully');

    // Test database connection
    await prisma.$connect();
    logger.info('Database connected successfully');

    // Initialize Socket.IO handlers
    initializeSocketHandlers(io);
    logger.info('Socket.IO handlers initialized');

  } catch (error) {
    logger.error('Failed to initialize services:', error);
    process.exit(1);
  }
}

// Start server
const PORT = process.env.PORT || 3001;

server.listen(PORT, async () => {
  await initializeServices();
  logger.info(`🚀 Server running on port ${PORT}`);
  logger.info(`🌐 Environment: ${process.env.NODE_ENV}`);
  logger.info(`📊 Dashboard: http://localhost:5173`);
  logger.info(`🔌 WebSocket: ws://localhost:${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Process terminated');
  });
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully');
  await prisma.$disconnect();
  server.close(() => {
    logger.info('Process terminated');
  });
});

export { io };
