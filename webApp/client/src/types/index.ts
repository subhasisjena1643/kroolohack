// Session Types
export interface Session {
  id: string;
  name: string;
  description?: string;
  startTime: string;
  endTime?: string;
  status: 'ACTIVE' | 'PAUSED' | 'COMPLETED' | 'CANCELLED';
  totalStudents?: number;
  classroomLayout?: any;
  createdAt: string;
  updatedAt: string;
}

// Metrics Types
export interface SessionMetric {
  id: string;
  sessionId: string;
  timestamp: string;
  attendanceCount: number;
  attendanceConfidence: number;
  overallEngagement: number;
  attentionScore: number;
  participationScore: number;
  frontZoneEngagement?: number;
  middleZoneEngagement?: number;
  backZoneEngagement?: number;
  noiseLevel: number;
  speakerActivity: number;
  rawData?: any;
}

// Real-time Metrics from AI Pipeline
export interface RealTimeMetrics {
  timestamp: string;
  sessionId: string;
  attendance: {
    total_detected: number;
    confidence_avg: number;
  };
  engagement: {
    overall_score: number;
    attention_score: number;
    participation_score: number;
    zones: {
      front: number;
      middle: number;
      back: number;
    };
  };
  audio: {
    noise_level: number;
    speaker_activity: number;
  };
}

// Alert Types
export interface Alert {
  id: string;
  sessionId: string;
  timestamp: string;
  type: 'DISENGAGEMENT' | 'LOW_ATTENDANCE' | 'HIGH_NOISE' | 'PARTICIPATION_DROP' | 'TECHNICAL_ISSUE';
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  zone?: string;
  message: string;
  triggerValue?: number;
  threshold?: number;
  acknowledged: boolean;
  acknowledgedAt?: string;
}

// Statistics Types
export interface SessionStats {
  timeRange: {
    start: string;
    end: string;
    duration: number;
    dataPoints: number;
  };
  attendance: {
    current: number;
    average: number;
    peak: number;
    low: number;
    confidence: number;
  };
  engagement: {
    current: number;
    average: number;
    peak: number;
    low: number;
    trend: 'increasing' | 'decreasing' | 'stable';
  };
  attention: {
    current: number;
    average: number;
    trend: 'increasing' | 'decreasing' | 'stable';
  };
  participation: {
    current: number;
    average: number;
    trend: 'increasing' | 'decreasing' | 'stable';
  };
  audio: {
    currentNoise: number;
    averageNoise: number;
    currentActivity: number;
    averageActivity: number;
  };
  zones: {
    [key: string]: {
      current: number;
      average: number;
      peak: number;
      low: number;
      trend: 'increasing' | 'decreasing' | 'stable';
    };
  };
}

// Socket Event Types
export interface SocketEvents {
  // Client to Server
  'join-session': (data: { sessionId: string; userId?: string; role?: string }) => void;
  'leave-session': () => void;
  'acknowledge-alert': (data: { alertId: string }) => void;
  'session-control': (data: { sessionId: string; action: 'pause' | 'resume' | 'stop' }) => void;

  // Server to Client
  'session-joined': (data: {
    sessionId: string;
    sessionState: any;
    latestMetrics: RealTimeMetrics;
    activeAlerts: Alert[];
    connectionCount: number;
  }) => void;
  'metrics-updated': (data: {
    sessionId: string;
    metrics: RealTimeMetrics;
    dbId?: string;
  }) => void;
  'alert-triggered': (data: {
    alert: Alert;
    source?: string;
  }) => void;
  'alert-acknowledged': (data: {
    alertId: string;
    acknowledgedAt: string;
  }) => void;
  'session-status-changed': (data: {
    sessionId: string;
    status: string;
    timestamp: string;
  }) => void;
  'user-joined': (data: {
    socketId: string;
    userId?: string;
    role?: string;
    connectionCount: number;
  }) => void;
  'user-left': (data: {
    socketId: string;
    connectionCount: number;
  }) => void;
  'error': (data: { message: string }) => void;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  error?: {
    message: string;
    stack?: string;
  };
  timestamp?: string;
  path?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  pagination: {
    total: number;
    limit: number;
    offset: number;
    hasMore: boolean;
  };
}

// Component Props Types
export interface DashboardProps {
  sessionId: string;
}

export interface MetricCardProps {
  title: string;
  value: number | string;
  unit?: string;
  trend?: 'up' | 'down' | 'stable';
  color?: 'primary' | 'success' | 'warning' | 'danger';
  icon?: React.ReactNode;
  className?: string;
}

export interface AlertBadgeProps {
  alert: Alert;
  onAcknowledge?: (alertId: string) => void;
  compact?: boolean;
}

export interface EngagementHeatmapProps {
  zones: {
    front: number;
    middle: number;
    back: number;
  };
  className?: string;
}

// Chart Data Types
export interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}

export interface TimeSeriesData {
  attendance: ChartDataPoint[];
  engagement: ChartDataPoint[];
  attention: ChartDataPoint[];
  participation: ChartDataPoint[];
  noise: ChartDataPoint[];
}
