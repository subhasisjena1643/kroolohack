import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Users, 
  Brain, 
  Eye, 
  Hand, 
  Volume2, 
  AlertTriangle, 
  Activity,
  Pause,
  Play,
  Square
} from 'lucide-react';
import { apiClient } from '../lib/api';
import { socketClient } from '../lib/socket';
import type { RealTimeMetrics, Alert, Session } from '../types';
import { formatDateTime, formatPercentage, cn } from '../lib/utils';
import MetricCard from './MetricCard';
import EngagementHeatmap from './EngagementHeatmap';
import AlertPanel from './AlertPanel';
import RealTimeChart from './RealTimeChart';
import QuizCreator from './QuizCreator';

interface DashboardProps {
  sessionId: string;
}

export default function Dashboard({ sessionId }: DashboardProps) {
  const [realTimeMetrics, setRealTimeMetrics] = useState<RealTimeMetrics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connectionCount, setConnectionCount] = useState(0);
  const [sessionStatus, setSessionStatus] = useState<string>('ACTIVE');
  const [activeQuiz, setActiveQuiz] = useState<any>(null);
  const [quizResults, setQuizResults] = useState<any>(null);

  // Fetch session data
  const { data: sessionData, isLoading } = useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => apiClient.getSession(sessionId),
    refetchInterval: 60000, // Refetch every minute
  });

  // Fetch session stats
  const { data: statsData } = useQuery({
    queryKey: ['session-stats', sessionId],
    queryFn: () => apiClient.getSessionStats(sessionId, { timeRange: '1h' }),
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  // Set up real-time socket connection
  useEffect(() => {
    const socket = socketClient.connect();
    if (!socket) return;

    // Wait for connection before joining session
    const handleConnect = () => {
      console.log('Socket connected, joining session...');
      socketClient.joinSession(sessionId, 'user-123', 'INSTRUCTOR');
    };

    if (socket.connected) {
      handleConnect();
    } else {
      socket.on('connect', handleConnect);
    }

    // Set up event listeners
    const handleSessionJoined = (data: any) => {
      console.log('Joined session:', data);
      setRealTimeMetrics(data.latestMetrics);
      setAlerts(data.activeAlerts || []);
      setConnectionCount(data.connectionCount || 0);
    };

    const handleMetricsUpdated = (data: any) => {
      console.log('Metrics updated:', data);
      setRealTimeMetrics(data.metrics);
    };

    const handleAlertTriggered = (data: any) => {
      console.log('Alert triggered:', data);
      setAlerts(prev => [data.alert, ...prev.slice(0, 9)]); // Keep last 10 alerts
    };

    const handleAlertAcknowledged = (data: any) => {
      console.log('Alert acknowledged:', data);
      setAlerts(prev => 
        prev.map(alert => 
          alert.id === data.alertId 
            ? { ...alert, acknowledged: true, acknowledgedAt: data.acknowledgedAt }
            : alert
        )
      );
    };

    const handleSessionStatusChanged = (data: any) => {
      console.log('Session status changed:', data);
      setSessionStatus(data.status);
    };

    const handleQuizResultsUpdated = (data: any) => {
      console.log('Quiz results updated:', data);
      setQuizResults(data);
    };

    const handleUserJoined = (data: any) => {
      setConnectionCount(data.connectionCount);
    };

    const handleUserLeft = (data: any) => {
      setConnectionCount(data.connectionCount);
    };

    const handleError = (data: any) => {
      console.error('Socket error:', data);
    };

    // Register event listeners
    socketClient.onSessionJoined(handleSessionJoined);
    socketClient.onMetricsUpdated(handleMetricsUpdated);
    socketClient.onAlertTriggered(handleAlertTriggered);
    socketClient.onAlertAcknowledged(handleAlertAcknowledged);
    socketClient.onSessionStatusChanged(handleSessionStatusChanged);
    socketClient.onUserJoined(handleUserJoined);

    // Quiz event listeners
    socket.on('quiz-results-updated', handleQuizResultsUpdated);
    socketClient.onUserLeft(handleUserLeft);
    socketClient.onError(handleError);

    // Cleanup
    return () => {
      socketClient.offSessionJoined(handleSessionJoined);
      socketClient.offMetricsUpdated(handleMetricsUpdated);
      socketClient.offAlertTriggered(handleAlertTriggered);
      socketClient.offAlertAcknowledged(handleAlertAcknowledged);
      socketClient.offSessionStatusChanged(handleSessionStatusChanged);
      socketClient.offUserJoined(handleUserJoined);
      socketClient.offUserLeft(handleUserLeft);
      socketClient.offError(handleError);
    };
  }, [sessionId]);

  const handleSessionControl = (action: 'pause' | 'resume' | 'stop') => {
    socketClient.controlSession(sessionId, action);
  };

  const handleAcknowledgeAlert = (alertId: string) => {
    socketClient.acknowledgeAlert(alertId);
  };

  const handleQuizCreated = (quiz: any) => {
    console.log('Quiz created:', quiz);
    setActiveQuiz(quiz);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  const session = sessionData?.data.session;
  const stats = statsData?.data.stats;

  if (!session) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="mx-auto h-12 w-12 text-danger-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">Session not found</h3>
      </div>
    );
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ACTIVE':
        return <Play className="w-5 h-5 text-success-600" />;
      case 'PAUSED':
        return <Pause className="w-5 h-5 text-warning-600" />;
      case 'COMPLETED':
        return <Square className="w-5 h-5 text-gray-600" />;
      default:
        return <Activity className="w-5 h-5 text-gray-600" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Session Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-3">
              {getStatusIcon(sessionStatus)}
              <h1 className="text-2xl font-bold text-gray-900">{session.name}</h1>
              <span className={cn(
                'px-3 py-1 rounded-full text-sm font-medium',
                sessionStatus === 'ACTIVE' ? 'bg-success-100 text-success-800' :
                sessionStatus === 'PAUSED' ? 'bg-warning-100 text-warning-800' :
                'bg-gray-100 text-gray-800'
              )}>
                {sessionStatus}
              </span>
            </div>
            {session.description && (
              <p className="mt-2 text-gray-600">{session.description}</p>
            )}
            <div className="mt-2 flex items-center space-x-4 text-sm text-gray-500">
              <span>Started: {formatDateTime(session.startTime)}</span>
              <span>•</span>
              <span>Viewers: {connectionCount}</span>
              {session.totalStudents && (
                <>
                  <span>•</span>
                  <span>Expected: {session.totalStudents} students</span>
                </>
              )}
            </div>
          </div>
          
          <div className="mt-4 sm:mt-0 flex space-x-2">
            {sessionStatus === 'ACTIVE' && (
              <button
                onClick={() => handleSessionControl('pause')}
                className="btn-warning flex items-center space-x-2"
              >
                <Pause className="w-4 h-4" />
                <span>Pause</span>
              </button>
            )}
            {sessionStatus === 'PAUSED' && (
              <button
                onClick={() => handleSessionControl('resume')}
                className="btn-success flex items-center space-x-2"
              >
                <Play className="w-4 h-4" />
                <span>Resume</span>
              </button>
            )}
            <button
              onClick={() => handleSessionControl('stop')}
              className="btn-danger flex items-center space-x-2"
            >
              <Square className="w-4 h-4" />
              <span>Stop</span>
            </button>
          </div>
        </div>
      </div>

      {/* Real-time Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Attendance"
          value={realTimeMetrics?.attendance.total_detected || stats?.attendance.current || 0}
          unit="students"
          trend={stats?.attendance.current > (stats?.attendance.average || 0) ? 'up' : 'stable'}
          color="primary"
          icon={<Users className="w-5 h-5" />}
        />
        
        <MetricCard
          title="Engagement"
          value={formatPercentage(realTimeMetrics?.engagement.overall_score || stats?.engagement.current || 0)}
          trend={stats?.engagement.trend === 'increasing' ? 'up' : 
                 stats?.engagement.trend === 'decreasing' ? 'down' : 'stable'}
          color={
            (realTimeMetrics?.engagement.overall_score || stats?.engagement.current || 0) >= 0.8 ? 'success' :
            (realTimeMetrics?.engagement.overall_score || stats?.engagement.current || 0) >= 0.6 ? 'warning' : 'danger'
          }
          icon={<Brain className="w-5 h-5" />}
        />
        
        <MetricCard
          title="Attention"
          value={formatPercentage(realTimeMetrics?.engagement.attention_score || stats?.attention.current || 0)}
          trend={stats?.attention.trend === 'increasing' ? 'up' : 
                 stats?.attention.trend === 'decreasing' ? 'down' : 'stable'}
          color="primary"
          icon={<Eye className="w-5 h-5" />}
        />
        
        <MetricCard
          title="Participation"
          value={formatPercentage(realTimeMetrics?.engagement.participation_score || stats?.participation.current || 0)}
          trend={stats?.participation.trend === 'increasing' ? 'up' : 
                 stats?.participation.trend === 'decreasing' ? 'down' : 'stable'}
          color="primary"
          icon={<Hand className="w-5 h-5" />}
        />
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Engagement Heatmap */}
        <div className="lg:col-span-2">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Zone Engagement
            </h3>
            <EngagementHeatmap
              zones={realTimeMetrics?.engagement.zones || { front: 0, middle: 0, back: 0 }}
            />
          </div>
        </div>

        {/* Alerts Panel */}
        <div className="space-y-6">
          <QuizCreator
            sessionId={sessionId}
            onQuizCreated={handleQuizCreated}
          />

          {/* Student Access */}
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-3">Student Access</h3>
            <p className="text-sm text-gray-600 mb-3">
              Share this link with students to join quizzes and polls:
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={`${window.location.origin}/student/${sessionId}`}
                readOnly
                className="flex-1 p-2 text-sm border border-gray-300 rounded bg-gray-50"
              />
              <button
                onClick={() => navigator.clipboard.writeText(`${window.location.origin}/student/${sessionId}`)}
                className="px-3 py-2 bg-primary-600 text-white text-sm rounded hover:bg-primary-700"
              >
                Copy
              </button>
            </div>
          </div>

          <AlertPanel
            alerts={alerts}
            onAcknowledge={handleAcknowledgeAlert}
          />
        </div>
      </div>

      {/* Real-time Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Engagement Over Time
          </h3>
          <RealTimeChart
            sessionId={sessionId}
            metric="engagement"
            color="#3b82f6"
          />
        </div>
        
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Attendance Tracking
          </h3>
          <RealTimeChart
            sessionId={sessionId}
            metric="attendance"
            color="#22c55e"
          />
        </div>
      </div>

      {/* Audio Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <MetricCard
          title="Noise Level"
          value={formatPercentage(realTimeMetrics?.audio.noise_level || stats?.audio.currentNoise || 0)}
          color={
            (realTimeMetrics?.audio.noise_level || stats?.audio.currentNoise || 0) > 0.7 ? 'danger' :
            (realTimeMetrics?.audio.noise_level || stats?.audio.currentNoise || 0) > 0.5 ? 'warning' : 'success'
          }
          icon={<Volume2 className="w-5 h-5" />}
          className="card"
        />
        
        <MetricCard
          title="Speaker Activity"
          value={formatPercentage(realTimeMetrics?.audio.speaker_activity || stats?.audio.currentActivity || 0)}
          color="primary"
          icon={<Activity className="w-5 h-5" />}
          className="card"
        />
      </div>
    </div>
  );
}
