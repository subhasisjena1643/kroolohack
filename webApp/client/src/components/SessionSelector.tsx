import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Plus, Play, Pause, Square, Users, Clock, AlertTriangle } from 'lucide-react';
import { apiClient } from '../lib/api';
import { Session } from '../types';
import { formatDateTime, formatRelativeTime, cn } from '../lib/utils';

interface SessionSelectorProps {
  onSessionSelect: (sessionId: string) => void;
}

export default function SessionSelector({ onSessionSelect }: SessionSelectorProps) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  const [newSessionDescription, setNewSessionDescription] = useState('');
  const [totalStudents, setTotalStudents] = useState<number | ''>('');

  const queryClient = useQueryClient();

  // Fetch sessions
  const { data: sessionsData, isLoading, error } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => apiClient.getSessions({ limit: 50 }),
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  // Create session mutation
  const createSessionMutation = useMutation({
    mutationFn: (data: { name: string; description?: string; totalStudents?: number }) =>
      apiClient.createSession(data),
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      setShowCreateModal(false);
      setNewSessionName('');
      setNewSessionDescription('');
      setTotalStudents('');
      // Automatically select the new session
      onSessionSelect(response.data.session.id);
    },
  });

  const handleCreateSession = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSessionName.trim()) return;

    createSessionMutation.mutate({
      name: newSessionName.trim(),
      description: newSessionDescription.trim() || undefined,
      totalStudents: totalStudents ? Number(totalStudents) : undefined,
    });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ACTIVE':
        return <Play className="w-4 h-4 text-success-600" />;
      case 'PAUSED':
        return <Pause className="w-4 h-4 text-warning-600" />;
      case 'COMPLETED':
        return <Square className="w-4 h-4 text-gray-600" />;
      case 'CANCELLED':
        return <AlertTriangle className="w-4 h-4 text-danger-600" />;
      default:
        return <Clock className="w-4 h-4 text-gray-600" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ACTIVE':
        return 'bg-success-100 text-success-800';
      case 'PAUSED':
        return 'bg-warning-100 text-warning-800';
      case 'COMPLETED':
        return 'bg-gray-100 text-gray-800';
      case 'CANCELLED':
        return 'bg-danger-100 text-danger-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="mx-auto h-12 w-12 text-danger-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">Error loading sessions</h3>
        <p className="mt-1 text-sm text-gray-500">
          Please check your connection and try again.
        </p>
      </div>
    );
  }

  const sessions = sessionsData?.data.sessions || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Classroom Sessions</h2>
          <p className="text-gray-600">Select a session to view real-time analytics</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn-primary flex items-center space-x-2"
        >
          <Plus className="w-4 h-4" />
          <span>New Session</span>
        </button>
      </div>

      {/* Sessions Grid */}
      {sessions.length === 0 ? (
        <div className="text-center py-12">
          <Users className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No sessions found</h3>
          <p className="mt-1 text-sm text-gray-500">
            Get started by creating your first classroom session.
          </p>
          <div className="mt-6">
            <button
              onClick={() => setShowCreateModal(true)}
              className="btn-primary"
            >
              Create Session
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sessions.map((session: Session) => (
            <div
              key={session.id}
              className="card hover:shadow-lg transition-shadow duration-200 cursor-pointer"
              onClick={() => onSessionSelect(session.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {session.name}
                  </h3>
                  {session.description && (
                    <p className="text-gray-600 text-sm mb-3 line-clamp-2">
                      {session.description}
                    </p>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  {getStatusIcon(session.status)}
                  <span
                    className={cn(
                      'px-2 py-1 rounded-full text-xs font-medium',
                      getStatusColor(session.status)
                    )}
                  >
                    {session.status}
                  </span>
                </div>
              </div>

              <div className="mt-4 space-y-2">
                <div className="flex items-center justify-between text-sm text-gray-500">
                  <span>Started</span>
                  <span>{formatRelativeTime(session.startTime)}</span>
                </div>
                {session.totalStudents && (
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <span>Expected Students</span>
                    <span>{session.totalStudents}</span>
                  </div>
                )}
                <div className="flex items-center justify-between text-sm text-gray-500">
                  <span>Created</span>
                  <span>{formatDateTime(session.createdAt)}</span>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-200">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onSessionSelect(session.id);
                  }}
                  className="w-full btn-primary text-sm"
                >
                  View Dashboard
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Session Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Create New Session
            </h3>
            
            <form onSubmit={handleCreateSession} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Session Name *
                </label>
                <input
                  type="text"
                  value={newSessionName}
                  onChange={(e) => setNewSessionName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  placeholder="e.g., CS101 - Introduction to Programming"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={newSessionDescription}
                  onChange={(e) => setNewSessionDescription(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  rows={3}
                  placeholder="Optional description of the session"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Expected Students
                </label>
                <input
                  type="number"
                  value={totalStudents}
                  onChange={(e) => setTotalStudents(e.target.value ? Number(e.target.value) : '')}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  placeholder="e.g., 30"
                  min="1"
                  max="200"
                />
              </div>

              <div className="flex space-x-3 pt-4">
                <button
                  type="button"
                  onClick={() => {
                    setShowCreateModal(false);
                    setNewSessionName('');
                    setNewSessionDescription('');
                    setTotalStudents('');
                  }}
                  className="flex-1 btn-secondary"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!newSessionName.trim() || createSessionMutation.isPending}
                  className="flex-1 btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {createSessionMutation.isPending ? 'Creating...' : 'Create Session'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
