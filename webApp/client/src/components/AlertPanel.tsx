import { useState } from 'react';
import { 
  AlertTriangle, 
  Volume2, 
  Users, 
  Brain, 
  X, 
  Check,
  Clock,
  Filter
} from 'lucide-react';
import type { Alert } from '../types';
import { formatRelativeTime, cn } from '../lib/utils';

interface AlertPanelProps {
  alerts: Alert[];
  onAcknowledge: (alertId: string) => void;
}

export default function AlertPanel({ alerts, onAcknowledge }: AlertPanelProps) {
  const [filter, setFilter] = useState<'all' | 'unacknowledged' | 'critical'>('unacknowledged');

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'DISENGAGEMENT':
        return <Brain className="w-4 h-4" />;
      case 'HIGH_NOISE':
        return <Volume2 className="w-4 h-4" />;
      case 'LOW_ATTENDANCE':
        return <Users className="w-4 h-4" />;
      case 'PARTICIPATION_DROP':
        return <Brain className="w-4 h-4" />;
      case 'TECHNICAL_ISSUE':
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <AlertTriangle className="w-4 h-4" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'CRITICAL':
        return 'bg-danger-100 text-danger-800 border-danger-200';
      case 'HIGH':
        return 'bg-danger-50 text-danger-700 border-danger-200';
      case 'MEDIUM':
        return 'bg-warning-100 text-warning-800 border-warning-200';
      case 'LOW':
        return 'bg-primary-50 text-primary-700 border-primary-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getSeverityDot = (severity: string) => {
    switch (severity) {
      case 'CRITICAL':
        return 'bg-danger-500 animate-pulse';
      case 'HIGH':
        return 'bg-danger-400';
      case 'MEDIUM':
        return 'bg-warning-500';
      case 'LOW':
        return 'bg-primary-500';
      default:
        return 'bg-gray-500';
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    switch (filter) {
      case 'unacknowledged':
        return !alert.acknowledged;
      case 'critical':
        return alert.severity === 'CRITICAL' || alert.severity === 'HIGH';
      case 'all':
      default:
        return true;
    }
  });

  const unacknowledgedCount = alerts.filter(alert => !alert.acknowledged).length;
  const criticalCount = alerts.filter(alert => 
    (alert.severity === 'CRITICAL' || alert.severity === 'HIGH') && !alert.acknowledged
  ).length;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 text-warning-600" />
          <h3 className="text-lg font-semibold text-gray-900">Alerts</h3>
          {unacknowledgedCount > 0 && (
            <span className="bg-danger-100 text-danger-800 text-xs font-medium px-2 py-1 rounded-full">
              {unacknowledgedCount} new
            </span>
          )}
        </div>
        
        {/* Filter Dropdown */}
        <div className="relative">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="text-sm border border-gray-300 rounded-md px-3 py-1 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="unacknowledged">Unacknowledged ({unacknowledgedCount})</option>
            <option value="critical">Critical ({criticalCount})</option>
            <option value="all">All ({alerts.length})</option>
          </select>
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-8">
            <Check className="mx-auto h-8 w-8 text-success-400" />
            <p className="mt-2 text-sm text-gray-500">
              {filter === 'unacknowledged' ? 'No unacknowledged alerts' : 
               filter === 'critical' ? 'No critical alerts' : 'No alerts'}
            </p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className={cn(
                'border rounded-lg p-3 transition-all duration-200',
                getSeverityColor(alert.severity),
                alert.acknowledged ? 'opacity-60' : 'shadow-sm'
              )}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3 flex-1">
                  {/* Severity Indicator */}
                  <div className="flex items-center space-x-2 mt-0.5">
                    <div className={cn(
                      'w-2 h-2 rounded-full',
                      getSeverityDot(alert.severity)
                    )} />
                    <div className={cn(
                      'text-gray-600',
                      alert.severity === 'CRITICAL' ? 'text-danger-600' :
                      alert.severity === 'HIGH' ? 'text-danger-500' :
                      alert.severity === 'MEDIUM' ? 'text-warning-600' :
                      'text-primary-600'
                    )}>
                      {getAlertIcon(alert.type)}
                    </div>
                  </div>

                  {/* Alert Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-xs font-medium uppercase tracking-wide">
                        {alert.type.replace('_', ' ')}
                      </span>
                      {alert.zone && (
                        <span className="text-xs bg-white bg-opacity-50 px-2 py-0.5 rounded">
                          {alert.zone} zone
                        </span>
                      )}
                    </div>
                    
                    <p className="text-sm font-medium mb-1">
                      {alert.message}
                    </p>
                    
                    <div className="flex items-center space-x-3 text-xs opacity-75">
                      <div className="flex items-center space-x-1">
                        <Clock className="w-3 h-3" />
                        <span>{formatRelativeTime(alert.timestamp)}</span>
                      </div>
                      <span className="font-medium">
                        {alert.severity}
                      </span>
                      {alert.triggerValue && alert.threshold && (
                        <span>
                          {Math.round(alert.triggerValue * 100)}% / {Math.round(alert.threshold * 100)}%
                        </span>
                      )}
                    </div>

                    {alert.acknowledged && alert.acknowledgedAt && (
                      <div className="mt-2 flex items-center space-x-1 text-xs text-success-600">
                        <Check className="w-3 h-3" />
                        <span>Acknowledged {formatRelativeTime(alert.acknowledgedAt)}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Action Button */}
                {!alert.acknowledged && (
                  <button
                    onClick={() => onAcknowledge(alert.id)}
                    className="ml-2 p-1 hover:bg-white hover:bg-opacity-50 rounded transition-colors duration-200"
                    title="Acknowledge alert"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Alert Summary */}
      {alerts.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-lg font-semibold text-gray-900">
                {alerts.length}
              </div>
              <div className="text-xs text-gray-500">Total Alerts</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-danger-600">
                {unacknowledgedCount}
              </div>
              <div className="text-xs text-gray-500">Unacknowledged</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
