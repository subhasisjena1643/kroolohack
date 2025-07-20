import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { apiClient } from '../lib/api';
import { formatTime, formatPercentage } from '../lib/utils';

interface RealTimeChartProps {
  sessionId: string;
  metric: 'engagement' | 'attendance' | 'attention' | 'participation';
  color: string;
  height?: number;
}

export default function RealTimeChart({ 
  sessionId, 
  metric, 
  color, 
  height = 200 
}: RealTimeChartProps) {
  const [chartData, setChartData] = useState<any[]>([]);

  // Fetch historical metrics
  const { data: metricsData } = useQuery({
    queryKey: ['session-metrics', sessionId, metric],
    queryFn: () => apiClient.getSessionMetrics(sessionId, { 
      limit: 50,
      interval: '1m'
    }),
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  useEffect(() => {
    if (metricsData?.data.metrics) {
      const processedData = metricsData.data.metrics
        .slice(-20) // Last 20 data points
        .map((item: any) => {
          let value = 0;
          switch (metric) {
            case 'engagement':
              value = item.overallEngagement;
              break;
            case 'attendance':
              value = item.attendanceCount;
              break;
            case 'attention':
              value = item.attentionScore;
              break;
            case 'participation':
              value = item.participationScore;
              break;
          }

          return {
            timestamp: item.timestamp,
            time: formatTime(item.timestamp),
            value: value,
            displayValue: metric === 'attendance' ? value : formatPercentage(value)
          };
        })
        .reverse(); // Show oldest to newest

      setChartData(processedData);
    }
  }, [metricsData, metric]);

  const getMetricLabel = () => {
    switch (metric) {
      case 'engagement':
        return 'Engagement';
      case 'attendance':
        return 'Attendance';
      case 'attention':
        return 'Attention';
      case 'participation':
        return 'Participation';
      default:
        return 'Metric';
    }
  };

  const formatTooltipValue = (value: number) => {
    if (metric === 'attendance') {
      return `${value} students`;
    }
    return formatPercentage(value);
  };

  const formatYAxisValue = (value: number) => {
    if (metric === 'attendance') {
      return value.toString();
    }
    return `${Math.round(value * 100)}%`;
  };

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-medium text-gray-900">{label}</p>
          <p className="text-sm text-gray-600">
            <span className="font-medium" style={{ color }}>
              {getMetricLabel()}: {formatTooltipValue(payload[0].value)}
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-48">
        <div className="text-center">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-500">Loading chart data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={chartData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="time" 
            stroke="#6b7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
          />
          <YAxis 
            stroke="#6b7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={formatYAxisValue}
            domain={metric === 'attendance' ? ['dataMin', 'dataMax'] : [0, 1]}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={{ fill: color, strokeWidth: 2, r: 3 }}
            activeDot={{ r: 5, stroke: color, strokeWidth: 2 }}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>
      
      {/* Chart Info */}
      <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
        <span>Last 20 data points</span>
        <span>Updates every 30s</span>
      </div>
    </div>
  );
}
