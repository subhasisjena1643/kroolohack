import { cn, formatPercentage } from '../lib/utils';

interface EngagementHeatmapProps {
  zones: {
    front: number;
    middle: number;
    back: number;
  };
  className?: string;
}

export default function EngagementHeatmap({ zones, className }: EngagementHeatmapProps) {
  const getEngagementColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-500';
    if (score >= 0.6) return 'bg-orange-500';
    if (score >= 0.4) return 'bg-orange-600';
    return 'bg-red-500';
  };

  const getEngagementLabel = (score: number) => {
    if (score >= 0.8) return 'High';
    if (score >= 0.6) return 'Good';
    if (score >= 0.4) return 'Low';
    return 'Very Low';
  };

  const zoneData = [
    { name: 'Front Zone', key: 'front', score: zones.front },
    { name: 'Middle Zone', key: 'middle', score: zones.middle },
    { name: 'Back Zone', key: 'back', score: zones.back },
  ];

  return (
    <div className={cn('space-y-4', className)}>
      {/* Simple Clean Zone Display */}
      <div className="bg-white rounded-lg p-4">
        {/* Teacher Area */}
        <div className="text-center mb-4">
          <div className="inline-block bg-gray-800 text-white px-3 py-1 rounded text-sm">
            📺 Teacher
          </div>
        </div>

        {/* Zone List */}
        <div className="space-y-3">
          {zoneData.map((zone) => (
            <div key={zone.key} className="flex items-center justify-between p-3 bg-gray-50 rounded">
              <div className="flex items-center space-x-3">
                <div className={cn('w-3 h-3 rounded-full', getEngagementColor(zone.score))} />
                <span className="font-medium text-gray-900">{zone.name}</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-lg font-bold text-gray-900">
                  {formatPercentage(zone.score)}
                </span>
                <span className={cn(
                  'px-2 py-1 rounded text-xs font-medium',
                  zone.score >= 0.8 ? 'bg-green-100 text-green-800' :
                  zone.score >= 0.6 ? 'bg-orange-100 text-orange-800' :
                  'bg-red-100 text-red-800'
                )}>
                  {getEngagementLabel(zone.score)}
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Overall Summary */}
        <div className="mt-4 p-3 bg-blue-50 rounded border border-blue-200">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-blue-800">Overall Engagement</span>
            <span className="text-lg font-bold text-blue-600">
              {formatPercentage((zones.front + zones.middle + zones.back) / 3)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
