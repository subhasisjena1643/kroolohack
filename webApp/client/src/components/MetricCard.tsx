import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '../lib/utils';

interface MetricCardProps {
  title: string;
  value: number | string;
  unit?: string;
  trend?: 'up' | 'down' | 'stable';
  color?: 'primary' | 'success' | 'warning' | 'danger';
  icon?: React.ReactNode;
  className?: string;
}

export default function MetricCard({
  title,
  value,
  unit,
  trend,
  color = 'primary',
  icon,
  className
}: MetricCardProps) {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-success-600" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-danger-600" />;
      case 'stable':
        return <Minus className="w-4 h-4 text-gray-600" />;
      default:
        return null;
    }
  };

  const getColorClasses = () => {
    switch (color) {
      case 'success':
        return {
          icon: 'text-success-600',
          value: 'text-success-600',
          bg: 'bg-success-50',
          border: 'border-success-200'
        };
      case 'warning':
        return {
          icon: 'text-warning-600',
          value: 'text-warning-600',
          bg: 'bg-warning-50',
          border: 'border-warning-200'
        };
      case 'danger':
        return {
          icon: 'text-danger-600',
          value: 'text-danger-600',
          bg: 'bg-danger-50',
          border: 'border-danger-200'
        };
      case 'primary':
      default:
        return {
          icon: 'text-primary-600',
          value: 'text-primary-600',
          bg: 'bg-primary-50',
          border: 'border-primary-200'
        };
    }
  };

  const colorClasses = getColorClasses();

  return (
    <div className={cn(
      'bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow duration-200',
      className
    )}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2">
            {icon && (
              <div className={cn(
                'p-2 rounded-lg',
                colorClasses.bg,
                colorClasses.border,
                'border'
              )}>
                <div className={colorClasses.icon}>
                  {icon}
                </div>
              </div>
            )}
            <div>
              <p className="text-sm font-medium text-gray-600">{title}</p>
              <div className="flex items-baseline space-x-2">
                <p className={cn(
                  'text-2xl font-bold',
                  colorClasses.value
                )}>
                  {value}
                </p>
                {unit && (
                  <p className="text-sm text-gray-500">{unit}</p>
                )}
              </div>
            </div>
          </div>
        </div>
        
        {trend && (
          <div className="flex items-center space-x-1">
            {getTrendIcon()}
          </div>
        )}
      </div>
      
      {/* Optional: Add a small chart or additional info */}
      <div className="mt-4">
        <div className={cn(
          'h-1 rounded-full',
          colorClasses.bg
        )}>
          <div 
            className={cn(
              'h-1 rounded-full transition-all duration-500',
              color === 'success' ? 'bg-success-600' :
              color === 'warning' ? 'bg-warning-600' :
              color === 'danger' ? 'bg-danger-600' :
              'bg-primary-600'
            )}
            style={{ 
              width: typeof value === 'string' && value.includes('%') 
                ? value 
                : `${Math.min(100, Math.max(0, Number(value) * 100))}%` 
            }}
          />
        </div>
      </div>
    </div>
  );
}
