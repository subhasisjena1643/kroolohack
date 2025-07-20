import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, Calendar as CalendarIcon } from 'lucide-react';

interface CalendarProps {
  className?: string;
  sessionSummaries?: any[];
  onDateClick?: (date: Date, event?: React.MouseEvent) => void;
}

const Calendar: React.FC<CalendarProps> = ({ className = '', sessionSummaries = [], onDateClick }) => {
  const [currentDate, setCurrentDate] = useState(new Date());
  
  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];
  
  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  
  const getDaysInMonth = (date: Date) => {
    const year = date.getFullYear();
    const month = date.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startingDayOfWeek = firstDay.getDay();
    
    const days = [];
    
    // Add empty cells for days before the first day of the month
    for (let i = 0; i < startingDayOfWeek; i++) {
      days.push(null);
    }
    
    // Add days of the month
    for (let day = 1; day <= daysInMonth; day++) {
      days.push(day);
    }
    
    return days;
  };
  
  const navigateMonth = (direction: 'prev' | 'next') => {
    setCurrentDate(prev => {
      const newDate = new Date(prev);
      if (direction === 'prev') {
        newDate.setMonth(prev.getMonth() - 1);
      } else {
        newDate.setMonth(prev.getMonth() + 1);
      }
      return newDate;
    });
  };
  
  const isToday = (day: number | null) => {
    if (!day) return false;
    const today = new Date();
    return (
      day === today.getDate() &&
      currentDate.getMonth() === today.getMonth() &&
      currentDate.getFullYear() === today.getFullYear()
    );
  };
  
  const hasEvent = (day: number | null) => {
    if (!day) return false;
    const dateToCheck = new Date(currentDate.getFullYear(), currentDate.getMonth(), day);
    const hasSession = sessionSummaries.some(summary =>
      new Date(summary.session.startTime).toDateString() === dateToCheck.toDateString()
    );
    if (hasSession) {
      console.log('🎯 CALENDAR: Day', day, 'has session');
    }
    return hasSession;
  };

  const handleDayClick = (day: number | null) => {
    if (!day || !onDateClick) return;
    const clickedDate = new Date(currentDate.getFullYear(), currentDate.getMonth(), day);
    console.log('🎯 CALENDAR: Day clicked:', day, 'Date:', clickedDate);
    onDateClick(clickedDate);
  };
  
  const days = getDaysInMonth(currentDate);
  
  return (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-100 rounded-lg">
            <CalendarIcon className="w-5 h-5 text-blue-600" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900">
            {monthNames[currentDate.getMonth()]} {currentDate.getFullYear()}
          </h3>
        </div>
        
        <div className="flex items-center space-x-1">
          <button
            onClick={() => navigateMonth('prev')}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ChevronLeft className="w-4 h-4 text-gray-600" />
          </button>
          <button
            onClick={() => navigateMonth('next')}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ChevronRight className="w-4 h-4 text-gray-600" />
          </button>
        </div>
      </div>
      
      {/* Day Headers */}
      <div className="grid grid-cols-7 gap-1 mb-2">
        {dayNames.map(day => (
          <div key={day} className="text-center text-sm font-medium text-gray-500 py-2">
            {day}
          </div>
        ))}
      </div>
      
      {/* Calendar Grid */}
      <div className="grid grid-cols-7 gap-1">
        {days.map((day, index) => (
          <div
            key={index}
            onClick={() => handleDayClick(day)}
            className={`
              aspect-square flex items-center justify-center text-sm relative cursor-pointer
              transition-all duration-200 rounded-lg
              ${day ? 'hover:bg-gray-50' : ''}
              ${isToday(day) ? 'bg-blue-600 text-white font-semibold hover:bg-blue-700' : ''}
              ${!isToday(day) && day ? 'text-gray-700' : ''}
              ${!day ? 'text-gray-300 cursor-default' : ''}
              ${hasEvent(day) && !isToday(day) ? 'hover:bg-green-50 hover:text-green-700 font-medium' : ''}
            `}
          >
            {day && (
              <>
                <span className="z-10">{day}</span>
                {hasEvent(day) && !isToday(day) && (
                  <div className="absolute bottom-1 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-green-500 rounded-full pointer-events-none"></div>
                )}
                {hasEvent(day) && isToday(day) && (
                  <div className="absolute bottom-1 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-white rounded-full pointer-events-none"></div>
                )}
              </>
            )}
          </div>
        ))}
      </div>
      
      {/* Legend */}
      <div className="flex items-center justify-center space-x-4 mt-4 pt-4 border-t border-gray-100">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-blue-600 rounded-full"></div>
          <span className="text-xs text-gray-600">Today</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span className="text-xs text-gray-600">Session Summary</span>
        </div>
      </div>
    </div>
  );
};

export default Calendar;
