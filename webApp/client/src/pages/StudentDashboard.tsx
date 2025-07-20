import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { socketClient } from '../lib/socket';
import { Users, Wifi, WifiOff, LogOut, GraduationCap, X, Clock, BookOpen } from 'lucide-react';
import { toast } from 'react-hot-toast';
import Calendar from '../components/Calendar';
import QuizCard from '../components/QuizCard';

export default function StudentDashboard() {
  const navigate = useNavigate();

  const fetchSessionSummaries = async () => {
    try {
      console.log('🎯 STUDENT: Fetching session summaries...');
      const response = await fetch('http://localhost:3001/api/session-summaries');
      const data = await response.json();
      console.log('🎯 STUDENT: Session summaries response:', data);
      if (data.success) {
        setSessionSummaries(data.data.summaries);
        console.log('🎯 STUDENT: Set session summaries:', data.data.summaries);
      }
    } catch (error) {
      console.error('Error fetching session summaries:', error);
    }
  };

  const handleDateClick = async (date: Date, event?: React.MouseEvent) => {
    console.log('🎯 STUDENT: Date clicked:', date);
    console.log('🎯 STUDENT: Available summaries:', sessionSummaries);

    // Find sessions for this date
    const dateStr = date.toDateString();
    console.log('🎯 STUDENT: Looking for date:', dateStr);

    const sessionsForDate = sessionSummaries.filter(summary => {
      const sessionDate = new Date(summary.session.startTime).toDateString();
      console.log('🎯 STUDENT: Comparing', sessionDate, 'with', dateStr);
      return sessionDate === dateStr;
    });

    console.log('🎯 STUDENT: Sessions found for date:', sessionsForDate);

    if (sessionsForDate.length > 0) {
      // Get detailed summary for the first session
      try {
        console.log('🎯 STUDENT: Fetching summary for session:', sessionsForDate[0].sessionId);
        const response = await fetch(`http://localhost:3001/api/session-summaries/${sessionsForDate[0].sessionId}`);
        const data = await response.json();
        console.log('🎯 STUDENT: Summary response:', data);
        if (data.success) {
          setSelectedSummary(data.data.summary);

          // Calculate popup position near the clicked element
          if (event) {
            const rect = (event.target as HTMLElement).getBoundingClientRect();
            setPopupPosition({
              x: rect.left + rect.width / 2,
              y: rect.top - 10
            });
          }

          setShowSummaryModal(true);
        }
      } catch (error) {
        console.error('Error fetching session summary:', error);
      }
    } else {
      console.log('🎯 STUDENT: No sessions found for this date');
    }
  };
  const [isConnected, setIsConnected] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [connectionCount, setConnectionCount] = useState(0);
  const [currentQuiz, setCurrentQuiz] = useState<any>(null);
  const [sessionSummaries, setSessionSummaries] = useState<any[]>([]);
  const [selectedSummary, setSelectedSummary] = useState<any>(null);
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [popupPosition, setPopupPosition] = useState<{x: number, y: number} | null>(null);

  useEffect(() => {
    // Check if user is logged in
    const userType = localStorage.getItem('userType');
    const email = localStorage.getItem('userEmail');

    if (!userType || userType !== 'student' || !email) {
      navigate('/');
      return;
    }

    setUserEmail(email);

    // Fetch session summaries for calendar
    fetchSessionSummaries();

    // Connect to socket for real-time features
    const socket = socketClient.connect();
    if (!socket) return;

    const handleConnect = () => {
      setIsConnected(true);
      console.log('🎯 STUDENT: Connected to socket');
      // Join ALL active sessions for demo
      fetch('http://localhost:3001/api/sessions')
        .then(res => res.json())
        .then(data => {
          if (data.success && data.data.sessions.length > 0) {
            console.log('🎯 STUDENT: Found sessions:', data.data.sessions.length);
            // Join all active sessions
            data.data.sessions.forEach(session => {
              console.log('🎯 STUDENT: Joining session:', session.id, session.name);
              socket.emit('join-session', { sessionId: session.id });
            });
          }
        })
        .catch(err => console.error('Error fetching sessions:', err));
    };

    const handleDisconnect = () => {
      setIsConnected(false);
      console.log('Student disconnected from socket');
    };

    const handleConnectionUpdate = (data: any) => {
      setConnectionCount(data.count || 0);
    };

    const handleQuizStarted = (quiz: any) => {
      console.log('🎯 STUDENT: Quiz received:', quiz);
      console.log('🎯 STUDENT: Quiz type:', typeof quiz, quiz);
      setCurrentQuiz(quiz);
      toast.success('📝 New quiz received!', {
        duration: 4000,
        position: 'top-center',
        style: {
          background: '#8b5cf6',
          color: 'white',
          padding: '16px',
          borderRadius: '12px',
          fontSize: '16px',
          fontWeight: '600'
        }
      });
    };

    // Set up event listeners
    console.log('🎯 STUDENT: Setting up socket listeners');
    if (socket.connected) {
      handleConnect();
    } else {
      socket.on('connect', handleConnect);
    }

    socket.on('disconnect', handleDisconnect);
    socket.on('connection-update', handleConnectionUpdate);
    socket.on('quiz-started', handleQuizStarted);
    socket.on('session-joined', (data) => {
      console.log('🎯 STUDENT: Successfully joined session:', data);
      setConnectionCount(data.connectionCount || 0);
    });

    // Also listen for global quiz broadcasts (backup)
    socket.on('global-quiz', handleQuizStarted);

    // Test event listener
    socket.on('test-event', (data) => {
      console.log('🎯 STUDENT: Test event received:', data);
    });

    return () => {
      socket.off('connect', handleConnect);
      socket.off('disconnect', handleDisconnect);
      socket.off('connection-update', handleConnectionUpdate);
      socket.off('quiz-started', handleQuizStarted);
      socket.off('session-joined');
      socket.off('global-quiz', handleQuizStarted);
    };
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('userType');
    localStorage.removeItem('userEmail');
    toast.success('Logged out successfully!');
    navigate('/');
  };



  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                <GraduationCap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Learning Dashboard
                </h1>
                <p className="text-sm text-gray-600">
                  Track progress and engage with interactive content
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  {isConnected ? (
                    <>
                      <Wifi className="w-4 h-4 text-green-500" />
                      <span className="text-green-600">Connected</span>
                    </>
                  ) : (
                    <>
                      <WifiOff className="w-4 h-4 text-red-500" />
                      <span className="text-red-600">Disconnected</span>
                    </>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <Users className="w-4 h-4 text-blue-500" />
                  <span className="text-blue-600">
                    {connectionCount} online
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2 text-sm text-gray-600">
                <span>{userEmail}</span>
              </div>

              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Calendar */}
          <div className="space-y-6">
            <Calendar
              sessionSummaries={sessionSummaries}
              onDateClick={handleDateClick}
            />

            {/* Debug Info */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Connection Status
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Socket Connected:</span>
                  <span className={isConnected ? 'text-green-600' : 'text-red-600'}>
                    {isConnected ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Current Quiz:</span>
                  <span className={currentQuiz ? 'text-green-600' : 'text-gray-600'}>
                    {currentQuiz ? 'Active' : 'None'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Connected Users:</span>
                  <span className="text-blue-600">{connectionCount}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Quiz */}
          <div className="space-y-6">
            {currentQuiz ? (
              <QuizCard currentQuiz={currentQuiz} />
            ) : (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <div className="text-center py-8">
                  <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl">📝</span>
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">No Active Quiz</h3>
                  <p className="text-gray-600">Wait for your teacher to start a quiz</p>
                </div>
              </div>
            )}

            {/* Recent Activity */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Recent Activity
              </h3>
              <div className="space-y-3">
                <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-900">
                      Completed Biology Quiz
                    </p>
                    <p className="text-xs text-gray-500">2 hours ago</p>
                  </div>
                  <span className="text-sm font-medium text-green-600">92%</span>
                </div>

                <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-900">
                      Joined Math Class
                    </p>
                    <p className="text-xs text-gray-500">1 day ago</p>
                  </div>
                </div>

                <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-900">
                      Submitted Chemistry Assignment
                    </p>
                    <p className="text-xs text-gray-500">2 days ago</p>
                  </div>
                  <span className="text-sm font-medium text-purple-600">Pending</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Session Summary Modal */}
      {showSummaryModal && selectedSummary && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl p-6 w-full max-w-3xl max-h-[85vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <BookOpen className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-gray-900">
                    {selectedSummary.session.name}
                  </h3>
                  <div className="flex items-center gap-2 text-sm text-gray-500 mt-1">
                    <Clock className="w-4 h-4" />
                    <span>{new Date(selectedSummary.session.startTime).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setShowSummaryModal(false)}
                className="text-gray-400 hover:text-gray-600 p-1"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="max-w-none">
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 rounded-xl mb-6 border border-blue-100">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🤖</span>
                  <h4 className="text-lg font-semibold text-blue-800">
                    AI-Generated Session Summary
                  </h4>
                </div>
                <p className="text-sm text-blue-700">
                  This summary was created using Google AI to help you catch up on what you missed during this session.
                </p>
              </div>

              <div className="text-gray-700 leading-relaxed space-y-4">
                {selectedSummary.summary.split('\n').map((line, index) => {
                  if (line.trim() === '') return null;

                  // Handle headers (lines starting with ##)
                  if (line.trim().startsWith('##')) {
                    return (
                      <h3 key={index} className="text-lg font-semibold text-gray-900 mt-6 mb-3">
                        {line.replace('##', '').trim()}
                      </h3>
                    );
                  }

                  // Handle bold sections (lines starting with **)
                  if (line.trim().startsWith('**') && line.trim().endsWith('**')) {
                    return (
                      <h4 key={index} className="font-medium text-gray-800 mt-4 mb-2">
                        {line.replace(/\*\*/g, '').trim()}
                      </h4>
                    );
                  }

                  // Handle bullet points (lines starting with •)
                  if (line.trim().startsWith('•')) {
                    return (
                      <div key={index} className="flex items-start gap-2 ml-4">
                        <span className="text-blue-500 mt-1">•</span>
                        <span>{line.replace('•', '').trim()}</span>
                      </div>
                    );
                  }

                  // Regular paragraphs
                  if (line.trim()) {
                    return (
                      <p key={index} className="text-gray-700">
                        {line.trim()}
                      </p>
                    );
                  }

                  return null;
                })}
              </div>
            </div>

            <div className="mt-6 pt-4 border-t flex justify-end">
              <button
                onClick={() => setShowSummaryModal(false)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
