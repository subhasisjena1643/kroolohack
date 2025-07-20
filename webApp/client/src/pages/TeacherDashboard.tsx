import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Dashboard from '../components/Dashboard';
import {
  Users,
  LogOut,
  Plus,
  X,
  Send,
  BarChart3
} from 'lucide-react';
import { socketClient } from '../lib/socket';

const queryClient = new QueryClient();

const TeacherDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [isConnected, setIsConnected] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [sessions, setSessions] = useState<any[]>([]);
  const [selectedSession, setSelectedSession] = useState<any>(null);
  const [showQuizModal, setShowQuizModal] = useState(false);
  const [quizData, setQuizData] = useState({
    question: '',
    options: ['', '', '', ''],
    correctAnswer: 0
  });
  const [quizResults, setQuizResults] = useState<any>(null);

  useEffect(() => {
    // Check if user is logged in as teacher
    const userType = localStorage.getItem('userType');
    const email = localStorage.getItem('userEmail');

    if (!userType || userType !== 'teacher' || !email) {
      navigate('/');
      return;
    }

    setUserEmail(email);
    setIsConnected(true);

    // Fetch sessions from API
    fetch('http://localhost:3001/api/sessions')
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setSessions(data.data.sessions);
          // DON'T auto-select - let user choose
        }
      })
      .catch(err => console.error('Error fetching sessions:', err));

    // Listen for quiz responses
    const socket = socketClient.connect();
    if (socket) {
      console.log('🎯 TEACHER: Socket connected for quiz responses');

      // Join session when selected
      if (selectedSession) {
        console.log('🎯 TEACHER: Joining session for responses:', selectedSession.id);
        socket.emit('join-session', { sessionId: selectedSession.id });
      }

      socket.on('quiz-response', (data: any) => {
        console.log('🎯 TEACHER: Quiz response received:', data);
        // Update quiz results in real-time
        setQuizResults(prev => {
          console.log('🎯 TEACHER: Updating quiz results, prev:', prev);
          if (!prev) {
            const newResults = { responses: [data], totalResponses: 1 };
            console.log('🎯 TEACHER: New results:', newResults);
            return newResults;
          }
          const updatedResults = {
            ...prev,
            responses: [...prev.responses, data],
            totalResponses: prev.totalResponses + 1
          };
          console.log('🎯 TEACHER: Updated results:', updatedResults);
          return updatedResults;
        });
      });
    }

    return () => {
      if (socket) {
        socket.off('quiz-response');
      }
    };
  }, [navigate, selectedSession]);

  const handleLogout = () => {
    localStorage.removeItem('userType');
    localStorage.removeItem('userEmail');
    toast.success('Logged out successfully!');
    navigate('/');
  };

  const createSession = async () => {
    const sessionData = {
      name: `Demo Session ${new Date().toLocaleTimeString()}`,
      description: 'Hackathon demo session',
      totalStudents: 25
    };

    try {
      const response = await fetch('http://localhost:3001/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionData)
      });

      const result = await response.json();
      if (result.success) {
        setSessions(prev => [result.data.session, ...prev]);
        setSelectedSession(result.data.session);
        toast.success('🎉 New session created!');
      }
    } catch (error) {
      console.error('Error creating session:', error);
      toast.error('Failed to create session');
    }
  };

  const selectSession = (session: any) => {
    setSelectedSession(session);
  };

  const handleSendQuiz = async () => {
    if (!selectedSession) {
      toast.error('Please select a session first!');
      return;
    }

    if (!quizData.question.trim()) {
      toast.error('Please enter a question!');
      return;
    }

    const validOptions = quizData.options.filter(opt => opt.trim());
    if (validOptions.length < 2) {
      toast.error('Please provide at least 2 options!');
      return;
    }

    try {
      // Send quiz via API (which will broadcast via socket)
      const response = await fetch(`http://localhost:3001/api/sessions/${selectedSession.id}/quiz`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: quizData.question,
          options: validOptions,
          correctAnswer: quizData.correctAnswer,
          type: 'quiz',
          duration: 60
        })
      });

      const result = await response.json();
      if (result.success) {
        console.log('🎯 TEACHER: Quiz sent successfully:', result.data.quiz);

        // Initialize results tracking
        setQuizResults({
          quiz: result.data.quiz,
          responses: [],
          totalResponses: 0,
          startTime: new Date()
        });

        toast.success('📝 Quiz sent to all students!');
        setShowQuizModal(false);
        setQuizData({ question: '', options: ['', '', '', ''], correctAnswer: 0 });
      } else {
        toast.error('Failed to send quiz!');
      }
    } catch (error) {
      console.error('Error sending quiz:', error);
      toast.error('Error sending quiz!');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="container">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-blue-600">
                🎓 Classroom Analytics
              </h1>
              {selectedSession && (
                <button
                  onClick={() => setSelectedSession(null)}
                  className="ml-4 px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm rounded-lg transition-colors"
                >
                  ← Back to Sessions
                </button>
              )}
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: isConnected ? '#059669' : '#dc2626'
                  }}
                />
                <span className="text-sm text-gray-600">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>

              <div className="flex items-center space-x-2">
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center"
                  style={{
                    backgroundColor: '#eff6ff',
                  }}
                >
                  <span className="text-blue-600 font-medium text-sm">T</span>
                </div>
                <span className="text-sm font-medium text-gray-700">{userEmail}</span>
              </div>

              {/* Quick Quiz Button - Next to Profile */}
              {selectedSession && (
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('Quiz button clicked!', selectedSession);
                    setShowQuizModal(true);
                  }}
                  className="flex items-center gap-2 px-3 py-2 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 transition-colors font-medium"
                >
                  <Plus className="w-4 h-4" />
                  Quiz
                </button>
              )}

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
      </header>

      {/* Main Content */}
      <main className="container py-8">
        {!selectedSession ? (
          // Session List
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Classroom Sessions</h2>
                <p className="text-gray-600">Select a session to view real-time analytics</p>
              </div>
              <button onClick={createSession} className="btn btn-primary">
                + New Session
              </button>
            </div>

            {sessions.length === 0 ? (
              <div className="text-center py-12">
                <h3 className="text-lg font-medium text-gray-900 mb-2">No sessions found</h3>
                <p className="text-gray-500 mb-6">Get started by creating your first classroom session.</p>
                <button onClick={createSession} className="btn btn-primary">
                  Create Session
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {sessions.map((session) => (
                  <div key={session.id} className="card cursor-pointer" onClick={() => selectSession(session)}>
                    <div className="flex justify-between items-start mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">{session.name}</h3>
                      <span
                        className="px-2 py-1 rounded text-xs font-medium"
                        style={{
                          backgroundColor: session.status === 'ACTIVE' ? '#ecfdf5' : '#f3f4f6',
                          color: session.status === 'ACTIVE' ? '#059669' : '#6b7280'
                        }}
                      >
                        {session.status}
                      </span>
                    </div>
                    {session.description && (
                      <p className="text-gray-600 text-sm mb-3">{session.description}</p>
                    )}
                    <div className="text-sm text-gray-500">
                      <div>Started: {new Date(session.startTime).toLocaleString()}</div>
                      {session.totalStudents && <div>Expected: {session.totalStudents} students</div>}
                    </div>
                    <button
                      className="btn btn-primary w-full mt-4"
                      onClick={(e) => {
                        e.stopPropagation();
                        selectSession(session);
                      }}
                    >
                      View Dashboard
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          // Original Analytics Dashboard
          <div className="space-y-6">
            {/* Quiz Results Panel (if active) */}
            {quizResults && (
              <div className="card">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">Live Quiz Results</h3>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                    <span className="text-sm text-red-600 font-medium">LIVE</span>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{quizResults.totalResponses}</div>
                    <div className="text-sm text-gray-600">Total Responses</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">
                      {quizResults.startTime ? Math.round((Date.now() - quizResults.startTime.getTime()) / 1000) : 0}s
                    </div>
                    <div className="text-sm text-gray-600">Elapsed Time</div>
                  </div>
                </div>

                {/* Response Distribution (Zoom-style) */}
                {quizResults.totalResponses > 0 && (
                  <div className="space-y-3">
                    <h4 className="font-medium text-gray-900">Response Distribution:</h4>
                    {['A', 'B', 'C', 'D'].map((option, index) => {
                      const count = quizResults.responses.filter((r: any) => r.selectedAnswer === option).length;
                      const percentage = Math.round((count / quizResults.totalResponses) * 100);
                      return (
                        <div key={option} className="flex items-center space-x-3">
                          <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center font-medium">
                            {option}
                          </div>
                          <div className="flex-1">
                            <div className="flex justify-between text-sm mb-1">
                              <span>{quizResults.quiz?.options?.[index] || `Option ${option}`}</span>
                              <span className="font-medium">{count} ({percentage}%)</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                                style={{ width: `${percentage}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Original Analytics Dashboard */}
            <QueryClientProvider client={queryClient}>
              <Dashboard sessionId={selectedSession.id} />
            </QueryClientProvider>
          </div>
        )}
      </main>

      {/* Quiz Modal - Zoom Style */}
      {showQuizModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-lg mx-4">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h3 className="text-xl font-semibold text-gray-900">Create Quick Quiz</h3>
                <p className="text-sm text-gray-500 mt-1">Send a live quiz to all students</p>
              </div>
              <button
                onClick={() => setShowQuizModal(false)}
                className="text-gray-400 hover:text-gray-600 p-1"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Question *
                </label>
                <textarea
                  value={quizData.question}
                  onChange={(e) => setQuizData(prev => ({ ...prev, question: e.target.value }))}
                  placeholder="What is the capital of France?"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                  rows={3}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Answer Options *
                </label>
                {quizData.options.map((option, index) => (
                  <div key={index} className="mb-3">
                    <input
                      type="text"
                      value={option}
                      onChange={(e) => {
                        const newOptions = [...quizData.options];
                        newOptions[index] = e.target.value;
                        setQuizData(prev => ({ ...prev, options: newOptions }));
                      }}
                      placeholder={`Option ${String.fromCharCode(65 + index)}`}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    />
                  </div>
                ))}
                <p className="text-xs text-gray-500 mt-2">
                  Students will see response distribution in real-time
                </p>
              </div>

              <div className="flex justify-end gap-3 pt-4 border-t">
                <button
                  onClick={() => setShowQuizModal(false)}
                  className="px-6 py-2 text-gray-600 hover:text-gray-800 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSendQuiz}
                  className="flex items-center gap-2 px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium"
                >
                  <Send className="w-4 h-4" />
                  Launch Quiz
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeacherDashboard;
