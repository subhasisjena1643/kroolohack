import React, { useState, useEffect } from 'react';
import { Clock, CheckCircle, Users } from 'lucide-react';

interface Quiz {
  quizId: string;
  question: string;
  options: string[];
  type: 'quiz' | 'poll';
  duration: number;
  startTime: string;
}

interface StudentQuizProps {
  quiz: Quiz | null;
  sessionId: string;
  onResponse: (answer: string) => void;
}

export default function StudentQuiz({ quiz, sessionId, onResponse }: StudentQuizProps) {
  const [selectedAnswer, setSelectedAnswer] = useState<string>('');
  const [submitted, setSubmitted] = useState(false);
  const [timeLeft, setTimeLeft] = useState(0);
  const [studentId] = useState(() => `student_${Math.random().toString(36).substr(2, 9)}`);

  useEffect(() => {
    if (!quiz) return;

    const startTime = new Date(quiz.startTime).getTime();
    const endTime = startTime + (quiz.duration * 1000);

    const timer = setInterval(() => {
      const now = Date.now();
      const remaining = Math.max(0, Math.floor((endTime - now) / 1000));
      setTimeLeft(remaining);

      if (remaining === 0) {
        clearInterval(timer);
        if (!submitted) {
          handleAutoSubmit();
        }
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [quiz, submitted]);

  const handleAutoSubmit = () => {
    if (selectedAnswer) {
      handleSubmit();
    }
  };

  const handleSubmit = async () => {
    if (!selectedAnswer || !quiz) return;

    try {
      const response = await fetch(`http://localhost:3001/api/quiz/${quiz.quizId}/response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          studentId,
          answer: selectedAnswer,
          sessionId
        })
      });

      if (response.ok) {
        setSubmitted(true);
        onResponse(selectedAnswer);
      }
    } catch (error) {
      console.error('Error submitting response:', error);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!quiz) {
    return (
      <div className="card text-center py-8">
        <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">Waiting for Quiz/Poll</h3>
        <p className="text-gray-600">Your teacher will start a quiz or poll soon.</p>
      </div>
    );
  }

  if (submitted) {
    return (
      <div className="card text-center py-8">
        <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">Response Submitted!</h3>
        <p className="text-gray-600">
          You answered: <span className="font-medium">{selectedAnswer}</span>
        </p>
        <p className="text-sm text-gray-500 mt-2">
          Wait for your teacher to share the results.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <span className="text-2xl">{quiz.type === 'poll' ? '📊' : '🧠'}</span>
          <h3 className="text-lg font-semibold text-gray-900">
            {quiz.type === 'poll' ? 'Poll' : 'Quiz'}
          </h3>
        </div>
        
        <div className="flex items-center gap-2 text-sm">
          <Clock className="w-4 h-4 text-orange-500" />
          <span className={`font-mono ${timeLeft <= 10 ? 'text-red-600' : 'text-orange-600'}`}>
            {formatTime(timeLeft)}
          </span>
        </div>
      </div>

      {/* Question */}
      <div className="mb-6">
        <h4 className="text-xl font-medium text-gray-900 mb-4">
          {quiz.question}
        </h4>
      </div>

      {/* Options */}
      <div className="space-y-3 mb-6">
        {quiz.options.map((option, index) => (
          <label
            key={index}
            className={`block p-4 border-2 rounded-lg cursor-pointer transition-all ${
              selectedAnswer === option
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
            }`}
          >
            <div className="flex items-center">
              <input
                type="radio"
                name="quiz-option"
                value={option}
                checked={selectedAnswer === option}
                onChange={(e) => setSelectedAnswer(e.target.value)}
                className="sr-only"
              />
              <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center mr-3 ${
                selectedAnswer === option
                  ? 'border-primary-500 bg-primary-500'
                  : 'border-gray-300'
              }`}>
                {selectedAnswer === option && (
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                )}
              </div>
              <span className="flex-1 text-gray-900">
                <span className="font-medium mr-2">
                  {String.fromCharCode(65 + index)}.
                </span>
                {option}
              </span>
            </div>
          </label>
        ))}
      </div>

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        disabled={!selectedAnswer || timeLeft === 0}
        className="w-full py-3 px-4 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {timeLeft === 0 ? 'Time\'s Up!' : 'Submit Answer'}
      </button>

      {/* Progress Bar */}
      <div className="mt-4">
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-1000 ${
              timeLeft <= 10 ? 'bg-red-500' : 'bg-primary-500'
            }`}
            style={{
              width: `${(timeLeft / quiz.duration) * 100}%`
            }}
          ></div>
        </div>
        <p className="text-xs text-gray-500 mt-1 text-center">
          {timeLeft > 0 ? `${timeLeft} seconds remaining` : 'Time expired'}
        </p>
      </div>
    </div>
  );
}
