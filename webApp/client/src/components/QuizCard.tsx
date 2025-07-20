import React, { useState } from 'react';
import { HelpCircle, CheckCircle, Clock, Send } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { socketClient } from '../lib/socket';

interface QuizCardProps {
  className?: string;
  currentQuiz?: any;
}

const QuizCard: React.FC<QuizCardProps> = ({ className = '', currentQuiz }) => {
  const [selectedAnswer, setSelectedAnswer] = useState<string>('');
  const [textAnswer, setTextAnswer] = useState<string>('');
  const [isSubmitted, setIsSubmitted] = useState(false);

  // Use real-time quiz data or fallback to demo (Zoom-style - no correct/incorrect)
  const question = currentQuiz ? {
    id: currentQuiz.quizId,
    text: currentQuiz.question,
    options: currentQuiz.options.map((opt: string, index: number) => ({
      id: String.fromCharCode(65 + index), // A, B, C, D
      text: opt
    }))
  } : {
    id: 1,
    text: 'Which organelle is known as the "powerhouse of the cell"?',
    options: [
      { id: 'A', text: 'Mitochondria' },
      { id: 'B', text: 'Nucleus' },
      { id: 'C', text: 'Ribosome' },
      { id: 'D', text: 'Chloroplast' }
    ]
  };

  // Reset form when new quiz arrives
  React.useEffect(() => {
    if (currentQuiz) {
      setSelectedAnswer('');
      setTextAnswer('');
      setIsSubmitted(false);
    }
  }, [currentQuiz]);
  
  const handleSubmit = () => {
    if (!selectedAnswer && !textAnswer.trim()) {
      toast.error('Please select an answer!');
      return;
    }

    setIsSubmitted(true);

    // Send response to teacher if it's a live quiz (Zoom-style - no right/wrong)
    if (currentQuiz) {
      const socket = socketClient.connect();
      if (socket) {
        const response = {
          quizId: currentQuiz.quizId,
          sessionId: currentQuiz.sessionId,
          studentId: `student_${Date.now()}`,
          selectedAnswer,
          answer: selectedAnswer,
          textAnswer,
          responseTime: Date.now() - new Date(currentQuiz.startTime).getTime(),
          timestamp: new Date().toISOString()
        };
        console.log('🎯 STUDENT: Sending quiz response:', response);
        socket.emit('quiz-response', response);
      }
    }

    // Simple confirmation like Zoom - no right/wrong feedback
    toast.success('✅ Response submitted!', {
      duration: 2000,
      position: 'top-center',
      style: {
        background: '#10B981',
        color: 'white',
        padding: '16px',
        borderRadius: '12px',
        fontSize: '16px',
        fontWeight: '600'
      }
    });
  };
  
  const resetQuiz = () => {
    setSelectedAnswer('');
    setTextAnswer('');
    setIsSubmitted(false);
  };
  
  return (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-purple-100 rounded-lg">
            <HelpCircle className="w-5 h-5 text-purple-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              {currentQuiz ? 'Live Quiz' : 'Quiz for Students'}
            </h3>
            {currentQuiz && (
              <div className="flex items-center gap-2 mt-1">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                <span className="text-xs text-red-600 font-medium">LIVE</span>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2 text-sm text-gray-500">
          <Clock className="w-4 h-4" />
          <span>{currentQuiz ? '1 min' : '5 min'}</span>
        </div>
      </div>
      
      {/* Question */}
      <div className="mb-6">
        <h4 className="text-base font-medium text-gray-900 mb-4">
          Question 1
        </h4>
        <p className="text-gray-700 leading-relaxed">
          {question.text}
        </p>
      </div>
      
      {/* Multiple Choice Options */}
      <div className="mb-6">
        <p className="text-sm font-medium text-gray-700 mb-3">CHOOSE AN ANSWER:</p>
        <div className="space-y-3">
          {question.options.map((option) => (
            <label
              key={option.id}
              className={`
                flex items-center p-3 rounded-lg border cursor-pointer transition-all
                ${selectedAnswer === option.id 
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                }
                ${isSubmitted && option.isCorrect 
                  ? 'border-green-500 bg-green-50' 
                  : ''
                }
                ${isSubmitted && selectedAnswer === option.id && !option.isCorrect 
                  ? 'border-red-500 bg-red-50' 
                  : ''
                }
              `}
            >
              <input
                type="radio"
                name="quiz-answer"
                value={option.id}
                checked={selectedAnswer === option.id}
                onChange={(e) => setSelectedAnswer(e.target.value)}
                disabled={isSubmitted}
                className="w-4 h-4 text-blue-600 border-gray-300 focus:ring-blue-500"
              />
              <span className="ml-3 text-gray-700">
                <span className="font-medium text-blue-600">{option.id}.</span> {option.text}
              </span>
              {isSubmitted && option.isCorrect && (
                <CheckCircle className="w-5 h-5 text-green-600 ml-auto" />
              )}
            </label>
          ))}
        </div>
      </div>
      
      {/* Text Answer Section */}
      <div className="mb-6">
        <p className="text-sm font-medium text-gray-700 mb-3">OR EXPLAIN YOUR ANSWER:</p>
        <textarea
          value={textAnswer}
          onChange={(e) => setTextAnswer(e.target.value)}
          disabled={isSubmitted}
          placeholder="Type your detailed answer here..."
          className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
          rows={4}
        />
      </div>
      
      {/* Submit Button */}
      <div className="flex justify-center">
        {!isSubmitted ? (
          <button
            onClick={handleSubmit}
            className="flex items-center space-x-2 px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            <Send className="w-4 h-4" />
            <span>Submit Answer</span>
          </button>
        ) : (
          <button
            onClick={resetQuiz}
            className="px-8 py-3 bg-gray-600 text-white font-medium rounded-lg hover:bg-gray-700 transition-colors"
          >
            Try Another Question
          </button>
        )}
      </div>
      
      {/* Results */}
      {isSubmitted && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Your Answer:</span>
            <span className="text-sm font-medium">
              {selectedAnswer ? `Option ${selectedAnswer}` : 'Text explanation provided'}
            </span>
          </div>
          {textAnswer && (
            <div className="mt-2 pt-2 border-t border-gray-200">
              <p className="text-sm text-gray-600 mb-1">Your explanation:</p>
              <p className="text-sm text-gray-800 italic">"{textAnswer}"</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QuizCard;
