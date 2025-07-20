import React, { useState } from 'react';
import { Plus, Send, Clock, Users } from 'lucide-react';

interface QuizCreatorProps {
  sessionId: string;
  onQuizCreated: (quiz: any) => void;
}

interface QuizOption {
  id: string;
  text: string;
}

export default function QuizCreator({ sessionId, onQuizCreated }: QuizCreatorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [question, setQuestion] = useState('');
  const [options, setOptions] = useState<QuizOption[]>([
    { id: '1', text: '' },
    { id: '2', text: '' }
  ]);
  const [type, setType] = useState<'quiz' | 'poll'>('poll');
  const [duration, setDuration] = useState(60);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const addOption = () => {
    const newId = (options.length + 1).toString();
    setOptions([...options, { id: newId, text: '' }]);
  };

  const updateOption = (id: string, text: string) => {
    setOptions(options.map(opt => opt.id === id ? { ...opt, text } : opt));
  };

  const removeOption = (id: string) => {
    if (options.length > 2) {
      setOptions(options.filter(opt => opt.id !== id));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || options.some(opt => !opt.text.trim())) {
      alert('Please fill in all fields');
      return;
    }

    setIsSubmitting(true);
    try {
      const response = await fetch(`http://localhost:3001/api/sessions/${sessionId}/quiz`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          options: options.map(opt => opt.text.trim()),
          type,
          duration
        })
      });

      if (response.ok) {
        const result = await response.json();
        onQuizCreated(result.data.quiz);
        
        // Reset form
        setQuestion('');
        setOptions([{ id: '1', text: '' }, { id: '2', text: '' }]);
        setIsOpen(false);
      } else {
        alert('Failed to create quiz/poll');
      }
    } catch (error) {
      console.error('Error creating quiz:', error);
      alert('Error creating quiz/poll');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) {
    return (
      <div className="card">
        <button
          onClick={() => setIsOpen(true)}
          className="w-full flex items-center justify-center gap-2 p-4 text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
        >
          <Plus className="w-5 h-5" />
          Create Quiz/Poll
        </button>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Create Quiz/Poll</h3>
        <button
          onClick={() => setIsOpen(false)}
          className="text-gray-400 hover:text-gray-600"
        >
          ×
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Type Selection */}
        <div className="flex gap-4">
          <label className="flex items-center">
            <input
              type="radio"
              value="poll"
              checked={type === 'poll'}
              onChange={(e) => setType(e.target.value as 'poll')}
              className="mr-2"
            />
            📊 Poll (Opinion)
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              value="quiz"
              checked={type === 'quiz'}
              onChange={(e) => setType(e.target.value as 'quiz')}
              className="mr-2"
            />
            🧠 Quiz (Knowledge)
          </label>
        </div>

        {/* Question */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Question
          </label>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your question..."
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            rows={3}
            required
          />
        </div>

        {/* Options */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Options
          </label>
          <div className="space-y-2">
            {options.map((option, index) => (
              <div key={option.id} className="flex gap-2">
                <span className="flex-shrink-0 w-8 h-10 bg-gray-100 rounded flex items-center justify-center text-sm font-medium">
                  {String.fromCharCode(65 + index)}
                </span>
                <input
                  type="text"
                  value={option.text}
                  onChange={(e) => updateOption(option.id, e.target.value)}
                  placeholder={`Option ${String.fromCharCode(65 + index)}`}
                  className="flex-1 p-2 border border-gray-300 rounded focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  required
                />
                {options.length > 2 && (
                  <button
                    type="button"
                    onClick={() => removeOption(option.id)}
                    className="text-red-500 hover:text-red-700 px-2"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
          </div>
          
          {options.length < 6 && (
            <button
              type="button"
              onClick={addOption}
              className="mt-2 text-primary-600 hover:text-primary-800 text-sm flex items-center gap-1"
            >
              <Plus className="w-4 h-4" />
              Add Option
            </button>
          )}
        </div>

        {/* Duration */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <Clock className="w-4 h-4 inline mr-1" />
            Duration (seconds)
          </label>
          <select
            value={duration}
            onChange={(e) => setDuration(Number(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value={30}>30 seconds</option>
            <option value={60}>1 minute</option>
            <option value={120}>2 minutes</option>
            <option value={300}>5 minutes</option>
          </select>
        </div>

        {/* Submit */}
        <div className="flex gap-3">
          <button
            type="button"
            onClick={() => setIsOpen(false)}
            className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isSubmitting}
            className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isSubmitting ? (
              'Creating...'
            ) : (
              <>
                <Send className="w-4 h-4" />
                Launch {type === 'poll' ? 'Poll' : 'Quiz'}
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
