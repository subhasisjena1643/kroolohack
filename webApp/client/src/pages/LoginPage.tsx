import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { User, Lock, GraduationCap, Users } from 'lucide-react';

const LoginPage: React.FC = () => {
  const [userType, setUserType] = useState<'teacher' | 'student'>('student');
  const [credentials, setCredentials] = useState({
    email: '',
    password: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Store user info in localStorage
      localStorage.setItem('userType', userType);
      localStorage.setItem('userEmail', credentials.email);
      
      if (userType === 'student') {
        toast.success('🎉 Welcome Student! Ready to learn?', {
          duration: 4000,
          position: 'top-center',
          style: {
            background: '#4F46E5',
            color: 'white',
            padding: '16px',
            borderRadius: '12px',
            fontSize: '16px',
            fontWeight: '600'
          }
        });
        navigate('/student-dashboard');
      } else {
        toast.success('Welcome Teacher! Your classroom awaits.', {
          duration: 3000,
          position: 'top-center',
          style: {
            background: '#059669',
            color: 'white',
            padding: '16px',
            borderRadius: '12px',
            fontSize: '16px',
            fontWeight: '600'
          }
        });
        navigate('/teacher-dashboard');
      }
    } catch (error) {
      toast.error('Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full mb-4">
            <GraduationCap className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Learning Dashboard
          </h1>
          <p className="text-gray-600">
            Track progress and engage with interactive content
          </p>
        </div>

        {/* User Type Selector */}
        <div className="flex bg-gray-100 rounded-lg p-1 mb-6">
          <button
            type="button"
            onClick={() => setUserType('student')}
            className={`flex-1 flex items-center justify-center py-2 px-4 rounded-md text-sm font-medium transition-all ${
              userType === 'student'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <User className="w-4 h-4 mr-2" />
            Student
          </button>
          <button
            type="button"
            onClick={() => setUserType('teacher')}
            className={`flex-1 flex items-center justify-center py-2 px-4 rounded-md text-sm font-medium transition-all ${
              userType === 'teacher'
                ? 'bg-white text-green-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <Users className="w-4 h-4 mr-2" />
            Teacher
          </button>
        </div>

        {/* Login Form */}
        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Email Address
            </label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="email"
                required
                value={credentials.email}
                onChange={(e) => setCredentials(prev => ({ ...prev, email: e.target.value }))}
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder={`Enter your ${userType} email`}
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="password"
                required
                value={credentials.password}
                onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your password"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-all ${
              userType === 'student'
                ? 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700'
                : 'bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700'
            } ${isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-lg'}`}
          >
            {isLoading ? (
              <div className="flex items-center justify-center">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                Signing in...
              </div>
            ) : (
              `Sign in as ${userType === 'student' ? 'Student' : 'Teacher'}`
            )}
          </button>
        </form>

        {/* Demo Credentials */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600 mb-2">Demo Credentials:</p>
          <div className="text-xs text-gray-500 space-y-1">
            <div>Student: student@demo.com / password</div>
            <div>Teacher: teacher@demo.com / password</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
