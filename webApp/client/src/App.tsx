import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import LoginPage from './pages/LoginPage';
import TeacherDashboard from './pages/TeacherDashboard';
import StudentDashboard from './pages/StudentDashboard';
import './App.css';

function App() {



  return (
    <Router>
      <div className="App">
        <Toaster position="top-center" />
        <Routes>
          <Route path="/" element={<LoginPage />} />
          <Route path="/teacher-dashboard" element={<TeacherDashboard />} />
          <Route path="/student-dashboard" element={<StudentDashboard />} />
          {/* Legacy routes for backward compatibility */}
          <Route path="/teacher" element={<TeacherDashboard />} />
          <Route path="/student/:sessionId" element={<StudentDashboard />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
