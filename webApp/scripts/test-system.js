const axios = require('axios');
const io = require('socket.io-client');

const API_BASE = 'http://localhost:3001/api';
const SOCKET_URL = 'http://localhost:3001';

async function testSystem() {
  console.log('🧪 Testing Classroom Engagement Analyzer System\n');

  try {
    // Test 1: Health Check
    console.log('1️⃣ Testing Health Check...');
    const healthResponse = await axios.get(`${API_BASE}/../health`);
    console.log('✅ Health Check:', healthResponse.data);
    console.log('');

    // Test 2: Create Session
    console.log('2️⃣ Creating Test Session...');
    const sessionData = {
      name: 'Test Session - CS101',
      description: 'Test session for system validation',
      totalStudents: 25
    };
    
    const createResponse = await axios.post(`${API_BASE}/sessions`, sessionData);
    const sessionId = createResponse.data.data.session.id;
    console.log('✅ Session Created:', sessionId);
    console.log('');

    // Test 3: WebSocket Connection
    console.log('3️⃣ Testing WebSocket Connection...');
    const socket = io(SOCKET_URL);
    
    await new Promise((resolve) => {
      socket.on('connect', () => {
        console.log('✅ WebSocket Connected:', socket.id);
        resolve();
      });
    });

    // Test 4: Join Session
    console.log('4️⃣ Joining Session via WebSocket...');
    socket.emit('join-session', {
      sessionId,
      userId: 'test-user',
      role: 'INSTRUCTOR'
    });

    await new Promise((resolve) => {
      socket.on('session-joined', (data) => {
        console.log('✅ Session Joined:', data.sessionId);
        resolve();
      });
    });

    // Test 5: Send Test Metrics
    console.log('5️⃣ Sending Test Metrics...');
    const testMetrics = {
      sessionId,
      metrics: {
        attendance: {
          total_detected: 23,
          confidence_avg: 0.89
        },
        engagement: {
          overall_score: 0.75,
          attention_score: 0.82,
          participation_score: 0.68,
          zones: {
            front: 0.85,
            middle: 0.72,
            back: 0.58
          }
        },
        audio: {
          noise_level: 0.35,
          speaker_activity: 0.78
        }
      },
      timestamp: new Date().toISOString()
    };

    const metricsResponse = await axios.post(`${API_BASE}/ai/metrics`, testMetrics);
    console.log('✅ Metrics Sent:', metricsResponse.data.data.metricId);

    // Test 6: Verify Real-time Update
    console.log('6️⃣ Waiting for Real-time Update...');
    await new Promise((resolve) => {
      socket.on('metrics-updated', (data) => {
        console.log('✅ Real-time Update Received:', {
          sessionId: data.sessionId,
          attendance: data.metrics.attendance.total_detected,
          engagement: Math.round(data.metrics.engagement.overall_score * 100) + '%'
        });
        resolve();
      });
    });

    // Test 7: Trigger Test Alert
    console.log('7️⃣ Triggering Test Alert...');
    const lowEngagementMetrics = {
      sessionId,
      metrics: {
        attendance: {
          total_detected: 20,
          confidence_avg: 0.85
        },
        engagement: {
          overall_score: 0.45, // Below threshold
          attention_score: 0.50,
          participation_score: 0.40,
          zones: {
            front: 0.60,
            middle: 0.45,
            back: 0.30 // Very low
          }
        },
        audio: {
          noise_level: 0.80, // High noise
          speaker_activity: 0.60
        }
      },
      timestamp: new Date().toISOString()
    };

    await axios.post(`${API_BASE}/ai/metrics`, lowEngagementMetrics);

    // Wait for alert
    await new Promise((resolve) => {
      socket.on('alert-triggered', (data) => {
        console.log('✅ Alert Triggered:', {
          type: data.alert.type,
          severity: data.alert.severity,
          message: data.alert.message
        });
        resolve();
      });
    });

    // Test 8: Get Session Stats
    console.log('8️⃣ Fetching Session Statistics...');
    const statsResponse = await axios.get(`${API_BASE}/metrics/session/${sessionId}/stats`);
    const stats = statsResponse.data.data.stats;
    console.log('✅ Session Stats:', {
      dataPoints: stats.timeRange.dataPoints,
      avgEngagement: Math.round(stats.engagement.average * 100) + '%',
      currentAttendance: stats.attendance.current
    });

    // Test 9: Get Alerts
    console.log('9️⃣ Fetching Session Alerts...');
    const alertsResponse = await axios.get(`${API_BASE}/alerts/session/${sessionId}`);
    const alerts = alertsResponse.data.data.alerts;
    console.log('✅ Session Alerts:', alerts.length + ' alerts found');

    // Cleanup
    console.log('🧹 Cleaning up...');
    socket.disconnect();
    await axios.delete(`${API_BASE}/sessions/${sessionId}`);
    console.log('✅ Test session deleted');

    console.log('\n🎉 All Tests Passed! System is working correctly.');
    console.log('\n📊 System Ready for Demo:');
    console.log('   • Backend API: ✅ Working');
    console.log('   • WebSocket: ✅ Working');
    console.log('   • Database: ✅ Working');
    console.log('   • Real-time Updates: ✅ Working');
    console.log('   • Alert System: ✅ Working');
    console.log('\n🚀 Ready for AI Pipeline Integration!');

  } catch (error) {
    console.error('❌ Test Failed:', error.message);
    if (error.response) {
      console.error('Response:', error.response.data);
    }
    process.exit(1);
  }
}

// Run tests
testSystem();
