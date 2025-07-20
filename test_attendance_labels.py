#!/usr/bin/env python3
"""
Test Attendance Labels and Alerts Display
Shows how the labels and alerts will appear on the camera window
"""

import cv2
import numpy as np
import time

def create_mock_attendance_result():
    """Create mock attendance data to test drawing functions"""
    return {
        'tracked_persons': [
            {
                'name': 'Subhasis',
                'roll_number': 'C001',
                'confidence': 0.85,
                'is_present': True
            },
            {
                'name': 'Sachin', 
                'roll_number': 'C002',
                'confidence': 0.78,
                'is_present': True
            },
            {
                'name': 'Unknown Person',
                'roll_number': 'unknown_person_1',
                'confidence': 0.80,
                'is_present': False
            }
        ],
        'attendance_count': 2,
        'total_recognized': 2,
        'active_alerts': [
            {
                'alert_type': 'disappearance',
                'name': 'Unknown Person',
                'roll_number': 'unknown_person_1',
                'duration': 15.0,
                'message': 'Unknown Person disappeared from view'
            }
        ],
        'frame_annotations': [
            {
                'type': 'rectangle',
                'bbox': [100, 100, 200, 200],
                'color': (0, 255, 0),
                'thickness': 2,
                'label': 'Subhasis (C001)',
                'confidence': 0.85
            },
            {
                'type': 'rectangle', 
                'bbox': [300, 150, 400, 250],
                'color': (0, 255, 0),
                'thickness': 2,
                'label': 'Sachin (C002)',
                'confidence': 0.78
            }
        ]
    }

def draw_attendance_annotations(frame, attendance_result):
    """Draw attendance system annotations on frame"""
    try:
        if not attendance_result:
            return
        
        # Draw face annotations with roll numbers
        annotations = attendance_result.get('frame_annotations', [])
        for annotation in annotations:
            if annotation.get('type') == 'rectangle':
                bbox = annotation.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    color = annotation.get('color', (0, 255, 0))
                    thickness = annotation.get('thickness', 2)
                    label = annotation.get('label', 'Unknown')
                    confidence = annotation.get('confidence', 0.0)
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw label background
                    label_text = f"{label} ({confidence:.2f})"
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw attendance summary panel
        draw_attendance_panel(frame, attendance_result)
        
    except Exception as e:
        print(f"Error drawing attendance annotations: {e}")

def draw_attendance_panel(frame, attendance_result):
    """Draw attendance summary panel"""
    try:
        frame_height, frame_width = frame.shape[:2]
        panel_x = 10
        panel_y = 170  # Below engagement panel
        panel_width = 350
        panel_height = 120
        
        # Background panel
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "AUTOMATED ATTENDANCE", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Attendance statistics
        present_count = attendance_result.get('attendance_count', 0)
        total_recognized = attendance_result.get('total_recognized', 0)
        active_alerts = len(attendance_result.get('active_alerts', []))
        
        cv2.putText(frame, f"Present: {present_count}", (panel_x + 10, panel_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Total Recognized: {total_recognized}", (panel_x + 10, panel_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Alert indicator
        alert_color = (0, 0, 255) if active_alerts > 0 else (0, 255, 0)
        cv2.putText(frame, f"Active Alerts: {active_alerts}", (panel_x + 10, panel_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
        
    except Exception as e:
        print(f"Error drawing attendance panel: {e}")

def draw_attendance_alerts(frame, attendance_result):
    """Draw attendance alerts on frame"""
    try:
        if not attendance_result:
            return
        
        active_alerts = attendance_result.get('active_alerts', [])
        if not active_alerts:
            return
        
        frame_height, frame_width = frame.shape[:2]
        
        # Alert panel position (center-top)
        panel_width = 400
        panel_height = min(60 + len(active_alerts) * 30, 200)
        panel_x = (frame_width - panel_width) // 2
        panel_y = 80
        
        # Background panel with red border for alerts
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 255), 3)
        
        # Alert title
        cv2.putText(frame, "⚠️ ATTENDANCE ALERTS ⚠️", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw individual alerts
        y_offset = panel_y + 50
        for i, alert in enumerate(active_alerts[:5]):  # Show max 5 alerts
            alert_type = alert.get('alert_type', 'unknown')
            name = alert.get('name', 'Unknown')
            roll_number = alert.get('roll_number', 'N/A')
            duration = alert.get('duration', 0.0)
            
            if alert_type == 'disappearance':
                alert_text = f"🚨 {name} ({roll_number}) - Missing {duration:.0f}s"
                # Flash effect for urgent alerts
                flash = int(time.time() * 3) % 2  # Flash every 0.33 seconds
                text_color = (0, 0, 255) if flash else (255, 255, 255)
            else:
                alert_text = f"⚠️ {name} ({roll_number}) - {alert_type}"
                text_color = (0, 165, 255)
            
            cv2.putText(frame, alert_text, (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            y_offset += 25
            
    except Exception as e:
        print(f"Error drawing attendance alerts: {e}")

def main():
    """Test the attendance label drawing functions"""
    print("🎓 TESTING ATTENDANCE LABELS AND ALERTS")
    print("="*50)
    print("This shows how labels and alerts will appear on camera")
    print("Press 'q' to quit, 's' to simulate alert")
    print("="*50)
    
    # Create a test frame (simulating camera feed)
    frame_width, frame_height = 640, 480
    
    # Create mock attendance data
    attendance_result = create_mock_attendance_result()
    
    alert_active = False
    
    while True:
        # Create a black frame (simulating camera feed)
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Add some text to simulate camera feed
        cv2.putText(frame, "SIMULATED CAMERA FEED", (frame_width//2 - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Draw attendance annotations (face labels)
        draw_attendance_annotations(frame, attendance_result)
        
        # Draw attendance alerts (if any)
        if alert_active:
            draw_attendance_alerts(frame, attendance_result)
        else:
            # Remove alerts for demo
            attendance_result['active_alerts'] = []
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to toggle alert", (10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Attendance Labels & Alerts Demo', frame)
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            alert_active = not alert_active
            if alert_active:
                # Add alert back
                attendance_result['active_alerts'] = [
                    {
                        'alert_type': 'disappearance',
                        'name': 'Unknown Person',
                        'roll_number': 'unknown_person_1',
                        'duration': 15.0 + (time.time() % 30),  # Simulate increasing duration
                        'message': 'Unknown Person disappeared from view'
                    }
                ]
                print("🚨 Alert activated!")
            else:
                attendance_result['active_alerts'] = []
                print("✅ Alert cleared!")
    
    cv2.destroyAllWindows()
    
    print("\n✅ Demo completed!")
    print("\n📋 WHAT YOU SAW:")
    print("1. 🎯 Face labels with roll numbers: 'Subhasis (C001)', 'Sachin (C002)'")
    print("2. 📊 Attendance panel: Shows present count, recognized count, alerts")
    print("3. 🚨 Alert panel: Shows disappearance alerts with flashing text")
    print("4. 🎨 Color coding: Green for recognized students, Red for alerts")
    
    print("\n🎓 IN REAL SYSTEM:")
    print("- Labels will appear on actual detected faces")
    print("- Alerts will show when students disappear for 5+ seconds")
    print("- 30-second timeout before marking as 'exited'")
    print("- Real-time updates with live camera feed")

if __name__ == "__main__":
    main()
