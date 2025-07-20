#!/usr/bin/env python3
"""
Create a demo image showing how the attendance labels will look
"""

import cv2
import numpy as np

def create_demo_image():
    """Create a demo image showing the attendance interface"""
    # Create frame
    frame_width, frame_height = 800, 600
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Add background gradient
    for y in range(frame_height):
        intensity = int(30 + (y / frame_height) * 20)
        frame[y, :] = [intensity, intensity, intensity]
    
    # Title
    cv2.putText(frame, "AUTOMATED ATTENDANCE SYSTEM - DEMO", (frame_width//2 - 200, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Simulate detected faces with labels
    
    # Face 1: Subhasis (C001)
    x1, y1, x2, y2 = 100, 80, 200, 180
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label_text = "Subhasis (C001) (0.85)"
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 0), -1)
    cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Face 2: Sachin (C002)
    x1, y1, x2, y2 = 300, 120, 400, 220
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label_text = "Sachin (C002) (0.78)"
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 0), -1)
    cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Face 3: Unknown person
    x1, y1, x2, y2 = 500, 100, 600, 200
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    label_text = "Unknown Person (0.80)"
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 255), -1)
    cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Attendance Panel
    panel_x, panel_y = 10, 250
    panel_width, panel_height = 350, 120
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 255, 255), 2)
    
    cv2.putText(frame, "AUTOMATED ATTENDANCE", (panel_x + 10, panel_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Present: 2", (panel_x + 10, panel_y + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "Total Recognized: 2", (panel_x + 10, panel_y + 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Active Alerts: 1", (panel_x + 10, panel_y + 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Alert Panel
    alert_panel_width = 450
    alert_panel_height = 80
    alert_panel_x = (frame_width - alert_panel_width) // 2
    alert_panel_y = 400
    
    cv2.rectangle(frame, (alert_panel_x, alert_panel_y), 
                 (alert_panel_x + alert_panel_width, alert_panel_y + alert_panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (alert_panel_x, alert_panel_y), 
                 (alert_panel_x + alert_panel_width, alert_panel_y + alert_panel_height), (0, 0, 255), 3)
    
    cv2.putText(frame, "⚠️ ATTENDANCE ALERTS ⚠️", (alert_panel_x + 10, alert_panel_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "🚨 Tahir (C003) - Missing 25s", (alert_panel_x + 10, alert_panel_y + 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Features list
    features_y = 500
    cv2.putText(frame, "✅ FEATURES:", (10, features_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "• Real-time face recognition with roll numbers", (10, features_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "• Automatic attendance logging to database", (10, features_y + 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "• 5-second grace period before disappearance alerts", (10, features_y + 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "• 30-second timeout before marking as 'exited'", (10, features_y + 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Color legend
    legend_x = 450
    cv2.putText(frame, "COLOR LEGEND:", (legend_x, features_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (legend_x, features_y + 10), (legend_x + 20, features_y + 25), (0, 255, 0), -1)
    cv2.putText(frame, "Recognized Students", (legend_x + 30, features_y + 22), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(frame, (legend_x, features_y + 30), (legend_x + 20, features_y + 45), (0, 255, 255), -1)
    cv2.putText(frame, "Unknown Persons", (legend_x + 30, features_y + 42), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(frame, (legend_x, features_y + 50), (legend_x + 20, features_y + 65), (0, 0, 255), -1)
    cv2.putText(frame, "Alert Indicators", (legend_x + 30, features_y + 62), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

def main():
    """Create and save demo image"""
    print("🎨 Creating demo image...")
    
    demo_frame = create_demo_image()
    
    # Save image
    cv2.imwrite('attendance_demo.png', demo_frame)
    print("✅ Demo image saved as 'attendance_demo.png'")
    
    # Display image
    cv2.imshow('Attendance System Demo', demo_frame)
    print("📺 Press any key to close the demo image...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n🎓 THIS IS HOW YOUR ATTENDANCE SYSTEM WILL LOOK:")
    print("="*60)
    print("✅ Face Detection: Green rectangles around recognized faces")
    print("✅ Roll Number Labels: 'Subhasis (C001)', 'Sachin (C002)'")
    print("✅ Confidence Scores: Shows recognition confidence (0.85)")
    print("✅ Attendance Panel: Real-time statistics")
    print("✅ Alert System: Red flashing alerts for missing students")
    print("✅ Color Coding: Green=recognized, Yellow=unknown, Red=alerts")
    print("="*60)

if __name__ == "__main__":
    main()
