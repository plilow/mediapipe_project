import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5
)

# Load the emojis
shhh_emoji = cv2.imread('emojis/shhh_emoji.jpg')
thinking_emoji = cv2.imread('emojis/thinking_emoji.png')
thumbsup_emoji = cv2.imread('emojis/thumbsup_emoji.png')

if shhh_emoji is None or thinking_emoji is None or thumbsup_emoji is None:
    print("Error: Could not load emoji images")
    exit(1)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera")
    exit(1)

print("Hand and face tracking started. Press 'q' to quit.")
print("Gestures: 'shh' (finger over lips), 'thinking' (hand under chin), 'thumbs up' (both hands thumbs up beside face)!")

# Gesture state tracking for smoothing
gesture_state = {
    'current': None,  # 'shh', 'thinking', 'thumbsup', or None
    'frames_detected': 0,
    'frames_required': 2,
    'hold_frames': 10
}

def is_finger_curled(hand_landmarks, finger_tip_id, finger_pip_id):
    """Check if a finger is curled by comparing tip and PIP joint y-coordinates"""
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    return tip.y > pip.y  # Tip below PIP means curled

def is_shh_gesture(hand_landmarks, face_center, frame_shape):
    """Check if hand is making 'shh' gesture - index finger vertical over lips"""
    h, w = frame_shape[:2]
    
    # Hand landmarks
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    index_mcp = hand_landmarks.landmark[5]
    wrist = hand_landmarks.landmark[0]
    
    # Check if index finger is extended (tip above PIP)
    index_extended = index_tip.y < index_pip.y
    
    # Check if index finger is VERTICAL (pointing up, not horizontal)
    finger_angle_y = abs(index_tip.y - index_mcp.y)
    finger_angle_x = abs(index_tip.x - index_mcp.x)
    is_vertical = finger_angle_y > finger_angle_x * 0.5
    
    # Check if other fingers are curled
    middle_curled = is_finger_curled(hand_landmarks, 12, 10)
    ring_curled = is_finger_curled(hand_landmarks, 16, 14)
    pinky_curled = is_finger_curled(hand_landmarks, 20, 18)
    
    # Thumb should also be somewhat curled/tucked
    thumb_tip = hand_landmarks.landmark[4]
    thumb_curled = thumb_tip.y > index_mcp.y
    
    # Check if index finger is in the mouth/lips area
    if face_center:
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        
        # Check both vertical and horizontal positioning separately
        vertical_dist = abs(index_pos[1] - face_center[1])
        vertical_ok = vertical_dist < h * 0.15
        
        horizontal_dist = abs(index_pos[0] - face_center[0])
        horizontal_ok = horizontal_dist < w * 0.12
        
        distance = ((index_pos[0] - face_center[0])**2 + 
                   (index_pos[1] - face_center[1])**2)**0.5
        distance_ok = distance < w * 0.18
        
        near_lips = distance_ok or (vertical_ok and horizontal_ok)
    else:
        near_lips = False
    
    return (index_extended and is_vertical and middle_curled and 
            ring_curled and pinky_curled and thumb_curled and near_lips)

def is_thinking_gesture(hand_landmarks, face_center, frame_shape):
    """Check if hand is making 'thinking' gesture - thumb and index diagonal forming support under chin"""
    h, w = frame_shape[:2]
    
    # Hand landmarks
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    index_mcp = hand_landmarks.landmark[5]
    middle_mcp = hand_landmarks.landmark[9]
    wrist = hand_landmarks.landmark[0]
    
    # Check thumb is extended
    thumb_to_middle_mcp = ((thumb_tip.x - middle_mcp.x)**2 + (thumb_tip.y - middle_mcp.y)**2)**0.5
    thumb_mcp_to_middle_mcp = ((thumb_mcp.x - middle_mcp.x)**2 + (thumb_mcp.y - middle_mcp.y)**2)**0.5
    thumb_extended = thumb_to_middle_mcp > thumb_mcp_to_middle_mcp * 1.5
    
    thumb_to_index_tip = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    thumb_separated_from_index = thumb_to_index_tip > 0.05
    
    # Check if index finger is extended
    index_extended = index_tip.y < index_pip.y
    
    # Check if index finger is DIAGONAL
    index_angle_y = abs(index_tip.y - index_mcp.y)
    index_angle_x = abs(index_tip.x - index_mcp.x)
    index_diagonal = 0.2 < (index_angle_y / (index_angle_x + 0.001)) < 5.0
    
    # Check if thumb is DIAGONAL
    thumb_angle_y = abs(thumb_tip.y - thumb_mcp.y)
    thumb_angle_x = abs(thumb_tip.x - thumb_mcp.x)
    thumb_diagonal = 0.2 < (thumb_angle_y / (thumb_angle_x + 0.001)) < 5.0
    
    # Check if other fingers are curled
    middle_curled = is_finger_curled(hand_landmarks, 12, 10)
    ring_curled = is_finger_curled(hand_landmarks, 16, 14)
    pinky_curled = is_finger_curled(hand_landmarks, 20, 18)
    
    # Check if hand is positioned under chin/face area
    if face_center:
        index_pos_y = int(index_tip.y * h)
        chin_y = face_center[1]
        fingertips_at_chin = (chin_y - h * 0.2) < index_pos_y < (chin_y + h * 0.15)
        
        hand_center_x = (int(index_tip.x * w) + int(thumb_tip.x * w)) // 2
        horizontal_distance = abs(hand_center_x - face_center[0])
        near_horizontally = horizontal_distance < w * 0.4
    else:
        fingertips_at_chin = False
        near_horizontally = False
    
    return (thumb_extended and thumb_separated_from_index and index_extended and 
            index_diagonal and thumb_diagonal and
            middle_curled and ring_curled and pinky_curled and 
            fingertips_at_chin and near_horizontally)

def is_thumbs_up(hand_landmarks):
    """Check if hand is making thumbs up gesture - thumb extended up, all fingers curled"""
    # Hand landmarks
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    index_mcp = hand_landmarks.landmark[5]
    wrist = hand_landmarks.landmark[0]
    
    # Thumb should be extended upward (tip above IP joint and MCP)
    thumb_extended_up = thumb_tip.y < thumb_ip.y and thumb_tip.y < thumb_mcp.y
    
    # Thumb should be pointing UP (more vertical than horizontal)
    thumb_angle_y = abs(thumb_tip.y - thumb_mcp.y)
    thumb_angle_x = abs(thumb_tip.x - thumb_mcp.x)
    thumb_vertical = thumb_angle_y > thumb_angle_x * 0.8
    
    # All four fingers should be curled
    index_curled = is_finger_curled(hand_landmarks, 8, 6)
    middle_curled = is_finger_curled(hand_landmarks, 12, 10)
    ring_curled = is_finger_curled(hand_landmarks, 16, 14)
    pinky_curled = is_finger_curled(hand_landmarks, 20, 18)
    
    return (thumb_extended_up and thumb_vertical and 
            index_curled and middle_curled and ring_curled and pinky_curled)

def is_double_thumbs_up_gesture(hands_list, face_center, frame_shape):
    """Check if both hands are doing thumbs up next to face"""
    if len(hands_list) != 2:
        return False
    
    h, w = frame_shape[:2]
    
    # Check if both hands are thumbs up
    thumbs_up_count = 0
    hand_positions = []
    
    for hand_landmarks in hands_list:
        if is_thumbs_up(hand_landmarks):
            thumbs_up_count += 1
            # Get hand position (use wrist)
            wrist = hand_landmarks.landmark[0]
            hand_x = int(wrist.x * w)
            hand_y = int(wrist.y * h)
            hand_positions.append((hand_x, hand_y))
    
    if thumbs_up_count != 2:
        return False
    
    # Check if both hands are beside face (one on each side)
    if not face_center:
        return False
    
    face_x, face_y = face_center
    
    # Both hands should be at roughly face height
    hands_at_face_height = all(abs(pos[1] - face_y) < h * 0.25 for pos in hand_positions)
    
    # Hands should be on opposite sides of face
    left_hand = min(hand_positions, key=lambda p: p[0])
    right_hand = max(hand_positions, key=lambda p: p[0])
    
    # Left hand should be to the left of face, right hand to the right
    left_is_left = left_hand[0] < face_x - w * 0.1
    right_is_right = right_hand[0] > face_x + w * 0.1
    
    # Hands should be reasonably close to face horizontally
    left_distance = abs(left_hand[0] - face_x)
    right_distance = abs(right_hand[0] - face_x)
    
    hands_near_face = (left_distance < w * 0.45 and right_distance < w * 0.45)
    
    return hands_at_face_height and left_is_left and right_is_right and hands_near_face

def update_gesture_state(detected_gesture):
    """Smooth gesture detection to prevent flickering"""
    global gesture_state
    
    if detected_gesture == gesture_state['current']:
        gesture_state['frames_detected'] = gesture_state['frames_required']
        return gesture_state['current']
    elif detected_gesture is not None:
        gesture_state['frames_detected'] += 1
        if gesture_state['frames_detected'] >= gesture_state['frames_required']:
            gesture_state['current'] = detected_gesture
            gesture_state['frames_detected'] = gesture_state['frames_required']
        return gesture_state['current']
    else:
        gesture_state['frames_detected'] -= 1
        if gesture_state['frames_detected'] <= 0:
            gesture_state['current'] = None
            gesture_state['frames_detected'] = 0
        return gesture_state['current']

try:
    while True:
        ret, frame = camera.read()
        
        if not ret:
            print("Error: Cannot read frame from camera")
            break
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands and face
        hand_results = hands.process(rgb_frame)
        face_results = face_detection.process(rgb_frame)
        
        detected_gesture = None
        face_center = None
        
        # Draw face detection
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                
                # Get face center
                face_x = int((bboxC.xmin + bboxC.width / 2) * w)
                face_y = int((bboxC.ymin + bboxC.height * 0.7) * h)
                face_center = (face_x, face_y)
                
                # Draw face box
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(frame, bbox, (255, 255, 0), 2)
        
        # Draw hand landmarks and check for gestures
        if hand_results.multi_hand_landmarks:
            num_hands = len(hand_results.multi_hand_landmarks)
            
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
                )
            
            # Check for gestures (priority: shh > thinking > double thumbs up)
            if any(is_shh_gesture(hand, face_center, frame.shape) for hand in hand_results.multi_hand_landmarks):
                detected_gesture = 'shh'
            elif any(is_thinking_gesture(hand, face_center, frame.shape) for hand in hand_results.multi_hand_landmarks):
                detected_gesture = 'thinking'
            elif is_double_thumbs_up_gesture(hand_results.multi_hand_landmarks, face_center, frame.shape):
                detected_gesture = 'thumbsup'
            
            cv2.putText(frame, f"HANDS: {num_hands}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO HANDS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Update gesture state with smoothing
        active_gesture = update_gesture_state(detected_gesture)
        
        # Create right panel - black, shh, thinking, or thumbs up emoji
        if active_gesture == 'shh':
            emoji_resized = cv2.resize(shhh_emoji, (w, h))
            right_panel = emoji_resized
            cv2.putText(frame, "SHHH!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        elif active_gesture == 'thinking':
            emoji_resized = cv2.resize(thinking_emoji, (w, h))
            right_panel = emoji_resized
            cv2.putText(frame, "THINKING!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        elif active_gesture == 'thumbsup':
            emoji_resized = cv2.resize(thumbsup_emoji, (w, h))
            right_panel = emoji_resized
            cv2.putText(frame, "THUMBS UP!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            # Black screen
            right_panel = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Combine frames side by side
        combined = np.hstack((frame, right_panel))
        
        cv2.imshow("Hand and Face Tracking", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    pass
finally:
    hands.close()
    face_detection.close()
    camera.release()
    cv2.destroyAllWindows()
    print("Program closed.")