import cv2

# Open webcam
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera. Make sure:")
    print("1. Your camera is connected")
    print("2. You have proper permissions")
    print("3. You're running this on a machine with a display")
    exit(1)

print("Camera opened successfully! Press Ctrl+C to exit.")

try:
    # Display continuous feed
    while True:
        ret, frame = camera.read()
        
        if not ret:
            print("Error: Cannot read frame from camera")
            break
        
        cv2.imshow("Camera", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nClosing camera...")
finally:
    camera.release()
    cv2.destroyAllWindows()