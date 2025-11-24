import cv2
import mediapipe as mp
import numpy as np

class BodyPartEducator:
    """
    A class to handle body part detection and educational visualization for children.
    """
    def __init__(self):
        # Initialize MediaPipe Holistic solution
        # Holistic detects Pose, Face, and Hands simultaneously
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True  # Enables detailed eye/iris detection
        )
        
        # Define Kid-Friendly Colors (BGR format)
        # Bright, neon-like colors are engaging for children
        self.COLOR_LEFT = (0, 255, 0)      # Neon Green (for User's Left side)
        self.COLOR_RIGHT = (255, 0, 255)   # Magenta (for User's Right side)
        self.COLOR_FACE = (255, 255, 0)    # Cyan (for Face)
        self.COLOR_TEXT = (255, 255, 255)  # White text
        self.COLOR_BG = (50, 50, 50)       # Dark Gray background for text boxes
        
        # Font settings for large, readable labels
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7  # Smaller for multiple participants
        self.FONT_THICKNESS = 2

    def detect_features(self, frame):
        """
        Processes the video frame to detect holistic landmarks.
        Args:
            frame: The BGR video frame from OpenCV.
        Returns:
            results: The MediaPipe detection results.
        """
        # MediaPipe requires RGB input
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image. 
        # IMPORTANT: We pass the original (unflipped) image to MediaPipe.
        # This ensures MediaPipe correctly identifies "Left" vs "Right" hands 
        # based on the person's anatomy, not the screen position.
        results = self.holistic.process(image_rgb)
        return results

    def _draw_label(self, img, text, x, y, color):
        """
        Helper function to draw a colorful label with a background box.
        """
        (text_w, text_h), baseline = cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
        
        # Draw a filled rectangle behind the text for better contrast
        # Adding some padding
        cv2.rectangle(img, 
                     (x, y - text_h - 10), 
                     (x + text_w, y + 10), 
                     color, 
                     -1) # Filled box
                     
        # Draw a white border around the box for "pop"
        cv2.rectangle(img, 
                     (x, y - text_h - 10), 
                     (x + text_w, y + 10), 
                     (255, 255, 255), 
                     2)
        
        # Draw the text on top
        cv2.putText(img, text, (x, y), self.FONT, self.FONT_SCALE, self.COLOR_TEXT, self.FONT_THICKNESS)

    def draw_labels(self, frame, results):
        """
        Draws bounding boxes and labels on the frame.
        Handles the coordinate transformation for the Mirror Effect.
        """
        h, w, c = frame.shape
        
        # 1. Create the Mirror View
        # We flip the image horizontally for the display.
        # This feels natural to the user (like a real mirror).
        output_image = cv2.flip(frame, 1)
        
        # 2. Coordinate Transformation Function
        # Since we flipped the image, we must also flip the X coordinates of the landmarks.
        # Original X: 0 (Left) -> 1 (Right)
        # Mirrored X: The pixel at x=0 is now at x=Width.
        # Formula: x_mirrored = (1 - x_original) * Width
        def get_coords(landmark):
            return int((1 - landmark.x) * w), int(landmark.y * h)

        vis_thresh = 0.5

        # --- HEAD ---
        if results.face_landmarks:
            # General Head Label (using forehead landmark 10)
            forehead = results.face_landmarks.landmark[10]
            fx, fy = get_coords(forehead)
            self._draw_label(output_image, "Head", fx - 20, fy - 50, self.COLOR_FACE)

        # --- UPPER BODY (ARMS & HANDS) ---
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Left Arm (11-13-15)
            if lm[11].visibility > vis_thresh and lm[13].visibility > vis_thresh:
                s1 = get_coords(lm[11]) # Shoulder
                e1 = get_coords(lm[13]) # Elbow
                cv2.line(output_image, s1, e1, self.COLOR_LEFT, 4)
                self._draw_label(output_image, "Left Arm", e1[0] + 10, e1[1], self.COLOR_LEFT)
                
                if lm[15].visibility > vis_thresh:
                    w1 = get_coords(lm[15]) # Wrist
                    cv2.line(output_image, e1, w1, self.COLOR_LEFT, 4)

            # Right Arm (12-14-16)
            if lm[12].visibility > vis_thresh and lm[14].visibility > vis_thresh:
                s2 = get_coords(lm[12])
                e2 = get_coords(lm[14])
                cv2.line(output_image, s2, e2, self.COLOR_RIGHT, 4)
                self._draw_label(output_image, "Right Arm", e2[0] - 100, e2[1], self.COLOR_RIGHT)
                
                if lm[16].visibility > vis_thresh:
                    w2 = get_coords(lm[16])
                    cv2.line(output_image, e2, w2, self.COLOR_RIGHT, 4)

        # --- HANDS ---
        # MediaPipe Holistic separates hands into .left_hand_landmarks and .right_hand_landmarks
        
        # Left Hand (The child's physical Left Hand)
        if results.left_hand_landmarks:
            # Use the wrist (landmark 0) as the anchor
            wrist = results.left_hand_landmarks.landmark[0]
            cx, cy = get_coords(wrist)
            
            # Draw a large circle
            cv2.circle(output_image, (cx, cy), 25, self.COLOR_LEFT, 4)
            # Label it. Position text slightly to the right of the hand in the mirror view
            self._draw_label(output_image, "Left Hand", cx + 30, cy + 40, self.COLOR_LEFT)

        # Right Hand (The child's physical Right Hand)
        if results.right_hand_landmarks:
            wrist = results.right_hand_landmarks.landmark[0]
            cx, cy = get_coords(wrist)
            
            cv2.circle(output_image, (cx, cy), 25, self.COLOR_RIGHT, 4)
            # Position text slightly to the left (subtracting width approx)
            self._draw_label(output_image, "Right Hand", cx - 120, cy + 40, self.COLOR_RIGHT)

        # --- LOWER BODY (LEGS & FEET) ---
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Left Leg (23-25-27) & Foot (27-31)
            if lm[23].visibility > vis_thresh and lm[25].visibility > vis_thresh:
                h1 = get_coords(lm[23]) # Hip
                k1 = get_coords(lm[25]) # Knee
                cv2.line(output_image, h1, k1, self.COLOR_LEFT, 4)
                
                if lm[27].visibility > vis_thresh:
                    a1 = get_coords(lm[27]) # Ankle
                    cv2.line(output_image, k1, a1, self.COLOR_LEFT, 4)
                    self._draw_label(output_image, "Left Leg", k1[0] + 10, k1[1], self.COLOR_LEFT)
                    
                    # Left Foot (Ankle to Foot Index 31)
                    if lm[31].visibility > vis_thresh:
                        f1 = get_coords(lm[31])
                        cv2.line(output_image, a1, f1, self.COLOR_LEFT, 4)
                        cv2.circle(output_image, f1, 10, self.COLOR_LEFT, -1)
                        self._draw_label(output_image, "Left Foot", f1[0] + 10, f1[1], self.COLOR_LEFT)

            # Right Leg (24-26-28) & Foot (28-32)
            if lm[24].visibility > vis_thresh and lm[26].visibility > vis_thresh:
                h2 = get_coords(lm[24])
                k2 = get_coords(lm[26])
                cv2.line(output_image, h2, k2, self.COLOR_RIGHT, 4)
                
                if lm[28].visibility > vis_thresh:
                    a2 = get_coords(lm[28])
                    cv2.line(output_image, k2, a2, self.COLOR_RIGHT, 4)
                    self._draw_label(output_image, "Right Leg", k2[0] - 100, k2[1], self.COLOR_RIGHT)
                    
                    # Right Foot
                    if lm[32].visibility > vis_thresh:
                        f2 = get_coords(lm[32])
                        cv2.line(output_image, a2, f2, self.COLOR_RIGHT, 4)
                        cv2.circle(output_image, f2, 10, self.COLOR_RIGHT, -1)
                        self._draw_label(output_image, "Right Foot", f2[0] - 100, f2[1], self.COLOR_RIGHT)

        return output_image

def main():
    # Open Webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize our Educator Class
    educator = BodyPartEducator()
    
    print("Starting BodyPartBuddy...")
    print("Press 'Esc' to exit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 1. Detect Features
        results = educator.detect_features(frame)
        
        # 2. Draw Labels (and handle mirror logic)
        output_frame = educator.draw_labels(frame, results)
        
        # 3. Display
        cv2.imshow('BodyPartBuddy - Magic Mirror', output_frame)
        
        # Exit on ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()