import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import random
import math

# --- Text-to-Speech Engine (Non-blocking) ---
class TTSHandler:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        # Initialize engine inside the thread to avoid COM threading issues on Windows
        try:
            engine = pyttsx3.init()
        except Exception as e:
            print(f"TTS Init Error: {e}")
            return

        while self.running:
            text = None
            with self.lock:
                if self.queue:
                    text = self.queue.pop(0)
            if text:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"TTS Error: {e}")
            else:
                time.sleep(0.1)

    def speak(self, text):
        with self.lock:
            # Clear queue if too long to avoid lag
            if len(self.queue) > 2:
                self.queue = []
            self.queue.append(text)

    def stop(self):
        self.running = False

# --- Visual Reward System ---
class ConfettiSystem:
    def __init__(self):
        self.particles = []
        self.colors = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 165, 255), (255, 255, 255)]

    def start(self, width, height):
        self.particles = []
        for _ in range(100):
            self.particles.append({
                'x': random.randint(0, width),
                'y': random.randint(-height, 0),
                'speed': random.randint(5, 15),
                'color': random.choice(self.colors),
                'size': random.randint(5, 10)
            })

    def update_and_draw(self, frame):
        h, w, _ = frame.shape
        active = False
        for p in self.particles:
            p['y'] += p['speed']
            if p['y'] < h:
                active = True
                cv2.circle(frame, (p['x'], int(p['y'])), p['size'], p['color'], -1)
        return active

# --- Game Logic ---
class SimonSaysGame:
    def __init__(self, tts):
        self.tts = tts
        self.state = "IDLE"  # IDLE, ISSUING, WAITING, SUCCESS, CELEBRATION
        self.score = 0
        self.current_command = None
        self.state_timer = 0
        self.command_start_time = 0
        self.confetti = ConfettiSystem()
        self.is_celebrating = False
        
        # Define Commands
        self.commands = [
            {"text": "Raise your Right Hand", "type": "pose", "check": self._check_raise_right_hand},
            {"text": "Raise your Left Hand", "type": "pose", "check": self._check_raise_left_hand},
            {"text": "Touch your Nose", "type": "pose", "check": self._check_touch_nose},
            {"text": "Clap your Hands", "type": "pose", "check": self._check_clap},
            {"text": "Cover your Eyes", "type": "pose", "check": self._check_cover_eyes},
        ]

    def update(self, frame, landmarks):
        current_time = time.time()
        h, w, _ = frame.shape

        # State Machine
        if self.state == "IDLE":
            # Wait for user to be ready (e.g., hands visible)
            if self._are_hands_visible(landmarks):
                self.state = "ISSUING"
                self.state_timer = current_time
                self.tts.speak("Welcome to Simon Says! Get ready!")
            else:
                cv2.putText(frame, "Show your hands to start!", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        elif self.state == "ISSUING":
            if current_time - self.state_timer > 3.0: # Wait a bit before command
                self.current_command = random.choice(self.commands)
                self.tts.speak(f"Simon Says... {self.current_command['text']}")
                self.state = "WAITING"
                self.command_start_time = current_time

        elif self.state == "WAITING":
            # Draw Command
            text = f"Command: {self.current_command['text']}"
            cv2.putText(frame, text, (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Check for success
            if self.current_command['check'](landmarks):
                self.state = "SUCCESS"
                self.score += 1
                self.tts.speak("Great Job!")
                self.state_timer = current_time
            
            # Timeout (5 seconds)
            elif current_time - self.command_start_time > 8.0:
                self.tts.speak("Time's up! Let's try another one.")
                self.state = "ISSUING"
                self.state_timer = current_time

        elif self.state == "SUCCESS":
            self.is_celebrating = True
            self.confetti.start(w, h)
            self.state = "CELEBRATION"
            self.state_timer = current_time

        elif self.state == "CELEBRATION":
            # Draw "CORRECT!" overlay
            cv2.putText(frame, "CORRECT!", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            
            still_active = self.confetti.update_and_draw(frame)
            if not still_active or (current_time - self.state_timer > 2.5):
                self.is_celebrating = False
                self.state = "ISSUING"
                self.state_timer = current_time

        # Draw Score
        cv2.putText(frame, f"Score: {self.score}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # --- Verification Logic ---
    def _are_hands_visible(self, results):
        return results.left_hand_landmarks or results.right_hand_landmarks

    def _check_raise_right_hand(self, results):
        # Right Hand raised: Right Wrist Y < Right Shoulder Y
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 16: Right Wrist, 12: Right Shoulder
            if lm[16].visibility > 0.5 and lm[12].visibility > 0.5:
                return lm[16].y < lm[12].y
        return False

    def _check_raise_left_hand(self, results):
        # Left Hand raised: Left Wrist Y < Left Shoulder Y
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 15: Left Wrist, 11: Left Shoulder
            if lm[15].visibility > 0.5 and lm[11].visibility > 0.5:
                return lm[15].y < lm[11].y
        return False

    def _check_touch_nose(self, results):
        # Distance between either wrist and nose < threshold
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            nose = lm[0]
            l_wrist = lm[15]
            r_wrist = lm[16]
            
            def dist(p1, p2):
                return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

            if nose.visibility > 0.5:
                if l_wrist.visibility > 0.5 and dist(nose, l_wrist) < 0.15: return True
                if r_wrist.visibility > 0.5 and dist(nose, r_wrist) < 0.15: return True
        return False

    def _check_clap(self, results):
        # Distance between wrists is small
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            l_wrist = lm[15]
            r_wrist = lm[16]
            
            if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5:
                d = math.sqrt((l_wrist.x - r_wrist.x)**2 + (l_wrist.y - r_wrist.y)**2)
                return d < 0.15
        return False

    def _check_cover_eyes(self, results):
        # Wrists near eyes (approximate using pose eye landmarks 2 and 5)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            l_eye = lm[2]
            r_eye = lm[5]
            l_wrist = lm[15]
            r_wrist = lm[16]
            
            def dist(p1, p2):
                return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

            # Check if hands are up near face height generally
            if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5:
                # Simplified: Hands are above shoulders and close to center x
                if l_wrist.y < lm[11].y and r_wrist.y < lm[12].y:
                    # Check proximity to eyes
                    # Increased threshold slightly to make it easier for kids
                    if dist(l_wrist, l_eye) < 0.25 or dist(r_wrist, r_eye) < 0.25:
                        return True
        return False


class BodyPartEducator:
    """
    A class to handle body part detection and educational visualization for children.
    """
    def __init__(self):
        # Initialize MediaPipe Holistic solution
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )
        
        # Define Kid-Friendly Colors (BGR format)
        self.COLOR_LEFT = (0, 255, 0)      # Neon Green
        self.COLOR_RIGHT = (255, 0, 255)   # Magenta
        self.COLOR_FACE = (255, 255, 0)    # Cyan
        self.COLOR_TEXT = (255, 255, 255)  # White
        
        # Font settings
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7
        self.FONT_THICKNESS = 2
        
        # Initialize Game Components
        self.tts = TTSHandler()
        self.game = SimonSaysGame(self.tts)

    def detect_features(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        return results

    def _draw_label(self, img, text, x, y, color):
        (text_w, text_h), baseline = cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
        cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y + 10), color, -1)
        cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y + 10), (255, 255, 255), 2)
        cv2.putText(img, text, (x, y), self.FONT, self.FONT_SCALE, self.COLOR_TEXT, self.FONT_THICKNESS)

    def draw_labels(self, frame, results):
        h, w, c = frame.shape
        
        # 1. Create the Mirror View
        output_image = cv2.flip(frame, 1)
        
        # 2. Coordinate Transformation Function
        def get_coords(landmark):
            return int((1 - landmark.x) * w), int(landmark.y * h)

        vis_thresh = 0.5

        # --- HEAD ---
        if results.face_landmarks:
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
        if results.left_hand_landmarks:
            wrist = results.left_hand_landmarks.landmark[0]
            cx, cy = get_coords(wrist)
            cv2.circle(output_image, (cx, cy), 25, self.COLOR_LEFT, 4)
            self._draw_label(output_image, "Left Hand", cx + 30, cy + 40, self.COLOR_LEFT)

        if results.right_hand_landmarks:
            wrist = results.right_hand_landmarks.landmark[0]
            cx, cy = get_coords(wrist)
            cv2.circle(output_image, (cx, cy), 25, self.COLOR_RIGHT, 4)
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
                    
                    if lm[32].visibility > vis_thresh:
                        f2 = get_coords(lm[32])
                        cv2.line(output_image, a2, f2, self.COLOR_RIGHT, 4)
                        cv2.circle(output_image, f2, 10, self.COLOR_RIGHT, -1)
                        self._draw_label(output_image, "Right Foot", f2[0] - 100, f2[1], self.COLOR_RIGHT)

        # --- GAME UPDATE ---
        # Pass the flipped image to the game manager for drawing overlays
        self.game.update(output_image, results)

        return output_image

    def close(self):
        self.tts.stop()

def main():
    cap = cv2.VideoCapture(0)
    educator = BodyPartEducator()
    
    print("Starting BodyPartBuddy - Simon Says Mode...")
    print("Press 'Esc' to exit.")
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            results = educator.detect_features(frame)
            output_frame = educator.draw_labels(frame, results)
            
            cv2.imshow('BodyPartBuddy - Simon Says', output_frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        educator.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
