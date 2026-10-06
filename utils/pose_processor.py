import math
import threading
import time

import av
import cv2
import mediapipe as mp
from streamlit_webrtc import VideoProcessorBase

from .ml_model import FormModel


class RehabilitationProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.form_model = FormModel()
        self.repetitions = 0
        self.angle = 180.0
        self.angle_history = []
        self.state = "extended"
        self.last_angle = None
        self.last_time = None
        self.feedback = "Stand where your left shoulder, elbow and wrist are visible."
        self.form_score = 0.0
        self.session_saved = False

    @staticmethod
    def _angle(a, b, c):
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag = math.hypot(*ba) * math.hypot(*bc)
        if mag == 0:
            return 180.0
        value = max(-1.0, min(1.0, dot / mag))
        return math.degrees(math.acos(value))

    def _process(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            cv2.putText(image, "Move into camera view", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        lm = results.pose_landmarks.landmark
        s = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        e = lm[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        w = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        angle = self._angle(s, e, w)
        now = time.time()

        with self.lock:
            self.angle = angle
            self.angle_history.append(angle)
            self.angle_history = self.angle_history[-30:]

            speed = 0.0
            if self.last_angle is not None and self.last_time is not None:
                dt = max(now - self.last_time, 1e-3)
                speed = abs(angle - self.last_angle) / dt
            self.last_angle = angle
            self.last_time = now

            stability = 1.0
            if len(self.angle_history) >= 8:
                mean = sum(self.angle_history) / len(self.angle_history)
                variance = sum((x - mean) ** 2 for x in self.angle_history) / len(self.angle_history)
                stability = max(0.0, min(1.0, 1.0 - math.sqrt(variance) / 35.0))

            if angle < 75 and self.state == "extended":
                self.state = "flexed"
            elif angle > 145 and self.state == "flexed":
                self.state = "extended"
                self.repetitions += 1

            probability = self.form_model.predict(angle, speed, stability)
            self.form_score = probability * 100.0

            if angle < 75:
                self.feedback = "Good flexion. Slowly return to the starting position."
            elif angle > 145:
                self.feedback = "Arm extended. Bend the elbow smoothly."
            elif speed > 180:
                self.feedback = "Slow down for a more controlled movement."
            else:
                self.feedback = "Keep your shoulder steady and move through a comfortable range."

        cv2.putText(image, f"Angle: {angle:.1f} deg", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"Reps: {self.repetitions}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"Form: {self.form_score:.0f}/100", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def recv(self, frame):
        return self._process(frame)

    @property
    def average_angle(self):
        return sum(self.angle_history) / len(self.angle_history) if self.angle_history else self.angle

    @property
    def max_range(self):
        return (max(self.angle_history) - min(self.angle_history)) if self.angle_history else 0.0

    @property
    def status(self):
        if not self.angle_history:
            return "Waiting"
        if self.form_score >= 75:
            return "Good"
        if self.form_score >= 50:
            return "Fair"
        return "Adjust"

    def has_session_data(self):
        return len(self.angle_history) >= 5

    def get_metrics(self):
        with self.lock:
            return {
                "angle": self.angle,
                "repetitions": self.repetitions,
                "form_score": self.form_score,
                "status": self.status,
                "feedback": self.feedback,
            }
