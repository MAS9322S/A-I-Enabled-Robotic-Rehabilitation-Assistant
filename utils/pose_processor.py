import math
import threading
import time

import av
import cv2
import mediapipe as mp
from streamlit_webrtc import VideoProcessorBase

from .ml_model import FormModel


class RehabilitationProcessor(VideoProcessorBase):
    """
    Multi-exercise rehabilitation processor.

    Supported exercises:
    - Elbow Flexion
    - Shoulder Flexion
    - Shoulder Abduction
    - Knee Flexion

    All exercises currently use the LEFT side of the body.
    """

    def __init__(self, exercise="Elbow Flexion"):
        self.lock = threading.Lock()

        self.exercise = exercise

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.form_model = FormModel()

        # Session metrics
        self.repetitions = 0
        self.angle = 180.0
        self.angle_history = []

        # Repetition state
        self.state = "start"

        # Movement tracking
        self.last_angle = None
        self.last_time = None

        # Feedback
        self.feedback = self._initial_feedback()

        # Form score
        self.form_score = 0.0

        # Session saving
        self.session_saved = False

    def _initial_feedback(self):
        if self.exercise == "Elbow Flexion":
            return "Keep your left shoulder, elbow and wrist visible."

        if self.exercise == "Shoulder Flexion":
            return "Keep your left hip, shoulder and elbow visible."

        if self.exercise == "Shoulder Abduction":
            return "Keep your left hip, shoulder and elbow visible."

        if self.exercise == "Knee Flexion":
            return "Keep your left hip, knee and ankle visible."

        return "Move into camera view."

    @staticmethod
    def _angle(a, b, c):
        """
        Calculate angle ABC using three MediaPipe landmarks.
        """
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)

        dot = ba[0] * bc[0] + ba[1] * bc[1]

        mag = math.hypot(*ba) * math.hypot(*bc)

        if mag == 0:
            return 180.0

        value = max(-1.0, min(1.0, dot / mag))

        return math.degrees(math.acos(value))

    def _get_angle(self, lm):
        """
        Calculate the appropriate joint angle for the selected exercise.
        """

        LEFT_SHOULDER = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
        LEFT_ELBOW = mp.solutions.pose.PoseLandmark.LEFT_ELBOW
        LEFT_WRIST = mp.solutions.pose.PoseLandmark.LEFT_WRIST

        LEFT_HIP = mp.solutions.pose.PoseLandmark.LEFT_HIP
        LEFT_KNEE = mp.solutions.pose.PoseLandmark.LEFT_KNEE
        LEFT_ANKLE = mp.solutions.pose.PoseLandmark.LEFT_ANKLE

        if self.exercise == "Elbow Flexion":
            shoulder = lm[LEFT_SHOULDER]
            elbow = lm[LEFT_ELBOW]
            wrist = lm[LEFT_WRIST]

            return self._angle(shoulder, elbow, wrist)

        if self.exercise == "Shoulder Flexion":
            hip = lm[LEFT_HIP]
            shoulder = lm[LEFT_SHOULDER]
            elbow = lm[LEFT_ELBOW]

            return self._angle(hip, shoulder, elbow)

        if self.exercise == "Shoulder Abduction":
            hip = lm[LEFT_HIP]
            shoulder = lm[LEFT_SHOULDER]
            elbow = lm[LEFT_ELBOW]

            return self._angle(hip, shoulder, elbow)

        if self.exercise == "Knee Flexion":
            hip = lm[LEFT_HIP]
            knee = lm[LEFT_KNEE]
            ankle = lm[LEFT_ANKLE]

            return self._angle(hip, knee, ankle)

        return 180.0

    def _update_repetitions(self, angle):
        """
        Exercise-specific repetition counting.
        """

        # ---------------------------------------
        # ELBOW FLEXION
        # ---------------------------------------
        if self.exercise == "Elbow Flexion":

            if angle < 75 and self.state == "extended":
                self.state = "flexed"

            elif angle > 145 and self.state == "flexed":
                self.state = "extended"
                self.repetitions += 1

        # ---------------------------------------
        # SHOULDER FLEXION
        # ---------------------------------------
        elif self.exercise == "Shoulder Flexion":

            # Arm approximately down
            if angle > 150 and self.state == "start":
                self.state = "down"

            # Arm raised forward
            elif angle < 70 and self.state in ("start", "down"):
                self.state = "up"

            # Return to starting position
            elif angle > 150 and self.state == "up":
                self.state = "down"
                self.repetitions += 1

        # ---------------------------------------
        # SHOULDER ABDUCTION
        # ---------------------------------------
        elif self.exercise == "Shoulder Abduction":

            # Arm down
            if angle > 150 and self.state == "start":
                self.state = "down"

            # Arm raised sideways
            elif angle < 70 and self.state in ("start", "down"):
                self.state = "up"

            # Return down
            elif angle > 150 and self.state == "up":
                self.state = "down"
                self.repetitions += 1

        # ---------------------------------------
        # KNEE FLEXION
        # ---------------------------------------
        elif self.exercise == "Knee Flexion":

            # Leg relatively straight
            if angle > 150 and self.state == "start":
                self.state = "straight"

            # Knee bent
            elif angle < 100 and self.state in ("start", "straight"):
                self.state = "bent"

            # Return to straight
            elif angle > 150 and self.state == "bent":
                self.state = "straight"
                self.repetitions += 1

    def _get_feedback(self, angle, speed):
        """
        Exercise-specific feedback.
        """

        if self.exercise == "Elbow Flexion":

            if angle < 75:
                return "Good flexion. Slowly return to the starting position."

            if angle > 145:
                return "Arm extended. Bend the elbow smoothly."

            if speed > 180:
                return "Slow down for a more controlled movement."

            return "Keep your shoulder steady and move through a comfortable range."

        if self.exercise == "Shoulder Flexion":

            if angle < 70:
                return "Good shoulder raise. Slowly lower your arm."

            if angle > 150:
                return "Arm is down. Slowly raise it forward."

            if speed > 180:
                return "Slow down and control the shoulder movement."

            return "Keep your elbow relaxed and raise your arm smoothly."

        if self.exercise == "Shoulder Abduction":

            if angle < 70:
                return "Good shoulder raise. Slowly lower your arm."

            if angle > 150:
                return "Arm is down. Raise it sideways smoothly."

            if speed > 180:
                return "Slow down for better control."

            return "Keep your body steady while raising your arm sideways."

        if self.exercise == "Knee Flexion":

            if angle < 100:
                return "Good knee flexion. Slowly straighten your leg."

            if angle > 150:
                return "Leg is relatively straight. Bend your knee slowly."

            if speed > 180:
                return "Slow down and control the knee movement."

            return "Keep your upper leg steady and move through a comfortable range."

        return "Perform the exercise slowly and with control."

    def _process(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Mirror camera
        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            cv2.putText(
                image,
                "Move into camera view",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24",
            )

        lm = results.pose_landmarks.landmark

        try:
            angle = self._get_angle(lm)
        except Exception:
            cv2.putText(
                image,
                "Adjust your position",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24",
            )

        now = time.time()

        with self.lock:

            self.angle = angle

            self.angle_history.append(angle)

            # Keep the last 30 angle measurements
            self.angle_history = self.angle_history[-30:]

            # ---------------------------------------
            # Movement speed
            # ---------------------------------------

            speed = 0.0

            if self.last_angle is not None and self.last_time is not None:

                dt = max(
                    now - self.last_time,
                    1e-3,
                )

                speed = abs(
                    angle - self.last_angle
                ) / dt

            self.last_angle = angle
            self.last_time = now

            # ---------------------------------------
            # Movement stability
            # ---------------------------------------

            stability = 1.0

            if len(self.angle_history) >= 8:

                mean = (
                    sum(self.angle_history)
                    / len(self.angle_history)
                )

                variance = (
                    sum(
                        (x - mean) ** 2
                        for x in self.angle_history
                    )
                    / len(self.angle_history)
                )

                stability = max(
                    0.0,
                    min(
                        1.0,
                        1.0 - math.sqrt(variance) / 35.0,
                    ),
                )

            # ---------------------------------------
            # Repetition counting
            # ---------------------------------------

            self._update_repetitions(angle)

            # ---------------------------------------
            # Form score
            # ---------------------------------------

            probability = self.form_model.predict(
                angle,
                speed,
                stability,
            )

            self.form_score = probability * 100.0

            # ---------------------------------------
            # Feedback
            # ---------------------------------------

            self.feedback = self._get_feedback(
                angle,
                speed,
            )

        # ---------------------------------------
        # Camera overlay
        # ---------------------------------------

        cv2.putText(
            image,
            f"Exercise: {self.exercise}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            f"Angle: {angle:.1f} deg",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            f"Reps: {self.repetitions}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            f"Form: {self.form_score:.0f}/100",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )

    def recv(self, frame):
        return self._process(frame)

    @property
    def average_angle(self):
        if not self.angle_history:
            return self.angle

        return (
            sum(self.angle_history)
            / len(self.angle_history)
        )

    @property
    def max_range(self):
        if not self.angle_history:
            return 0.0

        return (
            max(self.angle_history)
            - min(self.angle_history)
        )

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
