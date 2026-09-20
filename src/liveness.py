import math


class LivenessAnalyzer:
    """
    Basic on-device liveness analysis using facial landmarks.

    The analyzer looks for natural eye-blink activity over multiple frames.
    This is a prototype liveness signal, not a certified anti-spoofing system.
    """

    def __init__(self):
        self.blink_count = 0
        self.was_eye_closed = False

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    @classmethod
    def eye_aspect_ratio(cls, eye):
        """
        Calculate Eye Aspect Ratio (EAR).

        eye = [p1, p2, p3, p4, p5, p6]
        """

        vertical_1 = cls._distance(eye[1], eye[5])
        vertical_2 = cls._distance(eye[2], eye[4])
        horizontal = cls._distance(eye[0], eye[3])

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def update(self, left_eye, right_eye):
        """
        Process one video frame.

        Returns:
            {
                "left_ear": float,
                "right_ear": float,
                "blink_detected": bool,
                "blink_count": int,
                "status": str
            }
        """

        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)

        average_ear = (left_ear + right_ear) / 2.0

        # Eye closed threshold.
        closed = average_ear < 0.20

        blink_detected = False

        # Detect transition:
        # Open -> Closed -> Open
        if closed and not self.was_eye_closed:
            self.was_eye_closed = True

        elif not closed and self.was_eye_closed:
            self.blink_count += 1
            blink_detected = True
            self.was_eye_closed = False

        status = "LIVE SIGNAL DETECTED" if self.blink_count > 0 else "CHECKING LIVENESS"

        return {
            "left_ear": round(left_ear, 3),
            "right_ear": round(right_ear, 3),
            "blink_detected": blink_detected,
            "blink_count": self.blink_count,
            "status": status
        }

    def reset(self):
        """Reset the liveness state."""
        self.blink_count = 0
        self.was_eye_closed = False
