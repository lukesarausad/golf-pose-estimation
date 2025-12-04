#!/usr/bin/env python3
"""
Golf Swing Analyzer using MediaPipe Pose
Features phase detection, biomechanics analysis, and coaching feedback.
"""
import argparse
import cv2
import numpy as np
import os
from pathlib import Path

try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'mediapipe'])
    import mediapipe as mp

from swing_phase_detector import SwingPhaseDetector
from golf_swing_analyzer import GolfSwingAnalyzer


class MediaPipeGolfAnalyzer:
    """Golf swing analyzer using MediaPipe Pose"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0, 1, or 2 (2 = most accurate)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Phase detection and analysis
        self.phase_detector = SwingPhaseDetector()
        self.swing_analyzer = GolfSwingAnalyzer()

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def analyze_pose(self, landmarks):
        """Extract golf-specific biomechanics"""
        if not landmarks:
            return {}

        try:
            # Get key landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            # Calculate angles
            metrics = {
                'right_arm_angle': self.calculate_angle(right_shoulder, right_elbow, right_wrist),
                'left_arm_angle': self.calculate_angle(left_shoulder, left_elbow, left_wrist),
                'right_knee_angle': self.calculate_angle(right_hip, right_knee, right_ankle),
                'left_knee_angle': self.calculate_angle(left_hip, left_knee, left_ankle),
            }

            # Shoulder rotation (simplified)
            shoulder_slope = (right_shoulder.y - left_shoulder.y) / \
                           (right_shoulder.x - left_shoulder.x + 1e-6)
            metrics['shoulder_rotation'] = np.degrees(np.arctan(shoulder_slope))

            # Hip rotation (simplified)
            hip_slope = (right_hip.y - left_hip.y) / (right_hip.x - left_hip.x + 1e-6)
            metrics['hip_rotation'] = np.degrees(np.arctan(hip_slope))

            return metrics

        except Exception as e:
            return {}

    def process_video(self, video_path, output_path, show_metrics=True):
        """Process video and generate analyzed output"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f}s")

        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"\nProcessing video...")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.pose.process(frame_rgb)

            # Draw skeleton and analyze
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Calculate metrics
                metrics = self.analyze_pose(landmarks)

                if show_metrics and metrics:
                    # Extract wrist positions for phase detection
                    left_wrist = (
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * height
                    )
                    right_wrist = (
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height
                    )

                    # Detect current phase
                    current_phase = self.phase_detector.update(left_wrist, right_wrist, frame_idx)

                    # Store metrics for analysis (but don't show per-frame feedback)
                    self.swing_analyzer.analyze_phase(current_phase, metrics)
                    self.swing_analyzer.add_frame_data(frame_idx, current_phase, metrics)

                    # Draw phase label at top center with background
                    phase_text = f"Phase: {current_phase}"
                    text_size = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    text_x = (width - text_size[0]) // 2
                    # Draw semi-transparent background for phase label
                    padding = 10
                    cv2.rectangle(frame,
                                 (text_x - padding, 10),
                                 (text_x + text_size[0] + padding, 50),
                                 (0, 0, 0), -1)
                    cv2.putText(frame, phase_text, (text_x, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

                    # Draw metrics on left side with background
                    y_pos = 80
                    # Calculate background rectangle dimensions
                    max_text_width = 0
                    metric_texts = []
                    for key, value in metrics.items():
                        if value is not None:
                            text = f"{key.replace('_', ' ').title()}: {value:.1f}"
                            metric_texts.append(text)
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            max_text_width = max(max_text_width, text_size[0])

                    # Draw background rectangle for all metrics
                    if metric_texts:
                        padding = 5
                        bg_height = len(metric_texts) * 20 + padding
                        cv2.rectangle(frame,
                                     (5, y_pos - 15),
                                     (15 + max_text_width + padding, y_pos - 15 + bg_height),
                                     (0, 0, 0), -1)

                        # Draw metrics text
                        for text in metric_texts:
                            cv2.putText(frame, text, (10, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            y_pos += 20

            # Write frame
            out.write(frame)

            # Progress
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  Progress: {frame_idx}/{total_frames} ({progress:.1f}%)")

        cap.release()
        out.release()
        self.pose.close()

        # Export analysis data
        csv_path = str(Path(output_path).parent / "swing_analysis.csv")
        self.swing_analyzer.export_swing_data(csv_path)

        # Print summary
        self.swing_analyzer.print_summary()

        # Print top 3 deviations
        print("\n" + "=" * 60)
        print("TOP 3 AREAS FOR IMPROVEMENT")
        print("=" * 60)
        top_deviations = self.swing_analyzer.get_top_deviations(3)
        for i, dev in enumerate(top_deviations, 1):
            print(f"{i}. {dev['feedback']}")
        print("=" * 60)

        print(f"\nProcessing complete!")
        print(f"  Video saved to: {output_path}")
        print(f"  Analysis data: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze golf swings with MediaPipe')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input golf video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video')
    parser.add_argument('--no-metrics', action='store_true',
                        help='Disable biomechanics overlay')

    args = parser.parse_args()

    # Handle video path - check if it exists as-is, or in input/ folder
    video_path = Path(args.video)
    if not video_path.exists():
        # Try looking in input/ folder
        input_path = Path('input') / video_path.name
        if input_path.exists():
            video_path = input_path
            args.video = str(input_path)
        else:
            raise FileNotFoundError(f"Video not found: {args.video} or {input_path}")

    # Set default output to outputs folder
    if args.output is None:
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        args.output = str(Path('outputs') / f"{video_path.stem}_analyzed{video_path.suffix}")

    print("="*60)
    print("Golf Swing Analyzer (MediaPipe)")
    print("="*60)

    analyzer = MediaPipeGolfAnalyzer()
    analyzer.process_video(
        video_path=args.video,
        output_path=args.output,
        show_metrics=not args.no_metrics
    )


if __name__ == '__main__':
    main()
