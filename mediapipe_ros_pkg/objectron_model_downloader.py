import mediapipe as mp

if __name__ == "__main__":
    mp_objectron = mp.solutions.objectron
    mp_objectron.Objectron(
        static_image_mode=False,
        max_num_objects=5,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.99,
        model_name="Cup",
    )
