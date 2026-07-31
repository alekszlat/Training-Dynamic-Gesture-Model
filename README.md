# Training-Dynamic-Gesture-Model

Creating a model for tracking and identifying dynamic gestures based on a science paper

## TO DOs:

Step 1: Open webcam with OpenCV
Explanation: Use OpenCV to access the webcam and display the video feed in a window. This will allow you to capture real-time video frames for gesture recognition.

Step 2: Detect hand landmarks with MediaPipe
Explanation: Use MediaPipe's hand tracking solution to detect and extract hand landmarks from the video frames. This will provide the necessary data for gesture recognition.

Step 3: Save landmarks for one gesture
Explanation: Save the detected hand landmarks for a single gesture in a structured format for training purposes.

Step 4: Save landmarks for multiple gestures
Explanation: Extend the previous step to save hand landmarks for multiple gestures, creating a dataset that can be used to train the gesture recognition model.

Step 5: Pad all samples to fixed length
Explanation: Ensure that all gesture samples have a consistent length by padding shorter sequences with zeros or truncating longer sequences. This is important for training the model effectively.

Step 6: Train a simple baseline classifier
Explanation: Train a simple baseline classifier (only one movement) using the saved hand landmark data to establish a performance benchmark for gesture recognition.

Step 7: Train Transformer Encoder
Explanation: Train a Transformer Encoder model using the saved hand landmark data to improve gesture recognition performance. The Transformer architecture is well-suited for sequence data and can capture complex temporal dependencies in gestures.

Step 8: Evaluate with confusion matrix
Explanation: Evaluate the performance of the trained model using a confusion matrix to visualize the classification results and identify areas for improvement.

Step 9: Run real-time prediction
Explanation: Implement real-time gesture recognition by feeding the live video feed from the webcam into the trained model and displaying the predicted gestures on the screen.

Step 10: Map gestures to keyboard/mouse actions
Explanation: Create a mapping between recognized gestures and specific keyboard or mouse actions, allowing users to control their computer using dynamic gestures.
