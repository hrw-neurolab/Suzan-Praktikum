import cv2
import numpy as np
import tensorflow as tf
import os

# Suppress TensorFlow logging to keep the console clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Get the path for the TFLite model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "gesture_model_final.tflite")

# Initialize the TFLite Interpreter
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("STATUS: System Ready. Inference Pipeline Synchronized.")
except Exception as e:
    
    exit()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names in the exact alphabetical order used during training
class_names = ['GRIPPER_CLOSED', 'GRIPPER_OPEN', 'MOVE_BACKWARD', 'MOVE_DOWN', 
               'MOVE_FORWARD', 'MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'SAFTEY']

# Initialize the Webcam
cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for a more intuitive user experience
    frame = cv2.flip(frame, 1)

    # 1. COLOR SPACE CORRECTION:
    # Convert from BGR (OpenCV default) to RGB (Model training default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. IMAGE RESIZING:
    # Resize to 224x224 to match the MobileNetV2 input shape
    img_resized = cv2.resize(rgb_frame, (224, 224))
    
    # 3. NORMALIZATION :
    #  pixel range  [-1, 1] as expected by the model
    # Formula: (PixelValue / 127.5) - 1.0
    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
    
    # Add batch dimension (1, 224, 224, 3)
    input_data = np.expand_dims(img_normalized, axis=0)

    # 4. INFERENCE (Run the Model):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get the raw predictions (Softmax output)
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Identify the class with the highest probability
    idx = np.argmax(predictions)
    confidence = predictions[idx]

    # 5. VISUALIZATION:
    # Set a threshold of 70% to filter out "uncertain" guesses
    label = f"{class_names[idx]} ({int(confidence*100)}%)"
    
    # Green text for high confidence, Red for low confidence
    color = (0, 255, 0) if confidence > 0.100 else (0, 0, 255)
    
    # Draw the label on the original BGR frame for display
    cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Show the final window
    cv2.imshow('xArm Gesture Recognition - Final Stable Build', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Resource Cleanup
cap.release()
cv2.destroyAllWindows()