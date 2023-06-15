import cv2
import numpy as np
import tkinter as tk

window = tk.Tk()
window.title("Real Time")


def handle_button_press(event):
    window.destroy()


button = tk.Button(text="My simple app.")
button.bind("", handle_button_press)
button.pack()

# Load pre-trained object detection model
net = cv2.dnn.readNetFromDarknet('yolo/yolo.cfg', 'yolo/yolo.weights')

# Set classes for object labels
classes = []
with open('yolo/classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set colors for object bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load video stream (replace with webcam or video file source)
cap = cv2.VideoCapture('path/to/video')

# Create a writer object to save the output video
output_file = 'output.mp4'
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))

# Initialize counters and timers
frame_count = 0
total_objects = 0
start_time = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Process each detection
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Scale bounding box coordinates
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                center_x, center_y, width, height = box.astype('int')

                # Calculate top-left corner coordinates
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                # Update lists with detection results
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Process each detected object
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, width, height = box

        # Draw bounding box and label on frame
        color = colors[class_ids[i]]
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display processed frame
    cv2.imshow('Object Detection', frame)

    # Write frame to the output video file
    out.write(frame)

    # Update counters and timers
    frame_count
