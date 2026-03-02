from ultralytics import YOLO

model=YOLO('models//best.pt')  # Load a pretrained YOLOv8n model
results= model.predict(source='input_vedio\\08fd33_4.mp4',save=True)  # Perform inference on an input video



print(results[0])  # Print the results of the inference
for box in results[0].boxes:
    print(box)  # Print the bounding box coordinates
