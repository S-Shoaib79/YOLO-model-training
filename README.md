# YOLO-model-training
A YOLO model trained for image detection
# Brain Tumor Detection Using YOLOv8 and SAM Segmentation
 
 ## Project Overview
 
 This project focuses on detecting and segmenting brain tumors in MRI scans using two state-of-the-art deep learning models:
 
 - YOLOv8 – For fast and accurate object detection (tumor localization)  
 - SAM (Segment Anything Model) – For detailed pixel-level tumor segmentation
 
 ## Objective
 
 To automate brain tumor identification in medical images by combining the strengths of object detection and semantic segmentation.
 
 ## Approach
 
 - Trained YOLOv8 on a brain tumor dataset  
 - Used YOLOv8 to detect tumors in new MRI images  
 - Applied SAM for precise segmentation of the detected tumor regions  
 - Saved visual outputs (bounding boxes and masks) for evaluation
 
 ## Visual Examples
 
 - Original input MRI image  
 - Tumor bounding boxes (YOLOv8)  
 - Segmentation masks (SAM)  
 
 ## Code Usage and Explanations
 
 ```python
 # Import YOLO model
 from ultralytics import YOLO
 
 # Load YOLOv8 model (custom-trained)
 model = YOLO("/content/runs/detect/train4/weights/best.pt")
 
 # Detect tumor in a single image
 results = model("/content/images/glioma.jpg", save=True)
 results[0].show()
 ```
 
 This snippet loads the YOLOv8 model and performs tumor detection on a single image.
 
 ```python
 # Run detection on multiple images
 results = model("/content/images", save=True)
 
 # Print bounding box results
 for result in results:
     boxes = result.boxes
     print(boxes)
 ```
 
 This loop processes multiple MRI images and prints bounding boxes for detected tumors.
 
 ```python
 # Apply SAM segmentation
 from ultralytics import SAM
 sam = SAM("sam2_b.pt")
 
 # Segment using YOLOv8-detected boxes
 for result in results:
     class_ids = result.boxes.cls.int().tolist()
     if len(class_ids):
         boxes = result.boxes.xyxy
         sam_results = sam(result.orig_img, bboxes=boxes, verbose=False, save=True, device=0)
 ```
 
 This part uses SAM to segment tumors based on YOLO's bounding boxes.
 
 ## Notes and References
 
 - YOLOv8 Tutorial - Ultralytics  
 - SAM Model Overview - Meta AI  
 
 ## Video Demonstration
 
 Watch the full process here:  
 https://drive.google.com/drive/folders/1kXvE57XHe4PPbSpNWE_6tD5XucPwcxF-?usp=drive_link
