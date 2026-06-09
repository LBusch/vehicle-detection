# Vehicle Detection with Faster R-CNN

In this project a Faster R-CNN network is fine-tuned for vehicle detection. PyTorch's Faster R-CNN with a MobileNetV3 backbone is used for this task to strike a balance between model size and performance.
Pre-trained weights on the COCO dataset are used for the model. Then, the model is fine-tuned for 5 epochs on the UA-DETRAC dataset for vehicle detection. The UA-DETRAC dataset consists of over 40,000 frames from 100 challenging videos of traffic scenes.
The image sequences are captured in varying weather and lighting conditions. For the purposes of this project every 10th frame of each training and testing image sequence is used to create the dataset splits. The model achieves a Mean Average Precision of 0.77 at an IoU threshold of 0.5 on the test data spit.
The project includes a script to create , a training script that outputs, and a script for inference of a trained model on an image sequence.

# Contents
`create_dataset_jsons.py`:  creates train and test split jsons of the DETRAC dataset for fine-tuning. Expects DETRAC images and Annotations in /data. They can be downloaded from [here](https://sites.google.com/view/daweidu/projects/ua-detrac?authuser=0).

`train_detector.py`: script for fine-tuning the model on the created data split jsons. Outputs plots of useful object detection metrics such the model's precision-recall curve and the f1-confidence curve.

`inference.py`: script for performing inference of a trained model on a sequence of images. Outputs the detection results on the image frames in a video.

# Video

https://github.com/user-attachments/assets/109dba2f-3c04-4010-8d15-220dd8800bea

#  Plots
<img width="2539" height="1638" alt="f1_confidence" src="https://github.com/user-attachments/assets/ead02bf2-0f8c-4d5d-a51d-1973f9abe91f" />
<img width="2539" height="1638" alt="precision_recall" src="https://github.com/user-attachments/assets/1af0bdbb-99c3-4cda-9953-cde57af4bd9b" />



