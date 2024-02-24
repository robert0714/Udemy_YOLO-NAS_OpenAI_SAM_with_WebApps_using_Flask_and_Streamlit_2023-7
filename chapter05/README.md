# Chapter5. YOLO-NAS with SORT Object Tracking
### Conda Environment
* Install yolo-nas
  ```bash
    conda create -n yolo-nas python=3.10
    conda activate yolo-nas
    pip install super_gradients==3.5.0
    pip install easydict
    pip install scikit-image
    pip install filterpy
  ``` 
* Deep Sort with PyTorch:    
  https://github.com/ZQPei/deep_sort_pytorch
* Video sample source : 
  * links
    1. demo.mp4: https://drive.google.com/file/d/1crFwrpMF1OlaJ0ZCZjBNRo9llLEVR8VQ
    2. demo2.mp4: https://www.pexels.com/video/road-construction-4271760/
    3. demo3.mp4: https://drive.google.com/file/d/1cTIBNQ1R_7JAOURVv9cJ6P935ym_IkZ0
    4. demo4.mp4:  https://drive.google.com/file/d/1256pNK0nQnEDT6FRLQAraTRkOY7BSprq
    5. bikes.mp4: https://www.pexels.com/video/people-cycling-4277525/
    6. VehiclesEnteringandLeaving.mp4: https://www.pexels.com/video/road-systems-in-montreal-canada-for-traffic-management-of-motor-vehicles-3727445/
    7. video1.mp4: https://pixabay.com/videos/los-angeles-traffic-california-road-53125/
    8. ship1.mp4: https://www.pexels.com/video/top-view-of-a-boat-crossing-the-river-4884643/
  * we can use gdown (pip install gdown or in colab)  
       ```python
       !gdown "https://drive.google.com/uc?id=1crFwrpMF1OlaJ0ZCZjBNRo9llLEVR8VQ&confirm=t" 
       !gdown "https://drive.google.com/uc?id=1cTIBNQ1R_7JAOURVv9cJ6P935ym_IkZ0&confirm=t" 
       !gdown "https://drive.google.com/uc?id=1256pNK0nQnEDT6FRLQAraTRkOY7BSprq&confirm=t" 
       ```
## YOLO_NAS_Object_Tracking_Pycharm       
### Lecture_4_CaptureVideoFromCamera
* Type '1' in keyboard, it will stop.
  ```python
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break  
  ```
## YOLO_NAS_Object_Tracking_SORT_Custom  
In this  tutorial we will see how we can do object tracking using Sort algorithm on custom dataset.

### Dataset
Open 
  * https://universe.roboflow.com/k--stavrakakis/
  * https://universe.roboflow.com/k--stavrakakis/ships-google-earth

So in the first step I will train the model on the custom dataset and the dataset which I will be training or fine tuning.
The Uranus model is ship selection dataset, so I will train the Uranus model on the ship selection dataset so that after training the Uranus model on the ships, the train dataset, my model will be
 able to detect the ships from the Google satellite images.

So here you can see the the images of ships like you can see over here.These are all the images taken from the satellite images.
So these are the Google satellite images you can see over here.


So in this way, we have around 794 images.

And here we have the Google Earth images.
This dataset is available on Kaggle as well, and the author of this dataset has just converted this dataset.
Or you can say that it has just integrated this dataset and prepared this dataset so that it can be exported into the Google CoLab notebook and we can train the U.S.A. model on this dataset.

So the author of this dataset has prepared this dataset in the required format so we can train the model on this dataset.

* If I just go over here and if I just want to export this dataset from Roboflow into the Google CoLab 
notebook, so you just click on download dataset, select the **V5 PyTorch format** and click on Continue and Separate terms.
And then again, like the format and starting downloading and you can just copy this code from here.
And if you just paste this port into your notebook, you will be able to download this dataset from Roboflow into your Google CoLab notebook.
https://colab.research.google.com/github/robert0714/Udemy_YOLO-NAS_OpenAI_SAM_with_WebApps_using_Flask_and_Streamlit_2023-7/blob/main/chapter05/Train_YOLONAS_Custom_Dataset_Ships_detection_Complete_Final.ipynb

* if you are using Anaconda, Jupyter Notebook or Google CoLab Notebook, you can just copy this code and paste it in any of the cell and you will be able to download the dataset from Roboflow into the Google CoLab notebook.

* But if you are using any other like PyCharm, Spyder ID vs code or any other ID, you just need to go to the terminal from here and just add this code into your terminal.


