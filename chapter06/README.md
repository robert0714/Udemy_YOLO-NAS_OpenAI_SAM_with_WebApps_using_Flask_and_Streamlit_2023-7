# Chapter6. YOLO-NAS StreamLit Web App Development
Hello everyone.

Today we will learn how to build a computer vision interface from scratch using Streamlit in Python.
 
We will start up by writing code for object detection with Yolo nurse and then we will write a code for streamlined interface with Python and then combine it with Yolanda's for object detection in real
time.

There we will create three pages in our Streamlit app.

The first page, which will be the About Me page, will tell us little bit about the web app and the author.
 
The second page will allow us to do object detection on images.

The third page will allow us to do object detection on a video in real time.


So before we go ahead, let me tell you about Streamlit.
 
Streamlit turns writer scripts into shareable web apps in minutes, all in Python, all for free, no front end experience required.
 
So Streamlit is a way of getting professional looking web app or dashboards in which you can interact with your computer vision applications.
 
So let's get started.

### Conda Environment
* Install yolo-nas
  ```bash
    conda create -n yolo-nas python=3.10
    conda activate yolo-nas
    pip install super_gradients==3.5.0
    pip install easydict
    pip install scikit-image
    pip install filterpy
    pip install streamlit
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
## YOLO_NAS_StreamLit_Course
### Object Detection on Videos using YOLO-NAS
```bash
python object_detection.py
```
### Object Detection on Images using YOLO-NAS.
* Execute it
  ```bash
  python object_detection_image.py
  ```
* Type '0' in keyboard, it will stop.
  ```python
  cv2.imshow("Frame", resize_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```
### Streamlit with YOLO-NAS Integration
* Execute it
  ```python
  streamlit run streamlit_application.py
  ```
* Open the url: https://www.youtube.com/watch?v=FdZvMoP0dRU in your browser.

### Deploy Your Streamlit Web Application
* See https://github.com/AINeuronltd/Object_Detection_StreamLit_Deployment
* See http://share.streamlit.io

## Streamlit App to Count the Vehicles Entering and Leaving
* See object_detection_image_video_streamlit.py
* Execute it
  ```python
  streamlit run streamlit_application.py
  ```
* upload VehiclesEnteringandLeaving.mp4