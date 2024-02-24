# Chapter4. YOLO-NAS with DeepSORT Tracking
### Conda Environment
* Install yolo-nas
  ```bash
    conda create -n yolo-nas python=3.10
    conda activate yolo-nas
    pip install super_gradients==3.5.0
    pip install easydict
  ``` 
* Deep Sort with PyTorch:    
  https://github.com/ZQPei/deep_sort_pytorch
* Video sample source : 
  * links
    1. demo.mp4: https://drive.google.com/file/d/1crFwrpMF1OlaJ0ZCZjBNRo9llLEVR8VQ
    2. demo2.mp4: https://www.pexels.com/video/road-construction-4271760/
    3. demo3.mp4: https://drive.google.com/file/d/1cTIBNQ1R_7JAOURVv9cJ6P935ym_IkZ0
    4. demo4.mp4  https://drive.google.com/file/d/1256pNK0nQnEDT6FRLQAraTRkOY7BSprq
    5. bikes.mp4: https://www.pexels.com/video/people-cycling-4277525/
    6. VehiclesEnteringandLeaving.mp4: https://www.pexels.com/video/road-systems-in-montreal-canada-for-traffic-management-of-motor-vehicles-3727445/
  * we can use gdown (pip install gdown or in colab)  
       ```python
       !gdown "https://drive.google.com/uc?id=1crFwrpMF1OlaJ0ZCZjBNRo9llLEVR8VQ&confirm=t" 
       !gdown "https://drive.google.com/uc?id=1cTIBNQ1R_7JAOURVv9cJ6P935ym_IkZ0&confirm=t" 
       !gdown "https://drive.google.com/uc?id=1256pNK0nQnEDT6FRLQAraTRkOY7BSprq&confirm=t" 
       ```
## YOLO-NAS + DeepSORT Tracking
```
cd YOLO_NAS_DeepSORT_Video
git clone https://github.com/ZQPei/deep_sort_pytorch
python object_tracking.py
```
## YOLO-NAS + DeepSORT Tracking on Custom Dataset (Vehicles Detection)
You can see that we have assigned a unique ID to each of the detected object, like one for this person,
six for this bicycle, a 58 for this bicycle, 26 for this bicycle, 66 for this person, 21 for this
person.

Although you can see this person standing far away.

But using your object detection algorithm, we are able to detect the person.
So now here you can see that using Deepsort object tracking algorithm, we are able to detect the object
and using.

So using our object detection algorithm, we are able to detect the object and using Deepsort object
tracking algorithm, we have assigned a unique ID to each of the detected object, like you can see
here, one 670 with tourist handbag.

So these IDs are not assigned like in the numeric order, but these IDs are being assigned randomly
like 66, 81, 58, 26.
So these IDs are being assigned randomly.

They are not assigned in the sequence order.

So now you can see over here using object detection algorithm, using object Uranus object detection
algorithm, we detect object.


We create bounding boxes around each of the object, plus using Deepsort object tracking algorithm,
we assign a unique ID to each of the detected object using Deepsort Object tracking algorithm.

We track the detected object throughout all the frames until the object is in the scene, and when the 
object leaves the scene that ID is removed.

So a unique ID is being assigned to each of the detected object and we track that object using that 
ID throughout the frames.

So in using object detection using your nose, we detect the object, create bounding boxes around them
and using Deepsort object tracking algorithm, we assign a unique ID to each of the detected object

and that ID remains with that object until that object is in the scenes so that we can track that object,
you can see over 16, 14, ten 1106 and these IDs are not assigned randomly are not assigned in the
sequence order.

These IDs are assigned randomly.

So now I will just show you the CoLab code.

Using that CoLab code, we will be training the model on a custom dataset and then we will just add
this custom model weights over here and we will be doing object tracking using Yolo on a custom dataset.

### Dataset
Open 
  * https://universe.roboflow.com/drone-dataset-mvh8i/
  * https://universe.roboflow.com/drone-dataset-mvh8i/detection-bzujh

So here you can see the dataset.
I will be training a model on this dataset.

The dataset consists of 4680 images and we have ten different classes in our dataset.

So these are the drone images of boat camping, car, car, motorcycle pickup, plane, tractor truck 
van.

So these are the drone images of different vehicles.

And you can see over here, let me just go to the health check.

So the dataset consists of 4681 images with 20,554 annotations .

18,389 annotation belongs to the car class, so the car class is overrepresented, while 682 annotations belong
to the pickup and 421 annotations belong to the motorcycle class.

So the car class is overrepresented like it contains majority over 90% of the annotations, while 10% of the invitations belong to the remaining nine classes.

Okay, so I will be just training the model on this dataset.

So [here](https://colab.research.google.com/github/robert0714/Udemy_YOLO-NAS_OpenAI_SAM_with_WebApps_using_Flask_and_Streamlit_2023-7/blob/main/chapter04/Train_YOLONAS_Custom_Dataset_Drone_Images_Complete.ipynb) is our Google CoLab notebook.


So let us just go back to the **Google CoLab** or our **PyCharm** script and just add this model weights and

let's just this or let's integrate object tracking with Yolanda's own custom data set.


Uh, CoLab notebook file.
https://colab.research.google.com/github/robert0714/Udemy_YOLO-NAS_OpenAI_SAM_with_WebApps_using_Flask_and_Streamlit_2023-7/blob/main/chapter04/Train_YOLONAS_Custom_Dataset_Drone_Images_Complete.ipynb
