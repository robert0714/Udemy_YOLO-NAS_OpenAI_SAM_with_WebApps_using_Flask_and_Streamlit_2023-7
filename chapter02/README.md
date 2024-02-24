# Chapter2. YOLO-NAS Implementation  Windows
## Running YOLO-NAS on Windows
### Conda Environment
* Install yolo-nas
  ```bash
    conda create -n yolo-nas python=3.10
    conda activate yolo-nas
    pip install super-gradients 
  ``` 
* If you use ``super-gradients==3.1.0``, you need install  visualstudio2022buildtools  
  ```bash
  choco install -y visualstudio2022buildtools  
  ```
  And then got to control panel to add building tools. ex: cmake..etc
  * https://stackoverflow.com/questions/76764042/error-command-c-program-files-x86-microsoft-visual-studio-2022-buildtools-v
    ```bash
    pip install aiohttp==3.9.0b0
    ```  
    And then try again,

### Object Detection on Images
* Image sample source : 
  * links
    1. image.jpg:  https://www.pexels.com/zh-tw/photo/8853536/
    2. image1.jpg: https://www.pexels.com/zh-tw/photo/1402787/
    3. image2.jpg: https://drive.google.com/file/d/1cTIBNQ1R_7JAOURVv9cJ6P935ym_IkZ0
    4. image3.jpg  https://drive.google.com/file/d/1256pNK0nQnEDT6FRLQAraTRkOY7BSprq
### Object Detection on Videos
* Video sample source : 
  * links
    1. demo.mp4: https://drive.google.com/file/d/1crFwrpMF1OlaJ0ZCZjBNRo9llLEVR8VQ
    2. demo2.mp4: https://www.pexels.com/video/road-construction-4271760/
    3. demo3.mp4: https://drive.google.com/file/d/1cTIBNQ1R_7JAOURVv9cJ6P935ym_IkZ0
    4. demo4.mp4  https://drive.google.com/file/d/1256pNK0nQnEDT6FRLQAraTRkOY7BSprq
    5. bikes.mp4: https://www.pexels.com/video/people-cycling-4277525/
  * we can use gdown (pip install gdown or in colab)  
       ```python
       !gdown "https://drive.google.com/uc?id=1crFwrpMF1OlaJ0ZCZjBNRo9llLEVR8VQ&confirm=t" 
       !gdown "https://drive.google.com/uc?id=1cTIBNQ1R_7JAOURVv9cJ6P935ym_IkZ0&confirm=t" 
       !gdown "https://drive.google.com/uc?id=1256pNK0nQnEDT6FRLQAraTRkOY7BSprq&confirm=t" 
       ```

### Object Detection with YOLO-NAS on Live Webcam Feed
