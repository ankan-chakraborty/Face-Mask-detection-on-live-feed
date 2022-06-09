# Face-Mask-detection-on-live-feed
**This Project contains:**
* **Model Training:** How to fine tune Face-Mask-detection using MobileNet V2. 
* **Illumination Check:** Applying pure image processing technique to get rid of over-illuminated or under-exposed face image and only infer only on those images which are in good lighting condition. [Check Below](#illumination-check).
* **Real Time Inference:** And finally apply it on live feed for real time inference. 

### Sample Output Video
To check a sample real time mask detection output video, please <a href="https://github.com/ankan-chakraborty/Face-Mask-detection-on-live-feed/blob/main/Face-mask-detection.mp4">Click here</a>.



### Dataset 
Dataset is taken from: <a href='https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset'>Here</a>.

### Illumination Check
The following code snippet checks the illumination condition on each face and drops those faces before sending it to Mask Detection Classifier. Hence it reduces false classification and increases classification precision.

```
def illumination(image_array, bright_threshold = 0.5, dark_threshold = 0.2):
    bright_thres = bright_threshold
    dark_thres = dark_threshold
    frame = image_array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_part = cv2.inRange(gray, 0, 30)
    bright_part = cv2.inRange(gray, 220, 255)
    # use histogram
    # dark_pixel = np.sum(hist[:30])
    # bright_pixel = np.sum(hist[220:256])
    total_pixel = np.size(gray)
    dark_pixel = np.sum(dark_part > 0)
    bright_pixel = np.sum(bright_part > 0)
    
    if dark_pixel/total_pixel > bright_thres:
        print("Face is underexposed!")
        permission = False
        score = dark_pixel/total_pixel
        comment= "Illumination: Underexposed; Low Contrast or Brightness"
        return permission, score, comment
    
    if bright_pixel/total_pixel > dark_thres:
        print("Face is overexposed!")
        permission = False
        score = bright_pixel/total_pixel
        comment= "Illumination: Overexposed; High Contrast or Brightness"
        return permission, score, comment
    
    else:
        print('Normal Illumination!')
        permission = True
        score = None
        comment= "Illumination: Normal"
        return permission, score, comment
```

### Snapshot of Live Feed Output

**1. With Mask:**

   <img src ='https://github.com/ankan-chakraborty/Face-Mask-detection-on-live-feed/blob/main/Snapshots/With%20Mask.JPG' width = '800px'>
   
   
**2. Without Mask:**

   <img src ='https://github.com/ankan-chakraborty/Face-Mask-detection-on-live-feed/blob/main/Snapshots/Without%20Mask.JPG' width = '800px'>
   
   
