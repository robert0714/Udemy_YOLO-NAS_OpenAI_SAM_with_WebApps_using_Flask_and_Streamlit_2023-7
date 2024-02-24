import cv2
import torch
from super_gradients.training import models
import math
from sort import *


cap = cv2.VideoCapture("../Video/ship1.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

#model = models.get('yolo_nas_m', pretrained_weights="coco").to(device)

model = models.get('yolo_nas_s', num_classes= 1, checkpoint_path='weights/ckpt_best.pth')

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #Boat
        color = (85,45,255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None,categories=None, names=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[0]
        y2 += offset[0]
        cat = int(categories[i]) if categories is not None else 0
        color = compute_color_for_labels(cat)
        id = int(identities[i]) if categories is not None else 0
        label = str(id) + ":" + classNames[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1-20), (x1+w, y1), color, -1)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,255], 1)
    return img
count = 0
classNames = ['Boat']
out = cv2.VideoWriter('Output1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

tracker = Sort(max_age = 20, min_hits=3, iou_threshold=0.3)

while True:
    ret, frame = cap.read()
    count += 1
    if ret:
        detections = np.empty((0,6))
        result = list(model.predict(frame, conf=0.5))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            class_name = classNames[classname]
            conf = math.ceil((confidence*100))/100
            #label = f'{class_name}{conf}'
            #print("Frame N", count, "", x1, y1,x2, y2)
            #t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
            #c2 = x1 + t_size[0], y1 - t_size[1] -3
            #cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            #cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            currentArray = np.array([x1, y1, x2, y2, conf, cls])
            detections = np.vstack((detections, currentArray))
        tracker_dets = tracker.update(detections)
        if len(tracker_dets) >0:
            bbox_xyxy = tracker_dets[:,:4]
            identities = tracker_dets[:, 8]
            categories = tracker_dets[:, 4]
            draw_boxes(frame, bbox_xyxy, identities, categories)
        #resize_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()