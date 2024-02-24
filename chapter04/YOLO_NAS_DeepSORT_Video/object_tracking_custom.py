import cv2
import torch
from super_gradients.training import models
import numpy as np
import math
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort



cap = cv2.VideoCapture("Video/test4.mp4")

frame_width = int(cap.get(3))

frame_height = int(cap.get(4))

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)
model = models.get('yolo_nas_s', num_classes=10, checkpoint_path = 'weights/ckpt_best.pth')

cout = 0
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
classNames = ['boat','camping car','car','motorcycle','other','pickup','plane','tractor','truck','van'
              ]

cfg_deep = get_config()

cfg_deep.merge_from_file("deep_sort_pytorch\configs\deep_sort.yaml")
deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                    max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1,  x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[0]
        y2 += offset[0]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        label = str(id) + ":" + classNames[cat]
        (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.rectangle(img, (x1, y1-20), (x1+w, y1), (255, 144, 30), -1)
        cv2.putText(img, label, (x1, y1 -5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
    return img


while True:
    xywh_bboxs = []
    confs = []
    oids = []
    outputs = []
    ret, frame = cap.read()
    if ret:
        result = list(model.predict(frame, conf=0.35))[0]

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

            #t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

            #c2 = x1+t_size[0], y1 - t_size[1] -3

            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            #cv2.rectangle(frame, (x1, y1), c2,  (255, 0, 255), -1, cv2.LINE_AA)

            #cv2.putText(frame, label, (x1, y1-2), 0,1, [255, 255, 255], thickness=1,lineType = cv2.LINE_AA)

            #resize_frame= cv2.resize(frame, (0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            bbox_width = abs(x1-x2)
            bbox_height = abs(y1-y2)
            xcycwh = [cx, cy, bbox_width, bbox_height]
            xywh_bboxs.append(xcycwh)
            confs.append(conf)
            oids.append(int(cls))
        xywhs = torch.tensor(xywh_bboxs)
        confss= torch.tensor(confs)
        outputs = deepsort.update(xywhs, confss, oids, frame)
        if len(outputs)>0:
            bbox_xyxy = outputs[:,:4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            draw_boxes(frame, bbox_xyxy, identities, object_id)

        out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()