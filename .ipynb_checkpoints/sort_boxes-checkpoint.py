from typing import List, Dict
import cv2
import numpy as np


class XYCutingBox(object):
    def __init__(self, rx, ry, r_box=0.5, debug=False):
        self.rx = rx
        self.ry = ry
        self.r_box = r_box
        self.debug = debug
        
    
    def __call__(self, boxes: List[List[int]], image: np.ndarray=None) -> List[int]:
        for i in range(len(boxes)):
            boxes[i] = boxes[i] + [i]
        input_boxes = [[
            box[0], box[1], 
            box[0] + (box[2] - box[0]) * self.r_box, box[1] + (box[3] - box[1]) * self.r_box,
            box[-1]
        ] for box in boxes]
        input_boxes = [list(map(int, x)) for x in input_boxes]
        sorted_input_boxes = self.sort(input_boxes, image)
        flat_input_boxes = []
        for input_boxes in sorted_input_boxes:
            if len(boxes) == 1:
                flat_input_boxes.append(input_boxes[0])
            else:
                input_boxes = sorted(input_boxes, key=lambda x: x[0])
                flat_input_boxes += input_boxes
  
        map_boxes_idx = [box[-1] for box in flat_input_boxes]
        assert len(boxes) == len(map_boxes_idx), \
            f"Bug in Code sort boxes, len(boxes) = {len(boxes)}, len(map_boxes_idx) = {len(map_boxes_idx)}."
        return map_boxes_idx
        
    
    def flatten_list(self, x: List[list]) -> list:
        return [a for b in x for a in b]

    def sort(self, boxes: List[List[int]], image: np.ndarray=None) -> List[List[int]]:
        thresholds = self.find_XYthreshold(boxes)
        thresh_x = thresholds["x"]
        thresh_y = thresholds["y"]
        
        if self.debug and image is not None:
            x_min = min([box[0] for box in boxes])
            x_max = max([box[2] for box in boxes])
            y_min = min([box[1] for box in boxes])
            y_max = max([box[3] for box in boxes])
            for thr_x in thresh_x:
                cv2.rectangle(image, (thr_x, y_min), (thr_x, y_max), (255,0,0), 2)
            for thr_y in thresh_y:
                cv2.rectangle(image, (x_min, thr_y), (x_max, thr_y), (0,0,255), 2)

        if len(thresh_x) == 2 and len(thresh_y) == 2:
            return [boxes]
        else:
            sub_boxes = []
            for i in range((len(thresh_x) - 1) * (len(thresh_y) - 1)):
                sub_boxes.append([])

            indexes = [[i, j] for j in range(1, len(thresh_y), 1) for i in range(1, len(thresh_x), 1)]
            for k, box in enumerate(boxes):
                for i, j in indexes:
                    x_valid = (box[0] >= thresh_x[i-1] and box[2] <= thresh_x[i])
                    y_valid = (box[1] >= thresh_y[j-1] and box[3] <= thresh_y[j])
                    if x_valid and y_valid:
                        index = (j - 1) * (len(thresh_x) - 1) + (i - 1)
                        sub_boxes[index].append(box)
                        break
            while [] in sub_boxes:
                sub_boxes.remove([])
                
            return [self.flatten_list(self.sort(sub_box, image)) for sub_box in sub_boxes]


    def find_XYthreshold(self, boxes: np.ndarray) -> Dict[str, List[int]]:
        x_min = min([box[0] for box in boxes])
        x_max = max([box[2] for box in boxes])
        y_min = min([box[1] for box in boxes])
        y_max = max([box[3] for box in boxes])

        tuples_x = [[box[0], box[2]] for box in boxes]
        tuples_y = [[box[1], box[3]] for box in boxes]

        thresh_x = self.find_threshold(tuples_x, x_min, x_max, self.rx)
        thresh_y = self.find_threshold(tuples_y, y_min, y_max, self.ry)

        return {"x": thresh_x, "y": thresh_y}


    def find_threshold(self, tuples: List[List[int]], _min: float, _max: float, r: float) -> List[int]:
        thresh = []
        related = False
        for thr in range(_min + 1, _max - 1, r):
            valid = self.check_threshhold(tuples, thr)
            if valid:
                if not related:
                    thresh.append(thr)
                    related = True
            else:
                related = False
        return [_min - 1] + thresh + [_max + 1]


    def check_threshhold(self, tuples: List[List[int]], thresh: float) -> bool:
        valid = True
        for x1, x2 in tuples:
            if thresh > x1 and thresh < x2:
                valid = False
                break
        return valid


class YminXminSort(object):
    def __init__(self, debug=False):
        self.debug = debug
        
        
    def __call__(self, boxes: List[List[int]], image: np.ndarray=None) -> List[int]:
        for i in range(len(boxes)):
            boxes[i] = boxes[i] + [i]
        boxes = [list(map(int, x)) for x in boxes]
        sorted_boxes = sorted(boxes, key=lambda x: (x[1], x[0]))
  
        map_boxes_idx = [box[-1] for box in sorted_boxes]
        return map_boxes_idx
            