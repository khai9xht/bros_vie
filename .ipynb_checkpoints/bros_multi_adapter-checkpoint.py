from bros_infer_multi import BROSMultitaskInfer
from typing import Any, Dict, List
import numpy as np
from collections import defaultdict
from utils import draw_output
import cv2

from tta_process import tta_preprocess, tta_postprocess
from sort_boxes import XYCutingBox, YminXminSort

class BROSMultitaskAdapter(BROSMultitaskInfer):
    def __init__(self, config, sort_option="xycuting") -> None:
        super(BROSMultitaskAdapter, self).__init__(config)
        if sort_option == "xycuting":
            self.soft_boxes = XYCutingBox(rx=5, ry=5, r_box=0.8, debug=False)
        elif sort_option =="yminxmin":
            self.soft_boxes = YminXminSort(debug=False)
        else:
            raise f"sort_option is only one of these values: ['xycuting', 'yminxmin'] but receive: {sort_option}"
        
        
    def transform_input(self, origin_input , image, word_lv=False) -> Dict[str, Any]:
        """
        INPUT:
            origin_input: List[
                Tuple(List[float], str, float)    # box, text, confidence
            ]
            image: np.ndarray   (H x W x 3)
        OUTPUT:
            raw_data: {
               imageSize: {height: int, width: int},
               words: List[{
                   text: str,
                   boundingBox: List[int] (shape[4, 2]),
                   idx: int
               }]
            }   
        """
        h, w, _ = image.shape
        raw_data = {}
        raw_data["imageSize"] = {"height": h, "width": w}
        raw_data["words"] = []
        words_list = []
        bboxes = []
        for i, text_box in enumerate(origin_input):
            box, text, _, _ = text_box
            box = [int(x) for x in box]
            bbox = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
            if word_lv:
                text_split = text.split(" ")
                for word in text_split:
                    word_dict = {"text": word, "boundingBox": bbox, "idx": i}
                    words_list.append(word_dict)
            else:
                words_list.append({"text": text, "boundingBox": bbox, "idx": i})
            bboxes.append(box)
            
        softed_map_idx = self.soft_boxes(bboxes)
        for new_word_idx, cur_word_idx in enumerate(softed_map_idx):
            word_dict = words_list[cur_word_idx]
            word_dict["idx"] = cur_word_idx
            raw_data["words"].append(word_dict)
        return raw_data

    
    def get_metadata(self, origin_input):
        metadata = {}
        for idx, ocr in enumerate(origin_input):
            metadata[idx] = {
                "bbox" : ocr[0],
                "text" : ocr[1],
                "poly" : ocr[3],
                "id" : idx
            }
        return metadata
    
    
    def transform_output(self, hor_links_batch, ver_links_batch, 
        raw_inputs, origin_inputs, labels_batch
    ):
        """
        INPUT:
            hor_links_batch: List    #shape = (BN, :, 2), horizontal links
            ver_links_batch: List    #shape = (BN, :, 2), vertical links
            raw_inputs: List[Dict[str, any]]
            origin_inputs: List[Dict[str, any]]
            labels_batch: List[List[int]].  # shape = (BN, n)
        OUPUT:
            all_links_post: Dict[end_link: List[start_link]] 
            hor_links_post: Dict[start_link: List[end_link]]
            update_origin_inputs: same as origin_inputs (merge key has texts in different lines) 
            label_map_texts: List[int] (value is index of text in origin_inputs)
        """
        
        all_links_post = []
        hor_links_post = []
        label_map_texts = []
        update_origin_inputs = []
        
        for hor_links, ver_links, labels, raw_input, origin_input in \
                zip(hor_links_batch, ver_links_batch, labels_batch, raw_inputs, origin_inputs):
            all_links_map = defaultdict(list)
            hor_links_map = defaultdict(list)
            
            # map horizontal links from word level to line level
            for start_idx, end_idx in hor_links:
                map_start_text = raw_input["words"][start_idx]["idx"]
                map_end_text = raw_input["words"][end_idx]["idx"]
                hor_links_map[map_start_text].append(map_end_text)
            
            # map ser label from word level to line level
            label_map_text = [0] * len(origin_input)
            for i in range(len(labels)):
                if label_map_text[raw_input["words"][i]["idx"]] == 0:
                    label_map_text[raw_input["words"][i]["idx"]] = labels[i]
            
            # merge key has texts in different lines then update horizontal map & origin_inputs
            update_hor_links_map = defaultdict(list)
            for start_idx, ends_idx in hor_links_map.items():
                if label_map_text[start_idx] == 2:
                    key_ends_idx = [x for x in ends_idx if label_map_text[x] == 2]
                    
                    # soft boxes by top (y0) to concat line text correctly
                    if len(key_ends_idx) != 0:
                        start_box, start_text, conf, _ = origin_input[start_idx]
                        
                        key_ends_idx = sorted(
                            key_ends_idx, 
                            key=lambda x: origin_input[x][0][1] # 0 is box,  1 is y1 
                        )
                        texts = [origin_input[x][1] for x in key_ends_idx]
                        full_key_text = start_text  + " " + ' '.join(texts)
                        
                        end_boxes = [origin_input[x][0] for x in key_ends_idx]
                        end_boxes.append(start_box)
                        x_min = min([x[0] for x in end_boxes])
                        y_min = min([x[1] for x in end_boxes])
                        x_max = max([x[2] for x in end_boxes])
                        y_max = max([x[3] for x in end_boxes])
                        merge_box = (x_min, y_min, x_max, y_max)
                        
                        merge_poly = [
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max]
                        ]

                        # update merged text to origin_inputs
                        origin_input[start_idx] = (merge_box, full_key_text, conf, merge_poly)
                            
                        # remove texts of merged words
                        for idx in key_ends_idx:
                            box, text, conf, poly = origin_input[idx]
                            origin_input[idx] = (box, "", conf, poly)
                            
                        #remove link used to concat line text
                        [ends_idx.remove(x) for x in key_ends_idx]
                        
                # update horizontal links
                update_hor_links_map[start_idx] = ends_idx
                for idx in ends_idx:
                    all_links_map[idx].append(start_idx)
            
            # map all links from word level to line level
            for start_idx, end_idx in ver_links:
                map_start_text = raw_input["words"][start_idx]["idx"]
                map_end_text = raw_input["words"][end_idx]["idx"]
                all_links_map[map_end_text].append(map_start_text)
            
            all_links_post.append(all_links_map)
            hor_links_post.append(update_hor_links_map)
            label_map_texts.append(label_map_text)
            update_origin_inputs.append(origin_input)
                
        return all_links_post, hor_links_post, update_origin_inputs, label_map_texts
    
        
    def __call__(self, origin_input, image, tta=False, draw=False) -> List[List[List[int]]]:
        
        # preprocess & tranforms inputs, get raw outputs of model
        if tta:
            tta_inputs = tta_preprocess(origin_input)
            raw_inputs = [self.transform_input(x, image) for x in tta_inputs]
            hor_links, ver_links, labels = self.extract_link(raw_inputs)
            origin_inputs = tta_inputs
        else:
            raw_inputs = [self.transform_input(origin_input, image)]
            hor_links, ver_links, labels = self.extract_link(raw_inputs)
            origin_inputs = [origin_input]
        
        # transform outputs
        all_links_post, hor_links_post, update_origin_inputs, label_map_texts = self.transform_output(
            hor_links, ver_links, raw_inputs, origin_inputs, labels
        )
        if tta:
            all_links_map, hor_links_map, text_labels = tta_postprocess(
                all_links_post, hor_links_post, update_origin_inputs, label_map_texts
            )
            origin_input = update_origin_inputs[0]
        else:
            all_links_map, hor_links_map, origin_input, text_labels = \
                all_links_post[0], hor_links_post[0], update_origin_inputs[0], label_map_texts[0]
        
        # get metadata
        metadata = self.get_metadata(origin_input)
        
        if draw:   # visualize ser & link results
            draw_links = defaultdict(list)
            for key, values in all_links_map.items():
                for value in values:
                    draw_links[value].append(key)
            cp_image = draw_output(
                origin_input, image, draw_links, text_labels, color_idx=4)
            cp_image = draw_output(
                origin_input, cp_image, hor_links_map, text_labels, color_idx=5)
            random_name = str(np.random.rand())
        return all_links_map, hor_links_map, metadata, text_labels


