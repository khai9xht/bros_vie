"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import torch
from omegaconf import OmegaConf
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from typing import List


LABEL_COLORS = [(255, 255, 0), (0, 255, 255), (255, 0, 0), (0, 255, 0), (139, 128, 0), (0, 0, 255)]
fontpath = "./test/arial-unicode-ms.ttf"



def load_model_weight(net, pretrained_model_file):
    pretrained_model_state_dict = torch.load(pretrained_model_file, map_location="cpu")
    new_state_dict = {}
    for k, v in pretrained_model_state_dict.items():
        new_k = k
        if new_k.startswith("net."):
            new_k = new_k[len("net.") :]
        new_state_dict[new_k] = v
    net.load_state_dict(new_state_dict)


def get_config(default_conf_file):
    cfg = OmegaConf.load(default_conf_file)

    cfg_cli = _get_config_from_cli()
    if "config" in cfg_cli:
        cfg_cli_config = OmegaConf.load(cfg_cli.config)
        cfg = OmegaConf.merge(cfg, cfg_cli_config)
        del cfg_cli["config"]

    cfg = OmegaConf.merge(cfg, cfg_cli)

    return cfg


def _get_config_from_cli():
    cfg_cli = OmegaConf.from_cli()
    cli_keys = list(cfg_cli.keys())
    for cli_key in cli_keys:
        if "--" in cli_key:
            cfg_cli[cli_key.replace("--", "")] = cfg_cli[cli_key]
            del cfg_cli[cli_key]

    return cfg_cli

        
def parse_initial_words(itc_label, box_first_token_mask, class_names):
    itc_label_np = itc_label
    box_first_token_mask_np = box_first_token_mask

    outputs = [[] for _ in range(len(class_names))]
    for token_idx, label in enumerate(itc_label_np):
        if box_first_token_mask_np[token_idx] and label != 0:
            outputs[label].append(token_idx)

    return outputs


def parse_subsequent_words(stc_label, attention_mask, init_words, dummy_idx):
    max_connections = 50

    valid_stc_label = stc_label * attention_mask.astype(np.bool_)
    valid_stc_label = valid_stc_label
    stc_label_np = stc_label

    valid_token_indices = np.where(
        (valid_stc_label != dummy_idx) * (valid_stc_label != 0)
    )

    next_token_idx_dict = {}
    for token_idx in valid_token_indices[0]:
        next_token_idx_dict[stc_label_np[token_idx]] = token_idx

    outputs = []
    for init_token_indices in init_words:
        sub_outputs = []
        for init_token_idx in init_token_indices:
            cur_token_indices = [init_token_idx]
            for _ in range(max_connections):
                if cur_token_indices[-1] in next_token_idx_dict:
                    if (
                        next_token_idx_dict[cur_token_indices[-1]]
                        not in init_token_indices
                    ):
                        cur_token_indices.append(
                            next_token_idx_dict[cur_token_indices[-1]]
                        )
                    else:
                        break
                else:
                    break
            sub_outputs.append(tuple(cur_token_indices))

        outputs.append(sub_outputs)

    return outputs


def draw_polygon(image, bbox, color=(255,0,0), thickness = 1):
    bbox = bbox.astype(np.int32)
    tl = tuple(bbox[:2])
    tr = tuple(bbox[2:4])
    br = tuple(bbox[4:6])
    bl = tuple(bbox[6:])
    image = cv2.line(image, tl, tr, color = color, thickness = thickness)
    image = cv2.line(image, tr, br, color = color, thickness = thickness)
    image = cv2.line(image, br, bl, color = color, thickness = thickness)
    image = cv2.line(image, bl, tl, color = color, thickness = thickness)
    return image.copy()


def draw_links(image, bbox, link_map_tuples, ee_label, color=(0, 0, 255)) -> np.ndarray:
    height, width, _ =  image.shape
    bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] * width
    bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] * height
    bbox = bbox.astype(np.int32)
    
    for start_link, end_link in link_map_tuples:
        start_box = bbox[start_link]
        end_box = bbox[end_link]
        start_label = ee_label[start_link] if ee_label[start_link] >=0 else 0
        end_label = ee_label[end_link] if ee_label[end_link] >=0 else 0
        image = draw_polygon(image, start_box, color=LABEL_COLORS[start_label])
        image = draw_polygon(image, end_box, color=LABEL_COLORS[end_label])
        start_point = (int((start_box[0] + start_box[4]) / 2), int((start_box[1] + start_box[5]) / 2))
        end_point = (int((end_box[0] + end_box[4]) / 2), int((end_box[1] + end_box[5]) / 2))
        image = cv2.arrowedLine(image, start_point, end_point, color, 1)
    return image.copy()

def draw_ser(image, origin_input, labels, draw_text=False):
    img_cp = image.copy()
    for i, (box, text, _, _) in enumerate(origin_input):
        box = [int(x) for x in box]
        img_cp = cv2.rectangle(
            img_cp, 
            tuple(box[:2]), tuple(box[2:]),
            LABEL_COLORS[labels[i]], 1
        )
        if draw_text:
            font = ImageFont.truetype(fontpath, 20)
            img_pil = Image.fromarray(img_cp)
            draw = ImageDraw.Draw(img_pil)
            b, g, r = LABEL_COLORS[labels[i]]
            a = 0
            draw.text(tuple(box[:2]), text, font=font, fill=(b, g, r, a))
            img_cp = np.array(img_pil)
    return img_cp
    

def draw_output(origin_input, image, hor_links_map, labels, color_idx=4):
    image = draw_ser(image, origin_input, labels)
    cp_image = image.copy()
    for start_idx, ends_idx in hor_links_map.items():
        start_box = origin_input[start_idx][0]
        start_box = [int(x) for x in start_box]
        start_center = (
            int((start_box[0] + start_box[2]) / 2), 
            int((start_box[1] + start_box[3]) / 2)
        )
        for end_idx in ends_idx:
            end_box = origin_input[end_idx][0]
            end_box = [int(x) for x in end_box]
            end_center = (
                int((end_box[0] + end_box[2]) / 2), 
                int((end_box[1] + end_box[3]) / 2)
            )
            cp_image = cv2.arrowedLine(
                cp_image, start_center, end_center, LABEL_COLORS[color_idx], 1)
    return cp_image

def get_token_lv_ser_labels(itc_labels: np.ndarray,
                            stc_labels: np.ndarray,
                            box_first_tokens: np.ndarray,
                            attention_masks: np.ndarray,
                            max_seq_length: int,
                            class_names) -> List[int]:
    """
    From itc and stc predictions, construct ser label of each token
    """
    
    if itc_labels is None or stc_labels is None:
        return None
    class_words = []
    for itc_label, stc_label, box_first_token_mask, attention_mask in \
            zip(itc_labels, stc_labels, box_first_tokens, attention_masks):
        init_word = parse_initial_words(
            itc_label, box_first_token_mask, class_names
        )
        class_word = parse_subsequent_words(
            stc_label, attention_mask, init_word, max_seq_length
        )

        flat_class_word = np.ones(max_seq_length, np.int8) * 0
        for i, word_class in enumerate(class_word):
            word_class = [x for y in word_class for x in y]
            for word_idx in word_class:
                if flat_class_word[word_idx] == 0:
                    flat_class_word[word_idx] = i
        class_words.append(flat_class_word)
    return class_words

def get_entity_lv_ser_labels(
    token_lv_ser_labels: List[List[int]], box_first_tokens: np.ndarray, 
    attention_masks: np.ndarray, class_names: List[str], mode: str="first"):
    """
    voting SER labels among all tokens in a line
    Return: SER labels of entity boxes
    """
    entity_lv_ser_labels = []
    for token_lv_ser_label, box_first_token_mask, attention_mask in \
        zip(token_lv_ser_labels, box_first_tokens, attention_masks):
        entity_lv_ser_label = []
        
        # create entity id list (e.g [0, 1, 1, 1, 2, 2, 2, 2, ...])
        entity_lv_map_list = np.zeros(box_first_token_mask.shape, dtype=np.int32)
        entity_id = 0
        for i in range(entity_lv_map_list.shape[0]):
            if box_first_token_mask[i]:
                entity_id += 1
            entity_lv_map_list[i] = entity_id
            
        # use attention_mask to mask pad token entity_id to 0
        entity_lv_map_list = entity_lv_map_list * attention_mask
        num_entities = len(list(set(entity_lv_map_list))) # count <pad> entity id 0
        
        # use from entity id 1
        for entity_id in range(1, num_entities):
            entity_token_idxes = np.argwhere(entity_lv_map_list == entity_id).flatten()
            ser_label_by_entity = token_lv_ser_label[entity_token_idxes]
            
            if mode == "voting":
                # voting label among all tokens in a entity
                vote_classes_list = [0 for _ in range(len(class_names))]
                for idx, label in enumerate(ser_label_by_entity):
                    if idx == 0: # first position has higher weight
                        vote_classes_list[label] += 1.1
                    else:
                        vote_classes_list[label] += 1

                # change ser label of a entity's token based on voted class
                final_class = np.argmax(vote_classes_list)
            elif mode == "first":
                final_class = ser_label_by_entity[0]
                
            entity_lv_ser_label.append(final_class)
        entity_lv_ser_labels.append(entity_lv_ser_label)
         
    return entity_lv_ser_labels
