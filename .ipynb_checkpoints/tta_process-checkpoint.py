import copy
from text_augment import RandomLossCharacters
import numpy as np
from collections import defaultdict


def tta_preprocess(origin_input, p_text=0.1, num_samples=4):
    """
        Augment inputs to test TTA
        augmentation(s): Random Loss Charactera
        ratio of line texts used augmentation(s) in input: 1.0
    """
    tta_inputs = [origin_input]
    random_loss_chacs = RandomLossCharacters(r_loss=0.05, p=1)
    box_pad = (np.random.rand() - 0.5) * 1e-2
    for i in range(num_samples-1):
        input_cp = copy.deepcopy(origin_input)
        for i, (box, text, conf, poly) in enumerate(input_cp):
            prob = np.random.rand()
            if prob > p_text:
                continue
            new_text = random_loss_chacs(text)
            box = [x + box_pad for x in box] 
            input_cp[i] = (box, new_text, conf, poly)
        tta_inputs.append(input_cp)
    return tta_inputs


def tta_postprocess(all_links_post, hor_links_post, update_origin_inputs, label_map_texts):
    """
        Merge results from model prediction with TTA
    """

    # count nunmber of present times of each links in all results (only select couple which happens in origin input)
    dict_hor_links = defaultdict(int)
    dict_all_links = defaultdict(int)
            
    for i, all_links in enumerate(all_links_post):
        for end_idx, start_idxs in all_links.items():
            for start_idx in start_idxs:
                if i == 0:
                    dict_all_links[(end_idx, start_idx)] += 1
                elif (end_idx, start_idx) in dict_all_links:
                    dict_all_links[(end_idx, start_idx)] += 1
    
    for i, hor_links in enumerate(hor_links_post):
        for start_idx, end_idxs in hor_links.items():
            for end_idx in end_idxs:
                if i == 0:
                    dict_hor_links[(end_idx, start_idx)] += 1
                elif (end_idx, start_idx) in dict_hor_links:
                    dict_hor_links[(end_idx, start_idx)] += 1

    # get links have a half or more present times
    tta_all_links = defaultdict(list)
    tta_hor_links = defaultdict(list) 
    
    for (end_idx, start_idx), value in dict_all_links.items():
        if value >= len(all_links_post) / 2:
            tta_all_links[end_idx].append(start_idx)
            
    for (end_idx, start_idx), value in dict_hor_links.items():
        if value >= len(hor_links_post) / 2:
            tta_hor_links[start_idx].append(end_idx)

    tta_labels = label_map_texts[0]

    return tta_all_links, tta_hor_links, tta_labels