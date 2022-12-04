import torch
from typing import List, Any, Dict        
import itertools
import numpy as np
from utils import get_token_lv_ser_labels, get_entity_lv_ser_labels
from model import BROSSPADEREEModel


class BROSMultitaskInfer:
    def __init__(self, config) -> None:
        self.model = BROSSPADEREEModel(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()
        self.max_seq_length = config.model.max_seq_length
        self.tokenizer = self.model.tokenizer
        vocab = self.tokenizer.get_vocab()
        self.pad_token_id = vocab["<pad>"]
        self.cls_token_id = vocab["<s>"]
        self.sep_token_id = vocab["</s>"]
        self.unk_token_id = vocab["<unk>"]
        self.class_names = config.model.class_names
    
    def tokenize_text(self, text: str) -> List[int]:
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)) 
        return tokens
        
    def preprocess(self, raw_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        output: {
            "input_ids":             torch.Tensor((N, max_seq_length)),
            "bbox":                  torch.Tensor((N, max_seq_length, 8)),
            "attention_mask":        torch.Tensor((N, max_seq_length)),
            "are_box_first_tokens":  torch.Tensor((N, max_seq_length)),
        }
        """
        num_input = len(raw_inputs)
        input_ids = np.ones((num_input, self.max_seq_length), dtype=int) * self.pad_token_id
        bbox = np.zeros((num_input, self.max_seq_length, 8), dtype=np.float32)
        attention_mask = np.zeros((num_input, self.max_seq_length), dtype=int)
        are_box_first_tokens = np.zeros((num_input, self.max_seq_length), dtype=np.bool_)
        
        for raw_idx, raw_input in enumerate(raw_inputs):
            width = raw_input["imageSize"]["width"]
            height = raw_input["imageSize"]["height"]

            list_tokens, list_bbs, box_to_token_indices = self.convert_raw_data(
                raw_input, width, height
            )
            len_list_tokens = len(list_tokens)
            input_ids[raw_idx, :len_list_tokens] = list_tokens
            attention_mask[raw_idx, :len_list_tokens] = 1
            bbox[raw_idx, :len_list_tokens, :] = list_bbs
            bbox[raw_idx, :, [0, 2, 4, 6]] = bbox[raw_idx, :, [0, 2, 4, 6]] / width
            bbox[raw_idx, :, [1, 3, 5, 7]] = bbox[raw_idx, :, [1, 3, 5, 7]] / height

            st_indices = [
                indices[0] for indices in box_to_token_indices
                if indices[0] < self.max_seq_length
            ]
            are_box_first_tokens[raw_idx, st_indices] = True
        input_ids = torch.from_numpy(input_ids)
        bbox =torch.from_numpy(bbox)
        attention_mask = torch.from_numpy(attention_mask)
        are_box_first_tokens = torch.from_numpy(are_box_first_tokens)

        model_inputs = {
            "input_ids":             input_ids.to(self.device),
            "bbox":                  bbox.to(self.device),
            "attention_mask":        attention_mask.to(self.device),
            "are_box_first_tokens":  are_box_first_tokens,
        } 
        return model_inputs
    
    
    def convert_raw_data(self, raw_input: Dict[str, Any], width: int, height: int):
        list_tokens, list_bbs, box_to_token_indices = [], [], []
        cum_token_idx = 0
        cls_bbs = [0.0] * 8
        for word_idx, word in enumerate(raw_input["words"]):
            text = word["text"]
            
            tokens = self.tokenize_text(text=text)
            this_box_token_indices = []
            bb = word["boundingBox"]
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)
            if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
                break
            list_tokens += tokens
        
            # min, max clipping
            for coord_idx in range(4):
                bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
                bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

            bb = list(itertools.chain(*bb))
            bbs = [bb for _ in range(len(tokens))]
            for _ in tokens:
                cum_token_idx += 1
                this_box_token_indices.append(cum_token_idx)

            list_bbs.extend(bbs)
            box_to_token_indices.append(this_box_token_indices)
        
        sep_bbs = [width, height] * 4
        # For [CLS] and [SEP]
        list_tokens = (
            [self.cls_token_id]
            + list_tokens[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        
        if len(list_bbs) == 0:
            # When len(json_obj["words"]) == 0 (no OCR result)
            list_bbs = [cls_bbs] + [sep_bbs]
        else:  # len(list_bbs) > 0
            list_bbs = [cls_bbs] + list_bbs[: self.max_seq_length - 2] + [sep_bbs]
        
        return list_tokens, list_bbs, box_to_token_indices
    
    
    def forward(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            head_outputs, _ = self.model(input_data)
            # head_outputs={"el_outputs": torch.Tensor((N, max_seq_length, max_seq_length+1))}
        return head_outputs
    
    
    def postprocess_rel(self, 
        pr_el_labels: np.ndarray, box_first_tokens: np.ndarray \
    ) -> List[List[List[int]]]:
        """
            Convert output of link task -> link between tokens -> links between line texts
        """
        link_map_batch = []
        for pr_el, first_tk in zip (pr_el_labels, box_first_tokens):
            pr_valid_el = pr_el * first_tk
            pr_valid_el = pr_valid_el
            pr_el_np = pr_el
            valid_tk_indices = np.where(
                (pr_valid_el != self.max_seq_length) * (pr_valid_el != 0)
            )
            
            first_tk = first_tk
            map_idx = np.zeros(first_tk.shape, dtype=np.int32)
            _id = -1
            for i in range(map_idx.shape[0]):
                if first_tk[i]:
                    _id += 1
                map_idx[i] = _id

            link_map_tuples = []
            for token_idx in valid_tk_indices[0]:
                link_map_tuples.append([
                    map_idx[int(pr_el_np[token_idx])], 
                    map_idx[int(token_idx)]])
            link_map_batch.append(link_map_tuples)
        
        return link_map_batch
    
    def postprocess_ee(self, 
        pr_itc_labels: np.ndarray, pr_stc_labels: np.ndarray, \
        box_first_tokens: np.ndarray, attention_masks: np.ndarray, \
        map_mode: str = "first"
    ) -> List[int]:
        """
            Convert output of entity extraction task -> labels of token -> labels of line text
        """
        pr_class_tokens = get_token_lv_ser_labels(
            pr_itc_labels, pr_stc_labels,
            box_first_tokens, attention_masks,
            self.max_seq_length, self.class_names
        )
        entity_lv_ser_labels = get_entity_lv_ser_labels(
            pr_class_tokens, box_first_tokens, attention_masks, self.class_names, map_mode
        )
        return entity_lv_ser_labels
    
    
    def extract_link(self, raw_inputs: List[Dict[str, Any]]) -> List[List[List[int]]]:
        """
        input_data: {
           imageSize: {
               height: int,
               width: int
           },
           words: List[{
               text: str,
               boundingBox: List[int] (shape[4, 2])
           }]
        }   
        return: couple of box indices in each image of input batch
            format: List[List[List[int]]]
        """
        model_inputs = self.preprocess(raw_inputs)

        head_outputs = self.forward(input_data=model_inputs)
        box_first_tokens = model_inputs["are_box_first_tokens"].cpu().detach().numpy()
        attention_masks =model_inputs["attention_mask"].cpu().detach().numpy()
        pr_itc = head_outputs["itc_outputs"].cpu().detach().numpy()
        pr_stc, pr_hor, pr_ver = head_outputs["multi_outputs"].cpu().detach().numpy()
        
        pr_itc, pr_stc = np.argmax(pr_itc, -1), np.argmax(pr_stc, -1)
        pr_hor, pr_ver = np.argmax(pr_hor, -1), np.argmax(pr_ver, -1)
        
        
        link_map_hor_batch = self.postprocess_rel(pr_hor, box_first_tokens)
        del pr_hor
        link_map_ver_batch = self.postprocess_rel(pr_ver, box_first_tokens)
        del pr_ver
        ee_label = self.postprocess_ee(
            pr_itc, pr_stc, 
            box_first_tokens, attention_masks
        )
        del pr_itc, pr_stc,head_outputs, model_inputs
        torch.cuda.empty_cache()
        return link_map_hor_batch, link_map_ver_batch, ee_label

    
    
    
    
    