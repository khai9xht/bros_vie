import json, yaml
import os, re
from typing import List, Dict
import numpy as np
from fuzzywuzzy import fuzz
import random
from unidecode import unidecode
from transformers import AutoTokenizer


class ReplaceText(object):
    def __init__(self, tokenizer_path, examcodes_fp, threshold=90, p=0.5) -> None:
        self.__class__.__name__ = "replace_text"
        self.p = p
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, do_lower_case=True
        )
        self.text_corpus = self.load_text_corpus(examcodes_fp)
        
    def load_text_corpus(self, filepath) -> Dict[str, List[List[int]]]:
        assert os.path.exists(filepath), f"{filepath} doesn't exist."
        with open(filepath, 'r') as jfile:
            dict_data = yaml.safe_load(jfile)
        
        text_corpus = {}
        for key, value in dict_data.items():
            abb_tokens = [self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(word)) for word in value["abbreviation"]]
            text_corpus[key] = abb_tokens
            for text in value["keys"]:
                value_tokens = [
                    self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(word)
                ) for word in text.split(" ")]
                text_corpus[text] = value_tokens 
        return text_corpus
    
    def _process_text(self, text: str):
        return re.sub(" +", " ", re.sub("\W+", " ", unidecode(text))).lower()
    
    def check_text_in_corpus(self, key_text: str) -> List[str]:
        for text in list(self.text_corpus.keys()):
            if fuzz.ratio(self._process_text(key_text), self._process_text(text)
            ) > self.threshold:
                return True
        return False
    
    def __call__(self, text: str) -> str:
        if random.random() > self.p or not self.check_text_in_corpus(text):
            return text, None
        else:
            replaced_text = random.choice(list(self.text_corpus.keys()))
            return replaced_text, self.text_corpus[replaced_text]

    
class RandomLossCharacters(object):
    def __init__(self, r_loss=0.1, p=0.5) -> None:
        self.__class__.__name__ = "random_loss_characters"
        self.p = p
        self.r_loss = r_loss
    
    def __call__(self, text: str) -> str:
        if random.random() > self.p:
            return text
        else:
            prob_loss = np.random.rand(len(text))
            error_text = ''.join([text[i] for i in range(len(text)) if prob_loss[i] > self.r_loss])
            return error_text

    
class RandomMaskText(object):
    def __init__(self, min_words=4, r_mask=0.05, p=0.5) -> None:
        self.__class__.__name__ = "random_mask_text"
        self.p = p
        self.r_mask = r_mask
        self.mask = "<mask>"
        self.min_words = min_words
    
    def __call__(self, text: str) -> str: 
        if random.random() > self.p:
            return text
        else:
            words = text.split(" ")
            if len(words) < self.min_words:
                return text
            ids_prob = np.random.rand(len(words))
            for i, idx_prob in enumerate(ids_prob):
                if idx_prob < self.r_mask:
                    words[i] = self.mask
            masked_text = " ".join(words)
            return masked_text

