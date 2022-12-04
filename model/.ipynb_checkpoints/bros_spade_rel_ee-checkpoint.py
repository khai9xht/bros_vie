import torch
from torch import nn
from transformers import AutoTokenizer
import io


class BROSSPADEREEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg = cfg.model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encode_weight = cfg.pretrained.encode_path
        itc_weight = cfg.pretrained.itc_path
        multi_weight = cfg.pretrained.multi_path
        tokenizer_weight = cfg.model.tokenizer_path
        with open(encode_weight, 'rb') as ef:
            encode_buffer = io.BytesIO(ef.read())
            self.encode = torch.jit.load(encode_buffer, map_location=device)
        with open(itc_weight, 'rb') as itf:
            itc_buffer = io.BytesIO(itf.read())   
            self.itc = torch.jit.load(itc_buffer, map_location=device)
        with open(multi_weight, 'rb') as mf:
            multi_buffer = io.BytesIO(mf.read()) 
            self.multi = torch.jit.load(multi_buffer, map_location=device)
        self.tokenizer = AutoTokenizer.from_pretrained("configs/phoBert", local_files_only=True)


    def forward(self, batch):
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]

        last_hidden_states = self.encode(input_ids, bbox, attention_mask)
        last_hidden_states = last_hidden_states.transpose(0, 1).contiguous()

        itc_outputs = self.itc(last_hidden_states).transpose(0, 1).contiguous()
        multi_output = self.multi(last_hidden_states, last_hidden_states).squeeze(0)
        head_outputs = {"itc_outputs": itc_outputs, "multi_outputs": multi_output}
        
        self._get_loss(head_outputs, batch)
        losses = None
        return head_outputs, losses

    def _get_loss(self, head_outputs, batch):
        multi_output = head_outputs["multi_outputs"].cpu()

        stc_outputs = multi_output[0]
        hor_outputs = multi_output[1]
        ver_outputs = multi_output[2]
        
        self._get_stc_loss(stc_outputs, batch)
        self._get_el_loss(hor_outputs, batch, mode="hor")
        self._get_el_loss(ver_outputs, batch, mode="ver")


    def _get_stc_loss(self, stc_outputs, batch):
        inv_attention_mask = 1 - batch["attention_mask"]

        inv_attention_mask = inv_attention_mask.cpu()
        bsz, max_seq_length = inv_attention_mask.shape
        device = inv_attention_mask.device

        invalid_token_mask = torch.cat(
            [inv_attention_mask, torch.zeros([bsz, 1]).to(device)], axis=1
        ).bool()
        stc_outputs.masked_fill_(invalid_token_mask[:, None, :], -10000.0)

        self_token_mask = (
            torch.eye(max_seq_length, max_seq_length + 1).to(device).bool()
        )
        stc_outputs.masked_fill_(self_token_mask[None, :, :], -10000.0)
        invalid_token_mask = inv_attention_mask.cpu()
        self_token_mask = self_token_mask.cpu()
    
    
    def _get_el_loss(self, el_outputs, batch, mode):
        bsz, max_seq_length = batch["attention_mask"].shape
        device = batch["are_box_first_tokens"].device

        self_token_mask = (
            torch.eye(max_seq_length, max_seq_length + 1).to(device).bool()
        )
        box_first_token_mask = torch.cat([
            (batch["are_box_first_tokens"] == False),
            torch.zeros([bsz, 1], dtype=torch.bool).to(device),
        ], axis=1,)
        el_outputs.masked_fill_(box_first_token_mask[:, None, :], -10000.0)
        el_outputs.masked_fill_(self_token_mask[None, :, :], -10000.0)
        self_token_mask = self_token_mask.cpu()
        box_first_token_mask = box_first_token_mask.cpu()


