import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Iterable
from tqdm import tqdm
import traceback


class NLLCal:
    def __init__(self, model_path, device='cpu', batch_size=4, max_length=1024):
        if model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.nll_loss = torch.nn.NLLLoss(reduction='none')


    @torch.no_grad()
    def texts_to_encoded_iter(self, texts: List[str], batch_size=None) -> Iterable:
        if batch_size is None:
            batch_size = self.batch_size
        for i in range(0, len(texts), batch_size):
            text_list = texts[i:i+batch_size]

            try:
                encoded = self.tokenizer(text_list, 
                                        return_tensors='pt', 
                                        padding=True,
                                        truncation=True,
                                        max_length=self.max_length).to(self.model.device)
            except RuntimeError:
                traceback.print_exc()
                print(f'batch {i} failed')
                print(f'text_list: {text_list}')
                exit(0)
            yield encoded


    @torch.no_grad()
    def encoded_to_nll(self, encoded) -> List:
        assert self.model is not None
        ids = encoded['input_ids']

        output = self.model(ids, labels=ids)
        logits = output.logits.to(self.model.device)
        logits = logits.permute(0, 2, 1) # reshape logits from (B, L, V) to (B, V, L)
        shift_logits = logits[:, :, :-1]
        shift_targets = ids[:, 1:]

        nlls = self.nll_loss(self.log_softmax(shift_logits), shift_targets)
        mask = encoded['attention_mask'][:, 1:]
        nll_list = []
        for i in range(nlls.shape[0]): # Along B dimension
            raw = nlls[i, :]
            nll = torch.masked_select(raw, mask[i, :]>0)
            nll_list.append(nll)

        return nll_list


    @torch.no_grad()
    def texts_to_nll(self, texts: List[str], batch_size=None) -> List:
        """
        For quick experiment over a text input
        """
        if batch_size is None:
            batch_size = self.batch_size
        nll_list = []
        for encoded in tqdm(self.texts_to_encoded_iter(texts, batch_size), total=len(texts)//batch_size+1):
            nll_list.extend(self.encoded_to_nll(encoded))
        return nll_list

