import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_len, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_len]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+1+max_len]))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_text = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
        
        self.encoded_text = [encoded_text[:self.max_length] for encoded_text in self.encoded_text] 
        self.encoded_text = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) 
                             for encoded_text in self.encoded_text]
        
    def __getitem__(self, index):
        encoded = self.encoded_text[index]
        label = self.data.iloc[index]["Label"]

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_text:
            if len(encoded_text) > max_length:
                max_length = len(encoded_text)
        return max_length
