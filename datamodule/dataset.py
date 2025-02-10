import torch
from torch.utils.data import DataLoader, Dataset

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
