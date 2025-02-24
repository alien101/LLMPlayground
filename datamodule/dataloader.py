from datamodule.dataset import GPTDatasetV1
from torch.utils.data import DataLoader
import tiktoken


def create_dataloader(txt, batch_size=4, max_len=256, 
                      stride=128, shuffle=True, drop_last=True,
                      num_workers=0):
    
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt=txt,
                           tokenizer=tokenizer, 
                           max_len=max_len, 
                           stride=stride)
    
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader

if __name__ == "__main__":

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader(raw_text, batch_size=1,max_len=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    second=next(data_iter)
    print(first_batch)