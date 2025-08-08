"""
This script defines interactions with the smoke data
"""
import torch
from torch.utils.data import Dataset, DataLoader
from data.data_utils import SumTokenizer


class SmokeDataset(Dataset):
    def __init__(self, max_text_length=20, max_summary_length=10, total_examples=10, unk_text=False):
        self.max_text_length = max_text_length
        self.max_summary_length = max_summary_length
        self.tokenizer = SumTokenizer(max_text_length, max_summary_length)

        if unk_text:
            self.text = "A new text input unknown to the model."
            self.summary = "unknown input"
        else:
            self.text = "Example input to test whether model can memorize a single example"
            self.summary = "The summary of example input"
        self.total_examples = total_examples

    def __len__(self):
        return self.total_examples

    def __getitem__(self, i):
        # Tokenize input and target
        inputs = self.tokenizer(self.text, is_target=False)
        targets = self.tokenizer(self.summary, is_target=True)

        target_ids = targets["input_ids"].squeeze(0)

        # Shift targets right: [<s>, The, summary, ..., </s>] -> [<pad>, <s>, The, summary, ...]
        decoder_input_ids = torch.cat(
            [torch.tensor([1], device=target_ids.device), target_ids[:-1]]  # Assuming 1 is the <pad> token
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "input_mask": inputs["attention_mask"].squeeze(0),
            "target_ids": target_ids,  # Used for loss calculation
            "decoder_input_ids": decoder_input_ids,  # Shifted, used for model input
            "target_mask": targets["attention_mask"].squeeze(0)
        }


def create_smoke_loaders(config):
    """
    Create the data loaders for the smoke data
    :return:
    """
    batch_size = 2
    train_set = SmokeDataset(config['max_text_length'], config['max_summary_length'])
    test_set = SmokeDataset(config['max_text_length'], config['max_summary_length'], unk_text=True)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    return train_loader, None, test_loader
