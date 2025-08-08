"""
This script provides utilities used across the data sets
"""
from transformers import BartTokenizer


class SumTokenizer:
    def __init__(self, max_text_length=512, max_summary_length=128):
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.max_text_length = max_text_length
        self.max_summary_length = max_summary_length
        self.vocab_size = len(self.tokenizer)

    def __call__(self, text, is_target=False, padding=True, truncation=True):
        if is_target:
            return self.tokenizer(
                text,
                padding='max_length' if padding else False,
                truncation=truncation,
                max_length=self.max_summary_length,
                return_tensors='pt'
            )
        else:
            return self.tokenizer(
                text,
                padding='max_length' if padding else False,
                truncation=truncation,
                max_length=self.max_text_length,
                return_tensors='pt'
            )

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)