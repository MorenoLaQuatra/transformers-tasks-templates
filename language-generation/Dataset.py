from typing import List

import torch
import transformers


class Dataset(torch.utils.data.Dataset):
    """
    This class is inteded for sequence classification tasks.
    :param texts: List of texts to be tokenized.
    :param tokenizer: The tokenizer to be used for tokenizing the texts. It can be an instance of the transformers AutoTokenizer class or a custom tokenizer.
    :param max_length: The maximum length of the tokenized text.
    :param padding: The padding strategy to be used. Available options are available in the transformers library.
    :param truncation: Whether to truncate the text or not.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 1024,
        padding: str = "max_length",
        truncation: bool = True,
    ):

        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __getitem__(self, idx):
        """
        This function is called to get the tokenized text and the label for a given index.
        :param idx: The index of the text and label to be returned.
        :return: A dictionary containing the tokenized text (with attention mask) and the label.
        """
        item = self.tokenizer(
            f"{self.tokenizer.bos_token_id} {self.texts[idx]} {self.tokenizer.eos_token_id}",
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        return item

    def __len__(self):
        return len(self.texts)
