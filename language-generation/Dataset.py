from typing import List

import torch
import transformers


class Dataset(torch.utils.data.Dataset):
    """
    This class is inteded for sequence classification tasks.
    :param texts: List of texts to be tokenized.
    :param labels: List of labels for each text.
    :param tokenizer: The identifier for the tokenizer to be used.
                    It can be an identifier from the transformers library or a path to a local tokenizer.
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
