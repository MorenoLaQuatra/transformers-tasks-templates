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
        model_tag: str,
        max_length: int = 1024,
        truncation: bool = True,
        start_sequence: str = "<START>",
        end_sequence: str = "<END>",
    ):

        self.texts = texts
        self.max_length = max_length
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_tag)
        self.model_tag = model_tag
        self.truncation = truncation
        self.start_sequence = start_sequence
        self.end_sequence = end_sequence

    def __getitem__(self, idx):
        """
        This function is called to get the tokenized text and the label for a given index.
        :param idx: The index of the text and label to be returned.
        :return: A dictionary containing the tokenized text (with attention mask) and the label.
        """
        item = self.tokenizer(
            f"{self.start_sequence} {self.texts[idx]} {self.end_sequence}",
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors="pt",
        )

        return item

    def __len__(self):
        return len(self.texts)
