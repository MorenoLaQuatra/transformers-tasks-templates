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
        source_text: List[str],
        target_text: List[str],
        tokenizer,
        max_input_length: int = 1024,
        max_output_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
    ):

        self.source_text = source_text
        self.target_text = target_text
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation

    def __getitem__(self, idx):
        """
        This function is called to get the tokenized text and the label for a given index.
        :param idx: The index of the text and label to be returned.
        :return: A dictionary containing the tokenized text (with attention mask) and the label.
        """
        input = self.tokenizer(
            self.source_text[idx],
            max_length=self.max_input_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            output = self.tokenizer(
                self.target_text[idx],
                max_length=self.max_output_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors="pt",
            )

        item = {
            "input_ids": input["input_ids"].squeeze(),
            "attention_mask": input["attention_mask"].squeeze(),
            "labels": output["input_ids"].squeeze(),
        }

        return item

    def __len__(self):
        return len(self.source_text)
