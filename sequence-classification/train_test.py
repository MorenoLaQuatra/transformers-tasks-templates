import os
import argparse

import sklearn
import torch
import transformers

from ClassificationDataset import ClassificationDataset
from parsing_arguments import parse_arguments

# it removes the warning for the number of threads used for data loading
os.environ["TOKENIZERS_PARALLELISM"] = "true"

args = parse_arguments()

"""
############################################################################################################
Here you need to define the data you want to use for training, validation and testing.
Each split is composed of two lists: texts and labels.
texts is a list of strings, each string is a text.
labels is a list of integers, each integer is a label.
############################################################################################################
"""
"""
train_list_text = [...]  # replace with your list of texts
train_list_labels = [...]  # replace with your list of labels

val_list_text = [...]  # replace with your list of texts
val_list_labels = [...]  # replace with your list of labels

test_list_text = [...]  # replace with your list of texts
test_list_labels = [...]  # replace with your list of labels
"""

"""
Example using a dataset from datasets library.
"""
from datasets import load_dataset

dataset = load_dataset("emotion")

train_list_text = dataset["train"]["text"]
train_list_labels = dataset["train"]["label"]
val_list_text = dataset["validation"]["text"]
val_list_labels = dataset["validation"]["label"]
test_list_text = dataset["test"]["text"]
test_list_labels = dataset["test"]["label"]

"""
############################################################################################################
Instantiating the dataset objects for each split.
!!! The dataloaders are created inside the Trainer object !!!
############################################################################################################
"""
sequence_classification_train_dataset = ClassificationDataset(
    texts=train_list_text,
    labels=train_list_labels,
    model_tag=args.MODEL_TAG,
    max_length=args.MAX_LENGTH,
    padding="max_length",
    truncation=True,
)

sequence_classification_val_dataset = ClassificationDataset(
    texts=val_list_text,
    labels=val_list_labels,
    model_tag=args.MODEL_TAG,
    max_length=args.MAX_LENGTH,
    padding="max_length",
    truncation=True,
)

sequence_classification_test_dataset = ClassificationDataset(
    texts=test_list_text,
    labels=test_list_labels,
    model_tag=args.MODEL_TAG,
    max_length=args.MAX_LENGTH,
    padding="max_length",
    truncation=True,
)


""" Instantiate the model """
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    args.MODEL_TAG,
    num_labels=len(set(train_list_labels)),
)

"""
############################################################################################################
Creating the training arguments that will be passed to the Trainer object.
Most of the parameters are taken from the parser arguments.
############################################################################################################
"""
training_arguments = transformers.TrainingArguments(
    output_dir=args.CHECKPOINT_DIR,
    num_train_epochs=args.EPOCHS,
    per_device_train_batch_size=args.BATCH_SIZE,
    per_device_eval_batch_size=args.BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=args.LOG_DIR,
    logging_steps=args.LOGGING_STEPS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=args.LEARNING_RATE,
    dataloader_num_workers=args.DATALOADER_NUM_WORKERS,
    save_total_limit=args.SAVE_TOTAL_LIMIT,
    no_cuda=not (args.USE_CUDA),
    fp16=args.FP16,
)

"""
############################################################################################################
Defining the compute_metrics function that will be used to compute the metrics for the validation and testing sets.
The function takes as input a dictionary with the predictions and the labels and returns a dictionary with the metrics.
############################################################################################################
"""


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = sklearn.metrics.accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


"""
############################################################################################################
Instantiating the Trainer object.
It will take care of training and validation.
############################################################################################################
"""

trainer = transformers.Trainer(
    model=model,
    args=training_arguments,
    train_dataset=sequence_classification_train_dataset,
    eval_dataset=sequence_classification_val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

"""
############################################################################################################
Evaluate the model on the test set.
############################################################################################################
"""

trainer.evaluate(sequence_classification_test_dataset)
