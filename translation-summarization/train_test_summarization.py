import os
import argparse

import sklearn
import torch
import transformers

from Dataset import Dataset
from parsing_arguments import parse_arguments

# it removes the warning for the number of threads used for data loading
os.environ["TOKENIZERS_PARALLELISM"] = "true"

args = parse_arguments()

"""
############################################################################################################
Here you need to define the data you want to use for training, validation and testing.
Each split is composed of two lists: sources and targets.
sources is a list of strings, each string is a text.
targets is a list of strings, each integer is a label.
############################################################################################################
"""

"""
train_sources = [...]  # replace with your list of texts
train_targets = [...]  # replace with your list of labels (e.g., summary or translation)

val_sources = [...]  # replace with your list of texts
val_targets = [...]  # replace with your list of target text (e.g., summary or translation)

test_sources = [...]  # replace with your list of texts
test_targets = [...]  # replace with your list of target texts (e.g., summary or translation)
"""

"""
Example using a dataset from datasets library.
"""
from datasets import load_dataset

dataset = load_dataset("xsum")

train_sources = dataset["train"]["document"]
train_targets = dataset["train"]["summary"]
val_sources = dataset["validation"]["document"]
val_targets = dataset["validation"]["summary"]
test_sources = dataset["test"]["document"]
test_targets = dataset["test"]["summary"]

"""
############################################################################################################
Instantiating the dataset objects for each split.
!!! The dataloaders are created inside the Trainer object !!!
############################################################################################################
"""

summarization_train_dataset = Dataset(
    source_text = train_sources,
    target_text = train_targets,
    model_tag = args.MODEL_TAG,
    max_input_length = args.MAX_INPUT_LENGTH,
    max_output_length = args.MAX_OUTPUT_LENGTH,
    padding = args.PADDING,
    truncation = True,
)

summarization_val_dataset = Dataset(
    source_text = val_sources,
    target_text = val_targets,
    model_tag = args.MODEL_TAG,
    max_input_length = args.MAX_INPUT_LENGTH,
    max_output_length = args.MAX_OUTPUT_LENGTH,
    padding = args.PADDING,
    truncation = True,
)

summarization_test_dataset = Dataset(
    source_text = test_sources,
    target_text = test_targets,
    model_tag = args.MODEL_TAG,
    max_input_length = args.MAX_INPUT_LENGTH,
    max_output_length = args.MAX_OUTPUT_LENGTH,
    padding = args.PADDING,
    truncation = True,
)


""" Instantiate the model """
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    args.MODEL_TAG,
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
    metric_for_best_model="R2-F",
    greater_is_better=True,
)

"""
############################################################################################################
Defining the compute_metrics function that will be used to compute the metrics for the validation and testing sets.
The function takes as input a dictionary with the predictions and the labels and returns a dictionary with the metrics.
############################################################################################################
"""


rouge = load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    )

    return {
        "R1-P": round(rouge_output['rouge1'].mid.precision, 4),
        "R1-R": round(rouge_output['rouge1'].mid.recall, 4),
        "R1-F": round(rouge_output['rouge1'].mid.fmeasure, 4),
        "R2-P": round(rouge_output['rouge2'].mid.precision, 4),
        "R2-R": round(rouge_output['rouge2'].mid.recall, 4),
        "R2-F": round(rouge_output['rouge2'].mid.fmeasure, 4),
        "RL-P": round(rouge_output['rougeL'].mid.precision, 4),
        "RL-R": round(rouge_output['rougeL'].mid.recall, 4),
        "RL-F": round(rouge_output['rougeL'].mid.fmeasure, 4),
        "RLsum-P": round(rouge_output['rougeLsum'].mid.precision, 4),
        "RLsum-R": round(rouge_output['rougeLsum'].mid.recall, 4),
        "RLsum-F": round(rouge_output['rougeLsum'].mid.fmeasure, 4),
    }




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
