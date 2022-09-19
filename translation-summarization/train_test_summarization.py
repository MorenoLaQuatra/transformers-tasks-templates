import os
import argparse

import sklearn
import torch
import transformers
import datasets
import evaluate
from datasets import load_dataset
import pandas as pd

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
train_sources = [...]  # replace with your list of source texts
train_targets = [...]  # replace with your list of target texts (e.g., summary or translation)

val_sources = [...]  # replace with your list of source texts
val_targets = [...]  # replace with your list of target texts (e.g., summary or translation)

test_sources = [...]  # replace with your list of source texts
test_targets = [...]  # replace with your list of target texts (e.g., summary or translation)
"""


if args.DATASET_FILE is not None:
    # If the DATASET_FILE argument is not None, the dataset is loaded from the file.
    dataset = pd.read_csv(args.DATASET_FILE, sep="\t")
    sources = dataset["source_text"].tolist()
    targets = dataset["target_text"].tolist()
    train_sources, test_sources, train_targets, test_targets = train_test_split(
        sources, targets, test_size=0.2, random_state=42
    )
    val_sources, test_sources, val_targets, test_targets = train_test_split(
        test_sources, test_targets, test_size=0.5, random_state=42
    )
else:
    # Example using a dataset from datasets library.
    dataset = load_dataset("xsum")

    train_sources = dataset["train"]["document"]
    train_targets = dataset["train"]["summary"]
    val_sources = dataset["validation"]["document"]
    val_targets = dataset["validation"]["summary"]
    test_sources = dataset["test"]["document"]
    test_targets = dataset["test"]["summary"]


"""
############################################################################################################
Here you need to define the model and the tokenizer you want to use.
############################################################################################################
"""
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    args.MODEL_TAG,
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.MODEL_TAG,
)

tokenizer.save_pretrained(args.TOKENIZER_DIR)

"""
############################################################################################################
Instantiating the dataset objects for each split.
!!! The dataloaders are created inside the Trainer object !!!
############################################################################################################
"""

summarization_train_dataset = Dataset(
    source_text=train_sources,
    target_text=train_targets,
    tokenizer=tokenizer,
    max_input_length=args.MAX_INPUT_LENGTH,
    max_output_length=args.MAX_OUTPUT_LENGTH,
    padding="max_length",
    truncation=True,
)

summarization_val_dataset = Dataset(
    source_text=val_sources,
    target_text=val_targets,
    tokenizer=tokenizer,
    max_input_length=args.MAX_INPUT_LENGTH,
    max_output_length=args.MAX_OUTPUT_LENGTH,
    padding="max_length",
    truncation=True,
)

summarization_test_dataset = Dataset(
    source_text=test_sources,
    target_text=test_targets,
    tokenizer=tokenizer,
    max_input_length=args.MAX_INPUT_LENGTH,
    max_output_length=args.MAX_OUTPUT_LENGTH,
    padding="max_length",
    truncation=True,
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
    metric_for_best_model="R2",
    greater_is_better=True,
    hub_model_id=args.HUB_MODEL_ID,
    push_to_hub=args.PUSH_TO_HUB,
)

"""
############################################################################################################
Defining the compute_metrics function that will be used to compute the metrics for the validation and testing sets.
The function takes as input a dictionary with the predictions and the labels and returns a dictionary with the metrics.
############################################################################################################
"""


rouge = evaluate.load("rouge")


def compute_metrics(pred):

    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )

    return {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


"""
############################################################################################################
Instantiating the Trainer object.
It will take care of training and validation.
############################################################################################################
"""

trainer = transformers.Trainer(
    model=model,
    args=training_arguments,
    train_dataset=summarization_train_dataset,
    eval_dataset=summarization_val_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()

"""
############################################################################################################
Evaluate the model on the test set.
############################################################################################################
"""

trainer.evaluate(summarization_test_dataset)

"""
############################################################################################################
If the PUSH_TO_HUB argument is True, the model is pushed to the Hugging Face Hub.
The model is pushed to the user's namespace using the HUB_MODEL_NAME argument.
############################################################################################################
"""

if args.PUSH_TO_HUB:
    trainer.push_to_hub()
    tokenizer.push_to_hub(repo_id=args.HUB_MODEL_ID)
