import os
import argparse

import sklearn
import torch
import transformers
import evaluate
from sklearn.model_selection import train_test_split

from Dataset import Dataset
from parsing_arguments import parse_arguments



# it removes the warning for the number of threads used for data loading
os.environ["TOKENIZERS_PARALLELISM"] = "true"

args = parse_arguments()

"""
############################################################################################################
Here you need to define the data you want to use for training, validation and testing.
Each split is composed of a single list of strings 
sources is a list of strings.
############################################################################################################
"""

"""
train_sources = [...]  # replace with your list of texts
val_sources = [...]  # replace with your list of texts
test_sources = [...]  # replace with your list of texts
"""

"""
Example using a dataset from datasets library.
"""
from datasets import load_dataset

dataset = load_dataset("demelin/understanding_fables")

sources = dataset["test"]["story"]
sources = [s.replace("What is the moral of this story?", "") for s in sources]

train_sources, test_sources = train_test_split(sources, test_size=0.2, random_state=42)
val_sources, test_sources = train_test_split(test_sources, test_size=0.5, random_state=42)


"""
############################################################################################################
Instantiating the dataset objects for each split.
!!! The dataloaders are created inside the Trainer object !!!
############################################################################################################
"""

nlg_train_dataset = Dataset(
    texts = train_sources,
    model_tag = args.MODEL_TAG,
    max_length = args.MAX_LENGTH,
    truncation = True,
)

nlg_val_dataset = Dataset(
    texts = val_sources,
    model_tag = args.MODEL_TAG,
    max_length = args.MAX_LENGTH,
    truncation = True,
)

nlg_test_dataset = Dataset(
    texts = test_sources,
    model_tag = args.MODEL_TAG,
    max_length = args.MAX_LENGTH,
    truncation = True,
)


""" Instantiate the model """
model = transformers.AutoModelForCausalLM.from_pretrained(
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
    gradient_checkpointing=args.GRADIENT_CHECKPOINTING,
)

"""
############################################################################################################
Defining the compute_metrics function that will be used to compute the metrics for the validation and testing sets.
The function takes as input a dictionary with the predictions and the labels and returns a dictionary with the metrics.
############################################################################################################
"""

"""
In this case, we use the loss to evaluate the model. It is not needed to define a compute_metrics function.
"""



"""
############################################################################################################
Instantiating the Trainer object.
It will take care of training and validation.
############################################################################################################
"""

trainer = transformers.Trainer(
    model=model,
    args=training_arguments,
    train_dataset=nlg_train_dataset,
    eval_dataset=nlg_val_dataset,
)

trainer.train()

"""
############################################################################################################
Evaluate the model on the test set.
############################################################################################################
"""

trainer.evaluate(nlg_test_dataset)
