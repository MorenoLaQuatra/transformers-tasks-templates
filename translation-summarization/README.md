# Text classification task

This folder contains the training scripts and the dataset class for training a sequence-to-sequence model (e.g., Machine Translation or Text Summarization tasks).

Hereafter we use `[TASKNAME]` to indicate either `summarization` or `translation`.

Hereafter is a description of each file:

- `Dataset.py`: it contains the class for managing the dataset. The `__get_item__` method is required to retrieve the samples from the dataset. Text tokenization and padding are performed by the method `__get_item__`. If you want to add new preprocessing steps, you need to add this code inside the `__get_item__` method.
- `train_test_[TASKNAME].py`: it contains the script to train and test the model. It uses the Trainer class provided by the HuggingFace transformers library.
- `parsing_arguments.py`: it contains the class for parsing the arguments used to train and test the model. Feel free to add new arguments if you need to. You should also change the default values in `train_test_[TASKNAME].py` to match the new arguments. 

---

The example contains a script to train a sequence-to-sequence model on a toy dataset. If you want to train the model on a different dataset, you need to update the `train_test_[TASKNAME].py` script. In particular, you need to fill the following variables:

```python
train_sources = [...]  # replace with your list of source texts
train_targets = [...]  # replace with your list of target texts (e.g., summary or translation)

val_sources = [...]  # replace with your list of source texts
val_targets = [...]  # replace with your list of target texts (e.g., summary or translation)

test_sources = [...]  # replace with your list of source texts
test_targets = [...]  # replace with your list of target texts (e.g., summary or translation)
```

---

**Example command for training the summarization model**

You can use the following command to train a text summarization model on the [xsum](https://huggingface.co/datasets/xsum) dataset:

```python
python train_test_summarization.py \
    --MODEL_TAG facebook/bart-base \
    --BATCH_SIZE 32 \
    --EPOCHS 10 \
    --CHECKPOINT_DIR checkpoints \
    --LOG_DIR logs \
    --LOGGING_STEPS 100 \
    --MAX_INPUT_LENGTH 1024 \
    --MAX_OUTPUT_LENGTH 32 \
    --LEARNING_RATE 5e-5 \
    --DATALOADER_NUM_WORKERS 4 \
    --USE_CUDA \
    --SAVE_TOTAL_LIMIT 2 \
    --FP16 
```