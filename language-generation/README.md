# Text Generation
This folder contains the training scripts and the dataset class for training a text generation model (using transformer decoder to generate text, e.g., GPT2).

Hereafter is a description of each file:

- `Dataset.py`: it contains the class for managing the dataset. The `__get_item__` method is required to retrieve the samples from the dataset. Text tokenization are performed by the method `__get_item__`. If you want to add new preprocessing steps, you need to add this code inside the `__get_item__` method.
- `train_test_clf.py`: it contains the script to train and test the classification model. It uses the Trainer class provided by the HuggingFace transformers library.
- `parsing_arguments.py`: it contains the class for parsing the arguments used to train and test the model. Feel free to add new arguments if you need to. You should also change the default values in `train_test.py` to match the new arguments.

---

The example contains a script to train a text generation model on a toy dataset. If you want to train the model on a different dataset, you need to update the `train_test.py` script. In particular, you need to fill the following variables:

```python
train_sources = [...]  # replace with your list of texts
val_sources = [...]  # replace with your list of texts
test_sources = [...]  # replace with your list of texts
```

and remove the lines containing the following:
    
```python
"""
Example using a dataset from datasets library.
"""
from datasets import load_dataset

dataset = load_dataset("demelin/understanding_fables")

sources = dataset["test"]["story"]
sources = [s.replace("What is the moral of this story?", "") for s in sources]

train_sources, test_sources = train_test_split(sources, test_size=0.2, random_state=42)
val_sources, test_sources = train_test_split(test_sources, test_size=0.5, random_state=42)
```

---

**Example command for training the text generation model**

You can use the following command to fine-tune the classification model on the [demelin/understanding_fables](https://huggingface.co/datasets/demelin/understanding_fables) dataset:

```python
python train_language_generation.py \
    --MODEL_TAG distilgpt2 \
    --BATCH_SIZE 16 \
    --EPOCHS 5 \
    --CHECKPOINT_DIR checkpoints \
    --LOG_DIR logs \
    --LOGGING_STEPS 100 \
    --MAX_LENGTH 512 \
    --LEARNING_RATE 5e-5 \
    --DATALOADER_NUM_WORKERS 4 \
    --USE_CUDA \
    --SAVE_TOTAL_LIMIT 2 \
    --FP16 
```

It is worth noting that at each evaluation step, the loss is used to select the best checkpoint. In contrast to accuracy, the best checkpoint is selected by **minimizing** the loss.

## TODO

- [x] Modify the management of the tokenizer. It is currently loaded in the dataset class. It should be loaded in the `train_test.py` script.
- [ ] Add a demo notebook/script to generate text from a trained model.
- [ ] Add parameters to fine-tune the model on a custom dataset (e.g., csv file).