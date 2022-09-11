# Text classification task

This folder contains the training scripts and the dataset class for training a sequence classification model (e.g., Text Classification task).

Hereafter is a description of each file:

- `Dataset.py`: it contains the class for managing the dataset. The `__get_item__` method is required to retrieve the samples from the dataset. Text tokenization and padding are performed by the method `__get_item__`. If you want to add new preprocessing steps, you need to add this code inside the `__get_item__` method.
- `train_test_clf.py`: it contains the script to train and test the classification model. It uses the Trainer class provided by the HuggingFace transformers library.
- `parsing_arguments.py`: it contains the class for parsing the arguments used to train and test the model. Feel free to add new arguments if you need to. You should also change the default values in `train_test.py` to match the new arguments. 

**Example command for training the classification model**

You can use the following command to train the classification model on the [emotion dataset](https://huggingface.co/datasets/emotion):

```python
python train_test_clf.py \
    --MODEL_TAG distilbert-base-cased \
    --BATCH_SIZE 32 \
    --EPOCHS 10 \
    --CHECKPOINT_DIR checkpoints \
    --LOG_DIR logs \
    --LOGGING_STEPS 100 \
    --MAX_LENGTH 128 \
    --LEARNING_RATE 5e-5 \
    --DATALOADER_NUM_WORKERS 4 \
    --USE_CUDA \
    --SAVE_TOTAL_LIMIT 2 \
    --FP16 
```

### Regression example

The file `train_test_reg.py` contains an example script to train a regression model to predict a continuous score. 

Few things to notice here:

- The dataset class does not need any modification.
- The `num_classes` parameter must be set to 1.
- The metric to select the best checkpoint is the mean absolute error. In contrast to accuracy, the best checkpoint is selected by **minimizing** the metric.

**Example command for training the regression model**

The script uses the [hate speech dataset](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) to predict the score assigned by humans to hate speech annotations.

```python
python train_test_reg.py \
    --MODEL_TAG distilbert-base-cased \
    --BATCH_SIZE 32 \
    --EPOCHS 10 \
    --CHECKPOINT_DIR checkpoints_reg \
    --LOG_DIR logs_reg \
    --LOGGING_STEPS 100 \
    --MAX_LENGTH 128 \
    --LEARNING_RATE 5e-5 \
    --DATALOADER_NUM_WORKERS 4 \
    --USE_CUDA \
    --SAVE_TOTAL_LIMIT 2 \
    --FP16 
```