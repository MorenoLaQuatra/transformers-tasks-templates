# transformers-tasks-templates

This repository contains templates for fine-tuning 🤗 Transformers models on various tasks. It is intended to be used as a starting point  for training models on specific tasks. It is not NLP-specific, and the aim is to provide a template for tasks related to the audio, vision, and text domains (supported by the 🤗 Transformers library).

It is also possible to use the templates for training models on custom datasets following a specific format (slightly task-dependent).

---

An overview of the tasks and their implementation status is provided below.

- [x] Text classification (e.g., sentiment analysis)
- [x] Text regression (e.g., rating prediction)
- [x] Sequence-to-sequence (e.g., machine translation or text summarization)
- [x] Text generation (e.g., text completion)
- [ ] Audio classification (e.g., speech emotion recognition)
- [ ] Audio frame classification (e.g., speech activity detection)
- [ ] Image classification (e.g., recognizing the type of an image)

The repository contains a script to train and test the model. It uses the Trainer class provided by the HuggingFace Transformers library. Each folder is dedicated to a specific task (or a set of tasks). The folder contains the training script and the dataset class for training a sequence-to-sequence model.

- [Sequence Classification](sequence-classification/README.md)
- [Sequence to sequence](translation-summarization/README.md)
- [Text Generation](language-generation/README.md)

---

## Fine-tune a model on a custom dataset

Each folder contains a script that can be used to fine-tune specific type of models. You can also fine-tune a model on a custom dataset without needing to modify the script. If you provide a `.tsv` file with a specific format you can use the `--DATASET_FILE` parameter to specify the path to the file. The format of the file should be the following:

**Text Classification**
    
```tsv
text	label
I love this movie!	1
I hate this movie!	0
```

**Text Regression**

```tsv
text	label
I love this movie!	4.5
I hate this movie!	0.5
```

**Summarization**

```tsv
source_text	target_text
Very long text containing a lot of information.	Short summary of the text.
Very long text containing a lot of information.	Short summary of the text.
```

**Translation**

```tsv
source_text	target_text
The cat sat on the mat.	Il gatto è seduto sul tappeto.
The pen is on the table.	La penna è sul tavolo.
```

**Text Generation**

```tsv
text
Text that can be used to infer the next word.
Another text that can be used to infer the next word.
```

Each folder also the `data` subfolder that contains some examples of tsv files with the correct format.

---


## TODO

- [x] Modify the management of the tokenizer. It is currently loaded in the dataset class. It should be loaded in the `train_test.py` script.
- [ ] Translation example
- [ ] Add a demo notebook/script to run inference on the trained model (each task).
- [x] Add parameters to fine-tune the model on a custom dataset (e.g., csv file).
- [x] Update the main README.md file to include links to the task-specific README.md files and a description of the repository.
- [x] Manage push_to_hub() method to push the trained model to the hub (use argparse to specify the name of the repository).
- [ ] Add non-language tasks (e.g., audio classification, image classification, etc.).
- [ ] Make repository public.