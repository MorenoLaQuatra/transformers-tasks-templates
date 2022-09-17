# transformers-tasks-templates


## TODO

- [x] Modify the management of the tokenizer. It is currently loaded in the dataset class. It should be loaded in the `train_test.py` script.
- [ ] Translation example
- [ ] Add a demo notebook/script to run inference on the trained model (each task).
- [ ] Add parameters to fine-tune the model on a custom dataset (e.g., csv file).
- [ ] Update the main README.md file to include links to the task-specific README.md files and a description of the repository.
- [ ] Make repository public.
- [ ] Manage push_to_hub() method to push the trained model to the hub (use argparse to specify the name of the repository).
- [ ] Add non-language tasks (e.g., audio classification, image classification, etc.).