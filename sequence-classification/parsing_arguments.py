import argparse

"""
############################################################################################################
Parsing arguments for the sequence classification task.
############################################################################################################
"""


def parse_arguments():
    """
    This function parses the arguments for the sequence classification task.
    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--MODEL_TAG",
        type=str,
        default="bert-base-cased",
        help="The identifier for the model to be used. It can be an identifier from the transformers library or a path to a local model.",
    )

    parser.add_argument(
        "--BATCH_SIZE",
        type=int,
        default=32,
        help="The batch size to be used for training.",
    )

    parser.add_argument(
        "--EPOCHS",
        type=int,
        default=10,
        help="The number of epochs to be used for training.",
    )

    parser.add_argument(
        "--CHECKPOINT_DIR",
        type=str,
        default="checkpoints",
        help="The directory where the checkpoints will be saved.",
    )

    parser.add_argument(
        "--LOG_DIR",
        type=str,
        default="logs",
        help="The directory where the logs will be saved.",
    )

    parser.add_argument(
        "--LOGGING_STEPS",
        type=int,
        default=100,
        help="The number of steps after which the logs will be saved.",
    )

    parser.add_argument(
        "--MAX_LENGTH",
        type=int,
        default=512,
        help="The maximum length of the tokenized text.",
    )

    parser.add_argument(
        "--LEARNING_RATE",
        type=float,
        default=5e-5,
        help="The learning rate to be used for training.",
    )

    parser.add_argument(
        "--DATALOADER_NUM_WORKERS",
        type=int,
        default=4,
        help="The number of workers to be used for the dataloaders.",
    )

    parser.add_argument(
        "--SAVE_TOTAL_LIMIT",
        type=int,
        default=2,
        help="The maximum number of checkpoints that will be saved. The best checkpoint will always be saved.",
    )

    parser.add_argument(
        "--FP16",
        default=False,
        action="store_true",
        help="Whether to use 16-bit (mixed) precision instead of 32-bit.",
    )

    parser.add_argument(
        "--USE_CUDA", default=False, action="store_true", help="Enable cuda computation"
    )

    args = parser.parse_args()
    return args
