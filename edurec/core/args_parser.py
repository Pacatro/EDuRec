from argparse import ArgumentParser
from . import config


def build_parser() -> ArgumentParser:
    """
    Builds and returns an argument parser for the UcoRecSys command-line tool.

    The parser supports mutually exclusive execution modes (inference, training, evaluation, Surprise evaluation, and statistical tests),
    as well as grouped arguments for common, training, evaluation, and miscellaneous options.

    Returns:
        ArgumentParser: Configured argument parser for the CLI.
    """
    parser = ArgumentParser(
        prog="ucorecsys",
        description="Tool to train, evaluate, or perform inference with a recommendation model.",
    )

    # 1) Mutually exclusive group for execution mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-i",
        "--inference",
        metavar="MODEL_PATH",
        help="Run inference on a trained model (provide path to model file).",
    )
    mode_group.add_argument(
        "-t", "--train", action="store_true", help="Train the model."
    )
    mode_group.add_argument(
        "-e", "--eval", action="store_true", help="Evaluate the model."
    )
    mode_group.add_argument(
        "-s", "--surprise", action="store_true", help="Run Surprise evaluation."
    )
    mode_group.add_argument(
        "-st",
        "--stats_test",
        action="store_true",
        default=config.STATS_TEST,
        help=f"Run stats test (default: {config.STATS_TEST}).",
    )
    # 2) Common arguments (used in both training and evaluation)
    common_group = parser.add_argument_group("Common Options")
    common_group.add_argument(
        "-ds",
        "--dataset",
        choices=config.DATASETS_CHOICES,
        default=config.DATASET,
        help=f"Dataset to use (default: {config.DATASET}).",
    )
    common_group.add_argument(
        "-cv",
        "--cvtype",
        choices=config.CV_TYPES_CHOICES,
        default=config.CV_TYPE,
        help=f"Cross-validation type (default: {config.CV_TYPE}).",
    )
    common_group.add_argument(
        "--top_k",
        type=int,
        default=config.TOP_K,
        help=f"Top-k value for ranking metrics (default: {config.TOP_K}).",
    )

    # 3) Training-specific arguments
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"Training epochs (default: {config.EPOCHS}).",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Training batch size (default: {config.BATCH_SIZE}).",
    )
    train_group.add_argument(
        "--output_model",
        metavar="MODEL_OUT",
        default=config.OUTPUT_MODEL_PATH,
        help=f"Path to save trained model (default: {config.OUTPUT_MODEL_PATH}).",
    )
    train_group.add_argument(
        "-lr",
        type=float,
        default=config.LR,
        help=f"Learning rate (default: {config.LR}).",
    )

    # 4) Evaluation-specific arguments
    eval_group = parser.add_argument_group("Evaluation Options")
    # eval_group.add_argument(
    #     "--balance",
    #     action="store_true",
    #     default=config.BALANCE,
    #     help="Balance train/validation datasets.",
    # )
    eval_group.add_argument(
        "-k",
        "--k_splits",
        type=int,
        default=config.K,
        help=f"Number of CV splits (default: {config.K}).",
    )
    eval_group.add_argument(
        "--seeds",
        type=list,
        default=config.SEEDS,
        help=f"Random seeds (default: {config.SEEDS}).",
    )

    # 5) Miscellaneous options
    misc_group = parser.add_argument_group("Miscellaneous Options")
    misc_group.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Generate plots.",
        default=config.PLOT,
    )
    misc_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    return parser
