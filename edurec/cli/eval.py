# from typing import Annotated
#
# import typer
#
# from .. import settings
# from ..datasets import DatasetName
# from ..evaluation import eval_ghost, evaluate_recbole_models
#
# app = typer.Typer(no_args_is_help=True)
#
#
# @app.command(name="eval", help="Evaluate the GHOST model.")
# def eval_command(
#     dataset: Annotated[
#         DatasetName,
#         typer.Option("--dataset", "-d", help="Dataset to use"),
#     ] = DatasetName.MARS,
#     batch_size: Annotated[
#         int, typer.Option("--batch_size", "-b", help="Batch size")
#     ] = settings.RANKER_BATCH_SIZE,
#     remove_sparse: Annotated[
#         bool,
#         typer.Option(
#             "--remove_sparse",
#             "-R",
#             help="Remove users with less than MIN_INTERACTIONS interactions",
#         ),
#     ] = settings.REMOVE_SPARSE,
#     min_interactions: Annotated[
#         int,
#         typer.Option(
#             "--min_interactions",
#             "-I",
#             help="Minimum number of interactions per user",
#         ),
#     ] = settings.MIN_INTERACTIONS,
# ) -> None:
#     metrics = eval_ghost(
#         dataset=dataset,
#         batch_size=batch_size,
#         val_size=settings.VAL_RATIO,
#         test_size=settings.TEST_RATIO,
#         use_processed_data=settings.SAVE_DATA,
#         remove_sparse=remove_sparse,
#         min_interactions=min_interactions,
#         top_ks=[5, 10, 20],
#     )
#
#     if settings.state["verbose"]:
#         print(f"\n[EVAL] Evaluation results for dataset: {dataset.value}")
#         print(metrics.to_string())
#     else:
#         print(metrics.to_string())
#
#
# @app.command(name="sota", help="Evaluate RecBole models with cross-validation.")
# def eval_sota_command(
#     dataset: Annotated[
#         DatasetName,
#         typer.Option("--dataset", "-d", help="Dataset to use"),
#     ] = DatasetName.MARS,
#     epochs: Annotated[
#         int,
#         typer.Option("--epochs", "-e", help="RecBole training epochs per fold"),
#     ] = settings.EPOCHS,
#     batch_size: Annotated[
#         int,
#         typer.Option("--batch_size", "-b", help="Train and eval batch size"),
#     ] = settings.RANKER_BATCH_SIZE,
#     n_splits: Annotated[
#         int,
#         typer.Option("--n_splits", "-n", help="Number of CV folds"),
#     ] = 5,
#     val_size: Annotated[
#         float,
#         typer.Option("--val_size", "-v", help="Validation ratio inside each fold"),
#     ] = settings.VAL_RATIO,
#     patience: Annotated[
#         int,
#         typer.Option("--patience", "-p", help="Early stopping patience"),
#     ] = settings.RANKER_PATIENCE,
#     top_ks: Annotated[
#         list[int],
#         typer.Option("--top_k", "-k", help="Top-k values to evaluate"),
#     ] = [5, 10, 20],
#     models: Annotated[
#         list[str],
#         typer.Option(
#             "--model",
#             "-m",
#             help="RecBole models to evaluate. Defaults depend on the dataset.",
#         ),
#     ] = [],
#     remove_sparse: Annotated[
#         bool,
#         typer.Option(
#             "--remove_sparse",
#             "-R",
#             help="Remove users and items below the interaction threshold",
#         ),
#     ] = settings.REMOVE_SPARSE,
#     min_interactions: Annotated[
#         int,
#         typer.Option(
#             "--min_interactions",
#             "-I",
#             help="Minimum interactions per user/item after filtering",
#         ),
#     ] = settings.MIN_INTERACTIONS,
#     results_folder: Annotated[
#         str,
#         typer.Option(
#             "--results-folder",
#             help="Folder where RecBole evaluation results are stored",
#         ),
#     ] = settings.RESULTS_FOLDER,
# ) -> None:
#     summary, _ = evaluate_recbole_models(
#         dataset=dataset,
#         models=models or None,
#         n_splits=n_splits,
#         val_size=val_size,
#         batch_size=batch_size,
#         epochs=epochs,
#         patience=patience,
#         top_ks=top_ks,
#         results_folder=results_folder,
#         remove_sparse=remove_sparse,
#         min_interactions=min_interactions,
#     )
#     print(summary.to_string(index=False))
