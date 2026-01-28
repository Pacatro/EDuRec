# from typing import Annotated
# import pandas as pd
# import lightning as L
# import typer
# from pathlib import Path
#
# from ..data.datamodule import ELearningDataModule
# from ..training.engine import RecSys
# from ..training.model import EDuRecV1
# from .. import config
# from ..data.datasets import DatasetName, load_data
# from ..training.model_io import get_last_model
#
# app = typer.Typer(no_args_is_help=True)
#
#
# @app.command(help="Make predictions using a trained model.")
# def predict(
#     dataset: Annotated[
#         DatasetName,
#         typer.Option("--dataset", "-d", help="Dataset to use"),
#     ],
#     model_path: Annotated[
#         str | None,
#         typer.Option(
#             "--model_path",
#             "-m",
#             help="Path to trained model, if not provided, the most recent model will be used.",
#         ),
#     ] = None,
#     target: Annotated[
#         str, typer.Option("--target", "-t", help="Target column")
#     ] = config.RATING_COL,
#     batch_size: Annotated[
#         int, typer.Option("--batch_size", "-b", help="Batch size")
#     ] = config.BATCH_SIZE,
#     top_k: Annotated[
#         int, typer.Option("--top_k", "-k", help="Top-k value")
#     ] = config.TOP_K,
#     # balance: Annotated[
#     #     bool, typer.Option("--balance", "-B", help="Balance dataset")
#     # ] = config.BALANCE,
# ):
#     if model_path is None:
#         model_path = get_last_model(config.training_FOLDER)
#         print(f"No model path provided. Using the most recent model -> {model_path}")
#
#     predict_df = pd.read_csv("./data/predict_df.csv")
#
#     if config.state["verbose"]:
#         print(predict_df)
#
#     df = load_data(dataset)
#
#     dm = ELearningDataModule(
#         df,
#         predict_df=predict_df.drop(columns=["rating"]),
#         target=target,
#         batch_size=batch_size,
#         # balance=balance,
#     )
#
#     ranking = recommend(dm, top_k, model_path)
#
#     print(ranking)
#
#
# def recommend(
#     dm: ELearningDataModule, top_k: int, model_path: str | Path
# ) -> pd.DataFrame:
#     model = EDuRecV1(
#         n_users=dm.num_users,
#         n_items=dm.num_items,
#         cont_features=dm.cont_features,
#         cat_cardinalities=dm.cat_cardinalities,
#     )
#
#     recsys = RecSys.load_from_checkpoint(model_path, model=model, encoders=dm.encoders)
#
#     dm.setup("predict")
#     trainer = L.Trainer()
#     predictions = trainer.predict(recsys, datamodule=dm)
#
#     assert predictions is not None, "No predictions were made."
#
#     preds = pd.DataFrame(predictions[0]).sort_values(by=["user_id"])
#
#     return (
#         preds.sort_values(by=["prediction"], ascending=False)
#         .head(top_k)
#         .reset_index(drop=True)
#     )
