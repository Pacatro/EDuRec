from typing import Annotated
import numpy as np
import pandas as pd
import lightning as L
import typer

from ..core.datamodule import ELearningDataModule
from ..core.engine import UcoRecSys
from ..core.model import NeuralHybrid
from ..core import config
from ..core.datasets import DatasetName, load_data

app = typer.Typer(no_args_is_help=True)


@app.command(help="Make predictions using a trained model.")
def predict(
    model_path: Annotated[
        str, typer.Option("--model_path", "-m", help="Path to trained model")
    ],
    dataset: Annotated[
        DatasetName,
        typer.Option("--dataset", "-d", help="Dataset to use"),
    ],
    target: Annotated[
        str, typer.Option("--target", "-t", help="Target column")
    ] = config.TARGET_COL,
    batch_size: Annotated[
        int, typer.Option("--batch_size", "-b", help="Batch size")
    ] = config.BATCH_SIZE,
    top_k: Annotated[
        int, typer.Option("--top_k", "-k", help="Top-k value")
    ] = config.TOP_K,
    balance: Annotated[
        bool, typer.Option("--balance", "-B", help="Balance dataset")
    ] = config.BALANCE,
):
    predict_df = pd.read_csv("./data/predict_df.csv")

    if config.state["verbose"]:
        print(predict_df)

    df = load_data(dataset)

    dm = ELearningDataModule(
        df,
        predict_df=predict_df.drop(columns=["rating"]),
        target=target,
        batch_size=batch_size,
        balance=balance,
    )

    ranking = recommend(dm, top_k, model_path)

    print(ranking)


def generate_new_interactions(
    df: pd.DataFrame,
    samples: int = 50,
    interactions_per_user: int = 20,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    users = df[user_col].unique()
    items = df[item_col].unique()

    if samples == 0:
        return df

    rng = np.random.default_rng(seed=42)

    num_users_needed = max(1, samples // interactions_per_user)

    if num_users_needed > len(users):
        selected_users = rng.choice(users, size=num_users_needed, replace=True)
    else:
        selected_users = rng.choice(users, size=num_users_needed, replace=False)

    candidates_list = []

    for user in selected_users:
        user_items = rng.choice(items, size=interactions_per_user, replace=True)

        user_interactions = pd.DataFrame(
            {user_col: [user] * interactions_per_user, item_col: user_items}
        )

        candidates_list.append(user_interactions)

    candidates = pd.concat(candidates_list, ignore_index=True)

    positives = df[[user_col, item_col]].drop_duplicates()
    merged = candidates.merge(
        positives, on=[user_col, item_col], how="left", indicator=True
    )

    negatives = (
        merged[merged["_merge"] == "left_only"]
        .drop(columns="_merge")
        .drop_duplicates()
        .head(samples)  # Limitar al número exacto solicitado
        .reset_index(drop=True)
    )

    item_types = ["tutorial", "use_case", "webcast"]
    difficulties = ["Beginner", "Intermediate", "Advanced", "Undefined"]

    negatives["item_type"] = rng.choice(item_types, size=len(negatives))
    negatives["difficulty"] = rng.choice(difficulties, size=len(negatives))
    negatives["nb_views"] = rng.integers(0, 2000, size=len(negatives)).astype(float)
    negatives["watch_percentage"] = rng.integers(0, 101, size=len(negatives)).astype(
        float
    )

    return negatives


def extrac_user_interactions(
    df: pd.DataFrame, user_id: int, n_interactions: int = 10
) -> pd.DataFrame:
    user_interactions = df[df["user_id"] == user_id]
    return user_interactions.sample(n=n_interactions)


def recommend(dm: ELearningDataModule, top_k: int, model_path: str) -> pd.DataFrame:
    model = NeuralHybrid(
        n_users=dm.num_users,
        n_items=dm.num_items,
        cont_features=dm.cont_features,
        cat_cardinalities=dm.cat_cardinalities,
    )

    recsys = UcoRecSys.load_from_checkpoint(
        model_path, model=model, encoders=dm.encoders
    )

    dm.setup("predict")
    trainer = L.Trainer()
    predictions = trainer.predict(recsys, datamodule=dm)

    assert predictions is not None, "No predictions were made."

    preds = pd.DataFrame(predictions[0]).sort_values(by=["user_id"])

    return (
        preds.sort_values(by=["prediction"], ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
