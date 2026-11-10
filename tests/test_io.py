import pandas as pd

from edurec.recsys.io import save_metrics


def test_save_metrics_uses_one_row_and_the_dataset_folder(tmp_path):
    path = save_metrics({"ndcg@5": 0.5}, "coco", tmp_path)

    assert path == tmp_path / "coco" / "metrics.csv"
    assert pd.read_csv(path).to_dict(orient="records") == [{"ndcg@5": 0.5}]
