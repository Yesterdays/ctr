import json
import os
import logging
import sys
import pandas as pd
import click

from entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from models.model_fit_predict import (
    train_model,
    predict_model,
    evaluate_model,
    serialize_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--config")
def train(input_dir: str, output_dir: str, config: str):
    logger.info(f"input_dir: {input_dir}")
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"config: {config}")

    os.makedirs(output_dir, exist_ok=True)

    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(
        config
    )

    dftrain = pd.read_csv(os.path.join(input_dir, "train.csv"))
    dfval = pd.read_csv(os.path.join(input_dir, "test.csv"))

    train_features = dftrain.drop(["click"], axis=1)
    train_target = dftrain["click"]
    val_features = dfval.drop(["click"], axis=1)
    val_target = dfval["click"]

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    predicted_proba, preds = predict_model(model, val_features)
    metrics = evaluate_model(predicted_proba, preds, val_target)
    logger.info(f"preds/ targets shapes:  {(preds.shape, val_target.shape)}")

    with open(os.path.join(output_dir, "metrics.json"), "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Metric is {metrics}")

    serialize_model(model, os.path.join(output_dir, "catclf_saved.pkl"))


if __name__ == "__main__":
    train()
