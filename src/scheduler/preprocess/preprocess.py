import os
import logging
import sys
import pandas as pd
import click
from datetime import datetime

from entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)

from features.build_transformer import (
    build_ctr_transformer,
    build_transformer,
    extract_target,
    process_count_features,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--config")
def preprocess(input_dir: str, output_dir: str, config: str):
    logger.info(f"input_dir: {input_dir}")
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"config: {config}")

    os.makedirs(output_dir, exist_ok=True)

    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(
        config
    )

    data = pd.read_csv(os.path.join(input_dir, "sampled_train_50k.csv"))
    data["hour"] = data.hour.apply(lambda val: datetime.strptime(str(val), "%y%m%d%H"))

    transformer = build_transformer()
    processed_data = process_count_features(
        transformer, data, training_pipeline_params.feature_params
    )

    logger.info(
        f"processed_data:  {processed_data.shape} \n {processed_data.info()} "
        f"\n {processed_data.nunique()} \n {processed_data[training_pipeline_params.feature_params.count_features]}"
    )

    ctr_transformer = build_ctr_transformer(training_pipeline_params.feature_params)
    ctr_transformer.fit(processed_data)

    features = ctr_transformer.transform(processed_data)
    target = pd.DataFrame(
        extract_target(processed_data, training_pipeline_params.feature_params),
        columns=["click"],
    )

    data = pd.concat([features, target], axis=1)
    data.to_csv(os.path.join(output_dir, "processed.csv"), index=False)


if __name__ == "__main__":
    preprocess()
