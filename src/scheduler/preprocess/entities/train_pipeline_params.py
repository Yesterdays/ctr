import yaml
import logging
import sys

from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

from entities.feature_params import FeatureParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


PATH = "../configs/train_config.yaml"


@dataclass()
class TrainingPipelineParams:
    output_model_path: str
    output_transformer_path: str
    output_ctr_transformer_path: str
    metric_path: str
    feature_params: FeatureParams
    input_data_path: str = field(default="../data/raw/sampled_train_50k.csv")
    input_preprocessed_data_path: str = field(
        default="../data/raw/sampled_preprocessed_train_50k.csv"
    )


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)
        schema = TrainingPipelineParamsSchema().load(config_dict)
        logger.info("Check schema: %s", schema)
        return schema
