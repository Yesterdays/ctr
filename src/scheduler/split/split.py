import os
import logging
import sys
import click
from make_dataset.make_dataset import read_data, split_train_val_data

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--test-size")
def split(input_dir: str, output_dir: str, test_size: str):
    logger.info(f"input_dir: {input_dir}")
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"test_size: {test_size}")

    os.makedirs(output_dir, exist_ok=True)

    df = read_data(os.path.join(input_dir, "processed.csv"))

    train, test = split_train_val_data(df, test_size=float(test_size))

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)


if __name__ == "__main__":
    split()
