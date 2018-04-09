import sys
from pathlib import Path
from argparse import ArgumentParser

try:
    from image_dataset_viz import DatasetExporter
except ImportError:
    raise RuntimeError("image_dataset_viz is not found. "
                       "Install it using: pip install git+https://github.com/vfdev-5/ImageDatasetViz.git")

# Load common module
sys.path.insert(0, Path(__file__).absolute().parent.parent.as_posix())
from common.dataset import TrainvalFilesDataset, TestFilesDataset


if __name__ == "__main__":

    parser = ArgumentParser("Dataset viz")
    parser.add_argument("dataset_path", type=str, help="Path to a dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of datapoints to vizualize (default: None=all)")
    parser.add_argument("--test", action="store_true",
                        help="If dataset is test dataset")

    args = parser.parse_args()

    if args.test:
        dataset = TestFilesDataset(args.dataset_path)
        read_target_fn = None
        targets = None
    else:
        dataset = TrainvalFilesDataset(args.dataset_path)
        read_target_fn = lambda label: str(label)
        targets = dataset.labels

    de = DatasetExporter(n_cols=20, text_size=10, text_color=(255, 0, 255),
                         read_target_fn=read_target_fn,
                         img_id_fn=lambda f: Path(f).stem)
    de.export(dataset.images, targets, output_folder="../imaterialist_furniture_viz")
