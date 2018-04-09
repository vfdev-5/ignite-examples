from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

from joblib import Parallel, delayed


def task(fp):
    img = Image.open(fp)
    resized_img = img.resize((dim, dim), interpolation)
    resized_img.save((output_path / fp.name).as_posix())


if __name__ == "__main__":

    parser = ArgumentParser("Resize dataset")
    parser.add_argument("dataset_path", type=str, help="Path to a dataset")
    parser.add_argument("dim", type=int, help="Output image dimension")
    parser.add_argument("output_path", type=str, help="Output path")
    parser.add_argument("--interpolation", type=int, default=3,
                        help="interpolation: 0, 1, 2, 3, 4, 5 <-> PIL.Image.NEAREST ...")
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    dim = args.dim
    interpolation = args.interpolation
    output_path = Path(args.output_path)
    assert dataset_path.exists()
    assert dim > 0
    if not output_path.exists():
        output_path.mkdir(parents=True)

    files = dataset_path.glob("*.png")

    with Parallel(n_jobs=10) as parallel:
        parallel(delayed(task)(f) for f in tqdm(files))
