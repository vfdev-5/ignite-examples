from pathlib import Path
from argparse import ArgumentParser
from functools import partial

from tqdm import tqdm
from PIL import Image


from joblib import Parallel, delayed


def task(fp, output_path, dim, interpolation):
    # check if output exists:
    if (output_path / fp.name).exists():
        return
    try:
        img = Image.open(fp)
        resized_img = img.resize((dim, dim), interpolation)
        resized_img.save((output_path / fp.name).as_posix())
    except Exception as e:
        print("Problem with file: ", fp)
        print(e)


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

    files = list(dataset_path.glob("*.png"))

    predefined_task = partial(task, output_path=output_path, dim=dim, interpolation=interpolation)

    with Parallel(n_jobs=15) as parallel:
        parallel(delayed(predefined_task)(f) for f in tqdm(files))
