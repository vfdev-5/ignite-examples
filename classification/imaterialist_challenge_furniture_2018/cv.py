from argparse import ArgumentParser
from pathlib import Path

from train import run


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_cv_folder", type=str,
                        help="Folder with configuration files")
    args = parser.parse_args()
    print("Run CV")

    path = Path(args.config_cv_folder)
    assert path.exists(), "Path '{}' is not found".format(path.as_posix())

    config_files = sorted(path.glob("*.py"))
    for config_file in config_files:
        if config_file.name == "__init__.py":
            continue
        try:
            print("\n\n----- run {} -----\n".format(config_file.as_posix()))
            run(config_file.as_posix())
        except Exception as e:
            print("\n\n !!! Run {} failed !!!\n".format(config_file.as_posix()))
            print("\n{}".format(e))
