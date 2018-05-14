from argparse import ArgumentParser

from predict import run


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_files", type=str, nargs="+", help="Configuration files. See examples in configs/")
    args = parser.parse_args()
    print("Run multiple predictions")
    for config_file in args.config_files:
        try:
            print("\n\n----- run {} -----\n".format(config_file))
            run(config_file)
        except Exception as e:
            print("\n\n !!! Run {} failed !!!\n".format(config_file))
            print("\n{}".format(e))
