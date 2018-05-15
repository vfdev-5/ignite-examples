from pathlib import Path
from argparse import ArgumentParser

import pandas as pd


if __name__ == "__main__":

    parser = ArgumentParser("Update test predictions with leaked data")
    parser.add_argument("submission_csv", type=str, help="Path to a submission")
    parser.add_argument("test_with_labels_csv", default="../notebooks/test_with_labels.csv",
                        type=str, help="Test with labels csv")
    parser.add_argument("output_folder", type=str,
                        help="Output folder where to saved `fixed_submission_csv` file")

    args = parser.parse_args()
    submission_filepath = Path(args.submission_csv)
    assert submission_filepath.exists(), \
        "Submission file is not found at {}".format(submission_filepath.as_posix())

    test_with_labels_filepath = Path(args.test_with_labels_csv)
    assert test_with_labels_filepath.exists(), \
        "Test with labels csv file is not found at {}".format(test_with_labels_filepath.as_posix())

    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    submission_df = pd.read_csv(submission_filepath)
    test_with_labels_df = pd.read_csv(test_with_labels_filepath).sort_values('id')

    new_submission_df = submission_df.copy()

    mask_known_labels = test_with_labels_df['label'] >= 0
    mask = new_submission_df['id'].isin(test_with_labels_df[mask_known_labels]['id'])
    new_submission_df.loc[mask, 'predicted'] = test_with_labels_df[mask_known_labels]['label'].values

    new_submission_filepath = output_folder / 'fixed_{}'.format(submission_filepath.name)
    new_submission_df.to_csv(new_submission_filepath.as_posix(), index=False)
