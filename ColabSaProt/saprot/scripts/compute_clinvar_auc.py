import os
import argparse
import pandas as pd
import yaml

from glob import glob
from sklearn import metrics
from easydict import EasyDict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="running configurations", type=str, required=True)
    return parser.parse_args()


def main(args):
    with open(args.config, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))

    output_dir = "output/ClinVar"
    # file_name = "esm2_t33_650M_UR50D_foldseek_plddt70_iter2900448_mask70"

    label_name = "ClinVar_labels.csv"
    label_path = os.path.join(output_dir, label_name)

    list_variables_to_keep = ["protein_name", "mutations", "evol_indices"]
    all_evol_indices = pd.concat(
        [
            pd.read_csv(path, low_memory=False)[list_variables_to_keep]
            for path in glob(f"{config.model.kwargs.log_dir}/*.csv")
        ],
        ignore_index=True,
    )
    all_evol_indices = all_evol_indices.drop_duplicates()

    labels_dataset = pd.read_csv(label_path, low_memory=False)
    all_evol_indices_with_labels = pd.merge(
        all_evol_indices,
        labels_dataset[["protein_name", "mutations", "ClinVar_labels"]],
        how="right",
        on=["protein_name", "mutations"],
    )

    all_evol_indices_with_labels = all_evol_indices_with_labels[
        all_evol_indices_with_labels.ClinVar_labels != 0.5
        ]

    fpr, tpr, threshold = metrics.roc_curve(
        all_evol_indices_with_labels["ClinVar_labels"], all_evol_indices_with_labels["evol_indices"]
    )
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)


if __name__ == '__main__':
    main(get_args())