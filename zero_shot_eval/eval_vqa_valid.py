import os
import pandas as pd
import numpy as np
import json
import sklearn.metrics as skm
import glob
import warnings
import re


def normalize_answer(answer):
    """
    Normalize an answer to handle different formatting and case issues.
    """
    answer = (
        answer.strip().lower()
    )  # Remove leading/trailing spaces and convert to lowercase
    answer = answer.replace(".", "")  # Remove periods
    return answer


def text_to_indices(predicted: str) -> list:
    """
    Extract answer indices from the predicted text.
    Supports both (A), (B), ... and A), B) formats.

    Parameters:
    -----------
    predicted : str
        Model's predicted answer (entire text).

    Returns:
    --------
    list
        A list of indices corresponding to answer choices (0 for A, 1 for B, ...).
    """
    predicted_norm = normalize_answer(predicted)

    # Match both (a), (b), ..., and a), b), ... formats
    pattern = r"\(?([a-z])\)"
    found_options = re.findall(pattern, predicted_norm)

    # Get unique choices in sorted order
    unique_options = sorted(set(found_options))

    # Convert letters to index (A=0, B=1, ... Z=25)
    predicted_indices = [ord(letter) - ord("a") for letter in unique_options]

    return predicted_indices


def get_metrics(qtype, files, count_noanswer=True):
    if qtype != "objects":
        correct_list = []
        valid_count = 0

        for file in files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)

                data["inferred_indices"] = text_to_indices(data["inference"]["output"])

                correct_idx = data["correct_idx"]
                assert len(correct_idx) == 1
                correct_idx = correct_idx[0]

                if "inferred_indices" in data:
                    inferred_idx_list = data["inferred_indices"]
                    if len(inferred_idx_list) == 0:
                        if count_noanswer:
                            correct_list.append(0)
                        else:
                            continue
                    else:
                        valid_count += 1
                        inferred_idx = inferred_idx_list[0]
                        correct_list.append(1 if correct_idx == inferred_idx else 0)

                else:
                    scores = data["inferred_scores"]
                    if len(scores) > 0 and isinstance(scores[0], list):
                        scores = scores[0]
                    inferred_idx = scores.index(max(scores))
                    valid_count += 1
                    correct_list.append(1 if correct_idx == inferred_idx else 0)

            except Exception as e:
                warnings.warn(f"Error in {file}: {e}")

        acc = np.mean(correct_list) if correct_list else 0.0
        if not count_noanswer:
            valid_ratio = valid_count / len(files) if files else 0.0
            return "Acc", acc, valid_count, valid_ratio
        return "Acc", acc

    else:
        # object タイプの場合
        all_gt = []
        all_scores = []
        valid_count = 0

        for file in files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)

                data["inferred_indices"] = text_to_indices(data["inference"]["output"])
                total_option = len(data["options"])
                correct_indices = np.array(data["correct_idx"])
                gt_onehot = np.zeros(total_option)
                gt_onehot[correct_indices] = 1

                if "inferred_scores" in data:
                    infer_prob = np.array(data["inferred_scores"])
                    if infer_prob.ndim == 2:
                        infer_prob = infer_prob[0]
                    valid = True
                else:
                    infer_prob = np.zeros(total_option)
                    valid = False
                    if "inferred_indices" in data:
                        idxs = [i for i in data["inferred_indices"] if i < total_option]
                        if idxs:
                            infer_prob[idxs] = 1
                            valid = True

                if valid:
                    valid_count += 1
                else:
                    if not count_noanswer:
                        continue

                # 平坦化してスコアと GT を蓄積
                for i in range(total_option):
                    all_gt.append(gt_onehot[i])
                    all_scores.append(infer_prob[i])

            except Exception as e:
                warnings.warn(f"Error in {file}: {e}")

        ap = skm.average_precision_score(all_gt, all_scores) if all_gt else 0.0
        if not count_noanswer:
            valid_ratio = valid_count / len(files) if files else 0.0
            return "AP", ap, valid_count, valid_ratio
        return "AP", ap


infer_root = "./inferred_data"
out_dir = "./"
qtype_list = ["action", "process", "location", "state", "parts", "objects"]


overall_valid_data = []

metrics_by_category = []


exp_list = os.listdir(infer_root)
exp_list = sorted(exp_list)
for exp_id in exp_list:
    total_valid = 0
    total_files = 0
    exp_metric = {"exp_id": exp_id}
    print(f"Evaluating {exp_id}")
    for qtype in qtype_list:
        print(f"Computing {qtype} category...")
        files = sorted(glob.glob(f"{infer_root}/{exp_id}/{qtype}/*.json"))
        num_files = len(files)
        total_files += num_files

        try:
            metric_name, metric_value, valid_count, valid_ratio = get_metrics(
                qtype, files, count_noanswer=False
            )
        except Exception as e:
            warnings.warn(f"Error in processing {exp_id} {qtype}: {e}")
            metric_value = np.nan
            valid_count = 0

        total_valid += valid_count
        exp_metric[qtype] = metric_value

    overall_valid_ratio = total_valid / total_files if total_files > 0 else 0
    overall_valid_data.append(
        {
            "exp_id": exp_id,
            "total_files": total_files,
            "total_valid": total_valid,
            "valid_ratio": overall_valid_ratio,
        }
    )
    metrics_by_category.append(exp_metric)
    print(f"{exp_id}: {total_valid} / {total_files} = {overall_valid_ratio}")


df_metrics = pd.DataFrame(metrics_by_category)
df_metrics = df_metrics[["exp_id"] + qtype_list]
base_save_name = f"metrics_by_category"
save_name = f"{base_save_name}.csv"

print(df_metrics)

save_path_metrics = os.path.join(out_dir, save_name)
df_metrics.to_csv(save_path_metrics, index=False)
print("Metrics by category saved to:", save_path_metrics)
