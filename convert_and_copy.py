import os
import json
import pandas as pd
import glob
from tqdm import tqdm


def convert_dict(input_dict):
    output_dict = {
        "question": input_dict["question"],
        "options": input_dict["options"],
        "inference": input_dict["chat_0"],
        "correct_idx": input_dict["correct_idx"],
    }
    return output_dict


def read_json(p):
    with open(p, "r") as f:
        data = json.load(f)

    return data


def write_json(p, d):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(d, f, indent=4)


root = "/home/ace15208mc/Hand_Object_VQA/baselines_v2/results/mostly_debiased_data_0513/Qwen2_5/qwen2_5_72B_f16_480x854"
out_root = "/home/ace15208mc/Hand_Object_VQA/NeurIPS_Code_Submission/zero_shot_eval/inferred_data/Qwen2_5_70B_f16_480x854"
mapping_file = "/home/ace15208mc/Hand_Object_VQA/ego4d_hoi_collection/create_dataset_for_huggingface/qid_mapping.json"
split_file = "/home/ace15208mc/Hand_Object_VQA/ego4d_hoi_collection/create_dataset_for_huggingface/split_list_debiased.csv"
df = pd.read_csv(split_file)
df_split = df[df["split"] == "test"]
old_qid_list = df_split["old_qid"].to_list()

category_mapping = {
    "area": "parts",
    "how": "process",
    "object": "objects",
    "state": "state",
    "what": "action",
    "where": "location",
}

qid_map = read_json(mapping_file)
qid_map = {v: k for k, v in qid_map.items()}

for org_cat, new_cat in category_mapping.items():
    files = glob.glob(os.path.join(root, org_cat, "*.json"))
    for file in tqdm(files):
        old_qid = os.path.basename(file).split(".")[0]
        if not old_qid in old_qid_list:
            continue

        new_qid = qid_map[old_qid]
        infer_dict = read_json(file)
        out_dict = convert_dict(infer_dict)

        out_path = os.path.join(out_root, new_cat, new_qid + ".json")

        write_json(out_path, out_dict)
