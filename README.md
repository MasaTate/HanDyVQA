# Code for HanDyVQA Benchmark
We provide code that help reproduce:
- Zero-shot evaluation results (Qwen2.5-VL-72B) in Section 4.1
- Integration of HOI cues in Section 4.3

## Zero-shot Evaluation
To reduce computational requirements, we include precomputed model outputs from Qwen2.5-VL-72B along with the prompts passed to the model under `./zero_shot_eval/inferred_data/`.

Running the following code will produce per-category results.
```
cd zero_shot_eval
python eval_vqa_valid.py
```
Output:
```
                    exp_id    action   process  location    state    parts   objects
0  Qwen2_5_70B_f16_480x854  0.773024  0.729829  0.614056  0.71134  0.61217  0.734981
```

## Models for integrating HOI cues
We provide model code that integrate HOI cues in `./hoi_cue_integration/model.py`.
```
cd hoi_cue_integration
python model.py
```
Output:
```
========Input Features========
video_feats: torch.Size([1, 512])
left_hand_feats: torch.Size([1, 16, 21, 3])
right_hand_feats: torch.Size([1, 16, 21, 3])
left_bbox_feats: torch.Size([1, 16, 8, 4])
right_bbox_feats: torch.Size([1, 16, 8, 4])
left_obj_feats: torch.Size([1, 16, 8, 768])
right_obj_feats: torch.Size([1, 16, 8, 768])

========Integrated Feature========
Integrated Feature: torch.Size([1, 512])

========Simialrity with Text Feature========
Similarity matrix: tensor([[-0.0336]], grad_fn=<MmBackward0>)
```