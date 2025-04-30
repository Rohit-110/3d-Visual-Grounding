# EaSe: Evolving Symbolic 3D Visual Grounder with Weakly Supervised Reflection

[![Paper](https://img.shields.io/badge/arXiv-2502.01401-b31b1b.svg)](https://arxiv.org/abs/2502.01401)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Official repository for EaSe 3D Visual Grounding - a training-free, LLM/VLM-aided framework for 3D Visual Grounding (3DVG).

## 📖 About

EaSe (Evolving Symbolic 3D Visual Grounder) is a **training-free**, **LLM/VLM-aided** framework for **3D Visual Grounding (3DVG)**. It generates symbolic reasoning code to match natural language queries with 3D objects in complex scenes, using minimal supervision.

## ⚙️ Environment Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key
```

## 📦 Data Preparation

The `data/` directory should be organized as follows:

```
data
├── frames
│   ├── color
│   │   ├── 0.png
│   │   ├── 20.png
│   │   └── ...
├── referit3d
│   ├── annotations
│   ├── scan_data
├── symoblic_exp
│   ├── nr3d.jsonl
│   ├── scanrefer.json
├── test_data
│   ├── above
│   ├── behind
│   ├── ...
├── seg
├── nr3d_masks
├── scanrefer_masks
├── feats_3d.pkl
├── tables.pkl
```

### Data Downloads

| Resource | Description | Link |
|----------|-------------|------|
| `frames` | RGB images of the scenes | [download_link](https://drive.google.com/file/d/1VVnj3DAcOWqZhB6Vi0gWdzA9gTKQwrej/view) |
| `referit3d` | Processed ReferIt3D dataset from vil3dref | [Dropbox Link](https://www.dropbox.com/s/n0m5bpfvea1fg7w/referit3d.tar.gz?dl=0) |
| `seg` | Segmentation results of 3D point clouds for ScanRefer | [download_link](https://drive.google.com/file/d/1VRW_ew9Hwmsg-DRf22l_MHgFVB0UU1K0/view) |
| `nr3d_masks` | 2D GT object masks | [download_link](https://drive.google.com/file/d/1Z0pRv_UV7P_aNHsYHVkUz-lLaMCU2C9i/view) |
| `scanrefer_masks` | 2D predicted object masks | [download_link](https://drive.google.com/file/d/1v4nqJSOFVh7MAmyDo92Xze01U00yr1bB/view) |
| `feats_3d.pkl` | Predicted object labels for Nr3D from ZSVG3D | [SharePoint Link](https://cuhko365-my.sharepoint.com/:u:/g/personal/221019046_link_cuhk_edu_cn/ERMP88uTVCNLhzofKub7MsMBvaRAFXVr5abbQUjRYyYDiA?e=x6aKC9) |
| `tables.pkl` | Tables for code generation | [download_link](https://huggingface.co/datasets/yourusername/ease-dataset) |

## 🔧 (Optional) Relation Encoder Generation

Run the relation encoder optimization script:

```bash
python -m src.relation_encoders.run_optim
```

This will generate relation encoders for 6 relations: `left`, `right`, `between`, `corner`, `above`, `below`, `behind`.

After optimization completes, you'll get relation encoders and their accuracy metrics under `data/test_data/{relation_name}/trajs`. Select the best relation encoders for evaluation, or use the provided encoders in `src/relation_encoders`.

## ⚡ (Optional) Features Computation

```bash
python -m src.relation_encoders.compute_features \
    --dataset scanrefer \
    --output $OUTPUT_DIR \
    --label pred
```

- `--dataset`: Choose either `scanrefer` or `nr3d`
- `--label`: Choose either `gt` or `pred` (note: only `pred` label is supported for ScanRefer)

After running, features in `.pth` format will be generated in the `$OUTPUT_DIR` directory.

You can also download our pre-computed features:
- [Nr3D (pred label)](https://example.com/nr3d_pred)
- [Nr3D (gt label)](https://example.com/nr3d_gt)
- [ScanRefer](https://example.com/scanrefer)

## 🚀 Evaluation

### Nr3D Evaluation:

```bash
python -m src.eval.eval_nr3d \
    --features_path output/nr3d_features_per_scene_pred_label.pth \
    --top_k 5 \
    --threshold 0.9 \
    --label_type pred \
    --use_vlm 
```

### ScanRefer Evaluation:

```bash
python -m src.eval.eval_scanrefer \
    --features_path output/scanrefer_features_per_scene.pth \
    --top_k 5 \
    --threshold 0.1 \
    --use_vlm
```

Change `features_path` and `label_type` to evaluate with ground truth labels. Set `--use_vlm`, `--top_k` and threshold to use the VLM model for evaluation. Please refer to our paper for details on these parameters.

## 📄 Paper

**Title:** [Evolving Symbolic 3D Visual Grounder with Weakly Supervised Reflection](https://arxiv.org/abs/2502.01401)

**Authors:** Boyu Mi, Hanqing Wang, Tai Wang, Yilun Chen, Jiangmiao Pang (Shanghai AI Laboratory)

**Published:** February 2025

## 🙏 Acknowledgements

Thanks to the following repositories for their contributions:
* [ZSVG3D](https://github.com/CurryYuan/ZSVG3D)
* [ReferIt3D](https://github.com/referit3d/referit3d)
* [Vil3dref](https://github.com/cshizhe/vil3dref)

## 🔍 Awesome Concurrent Works

* [SeeGround](https://github.com/example/seeground)
* [CSVG](https://github.com/example/csvg)
* [3D-Visual-Grounding](https://github.com/example/3d-visual-grounding)

## 📝 License
