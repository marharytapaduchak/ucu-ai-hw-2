---
title: Homework 2 -- Multiple Object Video Detection
author: Marharyta Paduchak
course: Artificial Intelligence · UCU
date: 2026-11-16
---

# UCU AI Homework Report Template

## 1. Methodology and motivation

### 1.1 Data preparation

The task is to detect eight object classes in VisDrone video frames: person, bicycle, car, van, truck, tricycle, bus, and motor. The dataset contains 56 training videos, 7 independent validation videos, and 48 test videos (3,811 frames).

Exploratory data analysis ([Notebook 01a](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/01a_EDA.ipynb)):

- **Class imbalance:** Cars account for approximately 537,000 training annotations, compared with only 9,900 for buses.
- **Small objects:** Approximately 24% of objects are smaller than 32 × 32 pixels. The median bounding box is 44 × 49 pixels.
- **Dense scenes:** Frames contain approximately 45 objects on average, with up to 180 in a single frame.
- **Ignored regions:** Approximately 75% of frames contain `iscrowd=1` regions, which require special treatment during evaluation.
- **Distribution shift:** Cars represent approximately 45% of training objects but 28% of validation objects, while the proportion of persons increases from 29% to 44%.

These led to higher input resolutions, smaller detection anchors, careful handling of ignored regions, and explicit evaluation of small-object performance.

The dataset provides COCO annotations and YOLO labels. Label consistency was checked on 30 frames containing 1,454 bounding boxes, and sample annotations were visualized. COCO category IDs (1–8) were explicitly mapped to model class IDs (0–7) where required. During the later high-resolution experiments, ignored regions were also masked in training images to reduce incorrect background supervision.

### 1.2 Validation strategy

To prevent data leakage, I used **three-fold GroupKFold cross-validation grouped by video**, rather than random frame-level splitting.

The initial experiments used every sixth training frame (4,062 images), with a fixed random seed of 42. Each video appeared exclusively in either the training or validation portion of a fold.

For final training, I used all 56 training videos with denser frame sampling (every third or fourth frame, depending on the experiment). The seven official validation videos remained excluded from training and provided an independent holdout of 951 sampled frames.

All architectures were compared using **COCOeval mAP@[0.50:0.95]**, AP50, and AP_small on consistent validation image IDs, with `iscrowd` annotations preserved. The holdout was additionally examined across two groups of videos to assess robustness to different scene densities.

### 1.3 Models and tuning strategy

I evaluated four required object-detection architectures:

| Detection family | Model | Motivation | Notebooks |
|---|---|---|---|
| Two-stage, anchor-based | Faster R-CNN ResNet50-FPN | Region proposals and FPN features for multi-scale detection | [02](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/02_Faster_RCNN_Baseline.ipynb), [06](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/06_Faster_RCNN_Tuning.ipynb) |
| One-stage, anchor-based | RetinaNet ResNet50-FPN v2 | Focal Loss for class imbalance and dense detection | [03](notebooks/03_RetinaNet_3Fold_Baseline.ipynb), [07](notebooks/07_RetinaNet_Tuning.ipynb) |
| One-stage, anchor-free | YOLO11 | Efficient inference and dense object localization | [01b](notebooks/01b_YOLO_Baseline.ipynb), [05](notebooks/05_YOLO_Tuning.ipynb) |
| Query-based | RT-DETR-L | Transformer-based detection with bipartite matching | [04](notebooks/04_RTDETR_Baseline.ipynb), [08](notebooks/08_RTDETR_Tuning_Kaggle.ipynb) |

The initial benchmarks used COCO-pretrained weights, six training epochs, batch size 2, and consistent video-level splits. I first tested individual changes on fold 0 and then evaluated promising configurations across all three folds.

The experiments investigated input resolution, optimizer choice, learning rate, and RetinaNet's Focal Loss gamma. Later experiments introduced **Faster R-CNN v2 and FCOS** ([Notebook 17](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/17_FasterRCNNv2_FCOS_Training_Kaggle.ipynb)), using multi-scale training, smaller anchors, and improved ignored-region handling.

For the final pipeline, I investigated multi-scale inference, horizontal-flip test-time augmentation, and **Weighted Boxes Fusion (WBF)**. Candidate model families were selected according to holdout mAP improvements, followed by tuning of family weights and fusion parameters.

Experiments were run on Kaggle and Google Colab NVIDIA T4 GPUs.


## 2. Results and analysis

### 2.1 Baselines and hyperparameter tuning

All four required model families were evaluated using three-fold video-level cross-validation and COCO mAP@[0.50:0.95].

| Model | Baseline mAP | Best configuration | Tuned mAP |
|---|---:|---|---:|
| YOLO11 | 0.1061 | YOLO11s, 960 px, AdamW, LR 5e-4 | **0.1744** |
| RetinaNet | 0.1498 | 768 px, SGD, LR 0.002 | 0.1699 |
| Faster R-CNN v1 | 0.1456 | 768 px, SGD, LR 0.002 | 0.1671 |
| RT-DETR-L | 0.1558 | Original LR 1e-4 | 0.1558 |

YOLO11s achieved the strongest cross-validation result. Increasing model capacity and resolution substantially improved its performance. Larger input sizes also improved RetinaNet and Faster R-CNN, consistent with the large proportion of small objects identified during EDA.

Several unsuccessful experiments helped narrow the search:

| Experiment | mAP | Comparison |
|---|---:|---|
| RetinaNet, Focal Loss gamma 1.5 | 0.1516 | 0.1699 with the selected configuration |
| RetinaNet, AdamW 2e-4 (fold 0) | 0.1584 | 0.1595 baseline fold 0 |
| RT-DETR, LR 2e-4 | 0.1525 | 0.1558 baseline |
| RT-DETR, LR 5e-5 (fold 0) | 0.1412 | 0.1596 baseline fold 0 |
| YOLO11s, SGD 0.01 (fold 0) | 0.1632 | 0.1682 with AdamW |

These results suggested that increasing image resolution was more reliable than changing optimizers or loss parameters. The comparison code and complete experiments are available in Notebooks 05–10.

### 2.2 Incremental leaderboard improvements

The following experiments progressively improved the submitted predictions. Holdout mAP was measured on 951 frames from the seven official validation videos.

| Experiment | Holdout mAP | Public LB | Notebook |
|---|---:|---:|---|
| Initial YOLO submission | — | 0.1761 | [01b](notebooks/01b_YOLO_Baseline.ipynb) |
| YOLO11s, 18 epochs | 0.1843 | 0.1873 | [09](notebooks/09_YOLO_Leaderboard_Improvement.ipynb) |
| RT-DETR + RetinaNet CV ensemble | 0.2029 | 0.2078 | [14](notebooks/14_RTDETR_RetinaNet_Ensemble_Kaggle.ipynb) |
| RT-DETR + final RetinaNet | 0.2091 | 0.2120 | [15b](notebooks/15b_Stronger_Multi_Model_Ensemble_Colab.ipynb) |
| Multi-scale ensemble v1 | 0.2627 | 0.2420 | [16](notebooks/16_Ensemble_v1_MultiScale_WBF.ipynb) |
| Ensemble v2 with Faster R-CNN v2 | 0.2930 | 0.2900 | [17–18](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/18_Ensemble_v2_with_FasterRCNNv2_FCOS.ipynb) |
| **Final ensemble v3** | **0.2962** | **0.2941** | [19](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/19_Final_Ensemble_v3_FasterRCNNv2_MultiScale.ipynb) |

An attempted final YOLO11s retraining run failed before training because the pretrained checkpoint download returned HTTP 504 ([Notebook 11](notebooks/11_Final_YOLO11s_Training_Kaggle_FAILED.ipynb)). This motivated exploring alternative available checkpoints and model ensembles rather than relying on one architecture.

### 2.3 Resolution, architecture, and fusion experiments

Testing larger inference resolutions without retraining improved YOLO11s from 0.1842 at 960 px to **0.1923 at 1280 px with horizontal flip**. RetinaNet improved from 0.1756 at 768 px to 0.1821 at 960 px. However, larger resolutions were not universally beneficial: YOLO11s dropped to 0.1835 at 1536 px, while RT-DETR decreased from 0.1537 at 640 px to 0.1318 at 960 px.

A stronger Faster R-CNN v2 was subsequently trained with multi-scale augmentation, smaller anchors, and masked ignored regions ([Notebook 17](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/17_FasterRCNNv2_FCOS_Training_Kaggle.ipynb)). It achieved **0.2542 holdout mAP**, compared with 0.1807 for the best earlier Faster R-CNN v1 fold. FCOS achieved 0.1972. Since several training settings changed simultaneously, the Faster R-CNN improvement cannot be attributed to architecture alone.

The final ensemble was built incrementally using Weighted Boxes Fusion:

| Addition | Holdout mAP |
|---|---:|
| Faster R-CNN v2, 1088 px + flip | 0.2593 |
| + Three YOLO11s folds at 1280 px | 0.2745 |
| + Faster R-CNN v2 at 1216 px | 0.2833 |
| + Three Faster R-CNN v1 folds | 0.2867 |
| + Faster R-CNN v2 at 960 px + flip | 0.2894 |
| + Family weight tuning | 0.2910 |
| + Fusion parameter tuning | **0.2962** |

The selected fusion used IoU 0.5, coverage power 1.0, and a maximum of 500 detections per frame. RetinaNet, RT-DETR, FCOS, and the standalone 18-epoch YOLO model were excluded because they did not improve the final combination.

Temporal rescoring using neighbouring video frames was also tested but reduced mAP by approximately 0.001–0.035, depending on the configuration. RetinaNet horizontal-flip augmentation reduced mAP by approximately 0.012. Both approaches were therefore disabled.

### 2.4 Final evaluation and conclusions

| Ensemble | Holdout mAP | Public LB | Private LB |
|---|---:|---:|---:|
| v2 | 0.2930 | 0.2900 | 0.2505 |
| **v3 (selected)** | **0.2962** | **0.2941** | **0.2531** |

The final model exceeded the organizer's public baseline of **0.22163** by approximately **0.0725 mAP**. It also improved AP_small from approximately 0.092 in the earlier ensemble to 0.147.

The main gains came from higher-resolution detection, the stronger Faster R-CNN v2 training configuration, and combining complementary predictions through WBF. In contrast, increasing resolution excessively, adding weaker models indiscriminately, and temporal rescoring were not beneficial.

The private leaderboard scores were approximately 0.04 lower than public scores for both final ensembles. This indicates that the seven-video holdout did not fully capture the hidden test distribution. Nevertheless, ensemble v3 consistently outperformed v2 on holdout, public, and private evaluations and was selected as the final solution.


## 3. Final model and run instructions

### 3.1 Selected Model

The final solution is **Ensemble v3** ([Notebook 19](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/19_Final_Ensemble_v3_FasterRCNNv2_MultiScale.ipynb)), combining nine prediction sources through class-aware Weighted Boxes Fusion (WBF).

| Model / inference configuration | Ensemble weight |
|---|---:|
| Faster R-CNN v2, 960 px + horizontal flip | 1.0 |
| Faster R-CNN v2, 1088 px + horizontal flip | 0.5 |
| Faster R-CNN v2, 1216 px | 1.0 |
| YOLO11s, three CV folds, 1280 px | 2.0 total |
| Faster R-CNN v1, three CV folds, 768 px | 1.0 total |

WBF uses an IoU threshold of 0.50, coverage power of 1.0, and a maximum of 500 detections per frame. The final ensemble achieved **0.2962 holdout mAP**, **0.2941 public LB**, and **0.2531 private LB**.

### 3.2 Environment and reproducibility

Experiments were conducted in Google Colab and Kaggle using NVIDIA T4 GPUs, Python 3, PyTorch, and torchvision. The main additional dependencies are:

```bash
pip install "ultralytics==8.4.204" pycocotools faster-coco-eval scikit-learn pyyaml
```

Training uses **seed 42**, with video-level splits to prevent frame leakage. The original competition data is stored in `MyDrive/UCU_AI_HW2/visdrone_competition.zip`. Pretrained COCO weights are used to initialize the models.

### 3.3 Training and submission reproduction

Run the following notebooks in order:

| Step | Notebook | Purpose |
|---|---|---|
| 1 | [05 — YOLO Tuning](notebooks/05_YOLO_Tuning.ipynb) | Train three YOLO11s folds (960 px, AdamW, LR 5e-4, 6 epochs, batch 2); save `yolo_tuning_results.zip`. |
| 2 | [06 — Faster R-CNN Tuning](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/06_Faster_RCNN_Tuning.ipynb) | Train three Faster R-CNN v1 folds (768 px, SGD, LR 0.002, 6 epochs, batch 2); save checkpoints under `faster_rcnn_tuning/`. |
| 3 | [17 — Faster R-CNN v2 and FCOS](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/17_FasterRCNNv2_FCOS_Training_Kaggle.ipynb) | Train high-resolution detectors on all 56 videos (every fourth frame, batch 2, up to 8 epochs, seed 42). Use Kaggle T4 ×2 and download `hr_models_bundle.zip`. |
| 4 | [19 — Final Ensemble v3](https://github.com/marharytapaduchak/ucu-ai-hw-2/blob/main/notebooks/19_Final_Ensemble_v3_FasterRCNNv2_MultiScale.ipynb) | Load checkpoints, generate or reuse cached predictions, apply multi-scale inference and WBF, and export the final submission. |

Upload the model archives to `MyDrive/UCU_AI_HW2/` before running Notebook 19. Previously generated predictions can be reused from `ensemble_cache/`, with checkpoint fingerprints preventing mismatched cache reuse.

The final submitted predictions are generated at:

`MyDrive/UCU_AI_HW2/ensemble16_v3/submission_16b_final.csv`

The CSV contains one row per test image, with `image_id` and `PredictionString` columns. Each detection is encoded as `class confidence x y width height`, using zero-based class IDs.


