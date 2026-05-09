# DeepLearningFireDetection

**Group 16 — Deep Learning Fire and Smoke Detection**  
**Roll numbers:** `27100159`, `27100384`  
**Repository:** <https://github.com/SJK159/DeepLearningFireDetection>

This repository contains our deep learning project for wildfire **fire and smoke detection** using object detection and vision-language guidance. The project begins with a strong **YOLOv8n baseline**, then explores improvements based on **false-positive mining**, **CLIP-guided semantic supervision**, and **contrastive learning**. The final submission paper summarizes the baseline, the proposed improvements, the ablation studies, and the failure-case analysis.

---

## Project Motivation

Early wildfire detection is a high-impact computer vision problem because smoke and flame cues can appear before a fire becomes uncontrollable. However, fire and smoke detection is difficult in real-world images because the visual appearance of smoke overlaps with clouds, fog, haze, glare, waterfalls, dust, and distant atmospheric effects. Flame detection is also challenging when fire regions are small, occluded, low-resolution, or visually blended into an orange/yellow background.

Our motivation was therefore not only to build a detector that recognizes smoke and fire, but also to study **what fools the detector** and whether semantic information from a vision-language model can reduce those mistakes.

---

## Repository Structure

The important files and folders are organized as follows:

```text
DeepLearningFireDetection/
├── README.md
├── Group16_27100159_27100384.ipynb
├── group16_27100159_27100384-2.ipynb
├── dl-firedetection-5.ipynb
├── results_final/
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── val_batch0_labels.jpg
│   ├── val_batch0_pred.jpg
│   ├── val_batch1_labels.jpg
│   ├── val_batch1_pred.jpg
│   ├── val_batch2_labels.jpg
│   └── val_batch2_pred.jpg
└── result(fusionEpoch=25,lambda=0.5)/
    ├── final_baseline_vs_hybrid_lambda_0_5.csv
    ├── hybrid_training_history_lambda_0_5.csv
    ├── mined_fps.csv
    ├── BoxF1_curve.png
    ├── BoxPR_curve.png
    ├── BoxP_curve.png
    ├── BoxR_curve.png
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── val_batch*_labels.jpg
    └── val_batch*_pred.jpg
```

### File Guide

| File / folder | Purpose |
|---|---|
| `Group16_27100159_27100384.ipynb` | Baseline notebook containing the initial YOLOv8n training/evaluation workflow. |
| `group16_27100159_27100384-2.ipynb` | First improvement notebook containing early improvement experiments and analysis. |
| `dl-firedetection-5.ipynb` | Final notebook containing the completed experiments, final model logic, and final evaluation workflow. |
| `results_final/` | Final visual outputs, curves, confusion matrices, and validation prediction grids. |
| `result(fusionEpoch=25,lambda=0.5)/` | Final CLIP+YOLO hybrid run with `lambda = 0.5`, 25 epochs, 5 frozen epochs, mined false positives, training history, and result CSVs. |

---

## Problem Formulation

We formulate the task as **two-class object detection**:

- `smoke`
- `fire`

Given an input image, the model predicts bounding boxes, confidence scores, and class labels for visible smoke and fire regions. We evaluate performance using standard object detection metrics including **mAP@0.5**, **mAP@0.5:0.95**, precision, recall, F1 score, class-wise AP, confidence curves, precision-recall curves, and confusion matrices.

---

## Datasets Used

The project uses Kaggle-hosted wildfire/fire/smoke detection datasets and saved model artifacts. Because the image datasets are large, they are not all stored directly in this repository; the notebooks reference them through Kaggle input paths.

### 1. Smoke-Fire Detection YOLO Dataset

This dataset provides YOLO-format fire and smoke annotations. It is used for training, validation, evaluation, and false-positive mining. The validation images are also used to analyze the cases where the trained detector produces false positives.

Example path used in the notebook outputs:

```text
/kaggle/input/datasets/sayedgamal99/smoke-fire-detection-yolo/data/val/images/
```

### 2. D-Fire / wildfire fire-smoke detection data

This data is used as the main fire-smoke detection training source for the YOLOv8n baseline and hybrid experiments. The model is trained to identify smoke and fire regions under varied wildfire conditions.

### 3. FASDD / COCO-style fire-smoke split

This dataset source was used for additional fire/smoke examples and evaluation support where compatible with the notebook pipeline.

### 4. Saved baseline weights

The baseline weights were stored as a Kaggle dataset and loaded into later notebooks so that the hybrid model could be trained without retraining the full baseline from scratch.

Example saved-weight path:

```text
/kaggle/input/datasets/merc11123saad/bestpt-saad/best.pt
```

---

## State-of-the-Art Survey Summary

Our SOTA review focused on four main directions in fire and smoke detection:

1. **Traditional image-processing methods**  
   Earlier approaches often used color thresholds, motion cues, texture descriptors, and handcrafted features. These methods can work in controlled scenes but are brittle under changing lighting, haze, clouds, and complex outdoor backgrounds.

2. **CNN-based classification and detection**  
   Convolutional neural networks improve feature learning and generalization, but classification-only systems do not localize fire/smoke regions. Detection models are more appropriate because wildfire monitoring requires knowing where the smoke or fire appears.

3. **Modern one-stage object detectors**  
   YOLO-style detectors are widely used for real-time fire and smoke detection because they balance speed and accuracy. YOLOv8n is especially attractive for this project because it is lightweight, fast, and practical for deployment-oriented wildfire monitoring.

4. **Vision-language and transformer-based methods**  
   Models such as CLIP provide semantic image-text representations. We used this idea to test whether a frozen vision-language encoder could help distinguish actual fire/smoke from visually similar negatives such as fog, clouds, glare, and waterfalls.

The gap we focused on was **false-positive robustness**. A normal object detector may learn visual texture well, but it can still misclassify smoke-like background patterns. Our improvement strategy therefore used false-positive mining and CLIP-guided contrastive learning.

---

## Baseline Model: YOLOv8n

The baseline model is **YOLOv8n**, a lightweight one-stage object detector. We selected it because:

- it is computationally efficient;
- it is suitable for real-time detection;
- it supports YOLO-format fire/smoke datasets directly;
- it provides strong detection performance with relatively low training cost;
- it gives interpretable detection outputs through bounding boxes, confidence scores, PR curves, and confusion matrices.

The baseline was trained on the fire/smoke detection dataset and evaluated on validation data using mAP, precision, recall, F1, and class-wise AP.

### Baseline Result Summary

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | FPS | AP Smoke | AP Fire |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n baseline | **0.7057** | **0.3912** | **0.7112** | **0.6365** | **0.6718** | 56.33 | **0.7624** | **0.6490** |

The baseline remained the strongest final model. It performed especially well on smoke, achieving AP Smoke of approximately **0.7624**.

---

## Improvement 1: False-Positive Mining

After training the baseline, we inspected cases where the model produced false positives. The goal was to identify visual patterns that fooled the detector and use them to guide the next model improvement.

The mined false positives showed that the baseline was confused by:

- fog and haze;
- clouds and cloudy skies;
- bright glare and sunlight;
- waterfalls and white textured regions;
- distant atmospheric effects;
- non-fire smoke-like visual patterns.

The final mined false-positive file contains **35 mined false-positive image entries** with the number of detections and maximum confidence per image.

Example structure from `mined_fps.csv`:

```csv
path,n_detections,max_conf
.../AoF00122.jpg,1,0.3313
.../AoF00748.jpg,1,0.3541
.../WEB02828.jpg,3,0.6334
```

This motivated the second improvement: adding semantic guidance so the detector could learn not only what smoke/fire looks like, but also what smoke-like negatives look like.

---

## Improvement 2: CLIP + YOLOv8n Hybrid Architecture

The second improvement introduces a hybrid architecture combining YOLOv8n with a frozen CLIP encoder. The purpose was to inject semantic vision-language information into the detector while preserving YOLO's localization ability.

### High-Level Architecture

```text
Input image
   ├── YOLOv8n backbone + neck
   │      ├── P3 feature map
   │      ├── P4 feature map
   │      └── P5 feature map ─────┐
   │                              │
   │      Detection head          │
   │      L_detect                │
   │                              │
   └── Frozen CLIP image encoder ─┤
                                  ↓
                           Fusion head
                                  ↓
                         Contrastive head
                                  ↓
                         L_contrastive
                                  ↓
              L_total = L_detect + λ · L_contrastive
```

### Architecture Explanation

The hybrid model has two parallel branches:

1. **YOLOv8n detection branch**  
   The YOLO branch extracts multi-scale visual features using the backbone and neck. It produces feature maps at different pyramid levels, commonly referred to as `P3`, `P4`, and `P5`. These are used by the detection head to predict bounding boxes and class probabilities.

2. **Frozen CLIP branch**  
   The CLIP image encoder is kept frozen. This avoids destroying CLIP's pretrained semantic representation and reduces training instability. The CLIP branch provides semantic context about whether an image resembles fire, smoke, or visually similar negative examples.

3. **Fusion head**  
   The fusion head combines YOLO's spatial features with CLIP-derived context. In our implementation, the P5 YOLO feature map is used for fusion because it contains stronger high-level semantic information than lower-level maps.

4. **Contrastive head**  
   The contrastive head projects fused visual features into a CLIP-compatible embedding space. These projected features are compared against text/image anchors using cosine similarity.

5. **Training objective**  
   The model is trained using both detection loss and contrastive loss:

```text
L_total = L_detect + λ · L_contrastive
```

where:

```text
L_detect = CIoU + DFL + classification loss
```

and `λ` controls how strongly the contrastive branch affects training.

---

## Why This Architecture?

The baseline YOLOv8n model was already strong, but its false positives revealed that it relied heavily on visual texture. Smoke-like patterns such as clouds and haze could trigger incorrect detections. CLIP was introduced because it has pretrained semantic alignment between images and language, which can help compare visual regions against prompts such as fire, smoke, non-fire haze, cloud, fog, or background.

We chose this architecture because it allowed us to test a clear research question:

> Can semantic vision-language guidance reduce false positives in wildfire smoke and fire detection without sacrificing YOLO's localization strength?

The answer from our experiments was mixed. The hybrid trained successfully and some ablations improved over weaker hybrid runs, but the final hybrid did not outperform the YOLOv8n baseline.

---

## Training Strategy

The final hybrid run used a two-phase training schedule:

1. **Frozen phase**  
   Early epochs keep the YOLO pathway stable while training the new fusion/contrastive components.

2. **Joint phase**  
   Later epochs allow the hybrid model to update more of the trainable components jointly.

For the main hybrid run:

| Setting | Value |
|---|---:|
| Model | CLIP + YOLOv8n Hybrid |
| λ | 0.5 |
| α image anchor | 0.5 |
| Total epochs | 25 |
| Frozen epochs | 5 |
| Hybrid weights | `/kaggle/working/hybrid_weights.pt` |

The training history shows that detection loss and total loss decreased steadily across 25 epochs, which means optimization was successful even though final detection performance was weaker than the baseline.

---

## Code Snippets

### Loading the saved baseline weights

```python
from pathlib import Path

BASELINE_W = "/kaggle/input/datasets/merc11123saad/bestpt-saad/best.pt"

if Path(BASELINE_W).exists():
    print(f"Loading baseline weights from: {BASELINE_W}")
else:
    raise FileNotFoundError(
        f"Saved baseline weights not found at {BASELINE_W}.\n"
        "Either train the baseline first or fix the path."
    )
```

### Hybrid loss objective

```python
lambda_contrastive = 0.5

loss_total = loss_detect + lambda_contrastive * loss_contrastive
```

### Result logging

```python
import pandas as pd

results = {
    "model": "CLIP+YOLOv8n Hybrid",
    "lambda": 0.5,
    "alpha_img": 0.5,
    "epochs": 25,
    "freeze_epochs": 5,
    "mAP50": 0.4057,
    "mAP50_95": 0.1763,
    "precision": 0.5257,
    "recall": 0.4093,
    "f1": 0.4603,
}

pd.DataFrame([results]).to_csv("final_baseline_vs_hybrid_lambda_0_5.csv", index=False)
```

---

## Final Results

### Main Comparison

| Model | λ | α | Epochs | Frozen epochs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | FPS | AP Smoke | AP Fire |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n Baseline | — | — | — | — | **0.7057** | **0.3912** | **0.7112** | **0.6365** | **0.6718** | 56.33 | **0.7624** | **0.6490** |
| CLIP+YOLOv8n Hybrid | 0.5 | 0.5 | 25 | 5 | 0.4057 | 0.1763 | 0.5257 | 0.4093 | 0.4603 | 77.90 | 0.2909 | 0.5204 |

### Interpretation

The YOLOv8n baseline outperformed the final CLIP+YOLOv8n hybrid run. The baseline achieved stronger overall mAP, stronger F1, and much better smoke AP. The hybrid was faster in the reported run, but its accuracy dropped substantially, especially for smoke detection.

This suggests that naive CLIP-YOLO feature fusion can disrupt the detector's learned localization and class balance. The hybrid branch may overemphasize semantic fire cues while weakening subtle smoke localization.

---

## Ablation Studies

We ran ablations to understand how contrastive-anchor design and loss weighting affected the hybrid model.

### Alpha Ablation

`α` controls the contribution of mined false-positive image embeddings relative to negative text embeddings.

| α | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.4910 | 0.2147 | 0.5518 | 0.4952 | 0.5219 |
| **0.3** | **0.5190** | 0.2355 | **0.5751** | **0.5101** | **0.5407** |
| 0.5 | 0.5182 | **0.2392** | 0.5580 | 0.5079 | 0.5318 |
| 0.7 | 0.4724 | 0.2138 | 0.5154 | 0.4764 | 0.4951 |
| 1.0 | 0.4904 | 0.2238 | 0.5433 | 0.4878 | 0.5141 |

**Finding:** The best alpha value was `α = 0.3`. This suggests that negative text guidance was more stable than relying too heavily on mined false-positive image embeddings. Higher image-anchor weight did not consistently improve performance.

### Lambda Ablation

`λ` controls the strength of the contrastive loss in the total training objective.

| λ | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.4696 | 0.2073 | 0.5608 | 0.4586 | 0.5046 |
| 0.3 | 0.4629 | 0.2088 | 0.5535 | 0.4577 | 0.5011 |
| **0.5** | **0.5367** | **0.2513** | **0.5696** | **0.5354** | **0.5520** |
| 1.0 | 0.4961 | 0.2249 | 0.5357 | 0.5165 | 0.5260 |

**Finding:** In the controlled lambda ablation, `λ = 0.5` gave the strongest result. However, the final 25-epoch hybrid run with `λ = 0.5` still underperformed the baseline, showing that the hybrid is sensitive not only to λ but also to training schedule, anchor construction, and feature-fusion stability.

---

## Qualitative Findings

The validation prediction grids and mined false-positive examples reveal several important patterns:

### What Worked Well

- The baseline YOLOv8n model localized large smoke regions effectively.
- Fire was detected well in clear flame-heavy scenes.
- The detector produced interpretable bounding boxes and confidence curves.
- False-positive mining helped identify meaningful hard negatives.

### Failure Cases

- Clouds, fog, haze, glare, and waterfalls were sometimes detected as smoke.
- Diffuse smoke was harder to localize than dense smoke.
- Small or distant fire regions were sometimes fragmented into multiple boxes.
- The hybrid model often became more fire-biased and less reliable for smoke.
- Some background regions were misclassified as fire or smoke.

---

## Key Takeaways

1. **YOLOv8n is a strong baseline for this task.**  
   It achieved the best final results with mAP@0.5 of approximately `0.7057` and F1 of `0.6718`.

2. **False-positive mining is useful.**  
   It exposed the exact visual patterns that fooled the baseline and gave a strong motivation for semantic negative anchors.

3. **CLIP guidance is promising but unstable.**  
   The hybrid model trained successfully, but the final configuration did not outperform the detector-only baseline.

4. **Smoke detection is more sensitive than fire detection.**  
   Several hybrid variants shifted the model toward fire-like cues and weakened smoke localization.

5. **The best hybrid ablation remained below the baseline.**  
   The best hybrid configuration reached approximately `mAP@0.5 = 0.519`, whereas the baseline reached `mAP@0.5 = 0.7057`.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/SJK159/DeepLearningFireDetection.git
cd DeepLearningFireDetection
```

### 2. Open notebooks on Kaggle or Colab

The notebooks are designed around Kaggle input paths. For the smoothest reproduction, run them on Kaggle and attach the required datasets.

Recommended order:

```text
1. Group16_27100159_27100384.ipynb      # Baseline YOLOv8n
2. group16_27100159_27100384-2.ipynb    # Improvement 1
3. dl-firedetection-5.ipynb             # Final hybrid and results
```

### 3. Attach required Kaggle datasets

Attach the fire/smoke image datasets and saved baseline weights used in the notebooks. Make sure the paths match the paths defined inside the notebooks.

### 4. Run evaluation

Evaluation outputs include:

```text
BoxF1_curve.png
BoxPR_curve.png
BoxP_curve.png
BoxR_curve.png
confusion_matrix.png
confusion_matrix_normalized.png
val_batch*_labels.jpg
val_batch*_pred.jpg
```

### 5. Check saved result CSV files

The final hybrid run saves:

```text
final_baseline_vs_hybrid_lambda_0_5.csv
hybrid_training_history_lambda_0_5.csv
mined_fps.csv
```

---

## Limitations

- The CLIP branch was applied using a relatively simple fusion strategy.
- The model did not use temporal video consistency, even though wildfire monitoring often involves video streams.
- The hybrid approach used image-level semantic guidance rather than region-level CLIP supervision.
- The datasets contain many visually diverse examples, but real-world deployment would require broader validation across camera types, weather, geography, and lighting conditions.
- The final hybrid run did not outperform the baseline, so the hybrid should be treated as an experimental contribution rather than the final best detector.

---

## Future Work

Future improvements could include:

- region-level CLIP supervision instead of global image-level CLIP embeddings;
- stronger hard-negative clustering for fog, clouds, glare, and haze;
- temporal modeling for video-based wildfire detection;
- larger YOLO variants such as YOLOv8s or YOLOv8m;
- attention-based fusion instead of simple feature concatenation;
- class-balanced training to reduce smoke/fire imbalance;
- additional external validation on unseen wildfire camera feeds.

---

## Final Conclusion

This project demonstrates a complete fire and smoke detection pipeline using YOLOv8n and a CLIP-guided hybrid extension. The YOLOv8n baseline achieved the strongest overall result, while false-positive mining revealed important weaknesses caused by smoke-like background patterns. The CLIP+YOLO hybrid provided a useful experimental direction and improved under some ablations, but it did not surpass the baseline in the final configuration. The main contribution is therefore both practical and analytical: a strong detector baseline, a documented semantic-fusion attempt, and a clear failure-case study showing why wildfire smoke detection remains difficult.

---

## References

- Jocher et al., Ultralytics YOLOv8 documentation and implementation.
- Radford et al., “Learning Transferable Visual Models From Natural Language Supervision” (CLIP).
- Fire and smoke detection literature surveyed in the project State-of-the-Art review.
- YOLO-format wildfire/fire/smoke datasets used through Kaggle notebook inputs.
