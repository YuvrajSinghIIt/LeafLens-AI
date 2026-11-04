# EC-422 LeafLens AI — Presentation Guide

## Presentation flow (5–7 minutes)
1. **Problem:** Early crop disease screening is difficult without specialists.
2. **Scope:** Tomato leaves; Healthy, Bacterial Spot, Early Blight, Late Blight, Leaf Mold.
3. **Dataset:** 500 labeled PlantVillage images, 350 training, 75 validation, 75 test.
4. **Model:** MobileNetV3-Small CNN with transfer learning; eight epochs.
5. **Results:** 62/75 test images correct = 82.67% accuracy; macro F1 ≈ 0.831.
6. **Demo:** Upload a tomato leaf photo, click Analyze, show five class scores. Then open Model Insights.
7. **Limitations:** Small benchmark, correlated samples, limited five categories; not a validated farmer-ready tool.

## Explain these terms
- **CNN:** A neural network that learns edges, spots, textures and patterns from images.
- **Transfer learning:** Start from an existing image-feature extractor and train it for new categories.
- **Epoch:** One complete pass through the training set.
- **Accuracy:** Correct predictions / total test images.
- **Precision/Recall:** Correctness of predicted category vs proportion of actual instances found.
- **Confusion matrix:** Where true versus predicted classes agree or differ.
- **Softmax score:** Relative model score, not a guarantee or validated medical/agricultural confidence.

## Questions you might get
**Why MobileNetV3?** It is lightweight compared with many larger CNNs, making deployment practical.
**Why use transfer learning?** 500 images are too few to train an image network from scratch reliably.
**Can it diagnose any plant?** No, only tomato leaves in five classes.
**Why only 82.67%?** Test examples can be ambiguous; the model is trained on a small dataset.
**Is this real-world accurate?** We have not validated that. Controlled-image test performance may overstate field use.
**What next?** More diverse field images, improved splits, better calibration, expert validation.
