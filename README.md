# Alzheimer's MRI Classification – CNN, SVM and Hybrid Models

## 1. Project Overview

This repository contains the code, saved models, evaluation results and supporting materials for the research project:

**Widely Used Machine Learning Methods for Alzheimer's MRI Classification**

The project compares three machine learning methods for classifying Alzheimer's disease-related MRI images:

- Convolutional Neural Network (CNN)
- Support Vector Machine (SVM)
- Hybrid CNN–SVM model

Three experiments were conducted to compare model performance under different dataset and experimental settings.

This is an academic research project and is not intended for clinical diagnosis.

## 2. Overview of the Three Experiments

| Experiment | Dataset | Main Purpose |
|---|---|---|
| Experiment 1 | 6,400 MRI images | Initial comparison of CNN, SVM and Hybrid models |
| Experiment 2 | 5,000 MRI images | Evaluation using a subject-disjoint split and Raw/CLAHE preprocessing |
| Experiment 3 | 5,000 MRI images | Further comparison using aligned experimental settings |

### Experiment 1 – Initial Model Comparison

**Folder:** [Experiment_1](Experiment_1/)

The first experiment used 6,400 preprocessed MRI images divided into two classes: Non-Demented and Demented.

CNN, SVM and Hybrid CNN–SVM methods were evaluated using an image-level stratified split.

Subject identifiers were not available for this experiment, so subject-level separation could not be verified.

The original dataset was obtained through the HeyWhale community.

### Experiment 2 – Subject-Disjoint Evaluation

**Folder:** [Experiment_2](Experiment_2/)

Experiment 2 used 5,000 MRI images from 342 subjects, based on an OASIS-derived dataset.

The dataset was divided as follows:

| Dataset Split | Images |
|---|---:|
| Training | 3,476 |
| Validation | 637 |
| Testing | 887 |
| Total | 5,000 |

The experiment used a subject-disjoint split to prevent the same subject from appearing in different data partitions.

Three classification methods were evaluated under two image input conditions:

- Raw MRI images
- CLAHE-processed MRI images

Saved models, figures and supporting scripts are available in the Experiment 2 folder.

Additional records from a later re-evaluation of the saved Experiment 2 models are provided in:

[Experiment_2/recovered_evaluation](Experiment_2/recovered_evaluation/)

These records include dataset split information, model predictions, evaluation metrics and consistency checks.

**Verification limitation:** The recovered evaluation results have been cross-checked against available model and dataset records. However, some original training logs and configuration records remain unavailable. Therefore, the original full training verification is incomplete.

See the [Experiment 2 README](Experiment_2/README.md) for further details.

### Experiment 3 – Aligned Experimental Comparison

**Folder:** [Experiment_3](Experiment_3/)

Experiment 3 used the same 5,000 MRI images and subject-disjoint dataset split as Experiment 2.

The experiment further examined CNN, Pixel SVM and Hybrid CNN–SVM performance using aligned experimental settings.

Raw MRI images were used as the main control, while CLAHE preprocessing was also evaluated.

The folder contains experiment scripts, saved models, evaluation figures, configuration records and verification results.

The provided Experiment 3 verification record reports successful completion of its documented checks. Details of the verification scope are available in the experiment files.

## 3. Repository Structure

- [Experiment_1](Experiment_1/) – Initial experiment, using 6,400 MRI images
- [Experiment_2](Experiment_2/) – Subject-disjoint evaluation with 5,000 images
- [Experiment_3](Experiment_3/) – Parameter-aligned comparison using the same 5,000-image dataset
- [dataset](dataset/) – Preprocessed MRI image dataset associated with the initial experiment
- `requirements.txt` – Original project dependency information

Each experiment folder contains its own available code and results. Refer to the relevant experiment files for the corresponding settings and outputs.

## 4. Evaluation Methods

The models were compared using classification performance measures, including:

- Accuracy
- F1-score
- Confusion Matrix
- Area Under the ROC Curve (AUC)

Balanced Accuracy is also relevant when interpreting class-imbalanced test results.

The experiments were conducted under different experimental settings, so differences in the results should be interpreted with consideration of dataset composition, splitting strategy, preprocessing and model configuration.

## 5. Limitations

Several limitations should be considered:

- Experiment 1 used an image-level split, and subject-level independence could not be confirmed.
- Experiment 2 has verified saved-model evaluation records, but its complete original training verification remains unavailable.
- Experiments 2 and 3 used subject-disjoint partitions from the same dataset source. Their held-out test sets are not independent external clinical cohorts.
- Model performance on these datasets does not establish clinical diagnostic reliability.

## 6. Dataset Information

The initial 6,400-image dataset was obtained through the HeyWhale community. The exact original dataset listing and redistribution terms have not yet been independently verified.

The later experiments used an OASIS-derived MRI image collection.

Users should check the original dataset sources, licensing conditions and applicable usage restrictions before redistributing or reusing the images.

## 7. Research Purpose

This repository documents the experimental process used in the research project **Widely Used Machine Learning Methods for Alzheimer's MRI Classification**.

It is intended to support academic review of the implemented methods, experimental comparisons and available evaluation evidence.
