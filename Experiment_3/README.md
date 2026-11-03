# Experiment 3 – Alzheimer's MRI Classification

## 1. Overview

Experiment 3 is the parameter alignment experiment in the research project *Widely Used Machine Learning Methods for Alzheimer's MRI Classification*.

It compares three machine learning methods:

- Convolutional Neural Network (CNN)
- Pixel-based Support Vector Machine (Pixel SVM)
- Hybrid CNN–SVM model

This experiment uses aligned experimental settings to further compare the models following Experiments 1 and 2.

## 2. Dataset

The experiment uses 5,000 MRI images from an OASIS-derived dataset, covering 342 subjects.

The dataset and subject-disjoint split are the same as those used in Experiment 2.

| Dataset Split | Number of Images |
|---|---:|
| Training | 3,476 |
| Validation | 637 |
| Testing | 887 |
| Total | 5,000 |

The subject-disjoint split prevents images from the same subject appearing across the training, validation and testing sets.

The test set is held out from the same dataset source and is not an independent external clinical cohort.

## 3. Experimental Methods

Three classification methods were evaluated:

**CNN:** Learns image features directly through a convolutional neural network.

**Pixel SVM:** Applies a Support Vector Machine to image-based features.

**Hybrid CNN–SVM:** Uses CNN-extracted features with an SVM classifier.

Two image input conditions were included:

- Raw MRI images, used as the main control
- CLAHE-processed MRI images, used for additional comparison

The experiment adopted aligned model settings for a more controlled comparison. Detailed configurations are provided in the experiment scripts and public configuration files.

## 4. Experimental Results

The following Accuracy values are recorded in the Experiment 3 verification results.

| Model | Raw Accuracy | CLAHE Accuracy |
|---|---:|---:|
| CNN | 66.18% | 67.98% |
| Pixel SVM | 56.82% | 49.38% |
| Hybrid CNN–SVM | 67.76% | 65.50% |

All six evaluations used the same held-out test set of 887 images.

Additional evaluation metrics, prediction records and figures are available in the `results/` directory.

Accuracy should be interpreted alongside other available classification metrics, including AUC, particularly because the dataset is class-imbalanced.

## 5. Repository Structure

- `cnn_model.py` – CNN model implementation
- `prepare_dataset.py` – Dataset preparation
- `run_experiment.py` – Experiment execution
- `compare_second.py` – Comparison with Experiment 2
- `verify_results.py` – Verification script
- `build_report.py` – Report generation
- `requirements.txt` – Python dependencies
- `results/figures/` – Evaluation figures
- `results/models/` – Saved model files
- `results/raw/` – Raw-image evaluation results
- `results/processed/` – CLAHE-processed evaluation results
- `results/run_config_public.json` – Public run configuration
- `results/source_config_public.json` – Public dataset configuration
- `results/verification.json` – Verification records

## 6. Verification Status

The saved `results/verification.json` reports an overall status of **passed**, with no recorded failures.

The documented checks cover dataset consistency, subject-disjoint splitting, image and model-input hashes, saved prediction records, and comparisons with Experiment 2.

The verification record also confirms that each model evaluation includes 887 test predictions.

**Verification scope:** The recorded verification covers the Experiment 3 protocol and available artifacts, including metrics recomputed from saved prediction CSV files.

Some historical training details were not fully recorded. Therefore, successful verification of the documented checks should not be interpreted as complete independent reproduction of every original training step.

## 7. Research Context and Limitations

Experiment 3 forms the final experimental comparison in the three-experiment research project.

The results support comparison of CNN, Pixel SVM and Hybrid CNN–SVM methods under the documented experimental conditions.

The models were evaluated on a dataset-derived held-out test set rather than an independent external clinical dataset. The results should therefore be interpreted as academic experimental findings, not evidence of clinical diagnostic reliability.
