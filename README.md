# MRI Classification Code

This folder is meant to be easy to run. Each model has its own script, and each
script asks whether you want to test the saved model or train a fresh one.

## What is inside

- `dataset/MRI/`: the full MRI dataset, kept in the original four class folders.
- `test_data/`: a small set of 20 images for a quick check, five from each class.
- `models/`: saved model files.
  - `pixel_svm.joblib`: the pixel-based SVM model.
  - `cnn_best.pt`: the best CNN checkpoint.
  - `hybrid_svm.joblib`: the CNN feature extractor plus SVM classifier.
  - `image_normalization.json`: the image mean and standard deviation used at test time.
- `svm_model.py`: train or test the SVM model.
- `cnn_model.py`: train or test the CNN model.
- `hybrid_model.py`: train or test the CNN-SVM hybrid model.

## How to run

From this folder:

```powershell
python svm_model.py
python cnn_model.py
python hybrid_model.py
```

From the project root:

```powershell
python code\svm_model.py
python code\cnn_model.py
python code\hybrid_model.py
```

Each script will ask:

```text
1. Test the saved model
2. Train a new model
```

Choose `1` or type `test` to load the saved model from `models/` and print the
prediction for each image in `test_data/`.

Choose `2` or type `train` to train on `dataset/MRI/` and save the result back
to `models/`.
