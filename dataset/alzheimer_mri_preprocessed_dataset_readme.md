# Alzheimer's MRI Preprocessed Dataset

This dataset contains preprocessed brain MRI images for Alzheimer's disease
classification. All images are grayscale JPG files resized to 128 x 128 pixels.

The original folders use four classes:

- `Mild_Demented`: 896 images
- `Moderate_Demented`: 64 images
- `Non_Demented`: 3200 images
- `Very_Mild_Demented`: 2240 images

For the scripts in this project, the task is simplified into two labels:

- `Non_Demented`
- `Demented`, which combines `Very_Mild_Demented`, `Mild_Demented`, and `Moderate_Demented`

The dataset was collected from public sources and is used here only for an
algorithm comparison experiment. It is not a clinical diagnosis tool.
