## Why Use a Diffusion Model for ECG Generation and Atrial Substrate Classification? 🤔

The assessment of atrial substrate status is critical for cardiac patients, but current methods are often expensive, invasive, and complex. While personalized electrocardiographic (ECG) data offers a non-invasive alternative, the scarcity of ECG data annotated with atrial substrate status has significantly hindered the development of accurate deep learning models.

To address this challenge, our work introduces the **ECG Large Diffusion Model (ECG LDM)**. This approach is highly effective because:

  * **Tackling Data Scarcity.** By generating a massive number of high-quality, diverse, and realistic ECG samples, the diffusion model overcomes the primary bottleneck of limited real-world data.
  * **Capturing Patient-Specific Signatures.** The generated data is not random; it is conditioned to reflect specific atrial substrate states, providing a robust dataset for training a highly accurate diagnostic model. This allows our subsequent classifier, ECGNet, to learn the subtle mapping between ECG signals and the underlying atrial substrate state, achieving breakthrough diagnostic accuracy.

-----

## Overview 📚

This study proposes a novel three-stage framework to enable the automated, non-invasive diagnosis of atrial substrate status using ECG data. Our paradigm is designed to simulate and enhance the clinical diagnostic workflow:

1.  **Data Generation:** We employ the ECG Large Diffusion Model (ECG LDM) to generate massive amounts of synthetic ECG training data from a small initial sample set.
2.  **Reinforcement Learning:** We introduce a "Human-in-the-loop" reinforcement learning mechanism where expert knowledge from cardiologists is used to continuously refine the generative model, ensuring the synthetic data meets high standards of medical accuracy and clinical compliance.
3.  **Model Prediction:** Finally, we train a highly accurate diagnostic model, **ECGNet**, on this enhanced and expanded dataset (a fusion of real and synthetic data) to precisely classify atrial substrate status, achieving breakthrough diagnostic accuracy.

This entire framework provides a powerful AI solution for the non-invasive assessment of atrial substrate status, holding significant value for both theoretical research and clinical application.

-----

## Features ✨

  * **Unified Data Preprocessing:** All ECG signals undergo a standardized and rigorous preprocessing pipeline, including wavelet transform for denoising, normalization. This ensures data quality and stability for both generation and classification tasks.
  * **Separated Generator and Classifier:** The architecture cleanly separates the data generation module (ECG LDM) from the diagnostic module (ECGNet). This modular design allows the generator to focus solely on creating high-fidelity data, which then provides a robust foundation for training a state-of-the-art classifier.
  * **Comprehensive Baseline Comparisons:** The performance of our framework is extensively validated against other models. The ECG LDM is compared with four other mainstream generative models (e.g., D2GAN, WGAN), and the ECGNet classifier is benchmarked against traditional machine learning and deep learning models like Random Forest, CNN, and CNN-LSTM.

-----

## Datasets 📊

The dataset used in this study is a high-quality clinical ECG database collected from two major medical centers: Jiangsu Provincial People's Hospital (1,051 cases) and Sun Yat-sen Memorial Hospital of Sun Yat-sen University (171 cases), for a total of 1,222 samples. This dataset is clinically representative, encompassing a wide spectrum of atrial substrate states and providing a reliable foundation for model training and validation.

-----

## Our Experiments 🔬

Please check the experimental results and analysis from our [paper](https://www.google.com/search?q=%23).

-----

## Pre-trained Weights 🏋️

You can download the pre-trained weights for both the ECG LDM generator from the following link.

  * **ECG LDM:** [Download Here](https://www.google.com/search?q=%23)

Place the downloaded weights into the `./pretrain_weight/` directory to use them for data generation.

-----

## Package Usage ⚙️

### Requirements 📋

You can quickly install the corresponding dependencies.

```bash
pip install -r requirements.txt
```

### Running Experiments 🚀

Follow these steps to train the models and generate data. The commands are based on the provided shell scripts.

#### Step 1: Train the ECG LDM Generator

```bash
# Train the diffusion model on your dataset
sh train_generator.sh
```


#### Step 2: Generate Synthetic ECG Data

```bash
sh generate_good_label.sh

sh generate_bad_label.sh
```

#### Step 3: Train the Classifier

**Option A: Train the Classifier without Pre-trained UNet**

```bash
# Train the final classifier on the combined real and synthetic dataset
sh classification.sh
```

**Option B: Train with Pre-trained UNet**


```bash
# Train the classifier using the pre-trained diffusion UNet weights
sh classification_pretrain.sh
```


