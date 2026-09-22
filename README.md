# Spectrum Occupancy Forecasting Beyond Persistence: A Split-Learning Study

Official implementation of the paper **"Spectrum Occupancy Forecasting Beyond Persistence: A Split-Learning Study"** by Atik Mahabub and Shervin Vakili (INRS-EMT).

[![Dataset on IEEE DataPort](https://img.shields.io/badge/Dataset-IEEE%20DataPort-orange.svg)](https://dx.doi.org/10.21227/5cc8-wg20)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Spectrum occupancy forecasting enables dynamic spectrum access systems to proactively schedule secondary transmissions instead of reacting to primary-user activity. This repository provides a unified implementation for dataset preprocessing, sequence forecasters, and evaluation metrics, all integrated into a single executable script: `Main_Code.py`.

### Key Features
* **Unified Execution:** All operations—from raw data ingestion and sub-channel preprocessing to model training and metric computation—are handled seamlessly within `Main_Code.py`.
* **Strong Non-Learned Baselines:** Highlights the fact that while trivial persistence achieves high raw accuracy, a simple historical-mean baseline substantially outperforms it, resetting the true pass mark for learned models.
* **Architecture Gains:** Implements state-of-the-art sequence models (PatchTST, iTransformer, Crossformer, N-HiTS, DLinear, and more) to forecast spectrum occupancy dynamics.
* **Real-World RF Dataset:** Evaluated across real over-the-air (OTA) RF measurements in the 2.4 GHz and 5 GHz unlicensed bands.

---

## Dataset

This project utilizes over-the-air (OTA) spectrum measurements collected in a congested indoor academic-office environment.

* **Bands Monitored:** 2.4 GHz ISM and 5 GHz / lower-6 GHz U-NII unlicensed bands.
* **Sub-channels:** Uniform 32-sub-channel grid stitched across sequential sweeps.
* **Dataset Access:** You must download the dataset prior to running the code. It is publicly available on IEEE DataPort.

> **Dataset Link:** [A Multi-Environment Real-World Multi-Band RF Dataset for Spectrum Sensing and Occupancy Analysis for Allocations or Sharing](https://ieee-dataport.org/documents/multi-environment-real-world-multi-band-rf-dataset-spectrum-sensing-and-occupancy)

Please download the raw sweep files and place them in a `./data/raw/` directory (or update the path parameters when calling `Main_Code.py`).

---

## Installation

Clone the repository and set up your Python environment:

```bash
git clone [https://github.com/](https://github.com/INRS-ECCoLe/spectrum-occupancy-forecasting.git)
cd spectrum-occupancy-forecasting

# Create and activate a conda environment
conda create -n specforecast python=3.10 -y
conda activate specforecast

# Install dependencies (e.g., PyTorch, NumPy, Pandas, Scikit-learn)
pip install -r requirements.txt
