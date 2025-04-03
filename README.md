# LangGas: Introducing Language in Selective Zero-Shot Background Subtraction for Semi-Transparent Gas Leak Detection with a New Dataset
Wenqi Guo, Yiyang Du, Shan Du 

[![arXiv](https://img.shields.io/badge/arXiv-2503.02910-b31b1b.svg)](https://arxiv.org/abs/2503.02910)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/langgas-introducing-language-in-selective/segmentation-on-simgas)](https://paperswithcode.com/sota/segmentation-on-simgas?p=langgas-introducing-language-in-selective)

## Abstract
Gas leakage poses a significant hazard that requires prevention. Traditionally, human inspection has been used for detection, a slow and labour-intensive process. Recent research has applied machine learning techniques to this problem, yet there remains a shortage of high-quality, publicly available datasets. This paper introduces a synthetic dataset, SimGas, featuring diverse backgrounds, interfering foreground objects, diverse leak locations, and precise segmentation ground truth. We propose a zero-shot method that combines background subtraction, zero-shot object detection, filtering, and segmentation to leverage this dataset. Experimental results indicate that our approach significantly outperforms baseline methods based solely on background subtraction and zero-shot object detection with segmentation, reaching an IoU of 69\% overall. We also present an analysis of various prompt configurations and threshold settings to provide deeper insights into the performance of our method. The dataset will be available after publication. 

## Method Overview

## Dataset Overview
![image](https://github.com/user-attachments/assets/14901599-cd7b-45ba-924b-846bf46df31d)

## Results
| BGS        | VLM Filtering | Temporal Filtering | Seg.   | τ₍VLM₎ | Stationary Foreground I/P/R/FLA | Moving Foreground I/P/R/FLA | Overall I/P/R/FLA          |
|------------|----------------|---------------------|--------|--------|-------------------------------|-----------------------------|----------------------------|
| ✓          |                |                     | None   | -      | 0.56 / 0.64 / 0.83 / 0.85     | 0.38 / 0.53 / 0.58 / 0.69   | 0.5 / 0.61 / 0.73 / 0.79   |
| ✓          | ✓              |                     | SAM 2  | 0.09   | 0.67 / 0.81 / 0.79 / **0.88** | 0.54 / 0.79 / 0.65 / 0.83   | 0.62 / 0.80 / 0.74 / 0.86  |
|            | ✓              | ✓                   | SAM 2  | 0.19   | 0.22 / 0.39 / 0.28 / 0.57     | 0.46 / 0.65 / 0.59 / 0.74   | 0.31 / 0.49 / 0.4 / 0.63   |
| ✓          | ✓              | ✓                   | Trad.  | 0.12   | 0.57 / **0.85** / 0.65 / 0.83 | 0.35 / **0.88** / 0.37 / 0.72 | 0.49 / **0.86** / 0.55 / 0.79 |
| ✓          | ✓              | ✓                   | SAM 2  | 0.12   | **0.70** / 0.83 / **0.82** / 0.87 | **0.69** / 0.79 / **0.84** / **0.92** | **0.69** / 0.82 / **0.82** / **0.89** |

**Table: Ablation study of different components with IoU (I), Precision (P), Recall (R), and frame-level accuracy (FLA).** In the segmentation column (Seg.), traditional (Trad.) means Otsu [Otsu] combined with morphological transformations. This analysis corresponds to our ablation study, detailed in Section 4 of the paper.

Warning: Different methods of calculating IoU can produce inconsistent results. We used per video aggregation and then average across all videos. 
## Test on SimGas


### Step 1 Install required packages
Install pytorch according to https://pytorch.org/get-started/locally/

```bash
pip3 install opencv-python transformers tqdm Pillow bitesandbytes wandb
```

### Step 2: Download and Install SAM-2
```bash
git clone https://github.com/facebookresearch/sam2.git
mv sam2 .sam2
cd .sam2
pip3 install -e .
```

If installation fails, run:
```bash
echo -e '[build-system]\nrequires = [\n    "setuptools>=62.3.0,<75.9",\n    "torch>=2.5.1",\n    ]\nbuild-backend = "setuptools.build_meta"' > pyproject.toml
```
(See https://github.com/facebookresearch/sam2/issues/611 for more)
Then run:
```bash
pip3 install -e .
```

Download the checkpoints:
```bash
cd checkpoints
bash download_ckpts.sh
```

More details: https://github.com/facebookresearch/sam2

### Step 3. Download Dataset
Go to https://paperswithcode.com/dataset/simgas and download the dataset.

Put the videos of the downloaded dataset into the `simulated_gas` folder, and run `python3 dataprep.py`

## Step 4. Run the code
### Modify `owl_notracking.py` for SAM-2 path.
Change `sam2_checkpoint` to your SAM-2 checkpoint path. Do NOT change config path.

### Run the code
Run `python3 owl_notracking.py` with the following options
```bash
Usage: owl_notracking.py [OPTIONS]

Options:
  --video_id TEXT           Video ID, such as vid20
  --root_path TEXT          Root path to dataset (default: "../sim")
  --display TEXT            Display server for visualization (default: "localhost:10.0")
  --log_file TEXT           Output log file name (default: "results.csv")
  --temporal_filter         Enable temporal filtering (flag)
  --vlm_threashold FLOAT    Threshold for VLM decision (default: 0.06)
  --positive_prompt TEXT    Positive prompt for VLM analysis (default: "white steam")
```
To reproduce our results, use the following config with appropriate path
```bash
python3 owl_notracking.py --temporal_filter --vlm_threashold 0.12 [OPTIONS]
```

## Test on GasVid

### Preparation  
1. Download the dataset from the original author:  
   [https://drive.google.com/drive/folders/1JKEMtCGPSq2IqGk4uXZP9A6gr8fyJsGC](https://drive.google.com/drive/folders/1JKEMtCGPSq2IqGk4uXZP9A6gr8fyJsGC)

2. Remove all videos captured at depths greater than 18 meters as specified in the original paper.

3. Place the remaining videos in the `Videos` directory.

4. Run `convert_gasvid.sh` to extract frames.

### Running the Test  
1. Edit `run_gasvid.py` and `owl_gasvid.py` to set the correct dataset path.

2. To evaluate the entire dataset, run:  
   `python3 run.py`

3. To evaluate a single video, run:  
   `python3 owl_gasvid.py --video_id [video id]`

4. Output results are stored in the `gasvid_res_full` directory.