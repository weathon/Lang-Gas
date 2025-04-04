# 🌬️ LangGas: Introducing Language in Selective Zero-Shot Background Subtraction for Semi-Transparent Gas Leak Detection with a New Dataset  
👨‍🔬 Wenqi Guo, Yiyang Du, Shan Du  
🏫 University of British Columbia  
🐱 Group of Methane Emission Observation and Watch (MEOW), 🧪 Weasoft Software  

[![arXiv](https://img.shields.io/badge/arXiv-2503.02910-b31b1b.svg)](https://arxiv.org/abs/2503.02910)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/langgas-introducing-language-in-selective/segmentation-on-simgas)](https://paperswithcode.com/sota/segmentation-on-simgas?p=langgas-introducing-language-in-selective)

---

## 🧠 Abstract  
🚨 Gas leaks are dangerous and must be detected early!  
🧍‍♂️ Traditional methods rely on manual inspection — slow and tedious.  
🧠💻 We bring machine learning to the rescue but face a major issue: lack of high-quality datasets.  

📦 **SimGas** is our new synthetic dataset featuring:  
🖼️ Varied backgrounds  
🚧 Distracting foregrounds  
💨 Random leak locations  
🎯 Precise segmentation masks  

🔍 We introduce a zero-shot method combining:  
🧼 Background Subtraction  
👁️ Zero-Shot Object Detection  
🧹 Filtering  
✂️ Segmentation  

💥 Our results?  
🎯 IoU up to **69%** — outperforming baseline approaches.  
🧪 We analyze prompt choices + thresholds for deeper insight.  
📂 Dataset available after publication!

---

## 🧬 Method Overview  
![meow2](https://github.com/user-attachments/assets/02debfe3-7da5-47e3-8720-d70cf3aee802)

---

## 📊 Dataset Overview  
![image](https://github.com/user-attachments/assets/14901599-cd7b-45ba-924b-846bf46df31d)  
🔍 More previews 👉 [here](simgas_preview.md)

---

## 🗂️ Pre-computed Masks  
📁 `result_full` — SimGas masks + animated webp  
📁 `gasvid_res_full` — GasVid masks + webp  

🔍 Previews 👉 [SimGas](simgas_preview.md) | [GasVid](gasvid_preview.md)

---

## 🧪 Results

| ⚙️ BGS | 🔎 VLM | 🌀 Temporal | ✂️ Seg | 🔧 τ₍VLM₎ | 🧍‍♂️ Stationary (I/P/R/FLA) | 🏃‍♂️ Moving (I/P/R/FLA) | 📊 Overall (I/P/R/FLA) |
|-------|--------|-------------|--------|--------|-----------------------------|--------------------------|------------------------|
| ✅    | ❌      | ❌           | ❌     | x      | 0.56 / 0.64 / 0.83 / 0.85    | 0.38 / 0.53 / 0.58 / 0.69 | 0.50 / 0.61 / 0.73 / 0.79 |
| ✅    | ✅      | ❌           | SAM 2  | 0.09   | 0.67 / 0.81 / 0.79 / ⭐0.88   | 0.54 / 0.79 / 0.65 / 0.83 | 0.62 / 0.80 / 0.74 / 0.86 |
| ❌    | ✅      | ✅           | SAM 2  | 0.19   | 0.22 / 0.39 / 0.28 / 0.57    | 0.46 / 0.65 / 0.59 / 0.74 | 0.31 / 0.49 / 0.40 / 0.63 |
| ✅    | ✅      | ✅           | Trad.  | 0.12   | 0.57 / ⭐0.85 / 0.65 / 0.83   | 0.35 / ⭐0.88 / 0.37 / 0.72 | 0.49 / ⭐0.86 / 0.55 / 0.79 |
| ✅    | ✅      | ✅           | SAM 2  | 0.12   | ⭐0.70 / 0.83 / ⭐0.82 / 0.87  | ⭐0.69 / 0.79 / ⭐0.84 / ⭐0.92 | ⭐0.69 / 0.82 / ⭐0.82 / ⭐0.89 |

📌 *Ablation study for IoU (I), Precision (P), Recall (R), and Frame-Level Accuracy (FLA)*  
📎 Trad. = Otsu + Morph transforms  
⚠️ IoU depends on method; we use average per-video

---

## 🧪 Test on SimGas

### 📦 Step 1: Install Packages  
```bash
pip3 install opencv-python transformers tqdm Pillow bitesandbytes wandb
```

---

### 📥 Step 2: Download + Install SAM-2  
```bash
git clone https://github.com/facebookresearch/sam2.git
mv sam2 .sam2
cd .sam2
pip3 install -e .
```

📛 Installation failed? Try:
```bash
echo -e '[build-system]\nrequires = [\n    "setuptools>=62.3.0,<75.9",\n    "torch>=2.5.1",\n    ]\nbuild-backend = "setuptools.build_meta"' > pyproject.toml
```
Then:
```bash
pip3 install -e .
```

⬇️ Download checkpoints:
```bash
cd checkpoints
bash download_ckpts.sh
```

📘 More info: [SAM-2 GitHub](https://github.com/facebookresearch/sam2)

---

### 🎞️ Step 3: Download Dataset  
🔗 [https://paperswithcode.com/dataset/simgas](https://paperswithcode.com/dataset/simgas)  
🗂️ Place videos into `simulated_gas/`  
▶️ Run:
```bash
python3 dataprep.py
```

---

### 🚀 Step 4: Run the Code  
🛠️ Edit `owl_notracking.py` to set `sam2_checkpoint`  
💡 Do *not* touch config path

▶️ Run:
```bash
python3 owl_notracking.py [OPTIONS]
```

📘 Example:
```bash
python3 owl_notracking.py --temporal_filter [OPTIONS]
```

---

## 🌫️ Test on GasVid

### ⚙️ Preparation  
1️⃣ Download from:  
[GasVid Google Drive](https://drive.google.com/drive/folders/1JKEMtCGPSq2IqGk4uXZP9A6gr8fyJsGC)  
2️⃣ Delete videos >18m depth  
3️⃣ Put videos into `Videos/`  
4️⃣ Run:
```bash
bash convert_gasvid.sh
```

---

### ▶️ Running the Test  
🛠️ Edit `run_gasvid.py` + `owl_gasvid.py` paths  

📈 Evaluate full dataset:
```bash
python3 run.py
```

🎥 Single video:
```bash
python3 owl_gasvid.py --video_id [video id]
```

📂 Results in `gasvid_res_full/`
