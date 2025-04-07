# LangGas: Linguam Introducens in Subtractione Selectiva Fundi Zero-Shot ad Detectionem Gasorum Semi-Translucidorum cum Nova Collectione Data

Wenqi Guo, Yiyang Du, Shan Du  
Universitas Columbiae Britannicae  
Cohors Observationis et Custodiae Emissionum Methani (MEOW), Weasoft Software

[arXiv](https://arxiv.org/abs/2503.02910)  
[Papers with Code](https://paperswithcode.com/sota/segmentation-on-simgas?p=langgas-introducing-language-in-selective)

[Latina](latin_readme.md)  
[繁体中文](traditional_chinrse_readme.md)  
[Mewspeak](mewspeak_readme.md)

## Summarium

Effluxus gasorum periculum grave praebet quod praecavendum est. Tradi­tionaliter, inspectio humana adhibita est ad detegendum, processus lentus et laboriosus. Investigatio recentior technicas machinae discendi ad hunc finem applicavit, sed adhuc desunt collectanea datae publicae et altae qualitatis. In hoc documento introducimus collectionem syntheticam nomine SimGas, quae varias rationes fundorum, obiecta interponentia in primo plano, loca effluxus varia, et veritatem segmentationis accuratam continet. Proponimus methodum zero-shot quae coniungit subtractionem fundi, detectionem obiecti zero-shot, colationem, et segmentationem ad hanc collectionem utendam. Experimenta ostendunt nostram methodum multo melius se gerere quam methodi basales fundatae tantum in subtractione fundi et detectione obiecti zero-shot cum segmentatione, assequendo intersectionem super unione (IoU) 69% in summa. Praeterea, analysim praebemus de variis configurationibus incitamentorum et valoribus liminum ad clariorem intellectum methodi nostrae. Collectio datae post publicationem praesto erit.

## Methodi Conspectus

![meow2](https://github.com/user-attachments/assets/02debfe3-7da5-47e3-8720-d70cf3aee802)

## Collectionis Conspectus

![image](https://github.com/user-attachments/assets/14901599-cd7b-45ba-924b-846bf46df31d)  
Plura videre potes [hic](simgas_preview.md).

## Larvae Praecomputatae

`result_full` omnes larvas praecomputatas et imagines animatas in forma webp pro SimGas continet, et `gasvid_res_full` larvas praecomputatas et webp pro GasVid. Imagines animatas videre potes hic: [SimGas](simgas_preview.md) [GasVid](gasvid_preview.md).

## Resultata

| BGS        | VLM Filtratio | Filtratio Temporalis | Seg.   | τ₍VLM₎ | Stans Anterior I/P/R/FLA | Movens Anterior I/P/R/FLA | Summa I/P/R/FLA |
|------------|---------------|---------------------|--------|--------|--------------------------|--------------------------|-----------------|
| ✓          |               |                     | Nulla  | -      | 0.56 / 0.64 / 0.83 / 0.85 | 0.38 / 0.53 / 0.58 / 0.69 | 0.5 / 0.61 / 0.73 / 0.79 |
| ✓          | ✓             |                     | SAM 2  | 0.09   | 0.67 / 0.81 / 0.79 / **0.88** | 0.54 / 0.79 / 0.65 / 0.83 | 0.62 / 0.80 / 0.74 / 0.86 |
|            | ✓             | ✓                   | SAM 2  | 0.19   | 0.22 / 0.39 / 0.28 / 0.57 | 0.46 / 0.65 / 0.59 / 0.74 | 0.31 / 0.49 / 0.4 / 0.63 |
| ✓          | ✓             | ✓                   | Trad.  | 0.12   | 0.57 / **0.85** / 0.65 / 0.83 | 0.35 / **0.88** / 0.37 / 0.72 | 0.49 / **0.86** / 0.55 / 0.79 |
| ✓          | ✓             | ✓                   | SAM 2  | 0.12   | **0.70** / 0.83 / **0.82** / 0.87 | **0.69** / 0.79 / **0.84** / **0.92** | **0.69** / 0.82 / **0.82** / **0.89** |

**Tabula: Studium ablationis variarum partium cum IoU (I), Praecisione (P), Recuperatione (R), et Acuratione gradus imaginum (FLA).** In columna segmentationis (Seg.), traditum significat methodum Otsu cum transformationibus morphologicis. Haec analysis respondet studio nostro in Sectione 4 documenti.

Monitum: Modi diversi calculandi IoU possunt resultata incongrua producere. Nos aggregationem per video adhibuimus et deinde mediocrem omnium videorum fecimus.

## Experimentum in SimGas

### Gradus 1: Institutionem Praerequisitorum

Institutionem pytorch secundum https://pytorch.org/get-started/locally/

```bash
pip3 install opencv-python transformers tqdm Pillow wandb matplotlib scipy
```

Deinde, curras:

```bash
pip3 install bitsandbytes 'accelerate>=0.26.0'
```

### Gradus 2: Depone et Installa SAM-2

```bash
git clone https://github.com/facebookresearch/sam2.git
mv sam2 .sam2
cd .sam2
pip3 install -e .
```

Si institutio deficiat, curre:

```bash
echo -e '[build-system]\nrequires = [\n    "setuptools>=62.3.0,<75.9",\n    "torch>=2.5.1",\n    ]\nbuild-backend = "setuptools.build_meta"' > pyproject.toml
```

(Consule https://github.com/facebookresearch/sam2/issues/611 pro pluribus)

Deinde curre:

```bash
pip3 install -e .
```

Depone checkpoint:

```bash
cd checkpoints
bash download_ckpts.sh
```

Plura: https://github.com/facebookresearch/sam2

### Gradus 3: Depone Collectionem

I ad https://paperswithcode.com/dataset/simgas et depone collectionem.

Pone videos collectae in fasciculum `simulated_gas` et curre `python3 dataprep.py`

### Gradus 4: Exerce Codicem

#### Mutatio `owl_notracking.py` pro via SAM-2.

Mutetur `sam2_checkpoint = "../../.sam2/checkpoints/sam2.1_hiera_small.pt"` ad tuam viam checkpoint. Configuratio non mutetur.

#### Curre Codicem

Curre `python3 owl_notracking.py` cum sequentibus optionibus. Si usus es `dataprep.py`, radix via debet esse `sim`.

```bash
Usage: owl_notracking.py [OPTIONS]

Options:
  --video_id TEXT           ID video, ut vid20
  --root_path TEXT          Radix via ad collectionem (default: "../sim")
  --display TEXT            Servus visualizationis (default: "localhost:10.0")
  --log_file TEXT           Nomen tabellae log (default: "results.csv")
  --temporal_filter         Activa filtrationem temporalem (flag)
  --vlm_threashold FLOAT    Limes decisionis VLM (default: 0.12)
  --positive_prompt TEXT    Incitatio positiva pro analysi VLM (default: "white steam")
```

Ad eventus nostros replicandos, utere vexillo `--temporal_filter`

## Experimentum in GasVid

### Praeparatio

1. Depone collectionem ex auctore originali:  
   [https://drive.google.com/drive/folders/1JKEMtCGPSq2IqGk4uXZP9A6gr8fyJsGC](https://drive.google.com/drive/folders/1JKEMtCGPSq2IqGk4uXZP9A6gr8fyJsGC)

2. Remove omnes videos captae in profunditate maiori quam 18 metra, ut in documento originali specificatum est.

3. Pone reliqua videos in directory `Videos`.

4. Curre `convert_gasvid.sh` ad extrahendas imagines.

### Exerce Experimentum

1. Edit `run_gasvid.py` et `owl_gasvid.py` ad viam collectionis recte constituendam.

2. Ad totam collectionem aestimandam, curre:  
   `python3 run_gasvid.py`

3. Ad singulum video aestimandum, curre:  
   `python3 owl_gasvid.py --video_id [video id]`

4. Resultata output in directory `gasvid_res_full` conduntur.
