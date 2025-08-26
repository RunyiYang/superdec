<p align="center">
 <h1 align="center">SuperDec: 3D Scene Decomposition with Superquadric Primitives</h1>
<p align="center">
<a href="https://elisabettafedele.github.io/">Elisabetta Fedele</a><sup>1,2</sup>,
<a href="https://boysun045.github.io/boysun-website/">Boyang Sun</a><sup>1</sup>,
<a href="https://geometry.stanford.edu/?member=guibas">Leonidas Guibas</a><sup>2</sup>,
<a href="https://people.inf.ethz.ch/pomarc/">Marc Pollefeys</a><sup>1,3</sup>,
<a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>2</sup>
<br>
<sup>1</sup>ETH Zurich,
<sup>2</sup>Stanford University,
<sup>3</sup>Microsoft <br>
</p>
<h2 align="center">ICCV 2025 (<span style="color:
#c20000;"><strong>Oral</strong></span>)</h2>
<h3 align="center"><a href="https://github.com/elisabettafedele/superdec">Code</a> | <a href="https://arxiv.org/abs/2504.00992">Paper</a> | <a href="https://super-dec.github.io">Project Page</a> </h3>
<div align="center"></div>
</p>
<p align="center">
<a href="">
<img src="https://super-dec.github.io/static/figures/compressed/teaser/room0_1_bg.jpeg" alt="Logo" width="100%">
</a>
</p>
<p align="center">
<strong>SuperDec</strong> allows to represent arbitrary 3D scenes with a compact and modular set of superquadric primitives.
</p>
<br>


## 🚀 Quick Start

### Environment Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/elisabettafedele/superdec.git
cd superdec

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Build sampler (required for training only)
python setup_sampler.py build_ext --inplace
```

### Download Data

Download the ShapeNet dataset (73.4 GB):

```bash
bash scripts/download_shapenet.sh
```

The dataset will be saved to `data/ShapeNet/`.

### Download Pre-trained Models

Download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1_pEHMEWdsNjHX86blL7Zgjs239xPJ7j6?usp=share_link) and store them in `checkpoints`. Alternatively, you can download the individual folders using the links below.

| Model | Dataset | Normalized | Link |
|:------|:--------|:-----------:|:-----|
| shapenet | ShapeNet | ❌ | [shapenet](https://drive.google.com/drive/folders/1kXgJJ_6SvvJt6kh53rs30feAnD-i4SBL?usp=share_link) |
| normalized | ShapeNet | ✅ | [normalized](https://drive.google.com/drive/folders/1a-mV8FH6YSA0TQyDdvbeaicHf9tPfZrR?usp=share_link) |

> **Note:** We use the ShapeNet checkpoint to evaluate on ShapeNet and the Normalized checkpoint to evaluate on objects from generic 3D scenes.

### Project Structure
After having downloaded ShapeNet and the checkpoints, the following project structure is expected:
```
superdec/
├── checkpoints/          # Checkpoints storage
│   ├── normalized/       # Checkpoint and config for normalized objects
│   └── shapenet/         # Checkpoint and config for ShapeNet objects
├── data/                 # Dataset storage
│   └── ShapeNet/         # ShapeNet dataset
├── scripts/              # Utility scripts
├── superdec/             # Main package
├── trainer/              # Training scripts
└── requirements.txt      # Dependencies
```

## 🎯 Usage

### Training

**Single GPU training:**
```bash
python trainer/train.py
```

**Multi-GPU training (4 GPUs):**
```bash
torchrun --nproc_per_node=4 train/train.py
```

### Evaluation and Visualization

Generate results on ShapeNet test set:

```bash
# Convert results to NPZ format
python superdec/evaluate/to_npz.py

# Visualize results
python superdec/visualization/object_visualizer.py
```

> **Note:** Mesh generation may take time depending on the chosen resolution.



## 🙏  Acknowledgements
We adapted some codes from some awesome repositories including [superquadric_parsing](https://github.com/paschalidoud/superquadric_parsing), [CuboidAbstractionViaSeg](https://github.com/SilenKZYoung/CuboidAbstractionViaSeg), [volumentations](https://github.com/kumuji/volumentations), [LION](https://github.com/nv-tlabs/LION), [occupancy_networks](https://github.com/autonomousvision/occupancy_networks), and [convolutional_occupancy_networks](https://github.com/autonomousvision/convolutional_occupancy_networks). Thanks for making codes and data public available.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests. For more specific questions or collaborations, please contact [Elisabetta Fedele](mailto:efedele@ethz.ch).


## 🛣️ Roadmap

- [x] Core implementation and visualization
- [x] ShapeNet training and evaluation
- [ ] Instance segmentation pipeline
- [ ] Path planning 
- [ ] Grasping 
- [ ] Superquadric-conditioned image generation