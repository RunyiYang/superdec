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

Download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1_pEHMEWdsNjHX86blL7Zgjs239xPJ7j6?usp=share_link):

| Model | Dataset | Normalized | Link |
|:------|:--------|:-----------|:-----|
| ShapeNet | ShapeNet | ❌ | [📁 Download](https://drive.google.com/drive/folders/1kXgJJ_6SvvJt6kh53rs30feAnD-i4SBL?usp=share_link) |
| Normalized | ShapeNet | ✅ | [📁 Download](https://drive.google.com/drive/folders/1a-mV8FH6YSA0TQyDdvbeaicHf9tPfZrR?usp=share_link) |

> **Note:** We use the ShapeNet checkpoint to evaluate on ShapeNet and the Normalized checkpoint to evaluate on objects from generic 3D scenes.

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

## 📊 Results

Our method achieves state-of-the-art performance on ShapeNet benchmark for primitive-based object decomposition. Detailed quantitative results and comparisons are available in our [paper](https://arxiv.org/abs/2504.00992).

## 📁 Project Structure

```
superdec/
├── data/                 # Dataset storage
├── scripts/             # Utility scripts
├── superdec/           # Main package
│   ├── evaluate/       # Evaluation scripts
│   └── visualization/ # Visualization tools
├── trainer/            # Training scripts
└── requirements.txt   # Dependencies
```

## 🛣️ Roadmap

- [x] Core implementation and visualization
- [x] ShapeNet training and evaluation
- [ ] Instance segmentation pipeline
- [ ] Path planning integration
- [ ] Grasping applications
- [ ] Superquadric-conditioned image generation

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 Citation

If you find SuperDec useful in your research, please consider citing:

```bibtex
@inproceedings{fedele2025superdec,
  title={SuperDec: 3D Scene Decomposition with Superquadric Primitives},
  author={Fedele, Elisabetta and Sun, Boyang and Guibas, Leonidas and Pollefeys, Marc and Engelmann, Francis},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:

- [superquadric_parsing](https://github.com/paschalidoud/superquadric_parsing)
- [CuboidAbstractionViaSeg](https://github.com/SilenKZYoung/CuboidAbstractionViaSeg)
- [volumentations](https://github.com/kumuji/volumentations)
- [LION](https://github.com/nv-tlabs/LION)
- [occupancy_networks](https://github.com/autonomousvision/occupancy_networks)
- [convolutional_occupancy_networks](https://github.com/autonomousvision/convolutional_occupancy_networks)

Thanks to all authors for making their code and data publicly available!

## 📧 Contact

For questions or collaborations, please contact [Elisabetta Fedele](mailto:efedele@ethz.ch).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.