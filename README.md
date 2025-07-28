# Neural Two-Level Monte Carlo Real-Time Rendering

![](TeaserNew.png)

**Official implementation of "Neural Two-Level Monte Carlo Real-Time Rendering"**

**Mikhail Dereviannykh**, Dmitrii Klepikov, Johannes Hanika, Carsten Dachsbacher
Karlsruhe Institute of Technology

**Eurographics, 2025 - Computer Graphics Forum**
🏆 **"Best Paper Honorable Mention"** 🏆
📅 **To be presented at SIGGRAPH 2025**

[**Project Page**](https://mishok43.github.io/nirc/)

## Abstract

We introduce an efficient **Two-Level Monte Carlo** (subset of Multi-Level Monte Carlo, MLMC) estimator for real-time rendering of scenes with global illumination. Using MLMC we split the shading integral into two parts: the radiance cache integral and the residual error integral that compensates for the bias of the first one. For the first part, we developed the **Neural Incident Radiance Cache (NIRC)** leveraging the power of tiny neural networks as a building block, which is trained on the fly.

## Installation

This implementation is based on **Falcor 7.0**. Please follow the complete installation guide for Falcor and all its dependencies:

📖 **[Falcor Installation Guide](https://github.com/NVIDIAGameWorks/Falcor)**

### Additional Dependencies

This project requires the following additional dependencies beyond Falcor:

1. **tiny-cuda-nn**: Required for neural network training
   ```bash
   git clone --recursive https://github.com/NVlabs/tiny-cuda-nn
   cd tiny-cuda-nn/bindings/torch
   python setup.py install
   ```
   📖 **[tiny-cuda-nn Installation Guide](https://github.com/NVlabs/tiny-cuda-nn)**

2. **Python Dependencies**: Install required Python packages
   ```bash
   pip install torch numpy matplotlib tqdm pyexr pyyaml
   ```

## Usage

### Running Experiments

The main experiment script reproduces results from **Figure 6** and **Section 6.1** of our paper:

```bash
# Navigate to the scripts directory
cd scripts/ours

# Run the main experiment notebook
jupyter notebook ours.ipynb
```

### Available Scenes

Due to repository size limitations, only the **Cornell Box** scene is provided. The script can be adapted for other scenes by adding scene files to the appropriate data directory


## Third-Party Components

This project builds upon several open-source libraries:

### Zunis Library
We include a **modified version** of the Zunis library for Monte Carlo integration:
- **Original**: [Zunis GitHub Repository](https://github.com/ndeutschmann/zunis)
- **Modifications**: Adapted for real-time rendering and integration with NIRC

### Falcor Framework
- **Repository**: [Falcor GitHub](https://github.com/NVIDIAGameWorks/Falcor)
- **License**: See [Falcor LICENSE.md](https://github.com/NVIDIAGameWorks/Falcor/blob/master/LICENSE.md)

### tiny-cuda-nn
- **Repository**: [tiny-cuda-nn GitHub](https://github.com/NVlabs/tiny-cuda-nn)
- **License**: See [tiny-cuda-nn LICENSE](https://github.com/NVlabs/tiny-cuda-nn/blob/master/LICENSE)

## License

This project is made available under the most permissive licensing terms allowed by its dependencies:

- **Falcor Framework**: BSD 3-Clause License
- **tiny-cuda-nn**: BSD 3-Clause License
- **Zunis Library**: MIT License

The implementation code specific to our Neural Two-Level Monte Carlo method is released under the **BSD 3-Clause License** to maintain compatibility with all dependencies.

**Note**: Some NVIDIA RTX SDKs included with Falcor (DLSS, RTXDI, NRD) may have different licensing terms. Please refer to their respective licenses for commercial use.

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{dereviannykh2025neural,
  author = {Dereviannykh, Mikhail and Klepikov, Dmitrii and Hanika, Johannes and Dachsbacher, Carsten},
  title = {Neural Two-Level Monte Carlo Real-Time Rendering},
  journal = {Computer Graphics Forum},
  volume = {44},
  number = {2},
  pages = {e70050},
  year = {2025},
  doi = {https://doi.org/10.1111/cgf.70050},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.70050}
}
```

## Acknowledgments

We thank the developers of:
- **Falcor**: NVIDIA's real-time rendering framework
- **tiny-cuda-nn**: NVIDIA's efficient neural network library
- **Zunis**: Monte Carlo integration library by Nicolas Deutschmann
