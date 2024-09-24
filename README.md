# 🧬🤖 disease-gene-predictor

This is our semester project for the [VITMMA19 Deep Learning course](https://www.tmit.bme.hu/vitmma19). It leverages graph neural networks and real-life medical data from [DISGENET](https://disgenet.com/) to predict disease-gene associations.

The project will include a model and a learning framework based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), a semi-automatic LaTeX documentation solution and an MLaaS application as well.  

## 🎯 Milestones

1. Containerisation and Data  
  - [ ] Containerisation
  - [ ] Data collection
  - [ ] Data analyzation
  - [ ] Data preparation
  - [ ] Data cleaning
  - [ ] Evaluation methodology
2. Baseline
  - [ ] Baseline model
3. Improvement and MLaaS
  - [ ] Iterative improvement
  - [ ] Evaluation
  - [ ] MLaaS

## 🚀 Quick Start

### Development 

Requirements: 
- [poetry v1.8](https://python-poetry.org/)

After meeting the requirements, execute the following commands: 
```
poetry install
poetry shell
```

### Documentation

Requirements:
- [TeX Live](https://www.tug.org/texlive/)  
- [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)

After installation, the project documentation will build after every LaTeX file save. 

### Deployment

## 📚 Resources

### node2vec: Scalable Feature Learning for Networks
- [📒 Article](https://arxiv.org/pdf/1607.00653)
- [🎞️ Research Paper Walkthrough](https://www.youtube.com/watch?v=LpwGZG5j_q0)
- [🎞️ Stanford Graph ML](https://youtu.be/Xv0wRy66Big?si=lA87djJRxRTvdpPv&t=1049)

### Variational Graph Auto-Encoders
- [📒 Article](https://arxiv.org/pdf/1611.07308)
- [🎞️ Tutorial](https://www.youtube.com/watch?v=hZkLu2OaHD0)
- [🤖 TensorFlow example](https://github.com/tkipf/gae)
