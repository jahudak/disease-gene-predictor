# 🧬🤖 disease-gene-predictor

Team name: __Rocket Team__  
Team members: 
- Laszlo Patrik Abrok (JPWF8N)
- Janos Hudak (CP4EIQ) 

## ℹ️ Project description

This is our semester project for the [VITMMA19 Deep Learning course](https://www.tmit.bme.hu/vitmma19). It leverages graph neural networks and real-life medical data from [DISGENET](https://disgenet.com/) to predict disease-gene associations.

The project will include a model and a learning framework based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), a semi-automatic LaTeX documentation solution and an MLaaS application as well.  

## 📊 Data

The project contains DISGENET gene and disease association data, accessed through the API key provided by the Academic License. The source code includes a DisgenetClient class that, given a valid API key, will automatically build the dataset.

The dataset currently contains 300 diseases, 5024 genes and a total of 10379 associations. Diseases were queried using [ICD-10 identifiers](https://icd.who.int/browse10/2019/en) based on information from the [Genes and Disease](https://www.ncbi.nlm.nih.gov/books/NBK22183/) book via the provided REST API. 

After cleaning the data of unnecessary information, the following information will be used: 

Diseases
- ICD-10 category

Genes 
- DSI
- DPI

Disease-gene associations
- EI

Where this information was not provided, the minimum of the interpretation range was chosen due to the formulae of the metrics.

As an evaluation metric, we chose the most accurate determination of breast cancer disease-gene associations beyond basic accuracy.

## 🎯 Milestones

1. Containerisation and Data  
  - [x] Containerisation
  - [x] Data collection
  - [x] Data analyzation
  - [x] Data preparation
  - [x] Data cleaning
  - [x] Evaluation methodology
2. Baseline
  - [ ] Baseline model
3. Improvement and MLaaS
  - [ ] Iterative improvement
  - [ ] Evaluation
  - [ ] MLaaS

## 🚀 Quick Start

### Development 

#### Requirements: 
- [poetry v1.8](https://python-poetry.org/)

After meeting the requirements, execute the following commands: 
```
poetry install
poetry shell
```

If you want to run the application without docker, the `DISGENET_API_KEY` environment value must be set through the console, e.g. in PowerShell:

```
$env:DISGENET_API_KEY = "paste-your-api-key-here"
```

After this, execute the following command: 

```
python main.py
```

### Documentation

#### Requirements:
- [TeX Live](https://www.tug.org/texlive/)  
- [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)

After installation, the project documentation will build after every LaTeX file save. 

### Deployment

#### Requirements: 
- [Docker](https://www.docker.com/)

#### Build the image locally  
```
docker build -t disease-gene-predictor .
```

#### Start the container  
```
docker run --env-file .env -e DISGENET_API_KEY=<paste-your-api-key-here> disease-gene-predictor
```

## 📚 Resources

### node2vec: Scalable Feature Learning for Networks
- [📒 Article](https://arxiv.org/pdf/1607.00653)
- [🎞️ Research Paper Walkthrough](https://www.youtube.com/watch?v=LpwGZG5j_q0)
- [🎞️ Stanford Graph ML](https://youtu.be/Xv0wRy66Big?si=lA87djJRxRTvdpPv&t=1049)

### Variational Graph Auto-Encoders
- [📒 Article](https://arxiv.org/pdf/1611.07308)
- [🎞️ Tutorial](https://www.youtube.com/watch?v=hZkLu2OaHD0)
- [🤖 TensorFlow example](https://github.com/tkipf/gae)
