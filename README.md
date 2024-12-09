# 🧬🤖 disease-gene-predictor

Team name: __Rocket Team__  
Team members:

- Laszlo Patrik Abrok (JPWF8N)
- Janos Hudak (CP4EIQ)

For evaluation instructions, please see the "Deployment and Evaluation" part under Quick Start.

## ℹ️ Project description

This is our semester project for the [VITMMA19 Deep Learning course](https://www.tmit.bme.hu/vitmma19). It leverages graph neural networks and real-life medical data from [DISGENET](https://disgenet.com/) to predict disease-gene associations.

The project includes a VGAE model and a learning framework based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), a semi-automatic LaTeX documentation solution and a Gradio MLaaS application as well.  

## 📊 Data and Files

The project contains DISGENET gene and disease association data, accessed through the API key provided by the Academic License. The source code includes a DisgenetClient class that, given a valid API key, will automatically build the dataset.

After processing all possible ICD-10 diseases (listed in disease_ids.txt) the dataset contains 300 diseases, 5024 genes and a total of 10379 associations. As mentioned before, the diseases were queried using [ICD-10 identifiers](https://icd.who.int/browse10/2019/en) based on information from the [Genes and Disease](https://www.ncbi.nlm.nih.gov/books/NBK22183/) book via the provided REST API.

After cleaning the data of unnecessary information in the DisgenetClient, the following information will be used as per [DISGENET schemas](https://disgenet.com/About#schema):

Diseases

- [ICD-10 category](https://icd.who.int/browse10/2019/en)

Genes

- [HGNC symbol](https://www.genenames.org/)
- [DSI](https://disgenet.com/About#metrics) (Disease Specificity Index)
- [DPI](https://disgenet.com/About#metrics) (Disease Pleiotropy Index)

Disease-gene associations

- [EI](https://disgenet.com/About#metrics) (Evidence Index)

Where the above information was not provided, the minimum of the interpretation range was chosen due to the formulae of the metrics.

The project also includes a DisgenetDatamodule class, which is responsible for generating the train, test and validation datasets. It does this using the [pandas](https://pandas.pydata.org/) and [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) libraries, using [HeteroData](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html?highlight=heterodata#torch_geometric.data.HeteroData).

The application checks that it has the required DISGENET API key and the generated data. If the latter is not present, it will be generated in a comma separated values file in about two minutes using around 100 API calls with the paging mechanism. After that, the data module processes the data and creates the train, test and validation datasets, which are exposed as [DataLoaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) via the corresponding functions.

## ⚙️ Model and Optimalization 

Running the pipeline, model training and model evaluation have to start after the data preparation ended. (When dga_data.cvs is present the '[INFO] Disgenet data found. Skipping data creation.' shall appear on the commandline.') To expose the dataloaders to the model a custom HeteroDataset dataset class [Datasets](https://pytorch.org/docs/stable/data.html#dataset-types) was implemented. 

Our solution consists of a custom VGAE implementation, which we optimized manually and with [Optuna](https://optuna.org/). The optimalization scripts and results can be found on a separate branch (#27-optimize-hyperparameters). The application loads a [Gradio](https://www.gradio.app/) web interface, which can be used to start a run by setting any parameters. 

Our documentation is also included in the root directory as a pdf file with the name "Dokumentáció.pdf". 

## 🚀 Quick Start

### Development

#### Requirements

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

#### Requirements

- [TeX Live](https://www.tug.org/texlive/)  
- [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)

After installation, the project documentation will build after every LaTeX file save.

### Deployment and Evaluation

#### Requirements

- [Docker](https://www.docker.com/)

#### Build the image locally  

```
docker build -t disease-gene-predictor .
```

#### Start the container  

```
docker run -e DISGENET_API_KEY=<paste-your-api-key-here> disease-gene-predictor
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

## 📜 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details or visit the [official Apache License 2.0 page](http://www.apache.org/licenses/LICENSE-2.0).
