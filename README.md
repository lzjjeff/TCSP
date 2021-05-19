# TCSP

Code for ACL 2021 Findings paper: *[A Text-Centered Shared-Private Framework via Cross-Modal Prediction for Multimodal Sentiment Analysis]()*

![model](./img/TCSP.jpg)



## Dependencies

* Python 3.7.1
* PyTorch 1.7.1
* Numpy 1.19.2
* Scikit-Learn 0.24.1
* CUDA 10.2



## Getting Start

Hi, if you are familiar to git,  you can easily clone this project via:

```bash
git clone https://github.com/lzjjeff/TCSP.git
cd TCSP
```

Then, you can create a runnable environment via:

```bash
conda env create -f environment.yml
```



## Data

The data for experiment are placed in `./data/`, before training, you need to create data folds via:

```bash
mkdir data
cd data
mkdir MOSI MOSEI
```

You can download the processed MOSI and MOSEI datasets in *[GoogleDrive]()* / *[BaiduCloud]()* and place them to `./data/MOSI/` and`./data/MOSEI` .

For more specific introduction about the two datasets, please refer to *[CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK)*.



## Train

