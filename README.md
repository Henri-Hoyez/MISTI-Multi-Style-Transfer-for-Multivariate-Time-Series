# MISTI: Multi-Style Transfer for Multivariate Time Series
## Overview

This repository contains the code of the the research paper "MISTI: Multi-Style Transfer for Multivariate Time Series" We introduces a novel multi-style transfer algorithm designed for multivariate time series. The implementation provided here allows users to reproduce the paper's results and apply the method to new datasets.

## Installation

Install dependencies:
```
pip install -r requirements.txt
```

## Training

To train the model, use the following command:

```
python train.py --task TaskName --exp_name ExpName --epochs n_epochs --exp_folde ExpFolderName --epochs n_epochs
```

### Parameters:

- ``--task``: Specifies the task to be performed (e.g.,TimeShift). All task names can be found in ``configsget_data_config.py``

- ``--exp_folder``: The directory where experiment resultswill be stored.

- ``--exp_name``: A descriptive name for the experiment. It will also define the name of the saving folder. At theend of the training, the model and its trainingparameters will be saved in ``to_evaluate/ExpFolder/ExpName``.
- ``--epochs``: Number of training epochs.

## Logging

All logging information, including generation during training and loss values, is stored in the logs folder. Logs can be accessed and visualized using TensorBoard with the following command: 
```
tensorboard --logdir=logs
```

## Evaluation

The evaluation consists of two steps:

1. Generating Model Predictions Run generate.py, providing the path to the trained model as an argument: ``python generate.py path/to/trained_model``

2. Running Evaluation Once predictions are generated, run evaluate.py with the same folder argument: ``python evaluate.py path/to/trained_model``


## Training on a New Multivariate Time Series Dataset

To train the model on a new multivariate time series dataset, you need to create a new dataset configuration class in ``configs/get_data_config.py``.

```pyton
class NewDataset():
    def __init__(self) -> None:
        self.unsupervised = False  # Set to True if labels are unavailable
        self.drop_labels = False    # Set to True to ignore labels during training
        self.univariate = False     # Set to True for single-variable time series
        
        self.sequence_length_in_sample = 128  # Define sequence length
        self.granularity = 2  # Defines the sampling granularity
        self.n_feature = 6  # Number of input features
        
        self.overlap = 0.02  # Defines the overlap between sequences
        
        self.n_classes = 5  # Number of classes in the dataset
        self.n_styles = 3  # Number of style variations
        
        self.style_vector_size = 16  # Size of the style encoding vector
        
        self.learning_rate = 0.002  # Training learning rate
        self.batch_size = 64  # Training batch size
        
        self.n_wiener = 2  # Wiener process factor for noise modeling
        self.n_sample_wiener = self.sequence_length_in_sample // 8
        
        ##### Generator loss parameters
        self.l_reconstr = 2.0  # Reconstruction loss weight
        self.l_local = 6.0  # Local adversarial loss weight
        self.l_global = 6.0  # Global adversarial loss weight
        self.l_style_preservation = 1.5  # Style preservation loss weight
        
        ## Local vs. global reconstruction loss balance
        self.l_global_reconstr = 0.3
        self.l_local_reconstr = 1 - self.l_global_reconstr
        
        ##### Content encoder loss
        self.l_content = 2.0  # Content loss weight
        self.encoder_adv = 0.1  # Adversarial loss weight for encoder
        self.encoder_recontr = 2.0  # Reconstruction loss weight for encoder
        
        ##### Style Encoder
        self.l_disentanglement = 4.0  # Disentanglement loss weight
        self.l_triplet = 4.0  # Triplet loss weight
        self.triplet_r = 0.01  # Triplet margin factor
        
        ##### Dataset Paths
        self.content_dataset = "data/NewDataset/content.h5"  # Path to content dataset
        
        self.style_datasets = [
            "data/NewDataset/style1.h5",
            "data/NewDataset/style2.h5",
            "data/NewDataset/style3.h5"
        ]
```

Then modify the training command to use the new dataset:
```
python train.py --task NewDataset --exp_nameMyExperiment --epochs 100 --exp_folder new_dataset_training
```


## Usefull links:

- **CorMet:** https://github.com/Henri-Hoyez/CorMet
- **Synthetic Dataset:** https://github.com/Henri-Hoyez/ChemicalStateSpaceModel


## Citation 
```
Comming Soon.
```