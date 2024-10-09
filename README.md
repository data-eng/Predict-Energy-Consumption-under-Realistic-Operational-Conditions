
# Predict-Energy-Consumption-under-Realistic-Operational-Conditions  
Applying Transformer methods for predicting energy consumption. Founded by https://navgreen.gr.  

## Requirements
To get the necessary libraries refer to the `poetry` configuration provided.

## Data
Data is publicly available at: http://cogsys.imm.dtu.dk/propulsionmodelling/data.html.  

We aggregate the available data to 3 minutes, maintaining their mean and standard deviation.   
The data processing is available at [./data_creation](https://github.com/data-eng/Energy-Consumption-NavGreen/tree/master/data_creation) and the processed data at [./data_creation/data](https://github.com/data-eng/Energy-Consumption-NavGreen/tree/master/data_creation/data) 

## Training and Evaluation Pipeline

- [./base/hist_data_analysis/transformer/loader.py](https://github.com/data-eng/Energy-Consumption-NavGreen/blob/master/base/hist_data_analysis/transformer/loader.py) : Provides the *data loading* for the inference and training. It processes the **timeseries** with a fixed length,  adds the **time representation**, assures the correct ordering of the timeseries and handles the training, validation and testing **data splits**.

-  [./base/hist_data_analysis/transformer/model.py](https://github.com/data-eng/Energy-Consumption-NavGreen/blob/master/base/hist_data_analysis/transformer/model.py) : Provides the simple **attention-based**, Transformer model for the regression task.

-  [./base/hist_data_analysis/transformer/train_eval.py](https://github.com/data-eng/Energy-Consumption-NavGreen/blob/master/base/hist_data_analysis/transformer/train_eval.py) : Provides the **training and evaluation pipeline**. It, also, *stores* the training information and evaluation results and provides some basic visualization relevant to the task.
> To run the pipeline, one simply can execute this script.

- [./base/hist_data_analysis/transformer/utils.py](https://github.com/data-eng/Energy-Consumption-NavGreen/blob/master/base/hist_data_analysis/transformer/utils.py) : Provides utility functions for the *loading*, *storing* and *plotting* as well as different **loss function** implementations.
