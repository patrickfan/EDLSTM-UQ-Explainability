# EDLSTM-UQ-Explainability

We propose an explainable machine learning (ML) model with uncertainty quantification (UQ) to improve multi-step reservoir inflow forecasting. Traditional ML methods have challenges in forecasting inflows multiple days ahead, and lack explainability and UQ. To address these limitations, we introduce an encoder–decoder long short-term memory (ED-LSTM) network for multi-step forecasting, employ the SHapley Additive exPlanation (SHAP) technique for understanding the influence of hydrometeorological factors on inflow prediction, and develop a novel UQ method for prediction trustworthiness. 

# Prerequisite

To run the code, make sure these packages are installed in addition to the commonly used Numpy, Pandas, Matplotlib, sklearn, etc.
--- python (>=3.8, version 3.8.3 is used in this study)
--- TensorFlow (>=2.0, version 2.4.1 is used in this study)
--- Hyperopt (=0.2.5, used for hyper-parameters tuning)
--- SHAP

# Run the code
We use the ED-LSTM network as the encoder to extract the information from time-series data, then we train MLP networks for UQ based on the intermediate predictions from the LSTM encoder.

First run the LSTM-encoder for multi-step prediction:

`python main_PI3NN.py --data CRY --mode lstm_encoder --project ./examples/Reservoir_Inflow_proj/ --exp CRY --configs ./examples/Reservoir_Inflow_proj/CRY/configs_encoder.json
`


And run the PI3NN-MLP for UQ:

`python main_PI3NN.py --data CRY --mode PI3NN_MLP --project ./examples/Reservoir_Inflow_proj/ --exp CRY --configs ./examples/Reservoir_Inflow_proj/CRY/configs_PI3NN.json
`


For explainability:

`python multi_step_SHAP.py`

# Reference
Fan, M., Liu, S., Lu, D., Gangrade, S. and Kao, S.C., 2023. Explainable machine learning model for multi-step forecasting of reservoir inflow with uncertainty quantification. Environmental Modelling & Software, p.105849. 
https://doi.org/10.1016/j.envsoft.2023.105849
