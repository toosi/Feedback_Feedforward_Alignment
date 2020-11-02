# Symbiotic Learning


## Control Discriminators

## Control Autoencoders
We have two types of control autoencoders:
1. End-to-End reconstruction cost
    * Training: main_train_autoencoders.py
    * Evaluation: main_evaluate_autoencoders.py
1. Two-objective autoencoders: discriminative encoder and 
reconstrucntion
    * Training: main_train_autoencoders_twocosts.py
    * Evaluation: main_evaluate_autoencoders_twocosts.py  


### Things to remember:
After running the autoencoder with two cost functions:
    * In the main_train there was a second optimizer.step() that I removed after
    submission to NeurIPS' workshop.

    * In create_config, when creating modelB, algorithm was 'BP' (line 297), I changed it to FA

    * 10-19-2020: I changed the modules used in ResLNets from layerwise to simple to test wether I could replicate SLVanilla 81.
    * 10-19-2020: in modules layerwise, I added custom ReLU which has nonlinear backward, and ommited inplace=True

    * in both main_train_autoencoder and main_train_autoencoder_twocosts I had the schedulerF operate on 'max'imizing the accuracy which is obviously wrong, I corrected it on Oct 28th and ran TwoCostAEcontrolMNISTCorrSchedul  
## SL + BiHebb using PCGrad

In *main_train_PCGrad* I force modelF and modelB to tolerate each other's gradient usnig PCGrad.  