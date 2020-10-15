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
    subission to NeurIPS' workshop.

## SL + BiHebb using PCGrad

In *main_train_PCGrad* I force modelF and modelB to tolerate each other's gradient usnig PCGrad.  