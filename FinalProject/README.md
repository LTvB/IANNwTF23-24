# Final project of the IANNwTF course 
submitted by Lane von Bassewitz

## Abstract
Artificial neural networks (ANNs) are increasingly integrated into various aspects of our lives, showcasing remarkable performance across numerous domains. 
However, they exhibit a significant weakness in sequential learning, often leading to catastrophic forgetting. 
This phenomenon results in a drastic drop in network performance when learning new tasks. 
To mitigate catastrophic forgetting, several methods have been proposed. Following the approach of Masse et al. (2018), this project implemented three of those methods: Elastic Weight Consolidation, Synaptic Intelligence, and Context-dependent Gating, combined with a feedforward network trained on the permuted MNIST task. 
All three methods notably enhanced accuracy in sequential task learning. 
Furthermore, the combination of these methods had an additional positive impact on mean accuracy. However, it was not possible to replicate the results reported from \citet{masse_alleviating_2018} in the scope of this work.

## How to run the code
All code can be found in the main.py file. To run the code, call the function trainings_loop with the following parameters (the default values are given): 
- epochs -> number of epochs used in training 
- num_tasks -> number of tasks the model should be trained on
- use_EWC=False -> True when EWC should be used as stabilisation Method; Further parameters for this Method: lambda_=1000, num_ewc_sample=8192
- use_SI=False -> True when SI should be used as stabilisation Method; Further parameters for this Method: param_c=2, param_xi=0.01 
- use_XdG=False -> True when EWC should be used as stabilisation Method; Further parameters for this Method: gating_percentage=0.8

Example given: trainings_loop(epochs=20, num_tasks=10, use_SI=True, param_c=0.5, param_xi=0.01, use_XdG=True)
