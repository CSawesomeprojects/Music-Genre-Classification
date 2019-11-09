# A Deep Learning Approach for Music Genre Classification
This repository contains project code for _CPSC 554X: Machine Learning and Signal Processing_.  
[Download the FMA-small dataset and the metadata here.](https://github.com/mdeff/fma)

## Code Description
`main.ipynb` has the main source code for processing the dataset.  
`Spectrograms.ipynb` generates choice spectrograms from each genre label, showing how some unique differences can be seen in each genre. 

## Initial List of Experiments
- Testing to see if mel-spectrogram or spectrogram leads to better results.
- Testing the existing PRCNN method and BRNN+Attention method with FMA. 
- Trying out different methods of calculating attention scores, if we go with the attention-based method. 
- Trying out different ways of combining the RNN+CNN in the parallel method - other types of ensembling, perhaps. 
- Trying out different # of layers for the RNN and CNNs in our models. 
- Adding additional hand-crafted feature-based classification and use ensembling methods. 

_(and more as we continue to do our research/experiment with different approaches…)_  

