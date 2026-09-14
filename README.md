Convolutional Neural Network Pipeline 

Attribuition
The original CNN script and model architecture were developed by a member
of Network Analysis and Modelling group at IPK Gatersleben. 
This version has been adapted and optimized by me for the present
Aegilops speltoides dataset, including data preprocessing, sequence
formatting, class assignment, model training/evaluation, and downstream
cis/trans classification.

This project uses a 1D CNN to classify genes based on genomic sequence information and investigate whether sequence patterns learned from A-chromosome genes can be transferred to B-chromosome genes in Aegilops speltoides.

Workflow

1. Extracted fixed-length regulatory sequences surrounding genes
2. One-hot encoded DNA sequences
3. Trained a 1D CNN on A-chromosome sequences
4. Applied the trained model to B-chromosome sequences
5. Addressed class imbalance through random downsampling
5. Evaluated predictions using Accuracy, AUROC, Precision, Recall, F1-score and a confusion matrix
6. Predict class probabilities 
7. Cis/trnas classification

