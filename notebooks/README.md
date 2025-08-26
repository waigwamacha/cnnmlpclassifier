#### Notebooks Documentation

Below are brief explanations of the notebooks in this folder and their functions:

1. ```cnnmlp.ipynb```: this notebook has a CNN-MLP that takes in a T1w image as input and performs a classification task to predict participant sex in addition to performing regression to predict age. This architecture had been shown to improve brain age prediction in Joo et al., 2023.

2. ```cnnclassifier.ipynb```: this notebook has a CNN-MLP-Classifier. In addition to performing prediction of age, there is a classificationt task that attempts to predict the age class that a participant belongs to (the participants are placed in different classes based on their ages). The code in this notebook has been transformed into a script thats in the ```src/``` directory; this script can be run as ```python3 train.py```

3. ```test_bhrc.ipynb```: script to test on the BHRC dataset. The ```test_frb.ipynb``` and ```test_abcd.ipynb``` are similar to this script.


##### References

1. Joo, Y., Namgung, E., Jeong, H., Kang, I., Kim, J., Oh, S., Lyoo, I. K., Yoon, S., & Hwang, J. (2023). Brain age prediction using combined deep convolutional neural network and multi-layer perceptron algorithms. Scientific reports, 13(1), 22388. https://doi.org/10.1038/s41598-023-49514-2
