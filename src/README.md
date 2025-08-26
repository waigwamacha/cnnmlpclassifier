
#### Scripts Documentation 

1. ```cnn_classifier.py```: this is the class that defines the cnn-mlp-classifier model that we use.

2. ```dataset.py```: contains out Dataset class which we use to load the images and their labels as tensors for training/testing

3. ```data_setup.py```: this contains the ```generate_phenofile``` function that cleans the test csv and gets it ready for testing with the CNN-MLP-Classifier model.

4. ```train.py```: script to load train data to GPU in batches, train & validate it, save performance on the train and validation sets and save the best model

5. ```predict.py```: script to load best model, test images and their labels, and make predictions. The function returns a few outputs (loss, age, predicted age)

6. ```utils.py```: some helpful functions that are called by the scripts shown above


To run the train script in the background:

```{python}
nohup python src/train.py > {directory}/trialrun.txt 2>&1 &

```

