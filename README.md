submission to ISAI2024 "Implementing Deep Learning Pipeline Toward Explainable Osteoporosis Detection Using Knee Radiographs"
 : https://ieeexplore.ieee.org/document/10799234

Data used is 224x224 rgb images+clinical data from : https://www.kaggle.com/datasets/mohamedgobara/osteoporosis-database

best performance is F1-macro: 0.70, increased to 0.75 with 2-step classification 

to reproduce:
  1.install pip packages in requirements.txt
  2.Process dataset with segmenter.ipynb + some manual cleaning for outliers
  3.Run data integrity check with pathchecks.ipynb
  4.Run classification and GradCAM visualization with classification.ipynb
  5.(optional) view results with tensorboard, or open them in visualizer.ipynb

other files:
  resnetModel.py contains a replacement block for the final layer of Resnet (this is for GradCAM visualization)
  
two step classification is done(badly) in the branch "2-step classification", but code has not been cleaned

inference of all models are ran on CUDA 12.7 A100 GPU (total inference time should be around 1 hour for all 5 models)
