import os

#fastai expects every custom method to be in the same path when exporting and loading models
#this means that we need to have an shared file for both the deployment and the notebook
#An better solution to this would be to export the model in a more shared format instead of an fastai object
#Like the ONNX runtime format, wich wouldn't need the entire fastai api and pytorch to run

def label_func(f): 
    return os.path.basename(os.path.dirname(f))