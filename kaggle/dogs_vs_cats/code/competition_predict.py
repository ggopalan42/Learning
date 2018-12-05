#! /usr/bin/env python

# imports
# import the necessary packages
from sklearn.linear_model import LogisticRegression
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.resnet50 import preprocess_input, decode_predictions
from config import dogs_vs_cats_config as config
from imutils import paths
import numpy as np
from collections import namedtuple
import pickle
import os


def load_model(model_path = config.LOGISTIC_TRAINED_MODEL):
    ''' Load the logistic model and return it '''
    with open(model_path, 'rb') as infh:
        model = pickle.load(infh)
    return model


def load_test_images_paths(test_image_path = config.TEST_COMPETITION_PATH):
    ''' Get the paths of the test images '''
    test_image_paths = list(paths.list_images(test_image_path))
    return test_image_paths


def  image_thru_resnet50_xfer(res_model, image_path):
    ''' Do the preprocessing step of passing the image through the ResNet50 lower level
        nets (and not the final FC layers)
        That is, process the image exactly as it would before being fed to the logistic classifier
        during the training step'''
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    # Now pass the image through the res_model "predict". This will not predict the result, but will
    # instead pass it through the conv layers onel of resnet50 (and not the FC layers).
    # meaning, res_image will be the extracted "features" of the image (remember, we are doing xfer learning here)
    res_image = res_model.predict(image)
    return res_image


def predict_test_set():
    ''' Go through all of the test images and predict '''
    # Get the output in CSV format
    img_pred = namedtuple('img_pred', 'id label')
    header = img_pred('id', 'label')
    submission = [header]
    
    # Load the trained model
    logistic_model = load_model()
    # Load the resnet50 model
    res_model = ResNet50(weights="imagenet", include_top=False)
    # Get the test images paths
    test_image_paths = load_test_images_paths()
    # test_image_paths = test_image_paths[0:10]
    
    for image_path in test_image_paths:
        res_image = image_thru_resnet50_xfer(res_model, image_path)
        prediction = logistic_model.predict_proba(res_image.reshape(1,-1))
        image_fn = os.path.basename(image_path)
        image_id = os.path.splitext(image_fn)[0]
        image_pred = prediction[0][1]  # (second) Index 0 => Cat and Index 1 => Dog
        formatted_pred = '{:f}'.format(float(image_pred))
        submission.append(img_pred(image_id, formatted_pred))
        print('Predicted: {} as {}'.format(image_id, formatted_pred))
    return submission


def main():
    # Get the predictions
    submission = predict_test_set()

    # And now save it
    with open('gg_catsvdogs_sub_02.csv', 'wt') as outfh:
        for (id, label) in submission:
            outfh.write('{},{}\n'.format(id, label))

if __name__ == '__main__':
    main()
