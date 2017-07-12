import cPickle as pickle

import numpy as np
import lasagne
import theano
import theano.tensor as T
import skimage.io as io


def cnn_model(PATCH_SIZE,NUM_CLASSES, input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, PATCH_SIZE[0], PATCH_SIZE[1]),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2))
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2))
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2))
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2,2))
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=512, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=512, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotNormal())
    
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.elu)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.elu)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0),
            num_units=NUM_CLASSES,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def split(img, PATCH_SIZE):
    patches = [] 
    for y in xrange(0, img.shape[0], PATCH_SIZE[0]):
        for x in xrange(0, img.shape[1], PATCH_SIZE[1]):
            patch = img[y:y+PATCH_SIZE[0], x:x+PATCH_SIZE[1]]
            if patch.shape==PATCH_SIZE:
                patches.append(patch)
    patch_arr = np.array(patches, dtype=np.float32)
    patch_arr = np.expand_dims(patch_arr, axis=1)
    return patch_arr

def main():

    NUM_CLASSES = 51
    PATCH_SIZE = (32,32)

    input_var = T.tensor4('inputs')
    network = cnn_model(PATCH_SIZE, NUM_CLASSES, input_var)
    model_name = "JPEG_PU_MODEL.pkl"
    with open(model_name, 'rb') as f:
        params = pickle.load(f)
    lasagne.layers.set_all_param_values(network, params)
    print ("model", model_name, "loaded")

    prediction = lasagne.layers.get_output(network,deterministic=True)
    predict_function = theano.function([input_var],prediction)
   
    img_name = "" 
    img = io.imread(img_name,True)
    patches = split(img, PATCH_SIZE)
    preds = predict_function(patches)
    score = np.argmax(preds, axis=1).mean()
    print (score)

if __name__ == "__main__":
     main()
