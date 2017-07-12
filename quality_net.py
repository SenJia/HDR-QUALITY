
NUM_CLASSES = 51
PATCH_SIZE = (32,32)

def cnn_model(PATCH_SIZE,NUM_TARGETS, input_var=None):
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


def main():
    file_name = "JPEG_PU_MODEL.pkl"
    input_var = T.tensor4('inputs')
    network = cnn_model(PATCH_SIZE,NUM_TARGETS,input_var)

if __name__ == "__main__":
     main()
