#Prithvi Lakshminarayanan

#set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os

caffe_root = '/home/prithvi/Downloads/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
input_img_val='/home/prithvi/Downloads/caffe/VOCdevkit'
MAX_IMGS=10

def setup_model():


    sys.path.insert(0, caffe_root + 'python')
    print caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
        print 'CaffeNet found.'
    else:
        print 'Download pre-trained CaffeNet model...'
        #!../scripts/download_model_binary.py ../models/bvlc_reference_caffenet

    # display plots in this notebook
    #%matplotlib inline

    # set display defaults
    plt.rcParams['figure.figsize'] = (10, 10)        # large images
    plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

    caffe.set_mode_cpu()

    model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)


    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(50,        # batch size
                              3,         # 3-channel (BGR) images
                              227, 227)  # image size is 227x227
    return net

def predict(net, path): #'examples/images/cat.jpg'

    print "*****"
    # load the mean ImageNet image (as disconvolution_paramtributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    print "Predicting:", path
    image = caffe.io.load_image(path)
    transformed_image = transformer.preprocess('data', image)
    plt.imshow(image)
   # plt.show()

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

    print 'predicted class is:', output_prob.argmax()

    # load ImageNet labels
    labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
    if not os.path.exists(labels_file):
        print "!../ data / ilsvrc12 / get_ilsvrc_aux.sh"

    labels = np.loadtxt(labels_file, str, delimiter='\t')

    print 'Prediction:', labels[output_prob.argmax()]

    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

    print 'probabilities and labels:'
    zip(output_prob[top_inds], labels[top_inds])
    print "******"

def layer_info(net):

    # for each layer, show the output shape
    #for layer_name, blob in net.blobs.iteritems():
    #    print layer_name + '\t' + str(blob.data.shape)

    name_params={}
    for layer_name, param in net.params.iteritems():
        name_params[layer_name] = str(param[0].data.shape) + str(param[1].data.shape)
    #print "name_params", name_params

    for i in range(len(net._layer_names)):
        name=net._layer_names[i]
        param=''
        if name in name_params:
            param="Params:" + name_params[name]

        print "Name:", name + '\t' + 'Type:' + net.layers[i].type + '\t' + param

    def vis_square(data):
        """Take an array of shape (n, height, width) or (n, height, width, 3)
           and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

        # normalize data for display
        data = (data - data.min()) / (data.max() - data.min())

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                    (0, 1), (0, 1))  # add some space between filters
                   + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
        data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        plt.imshow(data);
        plt.axis('off')
        plt.show()

    # the parameters are a list of [weights, biases]
    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))
    sys.path.insert(0, caffe_root + 'python')

    feat = net.blobs['conv1'].data[0, :36]
    vis_square(feat)

    print "*******"

def main():
    net = setup_model()
    i=1


    predict(net, caffe_root + "/examples/images/cat.jpg")

    layer_info(net)


if __name__ == "__main__":
    main()
