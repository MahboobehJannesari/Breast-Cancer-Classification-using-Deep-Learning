#python3 Prediction.py --Test_dir=../Test_images  --Batch_Size=1


import tensorflow as tf
#from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.contrib.learn.python.learn.datasets.base import Dataset
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from nets import inception
from preprocessing import inception_preprocessing
import glob, os
slim = tf.contrib.slim
import timeit
start=timeit.default_timer()


# =========================================================================== #
# input parameters
# =========================================================================== #
tf.app.flags.DEFINE_integer('Batch_Size', 1,
                          'Size of Batch')

tf.app.flags.DEFINE_string(
    'Test_dir', './Test_Images',
    'Directory where imagesfor test.')


FLAGS = tf.app.flags.FLAGS

Test_dir=FLAGS.Test_dir
###########################################################################
#======convert Label.txt to a List===================================
def nestedClassification(img_paths):

    checkpoints_path = '/home/mehdi/Desktop/1.Projects/1.Image_Processing/1_.classification/inception_v3/all'
    fileName = "../Project_Data/labels.txt"
    names = []

    crimefile = open(fileName, 'r')
    for line in crimefile.readlines():
        if line.strip():
            names.append(line.strip().split(":")[-1])

    BATCH_SIZE=FLAGS.Batch_Size

############################################################################

    number_of_inputImage=len(img_paths)
    if BATCH_SIZE>=number_of_inputImage:
        batch_size=number_of_inputImage
    else:
        batch_size=BATCH_SIZE

    NUM_CLASSES = len(names)
    image_size=inception.inception_v3.default_image_size

    repeat_count=2

    checkpoints_path = tf.train.latest_checkpoint(checkpoints_path)
    X = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        model,_ = inception.inception_v3(X, num_classes = NUM_CLASSES, is_training = False)

    probabilities = tf.nn.softmax(model)
    init = slim.assign_from_checkpoint_fn(checkpoints_path,slim.get_model_variables('InceptionV3'))


    def input_parser(img_path):
        # # read the img from file
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_png(img_file,channels=3)

        img_decoded=tf.image.convert_image_dtype(img_decoded,dtype=tf.float32)
    
        processed_image = inception_preprocessing.preprocess_image(img_decoded, image_size, image_size,
                                                                   is_training=False)
        return processed_image




    data=tf.data.Dataset.from_tensor_slices(img_paths)
    print("===imaPath",img_paths)
    data = data.map(input_parser, num_parallel_calls=1)
    data= data.repeat(repeat_count)  # Repeats dataset this # times
    data = data.batch(batch_size)
    # data= data.repeat(repeat_count)  # Repeats dataset this # times

    iterator=data.make_initializable_iterator()
    next_batch = iterator.get_next()
    test_init_op = iterator.make_initializer(data)


    with tf.Session() as sess:

        init(sess)
        sess.run(test_init_op)
    #
        j = 0
        k=0
        epoch=int(number_of_inputImage/batch_size)
        for i in range(epoch+1):

            img_batch=(sess.run(next_batch))
            probabilities1 = sess.run(probabilities, feed_dict={X: img_batch})

            for prediction in range((probabilities1.shape)[0]):  # batch_size
                if j<number_of_inputImage:
                    print(img_paths[j])
                    j += 1
                    batchPredResult = probabilities1[prediction, 0:]


                    sorted_inds = [i[0] for i in sorted(enumerate(-batchPredResult), key=lambda x: x[1])]
                    for i in range(1):
                        index = sorted_inds[i]
                        print('Probability %0.4f => [%s]' % (batchPredResult[index], names[index]))
                       


image_list = []
for root, dirs, files in os.walk(Test_dir, topdown=False):
   for name in files:
       image_list.append(os.path.join(root, name))


nestedClassification(image_list)
stop=timeit.default_timer()
print("Run time :", stop-start)
