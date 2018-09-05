import numpy as np
from scipy.misc import imread as imread
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import random
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
import sol5_utils

#######   CONSTANTS   ########
GRAYSCALE = 1
RGB_DIM = 3
GRAY_MAX_LVL = 255
PATCH_SUBTRACT_VAL = 0.5
CONV_DIM = 3
ADAM_BETA_VAL = 0.9
VALID_SPLIT_PERCENT = 0.2
NOISE_MEAN = 0
TRAIN_MIN_SIGMA = 0
TRAIN_MAX_SIGMA = 0.2
PATCH_DENOISE_DIM = 24
DENOISE_OUT_CHANNELS = 48
PATCH_DEBLUR_DIM = 16
DEBLUR_OUT_CHANNELS = 32
TRAIN_BATCH_SIZE = 100
VALID_SAMPLES_NUM = 1000
SAMPLES_PER_EPOCH = 10000
NOISY_EPOCHS_NUM = 5
BLURRY_EPOCHS_NUM = 10
TRAIN_BATCH_SIZE_QUICK = 10
VALID_SAMPLES_NUM_QUICK = 30
SAMPLES_PER_EPOCH_QUICK = 30
EPOCHS_NUM_QUICK = 2
BLURRY_TRAIN_KER_SIZE = 7


def read_image(filename, representation):
    '''
    this function reads an image file and converts it into a given representation
    :param filename: string containing the image filename to read
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2)
    :return: an image, represented by a matrix of type np.float64 with
    intensities normalized to the range [0, 1], according to given
    representation.
    '''
    image = imread(filename)
    image = image.astype(np.float64)
    if representation == GRAYSCALE and image.ndim == RGB_DIM:
        image = rgb2gray(image)
    image /= GRAY_MAX_LVL
    return image


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    '''
    this function creates a generator object which outputs random tuples of original images batches and corrupted
    images batches
    :param filenames: A list of filenames of clean images
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent
    :param corruption_func: A function receiving a numpy's array representation of an image as a single argument,
                            and returns a randomly corrupted version of the input image
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract
    :return: generator object which outputs random tuples of original images batches and corrupted images batches
    '''

    images_dict = {}
    while True:
        height, width = crop_size
        source_batch = np.zeros((batch_size, 1, height, width))
        target_batch = np.zeros((batch_size, 1, height, width))
        index = 0
        for i in range(batch_size):
            rand_index = np.random.randint(0, len(filenames))
            if filenames[rand_index] in images_dict:
                img = images_dict[filenames[rand_index]]
            else:
                img = read_image(filenames[rand_index], 1)
                images_dict[filenames[rand_index]] = img
            corrupted_img = corruption_func(img)
            rand_row = np.random.randint(0, img.shape[0] - height - 1)
            rand_col = np.random.randint(0, img.shape[1] - width - 1)
            patch = img[rand_row : rand_row + height, rand_col : rand_col + width]
            new_corrupted_patch = corrupted_img[rand_row : rand_row + height, rand_col : rand_col + width]
            target_batch[index, 0, :, :] = patch - PATCH_SUBTRACT_VAL
            source_batch[index, 0, :, :] = new_corrupted_patch - PATCH_SUBTRACT_VAL
            index += 1
        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    '''
    this function takes as input a symbolic input tensor and the number of channels for each of its
    convolutional layers, and returns the symbolic output tensor of the layer configuration described above.
    :param input_tensor: a symbolic tensor
    :param num_channels: num of convolutional layer's output channels
    :return: the output of a Res block
    '''
    output = Convolution2D(num_channels, CONV_DIM, CONV_DIM, border_mode = 'same')(input_tensor)
    output = Activation ('relu')(output)
    output = Convolution2D(num_channels, CONV_DIM, CONV_DIM, border_mode='same')(output)
    merged_output = merge([input_tensor, output], mode='sum')
    return merged_output

def build_nn_model(height, width, num_channels, num_res_blocks):
    '''
    this function creates an untrained Keras model
    :param height: height of the input tensor
    :param width: width of the input tensor
    :param num_channels: num of convolutional layer's output channels
    :param num_res_blocks: number of Res blocks in the model
    :return: the created Keras model
    '''
    init_input = Input(shape=(GRAYSCALE, height, width))
    input_to_merge = Convolution2D(num_channels, CONV_DIM, CONV_DIM, border_mode='same')(init_input)
    input_to_merge = Activation('relu')(input_to_merge)
    input1 = resblock(input_to_merge, num_channels)
    for i in range(num_res_blocks - 1):
        input1 = resblock(input1, num_channels)
    input1 = merge([input_to_merge, input1], mode='sum')
    input1 = Convolution2D(1, CONV_DIM, CONV_DIM, border_mode='same')(input1)
    model = Model(input=init_input, output=input1)
    return model

def train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples):
    '''
    this function trains a given neural-network model
    :param model: a general neural network model for image restoration
    :param images: a list of file paths pointing to image files
    :param corruption_func: A function receiving a numpy's array representation of an image as a single argument,
                            and returns a randomly corrupted version of the input image
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param samples_per_epoch: The number of samples in each epoch
    :param num_epochs: The number of epochs for which the optimization will run
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch
    :return: a trained neural-network model
    '''
    crop_size = (model.input_shape[2], model.input_shape[3])
    split_index = int(len(images) * VALID_SPLIT_PERCENT)
    validation_images = images[:split_index]
    training_images = images[split_index:]
    validation_set = load_dataset(validation_images, batch_size, corruption_func, crop_size)
    training_set = load_dataset(training_images, batch_size, corruption_func, crop_size)
    model.compile(loss='mean_squared_error', optimizer = Adam(beta_2=ADAM_BETA_VAL))
    model.fit_generator(training_set, samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=validation_set, nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model):
    '''
    this function restores a corrupted image
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the [0; 1] range of type
                            float64
    :param base_model: a neural network trained to restore small patches
    :return: a restored corrupted image
    '''
    row, col = np.shape(corrupted_image)
    input = Input(shape=(GRAYSCALE, row, col))
    trained_model = base_model(input)
    new_model = Model(input=input, output=trained_model)
    corrupted_image = corrupted_image.reshape((GRAYSCALE, row, col)) - PATCH_SUBTRACT_VAL
    restored_img = new_model.predict(corrupted_image[np.newaxis,...])[0]
    restored_img = (restored_img + PATCH_SUBTRACT_VAL).reshape(row, col).astype(np.float64)
    return np.clip(restored_img, 0, 1)

def add_gaussian_noise(image, min_sigma, max_sigma):
    '''
    this function returns a corrupted image by adding a gaussian noise to it
    :param image: a grayscale image with values in the [0, 1] range of type float64
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal
    :return: a noisy corrupted image
    '''
    sigma = random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(NOISE_MEAN, sigma, size=np.shape(image))
    corrupted_img = image + noise
    corrupted_img = np.round((corrupted_img * GRAY_MAX_LVL)) / GRAY_MAX_LVL
    corrupted_img = np.clip(corrupted_img, 0, 1).astype(np.float64)
    return corrupted_img

def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    '''
    this function creates a trained neural-network model for image denoising
    :param num_res_blocks: num of residual blocks in the model
    :param quick_mode: an indicator which allows a quick model training
    :return: a trained model for image denoising
    '''
    model = build_nn_model(PATCH_DENOISE_DIM, PATCH_DENOISE_DIM, DENOISE_OUT_CHANNELS, num_res_blocks)
    noisy_images_list = sol5_utils.images_for_denoising()
    if (quick_mode):
        batch_size = TRAIN_BATCH_SIZE_QUICK
        samples_num = VALID_SAMPLES_NUM_QUICK
        samples_per_epoch = SAMPLES_PER_EPOCH_QUICK
        epoch_num = EPOCHS_NUM_QUICK
    else:
        batch_size = TRAIN_BATCH_SIZE
        samples_num = VALID_SAMPLES_NUM
        samples_per_epoch = SAMPLES_PER_EPOCH
        epoch_num = NOISY_EPOCHS_NUM

    corruption_func = lambda img: add_gaussian_noise(img, TRAIN_MIN_SIGMA, TRAIN_MAX_SIGMA)
    train_model(model, noisy_images_list, corruption_func, batch_size, samples_per_epoch, epoch_num,
                samples_num)
    return model

def add_motion_blur(image, kernel_size, angle):
    '''
    this function receives an image and performs a given motion blur on it
    :param image: a grayscale image with values in the [0,1] range of type float64
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined
    :param angle: an angle in radians in the range [0,Pi]
    :return: a motion blurred image
    '''
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    blurred_img = convolve(image, kernel)
    return blurred_img

def random_motion_blur(image, list_of_kernel_sizes):
    '''
    this function receives an image and performs a random motion blur on it
    :param image: a grayscale image with values in the [0,1] range of type float64
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined
    :param angle: an angle in radians in the range [0,Pi]
    :return: a motion blurred image
    '''
    angle = random.uniform(0, np.pi) #generates a random angle between [0,Pi]
    kernel_size = np.random.choice(list_of_kernel_sizes)
    return add_motion_blur(image, kernel_size, angle)

def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    '''
    this function creates a trained neural-network model for image deblurring
    :param num_res_blocks: num of residual blocks in the model
    :param quick_mode: an indicator which allows a quick model training
    :return: a trained model for image deblurring
    '''
    model = build_nn_model(PATCH_DEBLUR_DIM, PATCH_DEBLUR_DIM, DEBLUR_OUT_CHANNELS, num_res_blocks)
    blurry_images_list = sol5_utils.images_for_deblurring()
    if (quick_mode):
        batch_size = TRAIN_BATCH_SIZE_QUICK
        samples_num = VALID_SAMPLES_NUM_QUICK
        samples_per_epoch = SAMPLES_PER_EPOCH_QUICK
        epoch_num = EPOCHS_NUM_QUICK
    else:
        batch_size = TRAIN_BATCH_SIZE
        samples_num = VALID_SAMPLES_NUM
        samples_per_epoch = SAMPLES_PER_EPOCH
        epoch_num = BLURRY_EPOCHS_NUM
    corruption_func = lambda img: random_motion_blur(img, [BLURRY_TRAIN_KER_SIZE])
    train_model(model, blurry_images_list, corruption_func, batch_size, samples_per_epoch, epoch_num,
                samples_num)
    return model

