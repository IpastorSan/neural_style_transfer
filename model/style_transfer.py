#First try to implement style transfer based on "A neural Algorithm of Artistic Style" https://arxiv.org/pdf/1508.06576.pdf
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19

content_image = "C:/Users/pasto/projects/GAN_neural_style_transfer/images/darth_vader_content.jpg"
style_image = "C:/Users/pasto/projects/GAN_neural_style_transfer/images/pollock_style.jpg"

#content_image = tf.keras.preprocessing.image.load_img("temple_of_heaven_content.jpg")
#style_image = tf.keras.preprocessing.image.load_img("chinese_painting_style.jpg")

result_prefix = "vader_generated" #???

#Weights of the different weight components (leap of faith)
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

#Dimensions of generated picture
width, height = tf.keras.preprocessing.image.load_img(content_image).size
img_nrows = 400
img_ncols = int(width*img_nrows / height)

#························See the images·····················································
from IPython.display import Image, display
display(Image(content_image))
display(Image(style_image))

#························Image preprocessing/deprocessing utilities···························
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_nrows, img_ncols))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    #The images are converted from RGB to BGR,\
    # then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def unprocess_image(img):
    #Transforms a tensor into a valid image
    img = tf.reshape(img,(img_nrows, img_ncols, 3))
    img = img.numpy()
    #remove zero-center by mean pixel
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img

#····················Compute the losses································································

#Gram matrix of an integer tensor (featue-wise outer product)

def gram_matrix(x):
    #input is (1, Height, Width, C) ---> squeeze to (H,W,C)
    x = tf.squeeze(x)
    x = tf.transpose(x, (2,0,1))   #shape (C,H,W)
    features = tf.reshape(x, (tf.shape(x)[0], -1))  #shape (C, H*W)
    gram = tf.matmul(features, tf.transpose(features)) #dot product of features and trnsposed of features (autocorrelation)
    return gram

#The style loss is designed to transfer the characteristics of the "style image" into the content image
#It is based on the gram matrices (exlain further of why they capture the style) of the feature maps
# of the style image (in 5 points of VGG19) and the generated image

def style_loss(style, combination):
    s = gram_matrix(style)
    c = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(s-c)) / (4.0 * (channels ** 2) * (size**2)) #Look up this formula

#This loss keeps the "general shape" of the original image into the combined image

def content_loss(content, combination):
    return tf.reduce_sum(tf.square(combination - content))

#Finally, the third loss is designed to keep the generated image globally coherent

def total_loss(x):
    a = tf.square(x[:, :img_nrows - 1, :img_ncols -1, :] - x[:, 1:, :img_ncols - 1, :])
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25)) #locura total, mirar la formula