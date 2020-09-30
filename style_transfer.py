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

from IPython.display import Image, display
display(Image(content_image))
display(Image(style_image))


