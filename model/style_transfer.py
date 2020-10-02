#First try to implement style transfer based on "A neural Algorithm of Artistic Style" https://arxiv.org/pdf/1508.06576.pdf
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19

content_image_path = "C:/Users/pasto/projects/GAN_neural_style_transfer/images/darth_vader_content.jpg"
style_image_path = "C:/Users/pasto/projects/GAN_neural_style_transfer/images/pollock_style.jpg"

#content_image = tf.keras.preprocessing.image.load_img("temple_of_heaven_content.jpg")
#style_image = tf.keras.preprocessing.image.load_img("chinese_painting_style.jpg")

result_prefix = "vader_generated" #???

#Weights of the different weight components (leap of faith)
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

#Dimensions of generated picture
width, height = tf.keras.preprocessing.image.load_img(content_image_path).size
img_nrows = 400
img_ncols = int(width*img_nrows / height)

#························See the images·····················································
from IPython.display import Image, display
display(Image(content_image_path))
display(Image(style_image_path))

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

#Gram matrix of an integer tensor (feature-wise outer product)

def gram_matrix(x):
    #input is (1, Height, Width, C) ---> squeeze to (H,W,C)
    x = tf.squeeze(x)
    x = tf.transpose(x, (2,0,1))   #shape (C,H,W)
    features = tf.reshape(x, (tf.shape(x)[0], -1))  #shape (C, H*W)
    gram = tf.matmul(features, tf.transpose(features)) #outer product of features and its transposed
    return gram

#The style loss is designed to transfer the characteristics of the "style image" into the content image
#It is based on the gram matrices (explain further of why they capture the style) of the feature maps
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
    a = tf.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25)) #locura total, mirar la formula

#···················Models··························

#Model with pre-trained Imagenet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

#Dictionary to get the output of each "key" layer
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

#Model that returns the activation of every layer in VGGG19 (as a dictionary)
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=outputs_dict)

#Aqui hay margen para convertir las max pooling layers en average

#List of layers to use in the style loss (5, as per the paper)
style_layer_names = ["block1_conv1", "block2_conv1", "block3_conv1",\
                     "block4_conv1", "block5_conv1"]

#Layer to use for the content loss
content_layer_name = "block5_conv2"

def compute_loss(combination_image, content_image, style_image):
    input_tensor = tf.concat([content_image, style_image, combination_image], axis=0)
    #content_image = axis 0 index 0, style_image = axis 0 index 1, combination_image = axis 0 index 2,
    features = feature_extractor(inputs=input_tensor)

    #Initialize loss
    loss = tf.zeros(shape=())

    #Add content loss
    layer_features = features[content_layer_name] #extract layer by name
    content_image_features = layer_features[0, :, :, :] #gets all the weights up to block5_conv2, passed by the input of content image
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(content_image_features, combination_features)

    #add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    #add total variation loss
    loss += total_variation_weight * total_loss(combination_image)
    return loss

#···········Loss and gradient computation··········

@tf.function
def compute_loss_and_gradients(combination_image, content_image, style_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, content_image, style_image)
    gradients = tape.gradient(loss, combination_image)
    return loss, gradients

#············Training Loop···············
optimizer = tf.keras.optimizers.SGD(tf.keras.optimizers.schedules.ExponentialDecay(\
    initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96))

content_image = preprocess_image(content_image_path)
style_image = preprocess_image((style_image_path))
combination_image = tf.Variable(preprocess_image(content_image_path)) #look into generating a random image

iterations = 4000
for i in range(1, iterations + 1):
    loss, gradients = compute_loss_and_gradients(combination_image, content_image, style_image)
    optimizer.apply_gradients([(gradients, combination_image)]) #investigar
    if i % 100 == 0:
        print(f"Iteration {i}: loss={loss}")
        img = unprocess_image(combination_image.numpy())
        fname = result_prefix +"_at_iteration_"+str(i)+".png"
        tf.keras.preprocessing.image.save_img(fname, img)

