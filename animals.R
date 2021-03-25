# Example script using convolutional neural networks
# Uses subset of data taken from https://www.kaggle.com/alessiocorrado99/animals10/data#

# Remind students about
# install.packages("keras")
# library(keras)
# install_keras()
rm(list=ls())

library(keras)

# list of animals to model
animal_list <- c("Butterfly", "Cow", "Elephant", "Spider")

# number of output classes (i.e. fruits)
output_n <- length(animal_list)

# image size to scale down to (original images vary but about 600 x 800 px)
img_width <- 250
img_height <- 250
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
train_image_files_path <- "Training\\"
valid_image_files_path <- "Validation\\"

# optional data augmentation
train_data_gen = image_data_generator(
  rescale = 1/255 ,
  # rotation_range = 40,
  # width_shift_range = 0.2,
  # height_shift_range = 0.2,
  # shear_range = 0.2,
  # zoom_range = 0.2,
  # horizontal_flip = TRUE,
  # fill_mode = "nearest"
)

# Validation data shouldn't be augmented! But it should also be scaled.
valid_data_gen <- image_data_generator(
  rescale = 1/255
)  

# training images
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = animal_list,
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = animal_list,
                                                    seed = 42)

# Check that things seem to have been read in OK
cat("Number of images per class:")
table(factor(train_image_array_gen$classes))
train_image_array_gen$class_indices


# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32
epochs <- 10

# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>% 
  layer_activation("softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)


# fit
hist <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2,
  # callbacks = list(
  #   # save best model after every epoch
  #   callback_model_checkpoint("/Users/shiringlander/Documents/Github/DL_AI/Tutti_Frutti/fruits-360/keras/fruits_checkpoints.h5", save_best_only = TRUE),
  #   # only needed for visualising with TensorBoard
  #   callback_tensorboard(log_dir = "/Users/shiringlander/Documents/Github/DL_AI/Tutti_Frutti/fruits-360/keras/logs")
  # )
)

# 
# model %>% evaluate_generator(valid_image_array_gen, steps=valid_samples) # overall loss and accuracy
# model %>% predict_generator(valid_image_array_gen, steps=valid_samples) # too much output, does for all batches
# ifelse(predictions > 0.5, 1, 0)
#class_predictions <- predict_classes(model, )