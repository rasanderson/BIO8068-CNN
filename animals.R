library(torch)
library(torchvision)
library(luz)
library(ggplot2)

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

## ----initial setup-----------------------------------------------------------
# image size to scale down to (original images vary but about 600 x 800 px)
img_width <- 48
img_height <- 48
target_size <- c(img_width, img_height)

# path to image folders
# For Windows change to Training\\ and Validation\\
train_image_files_path <- "animals/Training"
valid_image_files_path <- "animals/Validation"

## ----augmentation skeleton---------------------------------------------------
# Rescale from 255 to between zero and 1
# Initially don't bother with augmentation and keep really simple
# Need to work out how to use transform_normalise to standardise to 256
train_transforms <- function(animal) {
  animal %>%
    transform_to_tensor() %>% 
    (function(x) x$to(device = device)) %>% 
    transform_resize(target_size) 
}
valid_transforms <- train_transforms

## ----data generator----------------------------------------------------------
# training images
train_ds <- image_folder_dataset(
  file.path(train_image_files_path),
  transform = train_transforms)
valid_ds <- image_folder_dataset(
  file.path(valid_image_files_path),
  transform = train_transforms)

train_ds$.length()
valid_ds$.length()
train_length <- train_ds$.length()
for(i in 1:train_length){
  print(c(i,train_ds[i][[1]]$size()))
}
valid_length <- valid_ds$.length()
for(i in 1:valid_length){
  print(c(i, valid_ds[i][[1]]$size()))
}

class_names <- train_ds$classes
length(class_names)
num_class <- length(class_names)
class_names

batch_size <- 24

train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = batch_size)
batch <- train_dl$.iter()$.next()
# batch is a list, the first item being the image tensors:
batch[[1]]$size()
# And the second, the classes:
batch[[2]]$size()



# How many batches?
train_dl$.length() 
valid_dl$.length()
batch <- train_dl$.iter()$.next()
train_dl$.iter()$.next()

# Classes are coded as integers, to be used as indices in a vector of class
# names. We’ll use those for labeling the images.
classes <- batch[[2]]
classes
train_ds[1]

# Size of single tensor. It should display as channels x width x height
train_ds[200][[1]]$size()

# Display images
par(mfrow = c(4,6), mar = rep(1, 4))
images <- as_array(batch[[1]]) %>% aperm(perm = c(1, 3, 4, 2))
images %>%
  purrr::array_tree(1) %>%
  purrr::set_names(class_names[as_array(classes)]) %>%
  purrr::map(as.raster) %>%
  purrr::iwalk(~{plot(.x); title(.y)})
par(mfrow = c(1,1))

# Define the model
torch_manual_seed(123)

# Calculation of final number of parameters-------------------------------------
# Assumes img_width == img_height
# nn_conv2d
# Two convolution layers, with kernel = 3, stride = 1, reduce the with and
# height dimensions by 2 for each 2d convolution, hence -2 -2
# nn_max_pool2d
# Max pooling halves size, hence / 2
# Images
# Working with 2D images, hence ^2
# self$conv2
# This sets number of output paramters as 64, hence * 64
# self$fc1
# Takes as input the final number of in_features calculated below
final_in_features <- ((img_width - 2 - 2) / 2) ^ 2 * 64
net <- nn_module(
  "animalsCNN",
  
  initialize = function() {
    # in_channels, out_channels, kernel_size, stride = 1, padding = 0
    self$conv1 <- nn_conv2d(3, 32, 3)   #  1 32 3 # 
    self$conv2 <- nn_conv2d(32, 64, 3)  # 32 64 3
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout2d(0.5)
    self$fc1 <- nn_linear(final_in_features, 128)    # 9216 128
    self$fc2 <- nn_linear(128, num_class) # 128 10
  },
  
  forward = function(x) {
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      self$conv2() %>%
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      self$dropout1() %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$dropout2() %>%
      self$fc2()
  }
)

# Training the model -----------------------------------------------------------
model <- net()
model$to(device = device)
optimizer <- optim_adam(model$parameters)

n_epochs <- 5

for (epoch in 1:n_epochs) {
  l <- c()
  coro::loop(for (b in train_dl){
    optimizer$zero_grad()
    output <- model(b[[1]]$to(device = device))
    loss <- nnf_cross_entropy(output, b[[2]]$to(device = device))
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
  })
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}



# net <- nn_module(
#   
#   "simple-cnn",
#   
#   initialize = function() {
#     # start with batch_size * channels * width * height
#     # 32 * 3 * 250 * 250 = 6e+06
#     # in_channels, out_channels, kernel_size, stride = 1, padding = 0
#     self$conv1 <- nn_conv2d(3, 64, kernel_size = 3) #nn_conv2d(1, 32, 3)
#     # Above converts to 32 * 32 * 249 * 249 image size ??!!
#     self$conv2 <- nn_conv2d(64, 32, kernel_size = 3) #nn_conv2d(32, 64, 3)
#     # Above converts to 32 * 64 * 248 * 248
#     self$dropout1 <- nn_dropout2d(0.25)
#     self$dropout2 <- nn_dropout2d(0.5)
#     # The nnf_max_pool2d(2) below converts to
#     # 32 * 64 * 124 * 124. 64 * 124 * 124 = 984064
#     self$fc1 <- nn_linear(984064, 32) # nn_linear(9216, 128)
#     self$fc2 <- nn_linear(32, out_features = 4) #nn_linear(128, 4)
#    },
#   
#   forward = function(x) {
#     x %>% 
#       self$conv1() %>%
#       nnf_relu() %>%
#       self$conv2() %>%
#       nnf_relu() %>%
#       nnf_max_pool2d(2) %>%
#       self$dropout1() %>%
#       torch_flatten(start_dim = 2) %>%
#       self$fc1() %>%
#       nnf_relu() %>%
#       self$dropout2() %>%
#       self$fc2()     
#   }
# )
#
#
## Training ----
#
#fitted <- net %>%
#  setup(
#    loss = nn_cross_entropy_loss(),
#    optimizer = optim_adam,
#    metrics = list(
#      luz_metric_accuracy()
#    )
#  ) %>%
#  fit(train_dl, epochs = n_epochs, valid_data = valid_dl)
#
#
#
#
#
#
#
#model <- net()
#model$to(device = device)
#
#optimizer <- optim_adam(model$parameters)
#
## this will be called for every batch, see training loop below
## loss <- nnf_cross_entropy(output, b[[2]]$to(device = device))
#
#
#
#for (epoch in 1:5) {
#  
#  l <- c()
#  
#  coro::loop(for (b in train_dl) {
#    # make sure each batch's gradient updates are calculated from a fresh start
#    optimizer$zero_grad()
#    # get model predictions
#    output <- model(b[[1]]$to(device = device))
#    # calculate loss
#    loss <- nnf_cross_entropy(output, b[[2]]$to(device = device))
#    # calculate gradient
#    loss$backward()
#    # apply weight updates
#    optimizer$step()
#    # track losses
#    l <- c(l, loss$item())
#  })
#  
#  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
#}
#
#
#
#
#
## train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
##                                                     train_data_gen,
##                                                     target_size = target_size,
##                                                     class_mode = "categorical",
##                                                     classes = animal_list,
#                                                     seed = 42)
# 
# # validation images
# valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
#                                                     valid_data_gen,
#                                                     target_size = target_size,
#                                                     class_mode = "categorical",
#                                                     classes = animal_list,
#                                                     seed = 42)


## ----check generator---------------------------------------------------------
# Check that things seem to have been read in OK
# cat("Number of images per class:")
# table(factor(train_image_array_gen$classes))
# cat("Class labels vs index mapping")
# train_image_array_gen$class_indices


## ----final setup-------------------------------------------------------------
# number of training samples
# train_samples <- train_image_array_gen$n
# # number of validation samples
# valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
# batch_size <- 32 # Typical default, though possibly a little high given small dataset
# epochs <- 10


## ----define CNN structure----------------------------------------------------
# initialise model
# model <- keras_model_sequential()

## add layers
#model %>%
#  # First convolutional layer
#  layer_conv_2d(filter = 32, kernel_size = c(3,3),
#                input_shape = c(img_width, img_height, channels),
#                activation = "relu") %>%
#
#  # Second convolutional hidden layer
#  layer_conv_2d(filter = 16, kernel_size = c(3,3),
#                activation = "relu") %>%
#
#  # Use max pooling
#  layer_max_pooling_2d(pool_size = c(2,2)) %>%
#  layer_dropout(0.25) %>%
#  
#  # Flatten max filtered output into feature vector 
#  # and feed into dense layer
#  layer_flatten() %>%
#  layer_dense(100, activation = "relu") %>%
#  layer_dropout(0.5) %>%
#  
#  # Outputs from dense layer are projected onto output layer
#  layer_dense(output_n, activation = "softmax") 
#
#
### ----check CNN structure-----------------------------------------------------
#print(model)
#
#
### ----compile-----------------------------------------------------------------
## Compile the model
#model %>% compile(
#  loss = "categorical_crossentropy",
#  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
#  metrics = "accuracy"
#)
#
#
## ----train model, eval=FALSE--------------------------------------------------
## Train the model with fit_generator
#history <- model %>% fit_generator(
#  # training data
#  train_image_array_gen,
#
#  # epochs
#  steps_per_epoch = as.integer(train_samples / batch_size),
#  epochs = epochs,
#
#  # validation data
#  validation_data = valid_image_array_gen,
#  validation_steps = as.integer(valid_samples / batch_size),
#
#  # print progress
#  verbose = 2
#)
#
#
## ----plot and save results, eval=FALSE----------------------------------------
#plot(history)
#save.image("animals.RData")
#model %>% save_model_hdf5("animals_simple.hdf5")
#
#
### ----augmented generator-----------------------------------------------------
#train_data_gen = image_data_generator(
#  rescale = 1/255 ,
#  rotation_range = 40,
#  width_shift_range = 0.2,
#  height_shift_range = 0.2,
#  shear_range = 0.2,
#  zoom_range = 0.2,
#  horizontal_flip = TRUE,
#  fill_mode = "nearest"
#)
#train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
#                                                    train_data_gen,
#                                                    target_size = target_size,
#                                                    class_mode = "categorical",
#                                                    classes = animal_list,
#                                                    seed = 42)
#
#
## ----fit augmented model, eval=FALSE------------------------------------------
## Copy the original model (this is not essential; could keep the same name)
#model_aug <- model
#
## Train the model with fit_generator
#history_aug <- model %>% fit_generator(
#  # training data
#  train_image_array_gen,
#
#  # epochs
#  steps_per_epoch = as.integer(train_samples / batch_size),
#  epochs = epochs,
#
#  # validation data
#  validation_data = valid_image_array_gen,
#  validation_steps = as.integer(valid_samples / batch_size),
#
#  # print progress
#  verbose = 2
#)
#
#plot(history_aug)
#
#
### ----one augmented image-----------------------------------------------------
#fnames <- list.files(train_image_files_path, full.names = TRUE, recursive = TRUE)
## Pick the third image (a butterfly) to look at
#img_path <- fnames[[3]]                                               
#
#img <- image_load(img_path, target_size = c(150, 150))                
#img_array <- image_to_array(img)                                      
#img_array <- array_reshape(img_array, c(1, 150, 150, 3))              
#
## Very simple augmentation generator
#augmentation_generator <- flow_images_from_data(                      
#  img_array,                                                          
#  generator = train_data_gen,                                                
#  batch_size = 1
#)
#
#op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))            
#for (i in 1:4) {                                                      
#  batch <- generator_next(augmentation_generator)                     
#  plot(as.raster(batch[1,,,]))                                        
#}                                                                     
#par(op)                                                               
