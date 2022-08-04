library(keras)

## ----initial setup---------------------------------------------------------------------------------
# list of animals to model
animal_list <- c("butterfly", "cow", "elephant", "spider")

# number of output classes (i.e. fruits)
output_n <- length(animal_list)

# image size to scale down to (original images vary but about 600 x 800 px)
img_width <- 250
img_height <- 250
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
# Note: you may need to omit the trailing \\ if using MacOS or Linux
train_image_files_path <- "Training\\"
valid_image_files_path <- "Validation\\"


## ----augmentation skeleton-------------------------------------------------------------------------
# Rescale from 255 to between zero and 1
train_data_gen = image_data_generator(
  rescale = 1/255
)

valid_data_gen <- image_data_generator(
  rescale = 1/255
)  


## ----data generator--------------------------------------------------------------------------------
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


## ----check generator-------------------------------------------------------------------------------
# Check that things seem to have been read in OK
cat("Number of images per class:")
table(factor(train_image_array_gen$classes))
cat("Class labels vs index mapping")
train_image_array_gen$class_indices


## ----final setup-----------------------------------------------------------------------------------
# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32 # Typical default, though possibly a little high given small dataset
epochs <- 10


## ----define CNN structure--------------------------------------------------------------------------
# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  # First convolutional layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3),
                input_shape = c(img_width, img_height, channels),
                activation = "relu") %>%

  # Second convolutional hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3),
                activation = "relu") %>%

  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n, activation = "softmax") 


## ----check CNN structure---------------------------------------------------------------------------
print(model)


## ----compile---------------------------------------------------------------------------------------
# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)


# ----train model, eval=FALSE-----------------------------------------------------------------------
# Train the model with fit_generator
history <- model %>% fit_generator(
  # training data
  train_image_array_gen,

  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size),
  epochs = epochs,

  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),

  # print progress
  verbose = 2
)


# ----plot and save results, eval=FALSE-------------------------------------------------------------
plot(history)
save.image("animals.RData")
model %>% save_model_hdf5("animals_simple.hdf5")


## ----augmented generator---------------------------------------------------------------------------
train_data_gen = image_data_generator(
  rescale = 1/255 ,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = animal_list,
                                                    seed = 42)


# ----fit augmented model, eval=FALSE---------------------------------------------------------------
# Copy the original model (this is not essential; could keep the same name)
model_aug <- model

# Train the model with fit_generator
history_aug <- model %>% fit_generator(
  # training data
  train_image_array_gen,

  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size),
  epochs = epochs,

  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),

  # print progress
  verbose = 2
)

plot(history_aug)


## ----one augmented image---------------------------------------------------------------------------
fnames <- list.files(train_image_files_path, full.names = TRUE, recursive = TRUE)
# Pick the third image (a butterfly) to look at
img_path <- fnames[[3]]                                               

img <- image_load(img_path, target_size = c(150, 150))                
img_array <- image_to_array(img)                                      
img_array <- array_reshape(img_array, c(1, 150, 150, 3))              

# Very simple augmentation generator
augmentation_generator <- flow_images_from_data(                      
  img_array,                                                          
  generator = train_data_gen,                                                
  batch_size = 1
)

op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))            
for (i in 1:4) {                                                      
  batch <- generator_next(augmentation_generator)                     
  plot(as.raster(batch[1,,,]))                                        
}                                                                     
par(op)                                                               


