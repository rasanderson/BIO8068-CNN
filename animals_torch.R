library(torch)
library(torchvision)
library(luz)
library(ggplot2)
library(yardstick)
library(dplyr)

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
valid_length <- valid_ds$.length()

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

## Display images
#par(mfrow = c(4,6), mar = rep(1, 4))
#images <- as_array(batch[[1]]) %>% aperm(perm = c(1, 3, 4, 2))
#images %>%
#  purrr::array_tree(1) %>%
#  purrr::set_names(class_names[as_array(classes)]) %>%
#  purrr::map(as.raster) %>%
#  purrr::iwalk(~{plot(.x); title(.y)})
#par(mfrow = c(1,1))

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
    self$fc1 <- nn_linear(final_in_features, 128)    # final_in_features=30976
    self$fc2 <- nn_linear(128, num_class)            # num_class=4
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
n_epochs <- 10

fitted <- net %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) %>%
  fit(train_dl, epochs = n_epochs, valid_data = valid_dl)

# Make predictions
pred <- predict(fitted, valid_ds) %>% 
  as_array() %>% 
  max.col()
truth <- valid_ds$samples[[2]] %>%
  as.vector()

# Calculate and display confusion matrix
confusion <- bind_cols(pred = pred, truth = truth) %>%
  mutate(across(everything(), ~factor(.x, levels = 1:4, labels = class_names))) %>%
  conf_mat(truth, pred)

autoplot(confusion, type = "heatmap") + 
  scale_fill_distiller(palette = 2, direction = "reverse")

# Calculate and display various accuracy metrics and ROC plots
# Above gives logits 
# keras uses categorical cross entropy (gives probabilities)
# torch uses nnf_cross_entropy which gives logits hence no activation
# in last layer of model structure. Use exp to convert logits back to probs.
pred2 <- exp(as.matrix(predict(fitted, valid_ds)))
probs <- pred2 / rowSums(pred2)
joint <- data.frame(as.factor(truth), probs, as.factor(pred))
colnames(joint) <- c("truth", class_names, "pred")
metrics(joint, truth, pred)
precision(joint, truth, pred)

joint %>%
  mutate(Resample = 1) %>% 
  group_by(Resample) %>%
  roc_auc(truth, butterfly:spider)

joint %>%
  mutate(Resample = 1) %>% 
  group_by(Resample) %>%
  roc_curve(truth, butterfly:spider) %>% 
  autoplot()