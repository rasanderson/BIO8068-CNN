library(torch)
library(torchvision)
library(torchdatasets)
library(luz)

dir <- "~/Downloads/dogs-vs-cats2" 
device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

# ds <- torchdatasets::dogs_vs_cats_dataset(
#   dir,
#   token = "~/.kaggle/kaggle.json",
#   transform = . %>%
#     torchvision::transform_to_tensor() %>%
#     torchvision::transform_resize(size = c(224, 224)) %>% 
#     torchvision::transform_normalize(rep(0.5, 3), rep(0.5, 3)),
#   target_transform = function(x) as.double(x) - 1
# )

train_transforms <- function(img) {
  img %>%
    #(function(x) x$to(device = device)) %>%
    torchvision::transform_to_tensor(.) %>% 
    torchvision::transform_resize(., size = c(28, 28)) # %>%
    #    torchvision::transform_resize(size = c(224, 224)) # %>%
    #torchvision::transform_normalise(rep(0.5, 3), rep(0.5, 3)) 
}

ds <- image_folder_dataset(
  file.path(dir),
  transform = train_transforms) #,
  #target_transform = function(x) as.double(x) - 1)


train_ids <- sample(1:length(ds), size = 0.6 * length(ds))
valid_ids <- sample(setdiff(1:length(ds), train_ids), size = 0.2 * length(ds))
test_ids <- setdiff(1:length(ds), union(train_ids, valid_ids))

train_ds <- dataset_subset(ds, indices = train_ids)
valid_ds <- dataset_subset(ds, indices = valid_ids)
test_ds <- dataset_subset(ds, indices = test_ids)

train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE, num_workers = 4)
valid_dl <- dataloader(valid_ds, batch_size = 64, num_workers = 4)
test_dl <- dataloader(test_ds, batch_size = 64, num_workers = 4)

# Note: this is using Alexnet
batch <- train_dl$.iter()$.next()
# Next line should show torch_tensor (1,1,.,.) 
#train_dl$.iter()$.next()
# batch is a list, the first item being the image tensors:
batch[[1]]$size()
# And the second, the classes:
batch[[2]]$size()


# net <- torch::nn_module(
#   
#   initialize = function(output_size) {
#     self$model <- model_alexnet(pretrained = TRUE)
#     
#     for (par in self$parameters) {
#       par$requires_grad_(FALSE)
#     }
#     
#     self$model$classifier <- nn_sequential(
#       nn_dropout(0.5),
#       nn_linear(9216, 512),
#       nn_relu(),
#       nn_linear(512, 256),
#       nn_relu(),
#       nn_linear(256, output_size)
#     )
#   },
#   forward = function(x) {
#     self$model(x)[,1]
#   }
#   
# )

torch_manual_seed(123)
num_class <- 2

net <- nn_module(
  "dogsvcats",
  
  initialize = function() {
    self$conv1 <- nn_conv2d(3, 32, 3)
    self$conv2 <- nn_conv2d(32, 64, 3)
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout2d(0.5)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, num_class)
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

n_epochs <- 3

## Next bit should work (from mnist example) but gives closure error
# fitted <- net %>%
#   setup(
#     loss = nn_cross_entropy_loss(),
#     optimizer = optim_adam,
#     metrics = list(
#       luz_metric_accuracy
#     )
#   ) %>%
#   #set_hparams(num_class = 2) %>% 
#   set_opt_hparams(lr = 0.003) %>% 
#   fit(train_dl, epochs = n_epochs, valid_data = test_dl)

model <- net()
model$to(device = device)
optimizer <- optim_adam(model$parameters)

for (epoch in 1:n_epochs) {
  l <- c()
  coro::loop(for (b in train_dl) {
    optimizer$zero_grad()
    output <- model(b[[1]])
    loss <- nnf_cross_entropy(output, b[[2]])
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
  })
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}


# Training
# 
# Below, you see four calls to luz, two of which are required in every setting,
#  and two are case-dependent. The always-needed ones are setup() and fit() :
#   
#   In setup(), you tell luz what the loss should be, and which optimizer to
#    use. Optionally, beyond the loss itself (the primary metric, in a sense,
#     in that it informs weight updating) you can have luz compute additional
#      ones. Here, for example, we ask for classification accuracy. (For a human
#       watching a progress bar, a two-class accuracy of 0.91 is way more
#        indicative than cross-entropy loss of 1.26.)
# 
# In fit(), you pass references to the training and validation dataloaders.
#  Although a default exists for the number of epochs to train for, you’ll
#   normally want to pass a custom value for this parameter, too.
# 
# The case-dependent calls here, then, are those to set_hparams() and
#  set_opt_hparams(). Here, set_hparams() appears because, in the model
#   definition, we had initialize() take a parameter, output_size. Any arguments
#    expected by initialize() need to be passed via this method.
# 
# set_opt_hparams() is there because we want to use a non-default learning rate
#  with optim_adam(). Were we content with the default, no such call would be
#   in order.
#   
# #   Default
# fitted <- net %>%
#   setup(
#     loss = nn_bce_with_logits_loss(),
#     optimizer = optim_adam,
#     metrics = list(
#       luz_metric_binary_accuracy_with_logits()
#     )
#   ) %>%
#   set_hparams(output_size = 1) %>%
#   set_opt_hparams(lr = 0.01) %>%
#   fit(train_dl, epochs = 3, valid_data = valid_dl)
# 
# # Save model weights and early stopping
# fitted <- net %>%
#   setup(
#     loss = nn_bce_with_logits_loss(),
#     optimizer = optim_adam,
#     metrics = list(
#       luz_metric_binary_accuracy_with_logits()
#     )
#   ) %>%
#   set_hparams(output_size = 1) %>%
#   set_opt_hparams(lr = 0.01) %>%
#   fit(train_dl,
#       epochs = 10,
#       valid_data = valid_dl,
#       callbacks = list(luz_callback_model_checkpoint(path = "./models"),
#                        luz_callback_early_stopping(patience = 2)))

# Training finished, we can ask luz to save the trained model:
#luz_save(fitted, "dogs-and-cats.pt")

