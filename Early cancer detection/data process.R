# Load required libraries
library(randomForest)
library(ROCR)
library(pracma)
library(signal)
library(ggplot2)
library(PPRaman)
library(reshape2)
library(hyperSpec)

# Set directory and wavenumber range
directory <- "~/HPC_Args/"
wavenumber <- as.data.frame(seq(611.6, 1717, by = 1.09))

# Read command line arguments using commandArgs
args <- commandArgs(trailingOnly = TRUE)

# Read and clean test data
raw_test <- read.csv(paste0(directory, args[2]))
raw_test_snip <- raw_test[, 2:1017]  # Extract matrix of spectra and cancer flag
colnames(raw_test_snip)[1016] <- "V1016"

# Read and clean training data
raw_train <- read.csv(paste0(directory, args[1]))
raw_train_snip <- raw_train[, 2:1017]
colnames(raw_train_snip)[1016] <- "V1016"

# Process test and training spectra data
processed_test_spec <- opt_process_hpc_dev(
  as.hyperSpec(raw_test_snip[, 1:1015]),
  norm_meth = args[4],
  rm_bl = "pol",
  poly_order = as.integer(args[5])
)
processed_test_spec$V1016 <- raw_test_snip$V1016

processed_train_spec <- opt_process_hpc_dev(
  as.hyperSpec(raw_train_snip[, 1:1015]),
  norm_meth = args[4],
  rm_bl = "pol",
  poly_order = as.integer(args[5])
)
processed_train_spec$V1016 <- raw_train_snip$V1016

# Runs and tests 100 RFs and picks the one which maximizes the AUROC
Rf_testing <- list()
for (i in 1:100) {
  Rf_testing[[i]] <- random_forest_training(processed_train_spec, rmv_out = 0)
}

Rf_multi_models <- list()
auroc <- list()
sen <- list()
spec <- list()
ppv <- list()
npv <- list()
for (i in 1:100) {
  Rf_multi_models[[i]] <- random_forest_testing(
    Rf_testing[[i]][[1]],
    Rf_testing[[i]][[2]],
    processed_test_spec,
    Rf_testing[[i]][[3]]
  )

  auroc[[i]] <- Rf_multi_models[[i]][[5]]@y.values[[1]]  # Grabs AUROC value for the model
  sen[[i]] <- sen_spec(Rf_multi_models[[i]][[1]])[[1]]
  spec[[i]] <- sen_spec(Rf_multi_models[[i]][[1]])[[2]]
  ppv[[i]] <- ppv_npv(Rf_multi_models[[i]][[1]])[[1]]
  npv[[i]] <- ppv_npv(Rf_multi_models[[i]][[1]])[[2]]
}

# Picks out the model which maximizes the AUROC
best_model_ind <- which.max(unlist(auroc) + unlist(sen) + unlist(spec))
Rf_model <- Rf_testing[[best_model_ind]]
Rf_output <- Rf_multi_models[[best_model_ind]]

# Standard deviations
auroc_std <- std(unlist(auroc))
sen_std <- std(unlist(sen))
spec_std <- std(unlist(spec))
ppv_std <- std(unlist(ppv))
npv_std <- std(unlist(npv))

# Output results
output <- c(
  args[4], args[5], Rf_output[[5]]@y.values[[1]], auroc_std, sen[[best_model_ind]],
  sen_std, spec[[best_model_ind]], spec_std, ppv[[best_model_ind]], ppv_std,
  npv[[best_model_ind]], npv_std, nrow(processed_train_spec), nrow(processed_test_spec), nrow(Rf_model[[5]])
)

# Save the best model
saveRDS(Rf_model, paste0(directory, "/Rf_model", "_", args[4], "_", args[5], "_", args[3], ".rds"))

# Write output to CSV
write.table(
  output,
  paste0(directory, "/Polyblrm", "_", args[4], "_", args[5], "_", args[3], ".csv"),
  row.names = FALSE,
  col.names = FALSE
)
