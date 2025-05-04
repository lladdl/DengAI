# --- Load libraries ---
library(readr)
library(dplyr)
library(tidyverse)
library(lubridate)
library(rsample)
library(recipes)
library(caret)
library(xgboost)
library(randomForest)
library(ggplot2) 
library(slider)
library(purrr)
library(foreach)
library(doParallel)

# --- Load data ---
dengue_labels_train <- read_csv("dengue_labels_train.csv")
dengue_features_train <- read_csv("dengue_features_train.csv")

# Merge features and labels
train_data <- dengue_features_train %>%
  left_join(dengue_labels_train, by = c("city", "year", "weekofyear"))

# Separate cities
train_sj <- train_data %>% filter(city == "sj")
train_iq <- train_data %>% filter(city == "iq")

# 1. Imputation function: average of previous and next week, then forward/backward fill
impute_local_avg <- function(x) {
  # Step 1: Fill using average of week before and after
  for (i in seq_along(x)) {
    if (is.na(x[i])) {
      before <- if (i > 1) x[i - 1] else NA
      after <- if (i < length(x)) x[i + 1] else NA
      x[i] <- mean(c(before, after), na.rm = TRUE)
    }
  }
  
  # Step 2: Forward fill
  for (i in seq_along(x)) {
    if (is.na(x[i]) && i > 1) {
      x[i] <- x[i - 1]
    }
  }
  
  # Step 3: Backward fill
  for (i in length(x):1) {
    if (is.na(x[i]) && i < length(x)) {
      x[i] <- x[i + 1]
    }
  }
  
  # Step 4: Final fallback — replace with overall mean if still NA
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  
  return(x)
}

# --- Updated PCA function ---
run_pca_extract <- function(data, variables, threshold = 0.9, group_name) {
  pca_result <- prcomp(data[, variables], scale. = TRUE)
  var_explained <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
  n_comp <- which(var_explained >= threshold)[1]
  
  scores <- as.data.frame(pca_result$x[, 1:n_comp])
  colnames(scores) <- paste0(group_name, "_PC", seq_len(n_comp))
  
  return(list(scores = scores, pca_model = pca_result))
}

# --- Variables to use ---
veg_vars <- c("ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw")
precip_vars <- c("precipitation_amt_mm", "reanalysis_precip_amt_kg_per_m2", 
                 "reanalysis_sat_precip_amt_mm", "station_precip_mm")
temp_vars <- c("reanalysis_air_temp_k", "reanalysis_avg_temp_k", 
               "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k",
               "reanalysis_min_air_temp_k", "reanalysis_relative_humidity_percent",
               "reanalysis_specific_humidity_g_per_kg", "reanalysis_tdtr_k",
               "station_avg_temp_c", "station_diur_temp_rng_c",
               "station_max_temp_c", "station_min_temp_c")

vars_to_transform <- c(veg_vars, precip_vars, temp_vars)

# --- Lag and rolling features ---
lag_weeks <- 1:3
rolling_window <- 4

train_sj <- train_sj %>%
  arrange(year, weekofyear) %>%
  mutate(across(all_of(vars_to_transform), 
                list(
                  lag1 = ~lag(., 1),
                  lag2 = ~lag(., 2),
                  lag3 = ~lag(., 3),
                  roll4 = ~slide_dbl(., mean, .before = 3, .complete = TRUE)
                ),
                .names = "{.col}_{.fn}"))

# Select all numeric columns you want to impute
vars_to_impute <- train_sj %>%
  select(where(is.numeric)) %>% 
  select(-c(year, weekofyear)) %>%  # exclude time identifiers
  names()

# Apply the impute function to each column
train_sj <- train_sj %>%
  mutate(across(all_of(vars_to_impute), ~impute_local_avg(.)))


# --- Group updated variables ---
veg_vars_all <- grep("^ndvi_", names(train_sj), value = TRUE)
precip_vars_all <- grep("^precipitation_amt_mm|^reanalysis_.*precip|^station_precip_mm", names(train_sj), value = TRUE)
temp_vars_all <- grep("^reanalysis_.*temp|^reanalysis_.*humidity|^station_.*temp", names(train_sj), value = TRUE)

# --- Perform PCA and extract scores ---
veg_pca_out <- run_pca_extract(train_sj, veg_vars_all, threshold = 0.9, group_name = "veg")
precip_pca_out <- run_pca_extract(train_sj, precip_vars_all, threshold = 0.9, group_name = "precip")
temp_pca_out <- run_pca_extract(train_sj, temp_vars_all, threshold = 0.9, group_name = "temp")

veg_scores <- veg_pca_out$scores
precip_scores <- precip_pca_out$scores
temp_scores <- temp_pca_out$scores

veg_pca <- veg_pca_out$pca_model
precip_pca <- precip_pca_out$pca_model
temp_pca <- temp_pca_out$pca_model

# --- Add PCA scores to dataset ---
train_sj <- bind_cols(train_sj, veg_scores, precip_scores, temp_scores)

# --- Standardize the PCA scores ---
pca_vars <- grep("^(veg|precip|temp)_PC", names(train_sj), value = TRUE)
train_sj <- train_sj %>%
  mutate(across(all_of(pca_vars), ~scale(.)[, 1]))

pca_vars <- grep("^(veg|precip|temp)_PC", names(train_sj), value = TRUE)

# # --- Plot scree function ---
# plot_scree <- function(pca_model, title) {
#   var_explained <- pca_model$sdev^2 / sum(pca_model$sdev^2)
#   scree_data <- data.frame(
#     PC = factor(seq_along(var_explained)),
#     Variance_Explained = var_explained
#   )
#   
#   ggplot(scree_data, aes(x = PC, y = Variance_Explained)) +
#     geom_bar(stat = "identity", fill = "steelblue") +
#     geom_line(aes(group = 1), color = "black", linetype = "dashed") +
#     geom_point(color = "red") +
#     labs(title = title, x = "Principal Component", y = "Proportion of Variance Explained") +
#     theme_minimal()
# }
# 
# # --- Scree plots ---
# plot_scree(veg_pca, "Vegetation PCA Scree Plot")
# plot_scree(precip_pca, "Precipitation PCA Scree Plot")
# plot_scree(temp_pca, "Temperature/Humidity PCA Scree Plot")
# 
# # Get the PCA predictor columns
# # Check cumulative variance explained for each PCA
# veg_var_explained <- cumsum(veg_pca$sdev^2 / sum(veg_pca$sdev^2))
# precip_var_explained <- cumsum(precip_pca$sdev^2 / sum(precip_pca$sdev^2))
# temp_var_explained <- cumsum(temp_pca$sdev^2 / sum(temp_pca$sdev^2))
# 
# # Print cumulative variance explained for each PCA
# print(veg_var_explained)
# print(precip_var_explained)
# print(temp_var_explained)
# 
# # Plot cumulative variance explained
# ggplot(data.frame(PC = seq_along(temp_var_explained), CumulativeVariance = temp_var_explained), aes(x = PC, y = CumulativeVariance)) +
#   geom_line() + 
#   geom_point() +
#   labs(title = "Cumulative Variance Explained by temp PCA", x = "Principal Component", y = "Cumulative Variance Explained") +
#   theme_minimal()
# 
# # You can repeat the above for precip_var_explained and temp_var_explained
# 
# 



#   #mtry = c(1, 2, 3, 4, 5,6,7,8,9,10),   # number of features at each split
#   nodesize = c(1, 5, 10)     # minimum number of samples per leaf
# )
# 
# # Perform grid search for Random Forest
# rf_tune <- train(
#   total_cases ~ ., 
#   data = train_sj[, c(pca_vars, "total_cases")],
#   method = "rf",
#   trControl = trainControl(method = "cv", number = 5), 
#   tuneGrid = rf_grid,
#   ntree=500,
#   mtry=8
# )
# 
# # View the best tuned parameters
# print(rf_tune$bestTune)

# Make sure your dataset has a date column
train_sj$week <- as.numeric(format(train_sj$week_start_date, "%U"))  # week number (0–52)

# Create cyclical features
train_sj$sin_week <- sin(2 * pi * train_sj$week / 52)
train_sj$cos_week <- cos(2 * pi * train_sj$week / 52)


# Arrange data
train_sj <- train_sj %>% arrange(week_start_date)

# Define the function for forward chaining
create_time_slices <- function(data, initial_size = 16, horizon = 1, fixed_window = TRUE) {
  n <- nrow(data)
  slices <- list()
  for (i in seq(initial_size, n - horizon, by = 1)) {
    if (fixed_window) {
      train <- data[1:i, ]
      test <- data[(i + 1):(i + horizon), ]
    } else {
      train <- data[1:(i + horizon), ]
      test <- data[(i + horizon + 1):(i + 2 * horizon), ]
    }
    slices[[length(slices) + 1]] <- list(train = train, test = test)
  }
  return(slices)
}

# Create time slices
time_slices <- create_time_slices(train_sj, initial_size = 16, horizon = 1)

# Prepare PCA predictor columns
pca_vars <- c(grep("^(veg|precip|temp)_PC", names(train_sj), value = TRUE),"sin_week", "cos_week")

# Initialize results data frame
results <- data.frame(fold = integer(0), MAE = numeric(0))
residual_data <- data.frame(
  fold = integer(),
  actual = numeric(),
  predicted = numeric(),
  residual = numeric(),
  week = as.Date(character())
)
# Loop over each fold
for (i in seq_along(time_slices)) {
  # Get train and test sets
  train_set <- time_slices[[i]]$train
  test_set <- time_slices[[i]]$test
  
  # Prepare xgboost DMatrix
  dtrain <- xgb.DMatrix(data = as.matrix(train_set[, pca_vars]), label = train_set$total_cases)
  dtest <- xgb.DMatrix(data = as.matrix(test_set[, pca_vars]), label = test_set$total_cases)
  
  # Train XGBoost model using cross-validation with early stopping
  set.seed(123)
  xgb_cv <- xgb.cv(
    params = list(
      objective = "reg:squarederror",
      eval_metric="mae",
      eta = 0.05,            # smaller learning rate
      max_depth = 7,         # slightly deeper trees
      subsample = 1,       # add randomness (good for generalization)
      colsample_bytree = 1,# use a random subset of variables at each tree
      gamma=0.5,
      min_child_weight=3
      
      ),
    data = dtrain,
    nrounds = 300,          # upper limit on rounds
    nfold = 5,              # k-fold cross-validation
    early_stopping_rounds = 20, # stop if no improvement after 20 rounds
    verbose = 1,            # show training progress
  )
  
  # Get the best iteration from the cross-validation
  best_iteration <- xgb_cv$best_iteration
  print(paste("Best iteration:", best_iteration))
  
  # Train final XGBoost model using the best iteration
  final_xgb_model <- xgboost(
    params = list(
      objective = "reg:squarederror",
      eval_metric="mae",
      eta = 0.05,            # smaller learning rate
      max_depth = 7,         # slightly deeper trees
      subsample = 1,       # add randomness (good for generalization)
      colsample_bytree = 1,# use a random subset of variables at each tree
      gamma=0.5,
      min_child_weight=3
    ),
    data = dtrain,
    nrounds = best_iteration,
    verbose = 1
  )
  
  # Predict on the test set
  predictions <- predict(final_xgb_model, newdata = dtest)
  actual <- test_set$total_cases
  residual <- actual - predictions
  
  # Store results
  residual_data <- rbind(residual_data, data.frame(
    fold = i,
    city = city,
    actual = actual,
    predicted = predictions,
    residual = residual,
    week = test_set$week_start_date
  ))
  # Calculate MAE
  mae <- mean(abs(predictions - test_set$total_cases))
  
  # Store results
  results <- rbind(results, data.frame(fold = i, MAE = mae))
}

# View results
print(results)

# Calculate average RMSE across all folds
mean_mae <- mean(results$MAE)
print(paste("Average XGBoost MAE with early stopping: ", mean_mae))


# Plot residuals
ggplot(residual_data, aes(x = week, y = residual)) +
  geom_line(color = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals Over Time",
       x = "Week",
       y = "Residual (Actual - Predicted)") +
  theme_minimal()
ggplot(residual_data, aes(x = predicted, y = residual)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs Predicted",
       x = "Predicted Cases",
       y = "Residual") +
  theme_minimal()

qqnorm(residual_data$residual)
qqline(residual_data$residual, col = "red")

ggplot(residual_data, aes(x = residual)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(title = "Histogram of Residuals",
       x = "Residual",
       y = "Frequency") +
  theme_minimal()

ggplot(residual_data, aes(x = week, y = residual, color = fold)) +
  geom_line(show.legend = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals Over Time by Fold",
       x = "Week",
       y = "Residual") +
  theme_minimal()
library(zoo)

residual_data$rolling_mean <- rollmean(residual_data$residual, k = 4, fill = NA)

ggplot(residual_data, aes(x = week)) +
  geom_line(aes(y = residual), alpha = 0.3) +
  geom_line(aes(y = rolling_mean), color = "red", size = 1) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals with Rolling Mean",
       x = "Week",
       y = "Residual") +
  theme_minimal()
residual_data$month <- format(residual_data$week, "%b")

ggplot(residual_data, aes(x = month, y = residual)) +
  geom_boxplot(fill = "lightblue") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals by Month",
       x = "Month",
       y = "Residual") +
  theme_minimal()


acf(residual_data$residual, main = "ACF of Residuals")

xgb.importance(model = final_xgb_model) %>%
  xgb.plot.importance(top_n = 10, rel_to_first = TRUE)

#Tuning
uno <- train_sj %>%
  filter(year=="2007")
residual_data$week_start_date <- residual_data$week

dos <- left_join(residual_data,uno,by="week_start_date")

dos <- dos %>%
  filter(year=="2007")

summary(dos)

ggplot(dos, aes(x = week_start_date, y = residual)) +
  geom_line(color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals Over Time (2007)",
       x = "Week",
       y = "Residual (Actual - Predicted)") +
  theme_minimal()

ggplot(dos, aes(x = week_start_date)) +
  geom_line(aes(y = total_cases), color = "black", linetype = "solid", size = 1.2) +
  geom_line(aes(y = predicted), color = "red", linetype = "dashed", size = 1.2) +
  labs(title = "Actual vs. Predicted Dengue Cases (2007)",
       x = "Week",
       y = "Total Cases") +
  theme_minimal()

ggplot(dos, aes(x = residual)) +
  geom_histogram(bins = 20, fill = "lightblue", color = "black") +
  labs(title = "Histogram of Residuals (2007)",
       x = "Residual",
       y = "Count") +
  theme_minimal()

ggplot(dos, aes(x = predicted, y = residual)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE, color = "red") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals vs. Predicted Values (2007)",
       x = "Predicted Cases",
       y = "Residual") +
  theme_minimal()

dos %>%
  arrange(desc(abs(residual))) %>%
  select(week_start_date, total_cases, predicted, residual) %>%
  head(5)

threshold <- quantile(dos$actual, 0.75)  # top 5% of case counts
dos$outbreak <- ifelse(dos$actual > threshold, 1, 0)



# Remove any non-numeric or ID columns (e.g., year, weekofyear, date, city)
features <- train_sj %>%
  select(-year, -weekofyear, -city, -week_start_date, -total_cases)

# Response variable
response <- train_sj$total_cases

# Create caret training data object
dtrain <- data.frame(features, total_cases = response)

cl <- makeCluster(detectCores() - 1)  # Use all but one core
registerDoParallel(cl)

xgb_grid <- expand.grid(
  nrounds = c(100, 200, 300, 400, 500, 600),
  max_depth = c(1, 3, 5, 7, 9),
  eta = c(0.05, 0.1, 0.3, 0.4, 0.5, 0.65, 0.75),
  gamma = c(0, 1, 0.5),
  min_child_weight = c(1, 5, 3),
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  subsample = c(0.3, 0.5, 0.7, 1.0)
)


xgb_trcontrol <- trainControl(
  method = "cv",             # Cross-validation
  number = 5,                # 5-fold
  verboseIter = TRUE,        # Show progress
  allowParallel = TRUE       # Speed up with parallel computing
)


set.seed(123)
xgb_tuned <- train(
  total_cases ~ .,
  data = dtrain,
  method = "xgbTree",
  trControl = xgb_trcontrol,
  tuneGrid = xgb_grid,
  metric = "MAE"
)
stopCluster(cl)


print(xgb_tuned)
plot(xgb_tuned)
best_params <- xgb_tuned$bestTune






