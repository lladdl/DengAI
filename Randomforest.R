# Load required packages
library(dplyr)
library(zoo)
library(randomForest)
library(ggplot2)

# --------------------------
# STEP 1: Feature Engineering
# --------------------------

# Week-based seasonality




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

train_sj$week <- as.numeric(format(train_sj$week_start_date, "%U"))
train_sj$sin_week <- sin(2 * pi * train_sj$week / 52)
train_sj$cos_week <- cos(2 * pi * train_sj$week / 52)

# Sort by time
train_sj <- train_sj %>% arrange(week_start_date)
# -------------------------------
# STEP 2: Time-Based Cross-Validation
# -------------------------------

# Define forward-chaining function
create_time_slices <- function(data, initial_size = 12, horizon = 1) {
  n <- nrow(data)
  slices <- list()
  for (i in seq(initial_size, n - horizon)) {
    train <- data[1:i, ]
    test <- data[(i + 1):(i + horizon), ]
    slices[[length(slices) + 1]] <- list(train = train, test = test)
  }
  return(slices)
}

# Create slices
time_slices <- create_time_slices(train_sj, initial_size = 52, horizon = 1)

# Select all predictor variables
predictor_vars <- c(grep(pca_vars, names(train_sj), value = TRUE), "sin_week", "cos_week")

# Storage for results
results <- data.frame(fold = integer(), MAE = numeric())
residual_data <- data.frame(
  fold = integer(), actual = numeric(), predicted = numeric(), residual = numeric(), week = as.Date(character())
)

# -------------------------------
# STEP 3: Random Forest CV Loop
# -------------------------------

for (i in seq_along(time_slices)) {
  train_set <- time_slices[[i]]$train
  test_set <- time_slices[[i]]$test
  
  # Train RF
  set.seed(123)
  rf_model <- randomForest(
    x = train_set[, predictor_vars],
    y = train_set$total_cases,
    ntree = 500,
    mtry = floor(sqrt(length(predictor_vars))),
    importance = TRUE
  )
  
  # Predict
  predictions <- predict(rf_model, newdata = test_set)
  actual <- test_set$total_cases
  residual <- actual - predictions
  
  # Store residuals and MAE
  residual_data <- rbind(residual_data, data.frame(
    fold = i, actual = actual, predicted = predictions, residual = residual, week = test_set$week_start_date
  ))
  
  mae <- mean(abs(residual))
  results <- rbind(results, data.frame(fold = i, MAE = mae))
}

# -------------------------------
# STEP 4: Evaluate Results
# -------------------------------

mean_mae <- mean(results$MAE)
print(paste("Average MAE across folds: ", round(mean_mae, 2)))

# -------------------------------
# STEP 5: Residual Visualization
# -------------------------------

ggplot(residual_data, aes(x = week, y = residual)) +
  geom_line(color = "darkred") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Random Forest Residuals Over Time", y = "Residual", x = "Week") +
  theme_minimal()

# -------------------------------
# STEP 6: Train Final Model on All Data
# -------------------------------

final_rf <- randomForest(
  x = train_sj[, predictor_vars],
  y = train_sj$total_cases,
  ntree = 500,
  mtry = 2,
  importance = TRUE
)

# Plot variable importance
varImpPlot(final_rf, main = "Random Forest Variable Importance")



# Load required packages
library(dplyr)
library(zoo)
library(randomForest)
library(ggplot2)

# --------------------------
# STEP 1: Feature Engineering
# --------------------------

# Week-based seasonality




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

train_iq <- train_iq %>%
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
vars_to_impute <- train_iq %>%
  select(where(is.numeric)) %>% 
  select(-c(year, weekofyear)) %>%  # exclude time identifiers
  names()

# Apply the impute function to each column
train_iq <- train_iq %>%
  mutate(across(all_of(vars_to_impute), ~impute_local_avg(.)))


# --- Group updated variables ---
veg_vars_all <- grep("^ndvi_", names(train_iq), value = TRUE)
precip_vars_all <- grep("^precipitation_amt_mm|^reanalysis_.*precip|^station_precip_mm", names(train_iq), value = TRUE)
temp_vars_all <- grep("^reanalysis_.*temp|^reanalysis_.*humidity|^station_.*temp", names(train_iq), value = TRUE)

# --- Perform PCA and extract scores ---
veg_pca_out <- run_pca_extract(train_iq, veg_vars_all, threshold = 0.9, group_name = "veg")
precip_pca_out <- run_pca_extract(train_iq, precip_vars_all, threshold = 0.9, group_name = "precip")
temp_pca_out <- run_pca_extract(train_iq, temp_vars_all, threshold = 0.9, group_name = "temp")

veg_scores <- veg_pca_out$scores
precip_scores <- precip_pca_out$scores
temp_scores <- temp_pca_out$scores

veg_pca <- veg_pca_out$pca_model
precip_pca <- precip_pca_out$pca_model
temp_pca <- temp_pca_out$pca_model

# --- Add PCA scores to dataset ---
train_iq <- bind_cols(train_iq, veg_scores, precip_scores, temp_scores)

# --- Standardize the PCA scores ---
pca_vars <- grep("^(veg|precip|temp)_PC", names(train_iq), value = TRUE)
train_iq <- train_iq %>%
  mutate(across(all_of(pca_vars), ~scale(.)[, 1]))

pca_vars <- grep("^(veg|precip|temp)_PC", names(train_iq), value = TRUE)

train_iq$week <- as.numeric(format(train_iq$week_start_date, "%U"))
train_iq$sin_week <- sin(2 * pi * train_iq$week / 52)
train_iq$cos_week <- cos(2 * pi * train_iq$week / 52)

# Sort by time
train_iq <- train_iq %>% arrange(week_start_date)
# -------------------------------
# STEP 2: Time-Based Cross-Validation
# -------------------------------

# Define forward-chaining function
create_time_slices <- function(data, initial_size = 12, horizon = 1) {
  n <- nrow(data)
  slices <- list()
  for (i in seq(initial_size, n - horizon)) {
    train <- data[1:i, ]
    test <- data[(i + 1):(i + horizon), ]
    slices[[length(slices) + 1]] <- list(train = train, test = test)
  }
  return(slices)
}

# Create slices
time_slices <- create_time_slices(train_iq, initial_size = 52, horizon = 1)

# Select all predictor variables
predictor_vars <- c(grep(pca_vars, names(train_iq), value = TRUE), "sin_week", "cos_week")

# Storage for results
results <- data.frame(fold = integer(), MAE = numeric())
residual_data <- data.frame(
  fold = integer(), actual = numeric(), predicted = numeric(), residual = numeric(), week = as.Date(character())
)

# -------------------------------
# STEP 3: Random Forest CV Loop
# -------------------------------

for (i in seq_along(time_slices)) {
  train_set <- time_slices[[i]]$train
  test_set <- time_slices[[i]]$test
  
  # Train RF
  set.seed(123)
  rf_model <- randomForest(
    x = train_set[, predictor_vars],
    y = train_set$total_cases,
    ntree = 500,
    mtry = floor(sqrt(length(predictor_vars))),
    importance = TRUE
  )
  
  # Predict
  predictions <- predict(rf_model, newdata = test_set)
  actual <- test_set$total_cases
  residual <- actual - predictions
  
  # Store residuals and MAE
  residual_data <- rbind(residual_data, data.frame(
    fold = i, actual = actual, predicted = predictions, residual = residual, week = test_set$week_start_date
  ))
  
  mae <- mean(abs(residual))
  results <- rbind(results, data.frame(fold = i, MAE = mae))
}

# -------------------------------
# STEP 4: Evaluate Results
# -------------------------------

mean_mae <- mean(results$MAE)
print(paste("Average MAE across folds: ", round(mean_mae, 2)))

