# Load libraries
library(readr)
library(dplyr)
library(tidyverse)
library(lubridate)
library(rsample)
library(recipes)
library(caret)
library(randomForest)
library(ggplot2)
library(slider)
library(purrr)
library(foreach)
library(doParallel)
library(softImpute)
library(Matrix)

# Load data
dengue_labels_train <- read_csv("dengue_labels_train.csv")
dengue_features_train <- read_csv("dengue_features_train.csv")

# Merge features and labels
train_data <- dengue_features_train %>%
  left_join(dengue_labels_train, by = c("city", "year", "weekofyear")) %>%
  mutate(week = isoweek(week_start_date),
         year = year(week_start_date),
         city_sj = if_else(city == "sj", 1, 0),
         city_iq = if_else(city == "iq", 1, 0),
         wet_season = case_when(
           city == "sj" & month(week_start_date) %in% 4:11 ~ 1,
           city == "iq" & month(week_start_date) %in% 11:12 | month(week_start_date) %in% 1:4 ~ 1,
           TRUE ~ 0
         ),
         sin_season = sin(2 * pi * week / 52),
         cos_season = cos(2 * pi * week / 52))

# Simulated population (if needed)
set.seed(123)
train_data <- train_data %>%
  group_by(city, year) %>%
  mutate(population = if_else(city == "sj",
                              2327000 + (year - 1990) * 500 + rnorm(n(), 0, 1000),
                              300000 + (year - 1990) * 300 + rnorm(n(), 0, 500))) %>%
  ungroup()

# Key environmental variables of interest
key_vars <- c(
  "reanalysis_specific_humidity_g_per_kg",
  "reanalysis_dew_point_temp_k",
  "station_avg_temp_c",
  "station_min_temp_c"
)

# Add lag features 1–12 weeks
for (lag in 1:12) {
  for (var in key_vars) {
    train_data <- train_data %>%
      group_by(city) %>%
      arrange(week_start_date) %>%
      mutate(!!paste0(var, "_lag", lag) := lag(.data[[var]], lag)) %>%
      ungroup()
  }
}

# Matrix Completion
env_vars <- c(
  "station_max_temp_c", "station_min_temp_c", "station_avg_temp_c",
  "station_precip_mm", "station_diur_temp_rng_c", "precipitation_amt_mm",
  "reanalysis_sat_precip_amt_mm", "reanalysis_dew_point_temp_k",
  "reanalysis_air_temp_k", "reanalysis_relative_humidity_percent",
  "reanalysis_specific_humidity_g_per_kg", "reanalysis_precip_amt_kg_per_m2",
  "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k",
  "reanalysis_avg_temp_k", "reanalysis_tdtr_k",
  "ndvi_se", "ndvi_sw", "ndvi_ne", "ndvi_nw"
)
X_env <- as.matrix(train_data[, env_vars])
X_env_scaled <- scale(X_env)
X_sparse <- as(X_env_scaled, "Incomplete")
fit_soft <- softImpute(X_sparse, rank.max = 10, lambda = 1)
X_completed <- complete(X_sparse, fit_soft)

# PCA
pca <- prcomp(X_completed, center = FALSE, scale. = FALSE)
cumvar <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
num_pc <- which(cumvar >= 0.90)[1]
pca_scores <- as.data.frame(pca$x[, 1:num_pc])
colnames(pca_scores) <- paste0("PC", 1:num_pc)

# Final dataset
train_final <- train_data %>%
  bind_cols(pca_scores) %>%
  arrange(city, week_start_date) %>%
  drop_na(total_cases)

# Time series cross-validation (12-week training window)
create_time_slices <- function(data, initial_size = 26, horizon = 1) {
  n <- nrow(data)
  slices <- list()
  for (i in seq(initial_size, n - horizon)) {
    train_idx <- 1:i
    test_idx <- (i + 1):(i + horizon)
    slices[[length(slices) + 1]] <- list(train = data[train_idx, ], test = data[test_idx, ])
  }
  return(slices)
}

pca_vars <- grep("^PC", names(train_final), value = TRUE)
feature_vars <- c(pca_vars, key_vars, paste0(rep(key_vars, each=12), "_lag", 1:12),
                  "population", "wet_season", "sin_season", "cos_season", "city_sj", "city_iq")

# Per-city modeling
cities <- unique(train_final$city)
results <- data.frame()
rf_predictions <- data.frame()
train_final <- train_final %>% drop_na(all_of(feature_vars))

for (city_name in cities) {
  city_data <- train_final %>% filter(city == city_name)
  time_slices <- create_time_slices(city_data, initial_size = 26, horizon = 1)
  
  for (i in seq_along(time_slices)) {
    train_set <- time_slices[[i]]$train
    test_set <- time_slices[[i]]$test
    
    rf_model <- randomForest(
      x = train_set[, feature_vars],
      y = train_set$total_cases,
      ntree = 500,
      mtry = floor(sqrt(length(feature_vars))),
      importance = TRUE
    )
    
    preds <- predict(rf_model, newdata = test_set[, feature_vars])
    
    fold_results <- data.frame(
      fold = i,
      date = test_set$week_start_date,
      city = city_name,
      actual = test_set$total_cases,
      predicted = preds
    )
    
    rf_predictions <- bind_rows(rf_predictions, fold_results)
    mae <- mean(abs(fold_results$predicted - fold_results$actual))
    results <- bind_rows(results, data.frame(fold = i, city = city_name, MAE = mae))
  }
}

# Output results
cat("Average MAE (Random Forest):\n")
print(results %>% group_by(city) %>% summarise(MAE = mean(MAE)))
