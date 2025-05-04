# ---- Load Libraries ----
library(tidyverse)
library(lubridate)
library(recipes)
library(slider)
library(softImpute)
library(Matrix)
library(xgboost)
library(splines)

# ---- Load Data ----
dengue_labels_train <- read_csv("dengue_labels_train.csv")
dengue_features_train <- read_csv("dengue_features_train.csv")
dengue_features_test <- read_csv("dengue_features_test.csv")

# ---- Merge and Process Train Data ----
train_data <- dengue_features_train %>%
  left_join(dengue_labels_train, by = c("city", "year", "weekofyear")) %>%
  mutate(week_start_date = as.Date(week_start_date), year = year(week_start_date))

test_data <- dengue_features_test %>%
  mutate(week_start_date = as.Date(week_start_date), year = year(week_start_date))

combined_data <- bind_rows(train_data, test_data)

# ---- Interpolate Population ----
pop_anchors <- tribble(
  ~city, ~year, ~population,
  "sj", 1990, 2327000, "sj", 2000, 2508000, "sj", 2010, 2478000,
  "iq", 1990, 247000,  "iq", 2000, 318000,  "iq", 2010, 391000
)

interpolated_pop <- pop_anchors %>%
  group_by(city) %>%
  group_modify(~ tibble(year = seq(min(.x$year), max(.x$year))) %>%
                 left_join(.x, by = "year") %>%
                 mutate(population = approx(year, population, xout = year, rule = 2)$y)) %>%
  ungroup()

combined_data <- combined_data %>%
  left_join(interpolated_pop, by = c("city", "year"))

# ---- Feature Engineering ----
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

combined_data <- combined_data %>%
  group_by(city) %>%
  arrange(week_start_date) %>%
  mutate(across(all_of(env_vars), list(
    lag1 = ~lag(.), lag2 = ~lag(., 2), lag3 = ~lag(., 3),
    roll4 = ~slide_dbl(., mean, .before = 3, .complete = TRUE)
  ), .names = "{.col}_{.fn}")) %>%
  ungroup()

# ---- Matrix Completion for Lagged/Rolling Variables ----
env_lag_vars <- grep("_(lag|roll)", names(combined_data), value = TRUE)

# Create matrix for imputation
X_env <- as.matrix(combined_data[, env_lag_vars])
# Convert to Matrix package format for softImpute
X_sparse <- as(X_env, "Incomplete")
# Perform matrix completion
fit_soft <- softImpute(X_sparse, rank.max = 10, lambda = 1)
X_completed <- complete(X_sparse, fit_soft)
X_completed <- as.data.frame(X_completed)
colnames(X_completed) <- env_lag_vars

# Replace the original variables with completed ones in combined_data
combined_data[, env_lag_vars] <- X_completed

# ---- Add PCA and Other Features to Combined Data ----
combined_data <- combined_data %>%
  mutate(
    week = week(week_start_date),
    sin1 = sin(2 * pi * week / 52), cos1 = cos(2 * pi * week / 52),
    sin2 = sin(4 * pi * week / 52)
  )

# ---- Add Spline & Season ----
spline_cols <- as.data.frame(ns(combined_data$year, df = 4))
colnames(spline_cols) <- paste0("spline_year_", seq_len(ncol(spline_cols)))

# PCA on the completed data
pca <- prcomp(X_completed, center = TRUE, scale. = TRUE)
cumvar <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
num_pc <- which(cumvar >= 0.75)[1]

pca_scores <- as.data.frame(pca$x[, 1:num_pc])
colnames(pca_scores) <- paste0("PC", 1:num_pc)

combined_data <- combined_data %>%
  mutate(
    season = case_when(
      city == "iq" & month(week_start_date) %in% c(11, 12, 1:4) ~ "wet",
      city == "iq" ~ "dry",
      city == "sj" & month(week_start_date) %in% 4:11 ~ "wet",
      TRUE ~ "dry"
    ),
    city = factor(city), season = factor(season)
  ) %>%
  bind_cols(spline_cols) %>%
  bind_cols(pca_scores)

# ---- Handle any remaining NAs in combined_data ----
# Check for any columns with NAs after imputation
na_columns <- combined_data %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "na_count") %>%
  filter(na_count > 0) %>%
  pull(column)

# If there are still NA columns, impute them with median/mode
if (length(na_columns) > 0) {
  cat("Imputing remaining NA values in columns:", paste(na_columns, collapse=", "), "\n")
  
  for (col in na_columns) {
    if (is.numeric(combined_data[[col]])) {
      # For numeric columns, use median
      median_val <- median(combined_data[[col]], na.rm = TRUE)
      combined_data[[col]][is.na(combined_data[[col]])] <- median_val
    } else if (is.factor(combined_data[[col]])) {
      # For factor columns, use mode (most frequent)
      mode_val <- names(sort(table(combined_data[[col]]), decreasing = TRUE))[1]
      combined_data[[col]][is.na(combined_data[[col]])] <- mode_val
    }
  }
}

# ---- Diagnostic checks after data processing ----
cat("Checking dimensions and NAs after data processing:\n")
cat("Rows in combined_data:", nrow(combined_data), "\n")
cat("Rows with NA in total_cases:", sum(is.na(combined_data$total_cases)), "\n")
cat("Rows with non-NA in total_cases:", sum(!is.na(combined_data$total_cases)), "\n")

# Verify that we have test data after processing
if (sum(is.na(combined_data$total_cases)) == 0) {
  cat("WARNING: No test data found! All rows have total_cases values.\n")
  cat("This likely means the test data is not being properly identified.\n")
  
  # We need to recover the test data
  cat("Attempting to recover test data from original sources...\n")
  
  # Get original test IDs from test_data
  test_ids <- test_data %>% select(city, year, weekofyear)
  
  # Match with combined_data using these IDs
  combined_data_with_test_flag <- combined_data %>%
    mutate(is_test = paste(city, year, weekofyear) %in% 
             paste(test_ids$city, test_ids$year, test_ids$weekofyear))
  
  cat("Number of identified test rows:", sum(combined_data_with_test_flag$is_test), "\n")
  
  # Re-split the data properly
  train_final <- combined_data_with_test_flag %>% filter(!is_test)
  test_final <- combined_data_with_test_flag %>% filter(is_test)
  
  cat("After recovery - Train rows:", nrow(train_final), "\n")
  cat("After recovery - Test rows:", nrow(test_final), "\n")
} else {
  # Original split should work fine
  train_final <- combined_data %>% filter(!is.na(total_cases))
  test_final <- combined_data %>% filter(is.na(total_cases))
}

# ---- Train XGBoost Model on All Training Data ----
feature_vars <- setdiff(names(train_final), c("week_start_date", "total_cases", "is_test"))

# Verify that we have usable features
cat("Number of feature variables:", length(feature_vars), "\n")
if (length(feature_vars) == 0) {
  stop("No feature variables found for training!")
}
train_x <- model.matrix(~ . - 1, data = train_final[, feature_vars])
dtrain <- xgb.DMatrix(data = train_x, label = train_final$total_cases)

params <- list(
  booster = "gbtree", objective = "reg:squarederror",
  eval_metric = "mae", eta = 0.05, max_depth = 6,
  subsample = 0.8, colsample_bytree = 0.8,
  lambda = 1, alpha = 0.5, gamma = 0.5
)

xgb_model <- xgb.train(
  params = params, data = dtrain, nrounds = 300,
  early_stopping_rounds = 20, watchlist = list(train = dtrain), verbose = 0
)

# ---- Predict on Test Set ----
# Create prediction dataframe
test_x <- model.matrix(~ . - 1, data = test_final[, feature_vars])
dtest <- xgb.DMatrix(data = test_x)
preds <- predict(xgb_model, dtest)

# Print diagnostics
cat("Number of rows in test_final:", nrow(test_final), "\n")
cat("Length of predictions:", length(preds), "\n")

# ---- Format Submission ----
submission <- test_final %>%
  select(city, year, weekofyear) %>%
  mutate(year=as.factor(year))%>%
  mutate(weekofyear=as.factor(weekofyear))%>%
  mutate(total_cases = round(pmax(preds, 0)) %>% as.integer())  # Ensure predictions are non-negative

# ---- Write Submission CSV ----
write_csv(submission, "submission_format.csv")

# ---- Diagnostic Information ----
cat("Number of rows in final submission:", nrow(submission), "\n")
cat("Number of unique cities in submission:", length(unique(submission$city)), "\n")
cat("Summary of predicted cases:", summary(submission$total_cases), "\n")