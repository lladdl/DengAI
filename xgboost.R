# ---- Load Libraries ----
library(tidyverse)
library(lubridate)
library(recipes)
library(slider)
library(softImpute)
library(Matrix)
library(xgboost)
library(splines)

# ---- Load and Merge Data ----
dengue_labels_train <- read_csv("dengue_labels_train.csv")
dengue_features_train <- read_csv("dengue_features_train.csv")

train_data <- dengue_features_train %>%
  left_join(dengue_labels_train, by = c("city", "year", "weekofyear")) %>%
  mutate(
    week_start_date = as.Date(week_start_date),
    year = year(week_start_date)
  )

# ---- Interpolate Population by City ----
pop_anchors <- tribble(
  ~city, ~year, ~population,
  "sj", 1990, 2327000, "sj", 2000, 2508000, "sj", 2010, 2478000,
  "iq", 1990, 247000,  "iq", 2000, 318000,  "iq", 2010, 391000
)

interpolated_pop <- pop_anchors %>%
  group_by(city) %>%
  group_modify(~ {
    tibble(year = seq(min(.x$year), max(.x$year))) %>%
      left_join(.x, by = "year") %>%
      mutate(population = approx(year, population, xout = year, rule = 2)$y)
  }) %>%
  ungroup()

train_data <- train_data %>%
  left_join(interpolated_pop, by = c("city", "year"))

# ---- Add Lag Features (1–12 weeks) for Key Environmental Variables ----
key_vars <- c(
  "reanalysis_specific_humidity_g_per_kg",
  "reanalysis_dew_point_temp_k",
  "station_avg_temp_c",
  "station_min_temp_c"
)

for (lag in 1:12) {
  for (var in key_vars) {
    train_data <- train_data %>%
      group_by(city) %>%
      arrange(week_start_date) %>%
      mutate(!!paste0(var, "_lag", lag) := lag(.data[[var]], lag)) %>%
      ungroup()
  }
}

# ---- Add Lag and Rolling Features (1–3 lag, 4-week rolling average) ----
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

train_data <- train_data %>%
  group_by(city) %>%
  arrange(week_start_date) %>%
  mutate(across(all_of(env_vars),
                list(
                  lag1 = ~lag(.), lag2 = ~lag(., 2), lag3 = ~lag(., 3),
                  roll4 = ~slide_dbl(., mean, .before = 3, .complete = TRUE)
                ),
                .names = "{.col}_{.fn}"
  )) %>%
  ungroup()

# ---- Matrix Completion on Lagged/Rolling Variables ----
env_lag_vars <- grep("_(lag|roll)", names(train_data), value = TRUE)

X_env <- scale(as.matrix(train_data[, env_lag_vars]))
X_sparse <- as(X_env, "Incomplete")
fit_soft <- softImpute(X_sparse, rank.max = 10, lambda = 1)
X_completed <- complete(X_sparse, fit_soft)
X_completed <- as.data.frame(X_completed)
colnames(X_completed) <- env_lag_vars

# ---- PCA on Completed Environmental Variables ----
pca <- prcomp(X_completed, center = FALSE, scale. = FALSE)
cumvar <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
num_pc <- which(cumvar >= 0.75)[1]

pca_scores <- as.data.frame(pca$x[, 1:num_pc])
colnames(pca_scores) <- paste0("PC", 1:num_pc)

# ---- Add Seasonality and Time Features ----
train_data <- train_data %>%
  mutate(
    week = week(week_start_date),
    sin1 = sin(2 * pi * week / 52), cos1 = cos(2 * pi * week / 52),
    sin2 = sin(4 * pi * week / 52) #cos2 = cos(4 * pi * week / 52)
  )

# ---- Add Spline Terms and Season ----
spline_cols <- as.data.frame(ns(train_data$year, df = 4))
colnames(spline_cols) <- paste0("spline_year_", seq_len(ncol(spline_cols)))

train_data <- train_data %>%
  mutate(
    season = case_when(
      city == "iq" & month(week_start_date) %in% c(11, 12, 1:4) ~ "wet",
      city == "iq" ~ "dry",
      city == "sj" & month(week_start_date) %in% 4:11 ~ "wet",
      TRUE ~ "dry"
    ),
    city = factor(city),
    season = factor(season)
  ) %>%
  bind_cols(spline_cols)

# ---- Final Feature Set ----
train_final <- train_data %>%
  bind_cols(pca_scores) %>%
  select(week_start_date, city, total_cases, population, season,
         starts_with("PC"), starts_with("sin"), starts_with("cos"), starts_with("spline_year_")) %>%
  filter(!is.na(total_cases))

# ---- Time-Series Cross-Validation Function ----
create_time_slices <- function(data, initial_size = 52, horizon = 1) {
  n <- nrow(data)
  map(seq(initial_size, n - horizon), ~ list(
    train = data[1:.x, ],
    test = data[(.x + 1):(.x + horizon), ]
  ))
}

# ---- Model Training and Evaluation ----
results <- tibble()
xgb_predictions <- tibble()

for (current_city in unique(train_final$city)) {
  city_data <- train_final %>% filter(city == current_city) %>% arrange(week_start_date)
  slices <- create_time_slices(city_data)
  
  for (i in seq_along(slices)) {
    train_set <- slices[[i]]$train
    test_set <- slices[[i]]$test
    feature_vars <- setdiff(names(train_set), c("week_start_date", "total_cases"))
    
    train_x <- model.matrix(~ . - 1, data = train_set[, feature_vars])
    test_x <- model.matrix(~ . - 1, data = test_set[, feature_vars])
    
    dtrain <- xgb.DMatrix(data = train_x, label = train_set$total_cases)
    dtest <- xgb.DMatrix(data = test_x, label = test_set$total_cases)
    
    params <- list(
      booster = "gbtree", objective = "reg:squarederror",
      eval_metric = "mae", eta = 0.05, max_depth = 6,
      subsample = 0.8, colsample_bytree = 0.8,
      lambda = 1, alpha = 0.5, gamma = 0.5
    )
    
    xgb_model <- xgb.train(
      params = params, data = dtrain, nrounds = 300,
      early_stopping_rounds = 20,
      watchlist = list(train = dtrain), verbose = 0
    )
    
    preds <- predict(xgb_model, dtest)
    
    fold_results <- tibble(
      fold = i,
      date = test_set$week_start_date,
      city = current_city,
      actual = test_set$total_cases,
      predicted = preds,
      residual = test_set$total_cases - preds
    )
    
    xgb_predictions <- bind_rows(xgb_predictions, fold_results)
    results <- bind_rows(results, tibble(
      fold = i, city = current_city,
      MAE = mean(abs(fold_results$predicted - fold_results$actual)),
    ))
  }
}

# ---- Evaluation and Feature Importance ----
cat("Average MAE (XGBoost with gbtree):\n")
print(results %>% group_by(city) %>% summarise(MAE = mean(MAE)))
print(results %>% summarise(MAE = mean(MAE)))
importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix, top_n = 20, rel_to_first = TRUE, xlab = "Relative Importance")
head(importance_matrix, 20)


library(ggplot2)

ggplot(xgb_predictions, aes(x = date, y = residual, color = city)) +
  geom_line() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  facet_wrap(~city, scales = "free_x") +
  labs(title = "XGBoost Residuals Over Time", y = "Residual (Actual - Predicted)", x = "Week")


ggplot(xgb_predictions, aes(x = residual, fill = city)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  facet_wrap(~city) +
  labs(title = "Histogram of Residuals", x = "Residual", y = "Count")


library(qqplotr)

ggplot(xgb_predictions, aes(sample = residual)) +
  stat_qq_band(distribution = "norm", alpha = 0.2) +
  stat_qq_line(distribution = "norm") +
  stat_qq_point(distribution = "norm") +
  facet_wrap(~city) +
  labs(title = "QQ Plot of Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles")


ggplot(xgb_predictions, aes(x = date, y = abs(residual), color = city)) +
  geom_line() +
  facet_wrap(~city, scales = "free_x") +
  labs(title = "Absolute Residuals Over Time", x = "Week", y = "|Residual|")


xgb_predictions <- xgb_predictions %>%
  group_by(city) %>%
  mutate(std_residual = scale(residual)[,1]) %>%
  ungroup()

ggplot(xgb_predictions, aes(x = predicted, y = std_residual, color = city)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  facet_wrap(~city) +
  labs(title = "Standardized Residuals vs Predicted", y = "Standardized Residual", x = "Predicted Cases")

# Classify outbreak severity
outbreaks_classified <- train_data %>%
  filter(city == "sj", year(week_start_date) >= 1994, year(week_start_date) <= 2001, total_cases > 50) %>%
  mutate(severity = case_when(
    total_cases > 100 ~ "severe",
    total_cases > 50 ~ "moderate"
  )) %>%
  select(week_start_date, severity)


# Step 1: Filter and reshape data
env_data_long <- train_data %>%
  filter(city == "sj", year(week_start_date) >= 1994, year(week_start_date) <= 2001) %>%
  select(
    week_start_date,
    reanalysis_specific_humidity_g_per_kg_roll4,
    station_avg_temp_c_roll4,
    station_precip_mm_roll4,
    ndvi_se_roll4,
    reanalysis_precip_amt_kg_per_m2_roll4,
    reanalysis_dew_point_temp_k_roll4
  ) %>%
  pivot_longer(-week_start_date, names_to = "variable", values_to = "value")

# Step 2: Compute z-scores and identify outliers
env_data_long <- env_data_long %>%
  group_by(variable) %>%
  mutate(
    z_score = (value - mean(value, na.rm = TRUE)) / sd(value, na.rm = TRUE),
    is_outlier = abs(z_score) > 2.5
  ) %>%
  ungroup()

# Step 3: Plot with outliers and outbreak markers
ggplot(env_data_long, aes(x = week_start_date, y = value)) +
  geom_line(aes(color = variable), linewidth = 0.6, alpha = 0.8) +
  geom_smooth(method = "loess", se = FALSE, color = "black", linewidth = 0.8, span = 0.2) +
  geom_point(data = filter(env_data_long, is_outlier),
             aes(x = week_start_date, y = value),
             color = "black", size = 2.2, shape = 21, fill = "white", stroke = 1) +
  # Outbreak lines
  geom_vline(data = filter(outbreaks_classified, severity == "moderate"),
             aes(xintercept = week_start_date),
             color = "orange", linetype = "dashed", linewidth = 0.5, alpha = 0.7, inherit.aes = FALSE) +
  geom_vline(data = filter(outbreaks_classified, severity == "severe"),
             aes(xintercept = week_start_date),
             color = "red", linetype = "solid", linewidth = 0.6, alpha = 0.9, inherit.aes = FALSE) +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  labs(
    title = "Outlier Environmental Conditions and Dengue Outbreaks (San Juan, 1994–2001)",
    subtitle = "Black points = environmental outliers (|z| > 2.5), Orange dashed = moderate outbreaks, Red solid = severe outbreaks",
    x = "Week Start Date", y = "Value"
  ) +
  theme_minimal(base_size = 12) +
  theme(strip.text = element_text(face = "bold"),
        legend.position = "none")

