library(tidyverse)
library(lubridate)
library(rsample)      # for time series cross-validation
library(recipes)      # for feature engineering
library(caret)        # for modeling framework
library(ggplot2)      # for EDA
library(readr)
library(dplyr)
dengue_labels_train <- read_csv("dengue_labels_train.csv")
dengue_features_train <- read_csv("dengue_features_train.csv")

# Merge features and labels
train_data <- dengue_features_train %>%
  left_join(dengue_labels_train, by = c("city", "year", "weekofyear"))

#Preview the Data
glimpse(train_data)
summary(train_data)
str(train_data)

#Check for Missing Values
train_data %>% summarise_all(~ mean(is.na(.)))
visdat::vis_miss(train_data)   # (optional: nice missing data heatmap)

#Plot Dengue Cases Over Time
train_data %>%
  ggplot(aes(x = as.Date(week_start_date), y = total_cases, color = city)) +
  geom_line() +
  theme_minimal()

#Plot Climate Variables Over Time
train_data %>%
  pivot_longer(cols = c(precipitation_amt_mm, reanalysis_air_temp_k, station_avg_temp_c),
               names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = as.Date(week_start_date), y = value, color = city)) +
  geom_line() +
  facet_wrap(~variable, scales = "free_y") +
  theme_minimal()

#Correlation Heatmap
library(corrplot)

numeric_vars <- train_data %>% select(where(is.numeric)) %>% drop_na()
cor_mat <- cor(numeric_vars)
cor_mat
corrplot(cor_mat, method = "color", type = "upper")

#Check for Seasonality by Month or Week
train_data %>%
  mutate(month = month(week_start_date)) %>%
  group_by(city, month) %>%
  summarise(mean_cases = mean(total_cases, na.rm = TRUE)) %>%
  ggplot(aes(x = factor(month), y = mean_cases, fill = city)) +
  geom_col(position = "dodge") +
  theme_minimal()

#Distribution of Target Variable
train_data %>%
  ggplot(aes(x = total_cases, fill = city)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  theme_minimal()

#City Differences
train_data %>%
  group_by(city) %>%
  summarise(
    mean_cases = mean(total_cases, na.rm = TRUE),
    median_cases = median(total_cases, na.rm = TRUE),
    sd_cases = sd(total_cases, na.rm = TRUE)
  )

#Feature Trends Before Outbreaks
# Quick scatterplot example
train_data %>%
  ggplot(aes(x = precipitation_amt_mm, y = total_cases, color = city)) +
  geom_point(alpha = 0.5) +
  theme_minimal()

#Outlier Detection
train_data %>%
  ggplot(aes(x = week_start_date, y = total_cases, color = city)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  facet_wrap(~city)
  theme_minimal()
  
  train_data <- train_data %>%
    mutate(weekofyear = ifelse(weekofyear == 53, 1, weekofyear))
  

  # 1. Ensure 'train_data' contains proper date info
  train_data <- train_data %>%
    mutate(date = as.Date(paste(year, weekofyear, 1, sep = "-"), format = "%Y-%U-%u"))
  
  # 2. Floor date to first day of month
  train_data <- train_data %>%
    mutate(year_month = floor_date(date, "month"))
  
  # 3. Group by city and year_month to sum cases
  monthly_cases <- train_data %>%
    mutate(year = year(year_month)) %>%
    group_by(city, year_month, year) %>%
    summarise(total_cases = sum(total_cases, na.rm = TRUE), .groups = "drop")
  
  # 4. Plot monthly totals
  ggplot(monthly_cases, aes(x = year_month, y = total_cases, color = city)) +
    geom_line(size = 1.2) +
    labs(title = "Monthly Dengue Cases by City",
         x = "Month",
         y = "Total Cases",
         color = "City") +
    facet_wrap(~year)+
    theme_minimal() +
    scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  