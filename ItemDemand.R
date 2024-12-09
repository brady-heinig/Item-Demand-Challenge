library(tidymodels)
library(vroom)
library(tidyverse)
library(ggplot2)
library(forecast)
library(gridExtra)
library(embed) 
library(modeltime) #Extensions of tidymodels to time series
library(timetk) #Some nice time series functions
library(prophet)


################################################################################
###################################### EDA #####################################
# read in data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

store1_Item1 <- train_data %>%
  filter(store==1, item==1)

ts_plot1 <- store1_Item1 %>%
  ggplot(mapping=aes(x=date, y=sales)) +
  geom_line() +
  geom_smooth(se=FALSE) + 
  ggtitle("Store 1, Item 1") +
  theme_minimal()

acf_plot1_month <- store1_Item1 %>%
  pull(sales) %>% 
  forecast::ggAcf(.) + 
  ggtitle("Store 1, Item 1")

acf_plot1_years <- store1_Item1 %>%
  pull(sales) %>% 
  forecast::ggAcf(., lag.max=2*365)+ 
  ggtitle("Store 1, Item 1")

store1_Item2 <- train_data %>%
  filter(store==2, item==3)

ts_plot2 <- store1_Item2 %>%
  ggplot(mapping=aes(x=date, y=sales)) +
  geom_line() +
  geom_smooth(se=FALSE)+ 
  ggtitle("Store 2, Item 3")

acf_plot2_month <- store1_Item2 %>%
  pull(sales) %>% 
  forecast::ggAcf(.)+ 
  ggtitle("Store 2, Item 3")

acf_plot2_years <- store1_Item2 %>%
  pull(sales) %>% 
  forecast::ggAcf(., lag.max=2*365)+ 
  ggtitle("Store 2, Item 3")

grid_plot <- grid.arrange(ts_plot1, acf_plot1_month, acf_plot1_years,
             ts_plot2, acf_plot2_month, acf_plot2_years,
             ncol = 3)

ggsave("panel_plot.png", plot = grid_plot, width = 15, height = 10, dpi = 300)

################################################################################
#################################### RF ########################################
################################################################################

filtered_data <- train_data %>%
  filter(store==7, item==9)
my_recipe <- recipe(sales ~ ., data=filtered_data) %>%
  step_rm(store, item) %>% 
  #step_lencode_glm(all_nominal_predictors(), outcome = vars(sales))  %>% 
  step_date(date, features="dow") %>% 
  step_date(date, features="month") %>% 
  step_date(date, features="year") %>% 
  step_mutate_at(date_month, fn=factor) %>% 
  step_mutate_at(date_dow, fn=factor) %>% 




# define model
forest_mod <- rand_forest(mtry = tune(), 
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# create workflow
forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

# set up grid of tuning values
forest_tuning_params <- grid_regular(mtry(range = c(1,40)),
                                     min_n(),
                                     levels = 5)
# set up k-fold CV
folds <- vfold_cv(filtered_data, v = 5, repeats=1)

CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=forest_tuning_params,
            metrics=metric_set(smape),
            control = control_grid(verbose=TRUE))

cv_metrics <- collect_metrics(CV_results)

# find best tuning params
bestTuneForest <- CV_results %>%
  show_best(metric = "smape")

best_cv_accuracy <- cv_metrics %>%
  filter(.metric == "smape") %>%
  summarise(mean_accuracy = mean(mean)) %>%
  pull(mean_accuracy)

# CV error is 1 - best accuracy
best_cv_error <- 1 - best_cv_accuracy

# finalize workflow and make predictions
forest_model <- rand_forest(mtry = bestTuneForest$mtry, 
                            min_n = bestTuneForest$min_n,
                            trees = bestTuneForest$levels) %>%
  set_engine("ranger") %>%
  set_mode("regression")

forest_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(forest_model) %>%
  fit(data=bike_train)

forest_preds <- predict(forest_wf, new_data=bike_test)

################################################################################
#################################### SARIMA ####################################
################################################################################

# Read in the Data and filter to store/item1
storeItemTrain <- train_data %>%
  filter(store==2, item==2)
storeItemTest <- test_data %>%
  filter(store==2, item==2)

#train <- train_data %>% filter(store==1, item==1)
cv_split <- time_series_split(storeItemTrain, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

## Create a recipe for the linear model part
arima_recipe <- recipe(sales ~ date, data=storeItemTrain) %>% 
  #step_rm(store, item) %>%
  step_date(date, features="dow") %>%
  step_date(date, features="month") %>%
  step_date(date, features="year") %>%
  step_mutate_at(date_month, fn=factor) %>%
  step_mutate_at(date_dow, fn=factor)

arima_model <- prophet_reg() %>%
  set_engine(engine = "prophet") #%>%
  #fit(sales ~ date, data = training(cv_split))

# ## Define the ARIMA Model
# arima_model <- arima_reg(seasonal_period=365,
#                          non_seasonal_ar=7, # default max p to tune
#                          non_seasonal_ma=7, # default max q to tune
#                          seasonal_ar=2, # default max P to tune
#                          seasonal_ma=2, #default max Q to tune
#                          non_seasonal_differences=2, # default max d to tune
#                          seasonal_differences=2 #default max D to tune
# ) %>%
# set_engine("auto_arima")

## Merge into a single workflow and fit to the training data
arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split))

## Calibrate (tune) the models (find p,d,q,P,D,Q)
cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))

## Visualize results
p2 <- cv_results %>%
modeltime_forecast(
                   new_data = testing(cv_split),
                   actual_data = training(cv_split)
) %>%
plot_modeltime_forecast(.interactive=FALSE)

## Now that you have calibrated (tuned) refit to whole dataset
fullfit <- cv_results %>%
  modeltime_refit(data=storeItemTrain)

## Predict for all the observations in storeItemTest1
p4 <-fullfit %>%
modeltime_forecast(
                   new_data = storeItemTest,
                   actual_data = storeItemTrain
) %>%
plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(p1,p2,p3,p4, nrows=2)


export(plot, file = "grid_plot.png")

################################################################################
############################ Final Predictions #################################
################################################################################

nStores <- max(train_data$store)
nItems <- max(train_data$item)
for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train_data %>%
      filter(store==s, item==i)
    storeItemTest <- test_data %>%
      filter(store==s, item==i)
    ## Fit storeItem models here
    arima_recipe <- recipe(sales ~ date, data=storeItemTrain) %>% 
      #step_rm(store, item) %>%
      step_date(date, features="dow") %>%
      step_date(date, features="month") %>%
      step_date(date, features="year") %>%
      step_mutate_at(date_month, fn=factor) %>%
      step_mutate_at(date_dow, fn=factor)
    
    arima_model <- prophet_reg() %>%
      set_engine(engine = "prophet") #%>%
    ## Predict storeItem sales
    ## Save storeItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

vroom_write(all_preds, file=..., delim=...)


###############################################################################
################################# Prophet #####################################
###############################################################################

# Read in the Data and filter to store/item1
storeItemTrain <- train_data %>%
  filter(store==2, item==2)
storeItemTest <- test_data %>%
  filter(store==2, item==2)

#train <- train_data %>% filter(store==1, item==1)
cv_split <- time_series_split(storeItemTrain, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

## Create a recipe for the linear model part
arima_recipe <- recipe(sales ~ date, data=storeItemTrain) %>% 
  #step_rm(store, item) %>%
  step_date(date, features="dow") %>%
  step_date(date, features="month") %>%
  step_date(date, features="year") %>%
  step_mutate_at(date_month, fn=factor) %>%
  step_mutate_at(date_dow, fn=factor)

arima_model <- prophet_reg() %>%
  set_engine(engine = "prophet") #%>%
#fit(sales ~ date, data = training(cv_split))

# ## Define the ARIMA Model
# arima_model <- arima_reg(seasonal_period=365,
#                          non_seasonal_ar=7, # default max p to tune
#                          non_seasonal_ma=7, # default max q to tune
#                          seasonal_ar=2, # default max P to tune
#                          seasonal_ma=2, #default max Q to tune
#                          non_seasonal_differences=2, # default max d to tune
#                          seasonal_differences=2 #default max D to tune
# ) %>%
# set_engine("auto_arima")

## Merge into a single workflow and fit to the training data
arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split))

## Calibrate (tune) the models (find p,d,q,P,D,Q)
cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))

## Visualize results
p2 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = training(cv_split)
  ) %>%
  plot_modeltime_forecast(.interactive=FALSE)

## Now that you have calibrated (tuned) refit to whole dataset
fullfit <- cv_results %>%
  modeltime_refit(data=storeItemTrain)

## Predict for all the observations in storeItemTest1
p4 <-fullfit %>%
  modeltime_forecast(
    new_data = storeItemTest,
    actual_data = storeItemTrain
  ) %>%
  plot_modeltime_forecast(.interactive=FALSE)

