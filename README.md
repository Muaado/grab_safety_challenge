# Grab Safety Challenge

This repo reflects the work I have carried out to fullfill the requirements of  [GRAB AI FOR SAFETY](https://www.aiforsea.com/safety). Machine Learning models I belive, shall avoid complexity by all means, and be able to express it self in the simplest of forms while been practically useful.

*To summarize :* This is a story of how a **random forest** model of **size 997KB** achieved a 10-fold CV **AUC** Score of  **72**  with  **21** Engineered **features**. 

---

- :rocket:  [Feature Engineering - And thought process behind it](https://github.com/Muaado/grab_safety_challenge/blob/master/Feature%20Engineering.md)
- :ticket:  [Instructions for using the saved model](https://github.com/Muaado/grab_safety_challenge/blob/master/RUN_SAVED_MODEl.ipynb)
- :checkered_flag:  [Full process - training and feature engineering](https://github.com/Muaado/grab_safety_challenge/blob/master/Training%20Process.ipynb)

---

### Findings and Adopted Strategies

The key findings and asdopted strategy used are summarised in the as below.

- Accelerometer and Gyroscope readings has noise, making it hard to decipher relevant signals. **Low Pass Filter was used to overcome the noise and smoothout the accelerometer and gyroscope readings.**

- Raw data in itself offers none or very little class seperation, Feature engineering is very important to create new dimensionality upon which class seperation can be improved.**Feature engineering was used extensively to allow for maximum possible class seperability while allowing for the creation of meaningful features which can be justified and explained. Use of descriptive statistics such as percentiles were avoided as this may prevent model from generalising well** [Refer Feature Engineering Section](#feature-engineering)

- Seperability of classes based on 1 second readings of accelerometer and gyrocsope is challenging as the significant difference in the magnitudes are relatively low. **A 4 second sliding window was adopted - Calculated Z crossing rate for each data point against the mean of the window, for both accelerometer and gyroscope data.** *The Z crossing rate is the number of times the signal crosses over the mean of the sliding window*

- Detected presence of outliers. Variable 'Second', in the collected data has outliers, where trip durations are unrealistic. **Outliers removed for this variable based on the lower and upper fence of the distribution**

- BookingID variable with value 0, duplicates in the label where two labels exisit for a signle BookingID. **Removed bookingID with value 0, removed duplicate bookingID from the labels**

- Since each bookingID has many records or data points, a proper aggregation strategy is required to reduce loss of information during the aggregation. **Sum , Maximum and Standard Deviation for the engineered features was used to keep loss of information to a minimal**

- The model is required to solve the problem of detecting dangerous drivers and be practical and efficient to implement. **Careful considerations have been made to reduce the computational time for preprocessing, feature engineering and model building with the use of vectorizations for calculations involving dataframes and limiting tree overgrowth by controlling factors like max_depth to finally reduce model size to 997 KB.**

- Class imbalance exisit in the data. Oversampling is not advisable with already noisy data. **Stratified sampling is used for model training and cross validation. Used class_weight  of random forest to balance out to remedy class imbalance issue.**
