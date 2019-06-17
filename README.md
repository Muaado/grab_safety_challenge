# Grab Safety Challenge

This repo reflects the work I have carried out to fullfill the requirements of the grab [GRAB AI FOR SAFETY](https://www.aiforsea.com/safety). Machine Learning models I belive, shall avoid complexity by all means, and be able to express it self in the simplest of forms while been practically useful.

![](https://i.ibb.co/wBSJbXw/sssss.png)

## Key findings - Adopted Strategies

The key findings and adopted strategy used are summarised in the as below.

- Accelerometer and Gyroscope readings has noise, making it hard to decipher relevant signals. ***Low Pass Filter was used to overcome the noise and smoothout the accelerometer and gyroscope readings.***

- Raw data in itself offers none or very little class seperation, Feature engineering is very important to create new dimensionality upon which class seperation can be improved.**Feature engineering was used extensively to allow for maximum possible class seperability while allowing for the creation of meaningful features which can be justified and explained. Use of descriptive statistics such as percentiles were avoided as this may prevent model from generalising well** [Refer Feature Engineering Section](##Feature-Engineering)

- Seperability of classes based on 1 second readings of accelerometer and gyrocsope is challenging as the significant difference in the magnitudes are relatively low. ***A 4 second sliding window was adopted - Calculated Z crossing rate for each data point against the mean of the window, for both accelerometer and gyroscope data*** *The Z crossing rate is the number of times the signal crosses over the mean of the sliding window*

- Variable 'Second', in the collected data has outliers, where trip durations are unrealistic. **Outliers removed based on the lower and upper fence of the distribution**

- BookingID variable with value 0, duplicates in the label where two labels exisit for a signle BookingID. **Removed bookingID with value 0, removed duplicate bookingID from the labels**

- Since each bookingID has many records or data points, a proper aggregation strategy is required to reduce loss of information during the aggregation. **Sum , Maximum and Standard Deviation for the engineered features was used to keep loss of information to a minimal**

`[Go to Real Cool Heading section](#real-cool-heading)`

## Feature Engineering
