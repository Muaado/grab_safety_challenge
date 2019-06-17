### Feature Engineering

Careful consideration have been made to keep the feature engineering process **SIMPLE** and **MEANINGFUL**. The engineered features are described as below.

Click here to go to the [Summary Table for Feature Engineering to know the final list of features](#summary-table-for-feature-engineering)

### Feature: Moving window Mean

The basis of the feature engineering was a 4-second sliding window which is calculated for each of the 3-axis data for accelerometer and gyroscope and is depicted in the figure below.

![moving window](https://i.ibb.co/HPL8fCf/111334455.png)

A1 ... A4 represents data points for a single axis of either accelerometer or gyroscope.

The moving window of 4 seconds was used because it is a reasonable time frame where individual events (driving behavior such as jerk, sudden brakes, sharp turns)  or anomalies in the  signal can be detected.

The decision to use a moving window approach was followed by analysis in to the signal from accelerometer. Diagram below shows a 4 second moving average used on the accelrometer (equivalent to a low pass filter LPF) to the left and raw data from accelearation_x to the right. The signal is more clean and noise free as observed with the moving average utilized.

![s](https://i.ibb.co/TrH5qCw/1133.jpg)

```python
mean_acc_gyro = merged.groupby('bookingID', as_index=False)\                  ['acceleration_x','acceleration_y','acceleration_z','gyro_x','gyro_y', 'gyro_z'].rolling(4).mean().reset_index()
```

The above code , calculates a moving average of 4 seconds. To extract the mean of the sliding window we then extract the information from the moving average with the code below, which takes every 4th row value of the rolling mean and assigns to its respective feature vector.

```python
merged['MEAN_ACC_X'] = mean_acc_gyro.acceleration_x[3::4]
merged['MEAN_ACC_Y'] = mean_acc_gyro.acceleration_y[3::4]
merged['MEAN_ACC_Z'] = mean_acc_gyro.acceleration_z[3::4]
merged['MEAN_GYRO_X'] = mean_acc_gyro.gyro_x[3::4]
merged['MEAN_GYRO_Y'] = mean_acc_gyro.gyro_y[3::4]
merged['MEAN_GYRO_Z'] = mean_acc_gyro.gyro_z[3::4]
```

### Feature: Zero-Crossing Rate (ZCR)

This feature is identified from previous research conducted on human activity recognition through smartphone sensors (Incel, 2015) .

Zero-Crossing Rate (ZCR) as defined by Incel (2015), is:

>  The number of points where a signal crosses through a speciﬁc value corresponding to half of the signal range

This feature was used to identify whether each point in a window, has crossed the window mean. This is implemented by the code as below;

```python
# Calculate Z-Crossing for accelerometer and gyroscope readings (how many times each point crosses the mean)

merged['CROSSOVER_ACC_X'] = (merged['acceleration_x'] > merged['MEAN_ACC_X'].bfill()).astype(int)
merged['CROSSOVER_ACC_Y'] = (merged['acceleration_y'] > merged['MEAN_ACC_Y'].bfill()).astype(int)
merged['CROSSOVER_ACC_Z'] = (merged['acceleration_z'] > merged['MEAN_ACC_Z'].bfill()).astype(int)

# Calculate Z-Crossing for accelerometer and gyroscope readings (how many times each point crosses the mean)

merged['CROSSOVER_GYRO_X'] = (merged['gyro_x'] > merged['MEAN_GYRO_X'].bfill()).astype(int)
merged['CROSSOVER_GYRO_Y'] = (merged['gyro_y'] > merged['MEAN_GYRO_Y'].bfill()).astype(int)
merged['CROSSOVER_GYRO_Z'] = (merged['gyro_z'] > merged['MEAN_GYRO_Z'].bfill()).astype(int)
```

This engineered feature is later aggregated by sum for each booking and used for training the model. 

Descriptive statistics indicate that this feature is promising as clear seperation of class can be observed at every descriptive statistic as shown in the figure below. The Z crossing rate for y axis accelerometer is relatively high for dangerous driving events as per the figure.

<img src="https://i.ibb.co/dDwYwLY/d33.png" alt="d33" border="1">

### 

### Feature: Pitch and Roll

Pitch and Roll is one of the most important features in terms of detecting abnormalities, as it understands the positioning of the car with respect to its center of gravity. The picture below illustrates what pitch and roll is.

<img src="https://i.ibb.co/WnTwNfy/eee1111111.png" alt="eee1111111" border="1">

The pitch and roll values are calculated for every data point. 

```python
merged['PITCH'] = np.arctan2(-merged.acceleration_x, \
                  np.sqrt(merged.acceleration_y *    \
                  merged.acceleration_y + merged.acceleration_z * merged.acceleration_z)) * 57.3

merged['ROLL'] = np.arctan2(merged.acceleration_y, merged.acceleration_z) * 57.3
```

*The formula to calculate pitch and roll was obtained from the wiki [DF Robot](https://wiki.dfrobot.com/How_to_Use_a_Three-Axis_Accelerometer_for_Tilt_Sensing)*

After calculating the pitch, a rolling window pitch and roll value is obtained so that changes in pitch and roll over the window can be detected. The end goal for this feature is to get the sum of ZCR (Z-Crossing Rate) for every booking. This was obtained in a similar fashion as in [the previous feature](###feature:-pitch-and-roll)

### Feature: Trip Distance

This feature was calculated using the formula  below, for each booking where, speed is the aggregated difference in speed and time is the aggregated intervals. The sum of the total distance was used as an aggregate feature for the final training.

$$
distance = speed / time
$$

### Feature: Total Time for Trip

The total time for the trip is the maximum of the field, 'Second' for each booking, or it can also be calculated as the sum of the intervals which was calculated in the previous feature. The sum aggregation of intervals was done for each booking.

### Feature: Bearing Rate

The bearing rate is the change in bearing at every data point divided by the time interval. This feature is aggregated with maximum, variance and sum for each booking.

### Feature: Acceleration

Acceleration was calculated with change in speed divided by the time interval and for every point. The maximum acceleration for each booking was used as a final feature. 



### Summary table for Feature Engineering

<img src="https://i.ibb.co/Wc4FnRL/table.png" alt="table" border="0">





### References

Incel, O. (2015). Analysis of Movement, Orientation and Rotation-Based Sensing for Phone Placement Recognition. *Sensors*, 15(10), pp.25474-25506.
