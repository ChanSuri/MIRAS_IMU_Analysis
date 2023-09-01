# Project
Unite! Blended Intensive Program on Communications engineering and data science. Learn about Communications and Data Science for industrial applications in this advanced bachelor's and master's level Erasmus+ Blended Intensive Program which will combine online learning at your home university and in-campus work at Aalto University Summer School. The Duration is 21st August to 1st September.

We did a project in a week based on the project MIRAS_IMU_Analysis_2022. 

## Presentation show: https://prezi.com/view/aJ1AseCZegWK0D66OdVt/

# MIRAS_IMU_2022

This directory contains the second part of the MIRAS IMU analysis done from 08.08.2022 until 15.10.2022 by Eileen Kränzle. 

## Data

The collected data can be found in the *data* directory. 

In this directory the data of 11 control subjects (A, B, C, D, E, F, G, H, I, J, K) can be found. From the left-handed subject G, data was collected from his right arm (GR) and his left arm (GL).
Further, the *draft* directory can be copied for collecting new data, since it contains the necessary structure and files for easy usage with the python scripts. There are also some additional directories (test, test_2, test_3) containing different test measurements of the sensors.

In each directory there are:

- Two screenshots named *orientation.pgn* and *reset heading.png* documenting the drift of the sensors: One screenshot was taken right after the heading reset and the other one after all measurements were done and the sensors were removed from the subject. There are corresponding recordings in these two positions available aswell.

- *raw_data*: the raw data of the recordings. For each recording one .mtb file is stored. This file is converted using the MT manager to .txt files, which results in one .txt file for each sensor with the following name convention: MT_*awinda base station id*-*run number*-000-*sensor id*.txt

- *overview.xlsx*: gives an overview of all the tasks performed and matches the tasks to the files available in the *raw_data* directory.
    - **task name**: should be unique, repetitive tasks are distinguished with numbers. *heading reset*, *calibration X*, *pointing X*, *BBT X*, *9HPT X* and *orientation* (with X as a number) are recognized in the scipt. All other task names are ignored.
    - **run number**: refering to the correct run number of the task
    - **Time**: time, when the measurement was done
    - **Test result**: for BBT number of blocks transfered, for 9HPT measured time in seconds
    - **Start time**: time point when actual task starts (is used for task cutting of FNT, BBT, 9HPT)
    - **Middle time**: only for 9HPT: time point when change from putting pegs in to removing pegs happens (is used for task cutting)
    - **End time**: time point when actual task ends (is used for task cutting of FNT, BBT, 9HPT)
    - **Repetitions**: number of repetitions (standard number 5 for FNT, 9 for 9HPT, number of blocks for BBT) (is used for repetition splitting, currently only used for FNT)

- *Study protocol*: contains questionaire & other particularities for the measurements with the corresponding subject

## Data Preprocessing

The data preprocessing is the same as in the first IMU analysis (*MIRAS-IMU analysis*) and consists of manual task cutting, gravity removal and a low pass filter. It is done in the *imu_analysis.ipynb* script. Min, Max, Mean and Standardization are used during data pre-processing.

## Feature extraction

The feature extraction is currently only done for the FNT with the same features as in the first IMU analysis. It is also done in the *imu_analysis.ipynb* script.

For the two new tasks (BBT and 9HPT), no feature extraction was performed yet, because there is no patient data available yet.

## Storage of extracted features

The results of the *imu_analysis.ipynb* script are stored in the *Results* directory. For each feature a .csv file is generated containing the feature for each subject, task, sensor, signal type and axis.

## Comparison

The comparison is done in the *comparison.ipynb* script, which produces plots that are stored in the *plots* directory. 

## Documentation
The main documentation is done in the same report as the first IMU analysis and therefore in the *MIRAS-IMU analysis* directory. For the measurements some pictures, videos and graphics are stored in the *pictures and videos* directory documenting the experiments. The last PowerPoint presentation in the *MIRAS-IMU analysis* directory is also about the measurements.
