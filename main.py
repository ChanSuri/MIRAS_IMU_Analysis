# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.stats
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, show, subplots
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from plots import multiple_bar_chart
from ds_charts import get_variable_types, multiple_bar_chart, choose_grid, HEIGHT

# %% FUNCTIONS

# Function to compute the confusion matrix.
def confusion_matrix(y_true, y_pred):
    df = pd.DataFrame([x for x in zip(y_true, y_pred)],
                      columns=['y_true', 'y_pred'])
    df[['samples']] = 1
    confusion = pd.pivot_table(df, index='y_true',
                               columns='y_pred',
                               values='samples',
                               aggfunc=sum)
    return confusion


# Function to display the classification performance.
def show_performance(best_report_train, best_report_val, best_confusion_train, best_confusion_val, title):
    # Evaluate the performance
    print('TRAINING CLASSIFICATION REPORT')
    print(best_report_train)

    print('\n\nVALIDATION CLASSIFICATION REPORT')
    print(best_report_val)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # Get the confusion matrix - training
    plt.suptitle(title)

    # Normalize by rows
    best_confusion_train = best_confusion_train.divide(best_confusion_train.sum(1), axis=0)
    # Visualize
    sns.heatmap(best_confusion_train, cmap='Blues', annot=True, fmt=".2f", vmin=0, vmax=1,
                ax=axs[0], cbar_kws={'label': 'Occurrences - Training'})
    axs[0].set_xlabel('Prediction')
    axs[0].set_ylabel('True')
    axs[0].set_title('Training Confusion Matrix')
    # Get the confusion matrix - validation
    # Normalize by rows
    best_confusion_val = best_confusion_val.divide(best_confusion_val.sum(1), axis=0)
    # Visualize
    sns.heatmap(best_confusion_val, cmap='Blues', annot=True, fmt=".2f", vmin=0, vmax=1,
                ax=axs[1], cbar_kws={'label': 'Occurrences - Validation'})
    axs[1].set_xlabel('Prediction')
    axs[1].set_ylabel('True')
    axs[1].set_title('Validation Confusion Matrix')

    plt.tight_layout()
    plt.show()

    return


def classification_task(title, clf, X_train_s, X_val_s, y_val, y_train):
    clf.fit(X_train_s, y_train)
    y_train_pred = clf.predict(X_train_s)
    y_val_pred = clf.predict(X_val_s)

    best_accuracy = accuracy_score(y_val, y_val_pred)
    best_report_train = classification_report(y_train, y_train_pred)
    best_report_val = classification_report(y_val, y_val_pred)
    best_confusion_train = confusion_matrix(y_train, y_train_pred)
    best_confusion_val = confusion_matrix(y_val, y_val_pred)

    show_performance(best_report_train, best_report_val, best_confusion_train, best_confusion_val, title)

    return best_accuracy

def calculate_slope(data):
    x = range(len(data))
    y = data.values
    slope, _, _, _, _ = scipy.stats.linregress(x, y)
    return slope

def compute_slope(y):
    output = linregress(list(range(len(y))), y)
    return output.slope


# %% START

df = pd.read_csv('DatasetPiccolo.csv', sep=';')
df = df.drop(columns=['PacketCounter'])
# df.info()

wind = 15 #rolling window

df_mean = df.groupby(['label']).rolling(window=wind, min_periods=1).mean().dropna().reset_index()
df_mean = df_mean.rename(columns={'Acc_X':'mean_Acc_X','Acc_Y':'mean_Acc_Y','Acc_Z':'mean_Acc_Z',
                                 'FreeAcc_X':'mean_FreeAcc_X','FreeAcc_Y':'mean_FreeAcc_Y','FreeAcc_Z':'mean_FreeAcc_Z',
                                 'Gyr_X':'mean_Gyr_X','Gyr_Y':'mean_Gyr_Y','Gyr_Z':'mean_Gyr_Z',
                                 'Mag_X':'mean_Mag_X','Mag_Y':'mean_Mag_Y','Mag_Z':'mean_Mag_Z',
                                 'Quat_q0':'mean_Quat_q0','Quat_q1':'mean_Quat_q1','Quat_q2':'mean_Quat_q2','Quat_q3':'mean_Quat_q3'})
df_mean = df_mean.drop('level_1',axis=1)                         

df_max = df.groupby(['label']).rolling(window=wind, min_periods=1).max().dropna().reset_index()
df_max = df_max.rename(columns={'Acc_X':'max_Acc_X','Acc_Y':'max_Acc_Y','Acc_Z':'max_Acc_Z',
                                 'FreeAcc_X':'max_FreeAcc_X','FreeAcc_Y':'max_FreeAcc_Y','FreeAcc_Z':'max_FreeAcc_Z',
                                 'Gyr_X':'max_Gyr_X','Gyr_Y':'max_Gyr_Y','Gyr_Z':'max_Gyr_Z',
                                 'Mag_X':'max_Mag_X','Mag_Y':'max_Mag_Y','Mag_Z':'max_Mag_Z',
                                 'Quat_q0':'max_Quat_q0','Quat_q1':'max_Quat_q1','Quat_q2':'max_Quat_q2','Quat_q3':'max_Quat_q3'})
df_max = df_max.drop('level_1',axis=1)

df_min = df.groupby(['label']).rolling(window=wind, min_periods=1).min().dropna().reset_index()
df_min = df_min.rename(columns={'Acc_X':'min_Acc_X','Acc_Y':'min_Acc_Y','Acc_Z':'min_Acc_Z',
                                 'FreeAcc_X':'min_FreeAcc_X','FreeAcc_Y':'min_FreeAcc_Y','FreeAcc_Z':'min_FreeAcc_Z',
                                 'Gyr_X':'min_Gyr_X','Gyr_Y':'min_Gyr_Y','Gyr_Z':'min_Gyr_Z',
                                 'Mag_X':'min_Mag_X','Mag_Y':'min_Mag_Y','Mag_Z':'min_Mag_Z',
                                 'Quat_q0':'min_Quat_q0','Quat_q1':'min_Quat_q1','Quat_q2':'min_Quat_q2','Quat_q3':'min_Quat_q3'})
df_min = df_min.drop('level_1',axis=1)

df_std = df.groupby(['label']).rolling(window=wind, min_periods=1).std().dropna().reset_index()
df_std = df_std.rename(columns={'Acc_X':'std_Acc_X','Acc_Y':'std_Acc_Y','Acc_Z':'std_Acc_Z',
                                 'FreeAcc_X':'std_FreeAcc_X','FreeAcc_Y':'std_FreeAcc_Y','FreeAcc_Z':'std_FreeAcc_Z',
                                 'Gyr_X':'std_Gyr_X','Gyr_Y':'std_Gyr_Y','Gyr_Z':'std_Gyr_Z',
                                 'Mag_X':'std_Mag_X','Mag_Y':'std_Mag_Y','Mag_Z':'std_Mag_Z',
                                 'Quat_q0':'std_Quat_q0','Quat_q1':'std_Quat_q1','Quat_q2':'std_Quat_q2','Quat_q3':'std_Quat_q3'})
df_std = df_std.drop('level_1',axis=1)
#df_slope = df.groupby(['label']).rolling(window=wind).apply(compute_slope).dropna()

"""
ys = df.label.to_numpy()
stride = ys.strides
slopes, intercepts = np.polyfit(np.arange(wind), 
                                as_strided(ys, (len(df)-wind+1, wind), 
                                           stride+stride).T,
                                deg=1)
"""

df_old = df

#df_final = df_mean.merge(df_max, how='inner')
df_final = pd.merge(df_mean, df_max, left_index=True, right_index=True).dropna().drop(columns=['label_y'])
#df_final = pd.merge(df_mean, df_max, how='inner', on='label')


# %% VISUALIZATION


# Ratio number of records (rows) / number of variables (features)
figure(figsize=(4,2))
values = {'nr records': df.shape[0], 'nr variables': df.shape[1]}
plt.bar(list(values.keys()), list(values.values()))
show()

# Dataset summary
summary5 = df.describe()
print(summary5)

# OUTLIERS
figure()
df.boxplot(rot=45)
show()
NR_STDEV: int = 2

numeric_vars = get_variable_types(df)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

outliers_iqr = []
outliers_stdev = []
summary5 = df.describe(include='number')

for var in numeric_vars:
    iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
    outliers_iqr += [
        df[df[var] > summary5[var]['75%']  + iqr].count()[var] +
        df[df[var] < summary5[var]['25%']  - iqr].count()[var]]
    std = NR_STDEV * summary5[var]['std']
    outliers_stdev += [
        df[df[var] > summary5[var]['mean'] + std].count()[var] +
        df[df[var] < summary5[var]['mean'] - std].count()[var]]

outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
figure(figsize=(12, HEIGHT))
multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
show()

# Distribution

numeric_vars = get_variable_types(df)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = choose_grid(len(numeric_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(df[numeric_vars[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
show()

# Bar Plot

num_pointing = sum(df['label'] == 'POINTING')
num_bbt = sum(df['label'] == 'BBT')
num_9hpt = sum(df['label'] == '9HPT')

tasks = ['POINTING', 'BBT', '9HPT']
values = [num_pointing, num_bbt, num_9hpt]

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(tasks, values, color='red',
        width=0.4)

plt.xlabel("Tasks")
plt.ylabel("No. of occurences")
plt.title("Dataset balancement")
plt.show()

# %% Corr Mat

# FEATURE CORRELATION MATRIX
tmp = df.drop(columns=['label'])
correlation_matrix = tmp.corr().abs()
plt.figure(figsize=(10, 10))
ax = sns.heatmap(correlation_matrix, cmap='Reds',
                 vmin=.0, vmax=1, cbar_kws={'label': 'Correlation'})

# %% STANDARDIZATION
features = df.drop(columns=['label'])

X = features.to_numpy()
y = df['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, train_size=.7, random_state=15)

# Standardize the dataset
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)


# %% CLASSIFICATION -- RANDOM FOREST
title = 'Classification Task - Random Forest'

clf = RandomForestClassifier()

accuracy_RF = classification_task(title, clf, X_train_s, X_val_s, y_val, y_train)

# %% CLASSIFICATION -- GAUSSIAN NAIVE BAYES
title = 'Classification Task - Gaussian Naive Bayes'

clf = GaussianNB()

accuracy_GNB = classification_task(title, clf, X_train_s, X_val_s, y_val, y_train)

# %% CLASSIFICATION -- SUPPORT VECTOR MACHINE
title = 'Classification Task - Support Vector Machine'

clf = SVC(random_state=42)

accuracy_SVM = classification_task(title, clf, X_train_s, X_val_s, y_val, y_train)

# %% CLASSIFICATION -- KNN CLASSIFIER
title = 'Classification Task - K - Nearest Neighbour'

clf = KNeighborsClassifier()

accuracy_KNN = classification_task(title, clf, X_train_s, X_val_s, y_val, y_train)

# %% Show the accuracy

accuracy = [accuracy_RF, accuracy_GNB, accuracy_SVM, accuracy_KNN]
classifiers = ['Random Forest', 'Gaussian Naive Bayes', 'Support Vector Machine', 'K-Nearest Neighbours']

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(classifiers, accuracy, color='green',
        width=0.2)
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Classifiers comparison")
plt.show()

