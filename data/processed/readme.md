This dataset (Finer-grained Affective Computing EEG Dataset, FACED) includes 32-channel EEG data and self-report emotion ratings when subjects were watching 28 emotion-evoking video clips targeted at nine emotions (anger, fear, disgust, sadness, amusement, inspiration, joy, tenderness and neutral emotion).

A total of 123 subjects (75 females, mean age = 23.2 years, ranging from 17 to 38 years) were recruited in the dataset, who were de-identified and indexed as S000∼S122.

The dataset includes the following file folders and files：

# Data/SubXXX：

For each subject, we provide raw EEG data and event data in the “.bdf” file format, self-reported emotion ratings in the MATLAB “.mat” format

# Processed_Data：

For each subject, we provide pre-processed EEG data in the Python “.pkl” format after the pre-processing pipeline (see details in Code/Readme.md) from the raw EEG data. The pre-processed EEG data is a 3-dimension matrix of VideoNum*ElecNum*(TrialDur\*SampRate). The number of video clips is 28. The order of video clips in the pre-processed data was reorganized according to the index of video clips as reported in Stimuli_info.xslx. The number of electrodes is 32. The order of electrodes is provided in Electrode_location.xlsx. The duration of each EEG trial is 30 seconds, and the sampling rate of pre-processed EEG data is 250 Hz.

# EEG_Features：

For each subject, we provide DE and PSD features in the Python “.pkl” format. The DE and PSD features were obtained from the pre-processed data within each non-overlapping second at 5 frequency bands (delta: 1–4 Hz, theta: 4–8 Hz, alpha: 8–14 Hz, beta: 14–30 Hz and gamma: 30–47 Hz).The formula to calculate DE and PSD followed the practice in the SEED dataset (https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html). For each subject, the DE and PSD feature is a 4-dimensional matrix of VideoNum*ElecNum*TrialDur\*FreqBand. Here, there are 5 frequency bands, corresponding to delta, theta, alpha, beta and gamma, respectively.

# Code： Codes for the pre-processing pipeline and technical validation (see more details in Code/Readme.md)

# Files containing recording details:

Dataset_description.md: Description of the dataset
Task_event.xslx: Event information during the experiment
Recording_info.csv: Age and gender，sampling rate, the units of EEG signals for each subject
Stimuli_info.xslx: The details of emotion-evoking video clips
Electrode_Location.xlsx: Electrode information during recording
DataStructureOfBehaviouralData.xlsx: The data structure of the behavioural data for each subject
