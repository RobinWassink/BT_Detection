# BT_Detection

This repository contains the code used in the ML framework part of the bachelor thesis "Implementation and Detection of Spectrum Data Falsification Attacks Affecting Crowdsensing Platforms".

## How to 

* Get de data from *link*. Add it to this directory so you get the following structure:
The following shows the outline structure of the this system.
   ```bash
      |—— data
        |—— 2022-07-06_13-06-52_fft_20000_30
          |—— raw
        |—— 2022-07-07_17-27-05_fft_200000_30
          |—— raw
        |—— 2022-07-08_22-41-52_rtlsdr_20000000_30
          |—— raw
        |—— 2022-07-10_04-08-51_rtlsdr_200000000_30
          |—— raw
        |—— 2022-07-11_16-41-53_rtlsdr_800000000_30
          |—— raw
      |—— preparation
        |—— get_features.py
        |—— scaling.py
      |—— ML
        |—— ml.py
      |—— visualization
        |—— performance_evaluation.py
   ```
   
* Create the features with
```
python preparation/get_features.py {FOLDER}
```
eg. 
```
python preparation/get_features.py 2022-07-07_17-27-05_fft_200000_30
```

* Train the models with
```
python visualization/performance_evaluation.py {FOLDER}
```

* Create final visualizations with
```
python visualization/performance_evaluation.py
```
