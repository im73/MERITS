# MERITS
MERITS : Medication Recommendation for Chronic Disease with Irregular Time-Series


## Overview
This repo contains code to MERITS, which is an attention-based encoder-decoder framework to combine the historical information of patients and medications from EMR. Besides, MERITS captures the irregular time-series dependencies with the neural ordinary differential equations (Neural ODE), and leverages a drug-drug interaction knowledge graph and two learned medication relation graphs to explore the co-occurrence and sequential correlations of the medications.  
We don't give out all the data because of privacy, but part of them are available for test. If you want to train the model by you own with full data, please contact us[zhangs@act.buaa.edu.cn] for further discussion.

## Requirements
- Pytorch == 1.7.1
- Python == 3.7.9

## Running the code with demo data
    ./run.sh
