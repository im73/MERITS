# MERITS
MERITS : Medication Recommendation for Chronic Disease with Irregular Time-Series


## Overview
This repo contains code to MERITS, which is an attention-based encoder-decoder framework to combine the historical information of patients and medications from EMR. Besides, MERITS captures the irregular time-series dependencies with the neural ordinary differential equations (Neural ODE), and leverages a drug-drug interaction knowledge graph and two learned medication relation graphs to explore the co-occurrence and sequential correlations of the medications.  
Sample data are provided in code/data/sample_data.pkl. To get access to the whole data set, please fill in the form [here](https://forms.gle/RFCVXVzsBX1kzhhYA).

## Requirements
- Pytorch == 1.7.1
- Python == 3.7.9

## Running the code with demo data
    ./run.sh
