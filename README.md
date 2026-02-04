# Bachelor_project

## Structure of this repo is (very) roughly as follows:

- Analysis folder contains both R and Python scrips for data analysis
- Data_Making folder contains Python scripts for cleaning, enriching and formatting data both for validation- and training sets.
- Failed_llm_identifications contains legacy Python scripts in which we tried to expand on blame detection using both Llama and Deepseek models, forcing rigid JSON output structure using the Outlines package. (This approach was fully abandoned)
- Inference folder contains Python scripts for doing actual inference on the data from Folketinget using the final BlameBERT language model after training was finished.
- Json_files folder contains much of the data, both for training, validation and inference in JSON formats. 
- Model_data folder contains the final validation set (gold labelled) and a raw .csv file which was output from BlameBERT runs.
- Training folder contains Python scripts with the final training pipeline for BlameBERT. (Also the exploratory training sessions where parameters were dialed in.)