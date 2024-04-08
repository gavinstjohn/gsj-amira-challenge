# Gavin St John Amira Data Challenge Submission
## Installation

### Requirements

- Python 3
- Environment Manager
  - Mamba (preferred)
  - Conda
- amira-data-challenge-v2 Data

### Data
Place ```labels.csv```, ```asr_data.csv```, ```arpabet_to_amirabet.json```, and ```all_story_words.dic``` into 
```bash
/gsj-amira-challenge/data/
```


### Environment

From the root directory (```/gsj-amira-challenge/```) 

#### 1.  Execute the installation script for your preferred package/environment manager:
```bash
./build_mamba_env.sh
```
or
```bash
./build_conda_env.sh
```

 #### 2. Activate environment
```bash
conda activate gsj-ac
```

You should now have ```(gsj-ac)``` as your active environment. 

## Usage
I built a clean, flexible, and robust dataset for central usage by a potential multi-faceted automatic mispronunciation detection system. 
### Build the dataset
Execute python script from root:
```bash
python build_dataset.py
```
This will result in a pickled python dictionary located at ```./dataset/gsj_ac_dataset.pickle```

### Browse the dataset
You can view and manipulate the dataset with light accompanying documentation using the notebook ```browse_dataset.ipynb``` or template work file ```quickstart.py```


## Writeup / Ruminations

I built what I think is the most important part of a system- the dataset. To me there are so many potential components which could/would make up a mispronuciation detection system and so built a flexible and robust dataset for that system. 

I wanted all text to have tokenized word and phoneme forms, as well as relevant asr confidence data if provided. 

As far as data cleaning, I removed entries where the transcription was vastly (2x) longer than the target phrase. To me this pointed to the reader being off-task or someone else talking in the background / talking to the reader. Or at the very least, introduced significant noise to the sample. I also removed entries where not all ASR data was present. Also removed were samples where the entire phrase was mispronounced, this suggested to me that something was wrong. Overall this resulted in data yield of 85% and total dataset size of 52109 samples.

This is a dataset which I would be eager to start using to build a mispronunciation detection system. It'd be easy to plug into a pytorch dataset or any other ML tool for analysis. 