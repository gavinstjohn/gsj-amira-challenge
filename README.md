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
Note: Personally I prefer mamba and that is the package/environment manager I used. However, you shouldn't have any issues using conda. 

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
python dataset_builder.py
```
This will result in a pickled python dictionary located at ```./dataset/gsj_ac_dataset.pickle```

### Browse the dataset
You can view and manipulate the dataset with accompanying documentation using the notebook ```dataset_browser.ipynb```

## 