# for interpretting asr data
import json

# for manipulating general data
import pandas as pd

# for tokenizing
import string

# for timing
from time import perf_counter

# for saving dataset
import pickle


def main():
    start_time = perf_counter()

    # grab labels.csv
    labels_path = "./data/labels.csv"
    labels_df = pd.read_csv(labels_path)

    # grab asr_data.csv
    asr_path = "./data/asr_data.csv"
    asr_df = pd.read_csv(asr_path)

    # data cleaning-
    # drop rows with missing asr data
    asr_df = asr_df.dropna(how="any", axis=0)

    # init dataset
    dataset = {}
    dataset_id = 0

    # iterate over each word in corpus
    for row_id in range(len(labels_df)):

        # grab current row from labels dataframe
        current_row = pd.DataFrame(labels_df.iloc[row_id : row_id + 1]).iloc[0]

        # outline dataset structure
        # fill in data from labels df + empty data for later
        dataset[dataset_id] = {
            "expected": {"word": current_row["expected_text"]},
            "label": current_row["label"],
            "metadata": {
                "activityId": current_row["activityId"],
                "storyId": current_row["storyId"],
                "phraseIndex": current_row["phraseIndex"],
                "word_index": current_row["word_index"],
            },
            "phrase": {},
            "asr": {
                "amazon_data": {},
                "kaldi_data": {},
                "kaldiNa_data": {},
                "wave2vec_transcript_words": {},
                "wave2vec_transcript_phonemes": {},
            },
        }

        # grab current metadata from above, used to query asr dataframe
        activityId = dataset[dataset_id]["metadata"]["activityId"]
        phraseIndex = dataset[dataset_id]["metadata"]["phraseIndex"]

        # asr_data is missing the following activityId,phraseIndex pairs.
        # 98D5EDA1373C11EC89641635D148,4 entirely and
        # 98D5EDA1373C11EC89641635D148,(0,1,2,3) for all translations but amazon
        # bypass this entire activityId
        if activityId == "98D5EDA1373C11EC89641635D148":
            del dataset[dataset_id]
            continue

        # query the relevant asr dataframe row
        current_asr_row = asr_df[asr_df["activityId"] == activityId].iloc[phraseIndex]

        # process story text
        story_text = current_asr_row["story_text"]
        exclude_apostrophe_punctuation = string.punctuation.replace("'", "")
        processed_story_text = story_text.lower().translate(
            str.maketrans("", "", exclude_apostrophe_punctuation)
        )

        # fill in story data from asr df
        dataset[dataset_id]["phrase"] = {"word": processed_story_text.split()}

        ## processing / formatting asr data

        # amazon
        # convert str to dict
        amazon_data = json.loads(current_asr_row["amazon_data"])
        # separate word and confidence data
        amazon_word_list, amazon_confidence_list = zip(*amazon_data["confidence"])
        # place in dataset dict
        dataset[dataset_id]["asr"]["amazon_data"] = {
            "word": [word.lower() for word in amazon_word_list],
            "word_confidence": amazon_confidence_list,
        }

        # some transcriptions are very long and show evidence of someone other
        # than the student talking. cleaning these by limiting samples to transcriptions
        # within 2x of the target phrase length
        transcript_length = len(dataset[dataset_id]["asr"]["amazon_data"]["word"])
        target_length = len(dataset[dataset_id]["phrase"]["word"])
        if transcript_length > 2 * target_length:
            del dataset[dataset_id]
            continue

        # kaldi
        # convert str to dict
        kaldi_data = json.loads(current_asr_row["kaldi_data"])["transcription"]
        # grab words & confidence values
        kaldi_word_list = [entry["word"] for entry in kaldi_data]
        kaldi_confidence_list = [entry["confidence"] for entry in kaldi_data]
        # place in dataset dict
        dataset[dataset_id]["asr"]["kaldi_data"] = {
            "word": kaldi_word_list,
            "word_confidence": kaldi_confidence_list,
        }

        # kaldiNa
        # convert str to dict
        kaldiNa_data = json.loads(current_asr_row["kaldiNa_data"])["transcription"]
        # grab words & confidence values
        kaldiNa_word_list = [entry["word"] for entry in kaldiNa_data]
        kaldiNa_confidence_list = [entry["confidence"] for entry in kaldiNa_data]
        # place in dataset dict
        dataset[dataset_id]["asr"]["kaldiNa_data"] = {
            "word": kaldiNa_word_list,
            "word_confidence": kaldiNa_confidence_list,
        }

        dataset_id += 1

    print(len(dataset))
    print(perf_counter() - start_time)
    breakpoint()

    # save dataset dict as pickle
    with open("./dataset/gsj_ac_dataset.pickle", "wb") as file:
        pickle.dump(dataset, file)


if __name__ == "__main__":
    main()
