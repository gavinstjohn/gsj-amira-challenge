import json
import pandas as pd
from time import perf_counter
import pickle
import string
from copy import copy


def build_dataset(dataset_path: str = "./dataset/gsj_ac_dataset.pickle"):
    """
    Builds amira mispronunciation detection dataset.

    Parameters
    ----------
    dataset_path : str
        Path to desired dataset save location.
    """
    start_time = perf_counter()

    # --- setup ---
    # grab labels.csv
    labels_path = "./data/labels.csv"
    labels_df = pd.read_csv(labels_path)

    # grab asr_data.csv
    asr_path = "./data/asr_data.csv"
    asr_df = pd.read_csv(asr_path)

    # rename phrase_index to phraseIndex for consistency
    asr_df = asr_df.rename(columns={"phrase_index": "phraseIndex"})

    # build phoneme map
    print(
        f"Building phoneme map... Time elapsed: {round(perf_counter()-start_time,2)} seconds"
    )
    phoneme_map = build_phoneme_map()
    # --- end setup ---

    # --- data cleaning ---
    print(
        f"Cleaning data... Time elapsed: {round(perf_counter()-start_time,2)} seconds"
    )
    # collect rows with missing data so we can remove them from labels data also
    asr_nulls = asr_df[asr_df.isnull().any(axis=1)]
    # add this row because it's missing from asr but not labels:
    asr_nulls.loc[asr_nulls.index.max() + 1] = {
        "activityId": "98D5EDA1373C11EC89641635D148",
        "phraseIndex": 4,
    }
    # drop rows with missing asr data
    asr_df = asr_df.dropna(how="any", axis=0)

    # drop rows in labels_df corresponding to null (activityId, phraseIndex) pairs
    merged_df = pd.merge(
        labels_df,
        asr_nulls,
        on=["activityId", "phraseIndex"],
        # how="outer",
        how="left",
        indicator=True,
    )
    labels_df = labels_df[merged_df["_merge"] == "left_only"].reset_index(drop=True)
    # --- end data cleaning ---

    # init dataset
    dataset = {}
    dataset_id = 0

    print(
        f"Building dataset... Time elapsed: {round(perf_counter()-start_time,2)} seconds"
    )
    # iterate over each word in corpus
    for row_id in range(len(labels_df)):

        # grab current row from labels dataframe
        current_row = pd.DataFrame(labels_df.iloc[row_id : row_id + 1]).iloc[0]

        # outline dataset structure
        # fill in data from labels df + empty data for later
        dataset[dataset_id] = {
            "expected": {
                "word": current_row["expected_text"],
                "phoneme": phonemize([current_row["expected_text"]], phoneme_map)[0],
            },
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
                "wav2vec_transcript_words": {},
                "wav2vec_transcript_phonemes": {},
            },
        }

        # grab current metadata from above, used to query asr dataframe
        activityId = dataset[dataset_id]["metadata"]["activityId"]
        phraseIndex = dataset[dataset_id]["metadata"]["phraseIndex"]

        # query the relevant asr dataframe row
        current_asr_activity = asr_df[asr_df["activityId"] == activityId]
        current_asr_row = current_asr_activity[
            current_asr_activity["phraseIndex"] == phraseIndex
        ]
        current_asr_row = current_asr_row.iloc[0]

        # process story text
        story_text = current_asr_row["story_text"]
        story_text_word = tokenize(story_text)
        story_text_phoneme = phonemize(story_text_word, phoneme_map)
        dataset[dataset_id]["phrase"] = {
            "word": story_text_word,
            "phoneme": story_text_phoneme,
        }

        # # processing / formatting asr data

        # amazon
        # convert str to dict
        amazon_data = json.loads(current_asr_row["amazon_data"])
        # separate word and confidence data
        amazon_word_list, amazon_confidence_list = zip(*amazon_data["confidence"])
        amazon_word = [word.lower() for word in amazon_word_list]
        # place in dataset dict
        dataset[dataset_id]["asr"]["amazon_data"] = {
            "word": amazon_word,
            "word_confidence": amazon_confidence_list,
            "phoneme": phonemize(amazon_word, phoneme_map),
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
        kaldi_word = [entry["word"] for entry in kaldi_data]
        # place in dataset dict
        dataset[dataset_id]["asr"]["kaldi_data"] = {
            "word": kaldi_word,
            "word_confidence": [entry["confidence"] for entry in kaldi_data],
            "phoneme": phonemize(kaldi_word, phoneme_map),
        }

        # kaldiNa
        # convert str to dict
        kaldiNa_data = json.loads(current_asr_row["kaldiNa_data"])["transcription"]
        kaldiNa_word = [entry["word"] for entry in kaldiNa_data]
        # place in dataset dict
        dataset[dataset_id]["asr"]["kaldiNa_data"] = {
            "word": kaldiNa_word,
            "word_confidence": [entry["confidence"] for entry in kaldiNa_data],
            "phoneme": phonemize(kaldiNa_word, phoneme_map),
        }

        # wav2vec word
        wav2vec_transcript_words = current_asr_row["wav2vec_transcript_words"]
        wav2vec_word = tokenize(wav2vec_transcript_words)
        dataset[dataset_id]["asr"]["wav2vec_transcript_words"] = {
            "word": wav2vec_word,
            "phoneme": phonemize(wav2vec_word, phoneme_map),
        }
        # wav2vec phoneme
        wav2vec_transcript_phonemes = current_asr_row["wav2vec_transcript_phonemes"]
        dataset[dataset_id]["asr"]["wav2vec_transcript_phonemes"] = {
            "phoneme": wav2vec_transcript_phonemes.split()
        }

        dataset_id += 1

    print(
        f"Dataset complete, initialized to {len(dataset)} samples. Elapsed time: {round(perf_counter()-start_time,2)} seconds"
    )

    print(f"Saving dataset to {dataset_path}")
    # save dataset dict as pickle
    with open(dataset_path, "wb") as file:
        pickle.dump(dataset, file)
    print(
        f"Dataset built and saved. Elapsed time: {round(perf_counter()-start_time,2)} seconds"
    )


def build_phoneme_map(
    word_to_arpabet_path: str = "./data/all_story_words.dic",
    arpabet_to_amirabet_path: str = "./data/arpabet_to_amirabet.json",
) -> dict:
    # init dict
    phoneme_map = {}
    # open relevant files
    word_to_arpabet_file = open(word_to_arpabet_path, "r")
    arpabet_to_amirabet_map = json.load(open(arpabet_to_amirabet_path, "r"))
    arpabet_to_amirabet_map["SIL"] = ""
    # iterate over each entry in dictionary file and translate
    for line in word_to_arpabet_file:
        # grab word and remove punctuation
        word = line.split()[0].translate(str.maketrans("", "", string.punctuation))
        # grab translation of word
        word_arpabet = line.split()[1:]
        # "SIL" is a '-' in the dictionary
        # part of the data cleaning done removes '-' from this dataset
        # if "SIL" in word_arpabet:
        # word_arpabet.remove("SIL")
        if len(word_arpabet) == 1:
            word_amirabet = arpabet_to_amirabet_map[word_arpabet[0]]
        else:
            # replace arpabet phoneme characters with amirabet characters
            word_amirabet = [arpabet_to_amirabet_map[pho] for pho in word_arpabet]
        phoneme_map[word.lower()] = "".join(word_amirabet)

    return phoneme_map


def phonemize(word_list: list[str], phoneme_map: dict) -> list[str]:
    """
    Translate a list of words from letters to AMIRABET phonemes.

    Parameters
    ----------
    word_list : list[str]
        List of english input words.

    Returns
    -------
    phoneme_list : list[str]
        List of english words written in phonemes.
    """
    word_list = copy(word_list)

    for id, word in enumerate(word_list):
        word_list[id] = word.translate(str.maketrans("", "", string.punctuation))

    # translate words
    phoneme_list = [
        phoneme_map[word] if word in phoneme_map.keys() else False for word in word_list
    ]

    return phoneme_list


def tokenize(text) -> list[str]:
    """
    Tokenize a given string by:
        Removing punctuation (except apostrophes).
        Make lowercase.
        Split from string to list of strings

    Parameters
    ----------
    text : str
        String containing one sentence of text.

    Returns
    -------
    list[str]
        Tokenized text.
    """

    exclude_apostrophe_punctuation = string.punctuation.replace("'", "")
    # remove punctuation (except apostrophes)
    processed_text = text.lower().translate(
        str.maketrans("", "", exclude_apostrophe_punctuation)
    )
    return processed_text.split()


def main():
    build_dataset()


if __name__ == "__main__":
    main()
