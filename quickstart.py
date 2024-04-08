import pickle


def main():
    # open dataset file
    dataset_path = "./dataset/gsj_ac_dataset.pickle"
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    print(dataset[0].keys())


if __name__ == "__main__":
    main()
