from information_retrieval.constructor import BSBIIndex

DATASET_PATH = 'information_retrieval/Dataset_IR/Train'
OUTPUT_DIR = 'information_retrieval/Dataset_IR/out'


if __name__ == '__main__':
    BSBI_instance = BSBIIndex(data_dir=DATASET_PATH, output_dir=OUTPUT_DIR)
    query = input('search >>  ')
    result = BSBI_instance.retrieve(query)
    print(result)
