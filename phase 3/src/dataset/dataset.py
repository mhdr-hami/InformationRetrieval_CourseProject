

class Dataset:
    def __int__(self, configs):
        self.train_set = None
        self.dev_set = None
        self.test_set = None

    def load_dataset_from_file(self):
        '''
        :param file_path:
        :return:
        '''
        ...

        return self

    def split_train_dev_test(self,train_per=70,dev=15,test=15):
        '''

        :param train_per: percentage of how many of samples should be in train set
        :param dev: percentage of how many of samples should be in dev set
        :param test: percentage of how many of samples should be in dev set
        :return:
        '''


def prepare_dataset(config):
    ds_cfg = config.get('dataset')
    dataset = Dataset(ds_cfg).load_dataset_from_file()
    dataset.split_train_dev_test()
    return