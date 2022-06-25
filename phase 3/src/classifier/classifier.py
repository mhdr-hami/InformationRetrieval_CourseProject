from typing import List,Dict,Any

class MyClassifier:
    def __int__(self,configs):
        ...


    def train(self):
        ...

    def inference(self,text_list: List[str],get_prob: bool=True)-> List[Dict[str,Any]]:
        ...

def get_model(config):
    model_cfg = config.get('model')
    model = MyClassifier(config=model_cfg)
    return model

def load_model(path):
    ...
