from fire import Fire
from src.classifier.classifier import get_model
from src.dataset.dataset import prepare_dataset
from src.utils import span_time,load_config_file


@span_time()
def trainer(config_path):
    config = load_config_file(config_path)
    dataset = prepare_dataset(config)
    model = get_model(config)
    model.train(dataset)
    model.save_model_results()


if __name__ == '__main__':
    Fire(trainer)

