from configs.MetricConfig import Metric

class DafaultArguments():
    def __init__(self) -> None:
        
        self.epochs = 100

        self.train_split = 0.7
        self.test_split  = 0.3
        self.valid_split = 0.2 

        self.met_params = Metric()

        self.tensorboard_root_folder = "logs"
        self.default_root_save_folder = "to_evaluate"
        self.experiment_folder = "page_blanche"
        self.exp_name = "Pas mal + loss modification."
        self.note = "Données standardizée, modèles plus petit."
        
        
        
        