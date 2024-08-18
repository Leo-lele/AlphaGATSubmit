from indicator.config import Config

from torch.utils.data import DataLoader

from indicator.datasetstock import Dataset_Stock
import logging

def get_loaders(
        config: Config,
        num_workers=0,
        pin_memory=False,
        train_transform=None,
        val_transform=None,
):
    train_dataset = Dataset_Stock(dataset=config.dataset,day_windows=config.day_windows,
                                  flag="train", device=config.device)
    valid_dataset = Dataset_Stock(dataset=config.dataset,day_windows=config.day_windows,
                                  flag="valid", device=config.device)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


    return train_dataloader, valid_dataloader

def get_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    #Create a file handler to log messages to a file.
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create a console handler to output logs to the console.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to the Logger.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class NoamOpt:
    "Optim wrapper that implements rate."
    # 512, 1, 400
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup == 0:
            return self.factor
        else:
            return self.factor * \
                   (self.model_size ** (-0.5) *
                    min(step ** (-0.5), step * self.warmup ** (-1.5)))



