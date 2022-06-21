"""
pytorch-lighting MINST image classification 
python mnist_train.py
"""

from pytorch_lightning import Trainer

from Config import Config
from model.MNISTClassifier import MNISTClassifier


def main(cfg):
    # 1. INIT LIGHTNING MODEL
    model = MNISTClassifier(cfg)

    # 2. INIT TRAINER
    trainer = Trainer(gpus=cfg.gpus)

    # 3. START TRAINING
    trainer.fit(model)

if __name__ == '__main__':
    cfg = Config.from_yaml('config/mnist_config.yml')
    main(cfg)
