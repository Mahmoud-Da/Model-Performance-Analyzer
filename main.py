import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import config
import data_setup
import engine
import model_builder
import utils


def run_experiments():
    """Runs all experiments defined in config.py."""

    # 1. SETUP DATA
    # -------------
    # Note: Assuming the same dataset for all experiments as defined in config
    print(f"[INFO] Using data: {config.DATA_NAME}")
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_classes=config.NUM_CLASSES,
        data_name=config.DATA_NAME
    )
    print(f"[INFO] Device: {config.DEVICE}")

    # RUN EXPERIMENT LOOP
    # ----------------------
    for i, exp_config in enumerate(config.EXPERIMENTS):
        print("\n" + "="*50)
        print(f"[INFO] Experiment number: {i+1} / {len(config.EXPERIMENTS)}")
        print(f"[INFO] Model: {exp_config['model_name']}")
        print(f"[INFO] Number of epochs: {exp_config['epochs']}")
        print("="*50)

        # Create model
        model = model_builder.create_model(model_name=exp_config['model_name'],
                                           num_classes=config.NUM_CLASSES,
                                           device=config.DEVICE)

        # Create loss function and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=config.LEARNING_RATE)

        # Create a new SummaryWriter for each experiment
        log_dir = os.path.join(config.LOG_DIR,
                               datetime.now().strftime("%Y-%m-%d"),
                               config.DATA_NAME,
                               exp_config['model_name'],
                               f"{exp_config['epochs']}_epochs")

        print(f"[INFO] Created SummaryWriter, saving to: {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)

        engine.train(model=model,
                     train_dataloader=train_dataloader,
                     test_dataloader=test_dataloader,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     epochs=exp_config['epochs'],
                     device=config.DEVICE,
                     writer=writer)

        save_filename = f"{exp_config['model_name']}_{config.DATA_NAME}_{exp_config['epochs']}_epochs.pth"
        utils.save_model(model=model,
                         target_dir=config.MODEL_SAVE_DIR,
                         model_name=save_filename)

        print("-" * 50 + "\n")


if __name__ == "__main__":
    run_experiments()
