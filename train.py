import os

from pytorch_lightning import Trainer

from cliport.agent import TwoStreamClipLingUNetLatTransporterAgent
from cliport.dataset.dataset import RavensDataset #, RavensMultiTaskDataset


def main():

    # Trainer
    max_epochs = 201000 // 1000
    trainer = Trainer(
        gpus=[0],
        max_epochs=max_epochs,
        automatic_optimization=False,
        check_val_every_n_epoch=max_epochs // 50,
    )

    # Config
    data_dir = "${root_dir}/data"
    task = "packing-boxes-pairs-seen-colors"
    agent_type = "two_stream_full_clip_lingunet_lat_transporter"
    n_demos = 1000
    n_val = 100
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    train_ds = RavensDataset(
        os.path.join(data_dir, '{}-train'.format(task)),
        n_demos=n_demos,
        augment=True
    )
    val_ds = RavensDataset(
        os.path.join(data_dir,'{}-val'.format(task)),
        n_demos=n_val,
        augment=False
    )

    # Initialize agent
    # agent = agents.names[agent_type](name, cfg, train_ds, val_ds)
    agent = TwoStreamClipLingUNetLatTransporterAgent(
        name,
        train_ds,
        val_ds
    )

    # Main training loop
    trainer.fit(agent)


if __name__ == '__main__':
    main()