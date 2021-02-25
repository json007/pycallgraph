import sys
ROOT_DIR = 'D:/code/pycallgraph'
# ROOT_DIR = '/mnt/d/code/pycallgraph'
sys.path.append(f"{ROOT_DIR}")
print(sys.path)


from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH, _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
    from torchvision.datasets.mnist import MNIST
else:
    from tests.helpers.datasets import MNIST


class LitAutoEncoder(pl.LightningModule):
    """
    >>> LitAutoEncoder()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28),
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



pl.seed_everything(1234)
# ------------
# args
# ------------
parser = ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--hidden_dim', type=int, default=128)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
# ------------
# data
# ------------
dataset = MNIST(_DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST(_DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
test_loader = DataLoader(mnist_test, batch_size=args.batch_size)



from pycallgraph import PyCallGraph,Config
from pycallgraph.output import GraphvizOutput
from pycallgraph import GlobbingFilter

graphviz = GraphvizOutput()
graphviz.output_file = f"{ROOT_DIR}/pl.png"

config = Config(**{"verbose":True})
config.trace_filter = GlobbingFilter(include=[
        'pytorch_lightning.*',
    ])

tracker_log = f"{ROOT_DIR}/tracker.pkl"

with PyCallGraph(config=config, output=graphviz, tracker_log=tracker_log, package_prefix = 'pytorch_lightning.'):
    # ------------
    # model
    # ------------
    model = LitAutoEncoder()
    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)
    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


print("done")