import numpy as np

from torch.nn.functional import nll_loss


class HardNegativeExamplesMining:

    def __init__(self, n_samples):
        self.loss = np.zeros(n_samples, dtype=np.float)

    def accumulate(self, evaluator, trainer, logger):
        data_loader = evaluator.state.dataloader
        y_pred, y = evaluator.state.output

        start_index = ((evaluator.state.iteration - 1) % len(data_loader)) * data_loader.batch_size
        end_index = min(start_index + data_loader.batch_size, len(self.loss))
        batch_loss = nll_loss(y_pred, y)
        self.loss[start_index:end_index] = batch_loss

    def update(self, evaluator, trainer, logger):
        trainer.batch_sampler.sampler.weights = self.loss * 5.0

