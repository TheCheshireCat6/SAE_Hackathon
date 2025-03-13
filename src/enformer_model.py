import torch
import pytorch_lightning as pl
import pandas as pd
from scipy.stats import pearsonr

from enformer_pytorch import Enformer, from_pretrained
from enformer_pytorch.finetune import HeadAdapterWrapper
from hanni_pytorch import utils

L = 393216
target_crop_length = 114688
# num_cell_types = 25

class EnformerModule(pl.LightningModule):
    def __init__(self, num_tracks, mode, learning_rate_phase_1=1e-3, learning_rate_phase_2=1e-4):
        super().__init__()
        self.enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough', use_tf_gamma=False)
        self.model = HeadAdapterWrapper(enformer=self.enformer, num_tracks=num_tracks, post_transformer_embed=False)
        self.mode = mode
        self.learning_rate_phase_1 = learning_rate_phase_1
        self.learning_rate_phase_2 = learning_rate_phase_2
        self.test_step_outputs = [[], []]  # two dataloaders,
        self.save_path = None
        self.num_tracks = num_tracks

    def set_save_path(self, save_path):
        self.save_path = save_path

    def forward(self, x, target=None):  # enformer should only be frozen during phase 1
        preds = self.model(x)
        if not utils.exists(target):
            return preds
        return self.model(seq=x, target=target)

    def training_step(self, batch):
        x, y_processed = utils.process_batch(batch, L, target_crop_length)
        loss = self.model(x, target=y_processed)
        return loss

    def validation_step(self, batch):
        x, y_processed = utils.process_batch(batch, L, target_crop_length)
        loss = self(x, target=y_processed)
        # Log validation loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def test_step(self, batch):
        x, y_processed = utils.process_batch(batch, L, target_crop_length)
        output = self(x)
        loss = self(x, target=y_processed)
        # Append y and y_hat to list
        self.test_step_outputs[0].append(y_processed) # ground truth
        self.test_step_outputs[1].append(output)      # predictions

        # Log test loss
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Return the output and y_processed for correlation calculation
        return {'test_loss': loss, 'output': output, 'y_processed': y_processed}


    def on_test_epoch_end(self):
        # Concatenate all outputs and y_processed from each batch
        all_y_processed = torch.cat([tmp for tmp in self.test_step_outputs[0]], dim=0)
        all_outputs = torch.cat([tmp for tmp in self.test_step_outputs[1]], dim=0)

        # Compute Pearson correlation
        pearson_corr = pearsonr(all_outputs.flatten().cpu().numpy(), all_y_processed.flatten().cpu().numpy())[0]

        results = []

        for i in range(self.num_tracks):

            results += [{
                  "cell_type": i,
                  f"pcc": pearsonr(all_outputs.cpu().numpy()[:, :, i].flatten(), all_y_processed.cpu().numpy()[:, :, i].flatten())[0]
            }]

        results = pd.DataFrame(results)
        print(f'{self.save_path}/test_performance.csv')
        results.to_csv(f'{self.save_path}/test_performance.csv', index=False)

        self.test_step_outputs.clear()  # free memory

        # Log Pearson correlation
        self.log('test_pearson_corr', pearson_corr)

        # Optionally, you can return this value too
        return {'test_pearson_corr': pearson_corr}, results


    def _configure_optim_phase_1(self):
        # Freeze original layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze new output tracks
        for param in self.model.to_tracks.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(self.model.to_tracks.parameters(), lr=self.learning_rate_phase_1)
        return optimizer

    def _configure_optim_phase_2(self):
        """
        Configure optimizer for phase 2: Unfreezes all layers except normalization layers.
        """
        for name, param in self.model.named_parameters():
            layers_list = name.split('.')[:-1]
            layer = utils.get_submodule(self.model, layers_list)

            if isinstance(layer, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm)):
                param.requires_grad = False  # Keep normalization layers frozen
            else:
                param.requires_grad = True  # Unfreeze other layers

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=self.learning_rate_phase_2)
        return optimizer

    def configure_optimizers(self):
        if self.mode == 'phase_1':
            return self._configure_optim_phase_1()
        elif self.mode == 'phase_2':
            return self._configure_optim_phase_2()