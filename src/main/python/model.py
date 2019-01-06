import torch
import trainer
from torch import nn

GRU_DEPTH = 2
HIDDEN_LAYER_SIZE = 2

            
def forward_sequence(model, seq_len, x, use_output_label=False):
    gru_hidden=None
    y = torch.Tensor(x.shape)
    for i in range(seq_len - 1):
        if use_output_label or i == 0:
            x_current = x[i]
        else:
            x_current = out
        out, gru_hidden = model.forward(x_current, gru_hidden)
        y[i] = out 
    return y 
    

class PredefinedFeatureModelTrainer(trainer.BaseTrainer):
    def __init__(self, learning_rate, model):
        parameters = model.parameters()
        self.model = model
        super(PredefinedFeatureModel, self).__init__(
            learning_rate, parameters
        )

    def get_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def get_loss(self, seq_len, x, y):
        loss = 0.
        logits_list, _ = forward_sequence(
            seq_len, x
        )
        y = forward_sequence(self.model, seq_len, x, use_output_label=False):
        loss = torch.sum(
            self.criterion(
                x[1:],
                y
                )
            )
        )
        return loss


class PredefinedFeatureModelGRU(nn.Module):
    def __init__(self, gru_depth, gru_hidden_size):
        # Weights for setting initial hidden input state based on first input.
        self.layer_initial_hidden_0 = nn.Linear(TOTAL_FEATURE_COUNT, gru_hidden_size)
        self.act_initial_hidden_0 = nn.LeakyReLU()
        self.layer_initial_hidden_1 = nn.Linear(gru_hidden_size, gru_hidden_size)
        self.act_initial_hidden_1 = nn.LeakyReLU()
        self.gru = nn.GRU(
            TOTAL_FEATURE_COUNT, gru_hidden_size, gru_depth 
        )
        # Output layers for transforming the hidden state into the actual output.
        self.layer_output_0 = nn.Linear(TOTAL_FEATURE_COUNT, gru_hidden_size)
        self.act_output_0 = nn.LeakyReLU()
        self.layer_output_1 = nn.Linear(gru_hidden_size, TOTAL_FEATURE_COUNT - FORCE_FEATURE_COUNT)

    def forward(self, x, gru_hidden=None, first_iteration=True):
        batch_size = x.size(0)
        if first_iteration:
            out_initial_hidden_0 = self.layer_initial_hidden_0(x)
            gru_hidden = self.layer_initial_hidden_1(out_initial_hidden_0)
        out_gru, gru_hidden = self.gru(x, gru_hidden)
        out_0 = self.layer_output_0(out_gru)
        out_1 = self.layer_output_1(out_0)
        return out_1, gru_hidden
