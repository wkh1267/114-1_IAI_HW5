import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

class HW5Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.setup_model(**kwargs)

    def setup_model(
            self,
            input_size: int = 8,
            hidden_size: int = 16,
            num_layers: int = 2,
            lr: float = 1e-3,
            device: torch.device = torch.device('cpu')
            ) -> None:
        """Setup model architecture here. To improve the performance of
        the model, You can try different hyperparameters here, or change
        the network architecture. Note that the forward function might
        not be compatible with the new architecture.

        Parameters
        ----------
        input_size: int
            input size of the LSTM network.

        hidden_size: int
            hidden size of the LSTM network.

        num_layers: int
            number of layers of the LSTM network.

        lr: float
            learning rate of the optimizer.

        device: torch.device
            context-manager that changes the selected device.
        """
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.BCEWithLogitsLoss()
        self.device = device
        self.to(device)
        return

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        batch: torch.Tensor
            input tensor of shape (batch_size, n_time_steps, input_size).

        Returns
        -------
        torch.Tensor:
            output tensor of shape (batch_size, 1).
        """
        embed, _ = self.lstm(batch)
        embed = torch.mean(embed, dim=1)
        output = self.output_layer(embed)
        output = torch.squeeze(output)
        return output

    def train_epochs(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            patience: int = 5,
            max_n_epochs: int = 15) -> None:
        """Train the model with max number of epochs. You should finish the TODOs in this function.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            dataloader for training data.

        val_loader: torch.utils.data.DataLoader
            dataloader for validation data.

        patience: int
            number of epochs to wait before early stop, default to 5.

        max_n_epochs: int
            maximum number of epochs to train, default to 10.
        """
        # initialize the best loss  (used for early stopping)
        best_loss = np.inf
        early_stop_counter = patience


        for epoch in range(max_n_epochs):
            self.train()
            train_loss_logs = []
            for mels, labels in tqdm(train_loader):
                mels = mels.to(self.device)
                labels = labels.to(self.device)
                # TODO (5P): change the following, make forward pass, and save the result to y_pred.
                # CHECK: self.forward
                # CHECK: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim

                # TODO (5P): change the following, compute loss and perform back propagation.
                # CHECK: self.loss
                # CHECK: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
                # CHECK: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim

                # TODO (5P): update the weights of the network through gradient descent.
                # CHECK: self.optimizer
                # CHECK: https://pytorch.org/docs/stable/optim.html
                # CHECK: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim

                # log training loss
                train_loss_logs.append(train_loss.item())

            # take the average of training loss
            average_train_loss = np.mean(np.array(train_loss_logs))

            self.eval()
            val_loss_logs = []
            with torch.no_grad():
              for mels, labels in tqdm(val_loader):
                    mels = mels.to(self.device)
                    labels = labels.to(self.device)
                    # TODO (5P): change the following, make forward pass, and save the result to y_pred.
                    # CHECK: self.forward
                    # CHECK: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim

                    # TODO (5P): change the following, compute loss.
                    # CHECK: self.loss
                    # CHECK: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

                    # log validation loss
                    val_loss_logs.append(val_loss.item())

            # take the average of validation loss
            average_val_loss = np.mean(np.array(val_loss_logs))
            print(f'{epoch+1:03}\ttrain_loss: {average_train_loss:.2f}\tval_loss: {average_val_loss:.2f}')

            if average_val_loss < best_loss:
               # TODO (5P): implement early stopping here, if the validation loss is smaller than the
               # current best. Save the model, update the best loss and reset the early stop counter.
               # CHECK: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
               # Please name the model’s state_dict as best_model.ckpt)
               pass
            else:
               pass
        return

    def predict_prob(self, dataloader: torch.utils.data.DataLoader) -> np.array:
        """Predict the probability of the audio class based on trained.
        the output should be of shape (n_samples,). Each element is the
        predicted probability for the sample being AI generated recoding.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
            dataloader to be predicted.

        Returns
        -------
        np.array:
            the output should be of shape (n_samples,). Each element is the
            predicted probability for the sample being AI generated recoding.
            e.g. np.array([0.03111489, 0.6505939 , 0.16935825, ...], dypte=np.float32)
        """
        self.eval()
        probs = []
        with torch.no_grad():
          for mels, labels in dataloader:
              # TODO (10P): change this. The model only predict the
              # logits of the probability of AI generated recording. You
              # need to compute the probability of both real recording and
              # AI generated recording.
              # CHECK: https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html

        probs_np = np.zeros(len(dataloader))
        return probs_np

    def predict(self, dataloader: torch.utils.data.DataLoader) -> np.array:
        """Predict the output based on trained model, the output should
        be of shape (n_samples,). Output 0 for the sample if the model
        predict the sample to be real recording, 1 otherwise.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
            dataloader to be predicted.

        Returns
        -------
        np.array:
            output 0 for the sample if the model predict the sample to
            be real recording, 1 otherwise.
            e.g. np.array([0, 1, 0], dypte=np.float32)
        """
        # TODO (10P): change this, it would be easier to implement
        # `predict_prob` first, and start from the output of it.

        return np.zeros(len(dataloader))

    def evaluate(self, y_true: np.array, y_prob: np.array) -> float:
        """Evaluate the output based on the ground truth, the output
        should be the area under precision recall curve.

        Parameters
        ----------
        y_true: np.array
            the ground truth of the data.

        y_pred: np.array
            the output of model.predict_prob.

        Returns
        -------
        float:
            the area under precision recall curve.
        """
        # TODO (10P): return value should be the AUROC given y_true and y_prob, and you need to plot the ROC curve and store as a .png file
        # CHECK: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        # CHECK: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

        # Compute ROC curve and AUROC


        # Plot ROC curve

        return 0.0