import numpy as np
import torch
import torch.nn as nn


class RelationalNet(nn.Module):
    def __init__(self, n_units_row, n_units_col, n_units_out):
        super(RelationalNet, self).__init__()
        self.layers_in_row = nn.ModuleList()  # Input layers for row
        for i in range(n_units_row.shape[0] - 1):
            self.layers_in_row.append(nn.Linear(n_units_row[i], n_units_row[i + 1]))
            self.layers_in_row.append(nn.Sigmoid())
        for i in np.arange(0, 2 * (n_units_row.shape[0] - 1), 2):
            interval_u = 1 / np.sqrt(n_units_row[i // 2])
            nn.init.uniform_(self.layers_in_row[i].weight, a=-interval_u, b=interval_u)
            nn.init.zeros_(self.layers_in_row[i].bias)

        self.layers_in_col = nn.ModuleList()  # Input layers for column
        for i in range(n_units_col.shape[0] - 1):
            self.layers_in_col.append(nn.Linear(n_units_col[i], n_units_col[i + 1]))
            self.layers_in_col.append(nn.Sigmoid())
        for i in np.arange(0, 2 * (n_units_col.shape[0] - 1), 2):
            interval_u = 1 / np.sqrt(n_units_col[i // 2])
            nn.init.uniform_(self.layers_in_col[i].weight, a=-interval_u, b=interval_u)
            nn.init.zeros_(self.layers_in_col[i].bias)

        self.layers_out = nn.ModuleList()  # Output layers
        for i in range(n_units_out.shape[0] - 1):
            self.layers_out.append(nn.Linear(n_units_out[i], n_units_out[i + 1]))
            self.layers_out.append(nn.Sigmoid())
        for i in np.arange(0, 2 * (n_units_out.shape[0] - 1), 2):
            interval_u = 1 / np.sqrt(n_units_out[i // 2])
            nn.init.uniform_(self.layers_out[i].weight, a=-interval_u, b=interval_u)
            nn.init.zeros_(self.layers_out[i].bias)

    def forward(self, x_row, x_col):
        for layers_in_row in self.layers_in_row:
            x_row = layers_in_row(x_row)
        for layers_in_col in self.layers_in_col:
            x_col = layers_in_col(x_col)

        y = torch.cat((x_row, x_col), 1)  # Concatenate the row and column outputs
        for layers_out in self.layers_out:
            y = layers_out(y)

        return y, x_row, x_col


class Loss:
    def __init__(self, net):
        self.net = net

    def calc_loss(self, x_row, x_col, y, lambda_reg):
        y_model, _, _ = self.net(x_row, x_col)
        criterion = torch.nn.MSELoss()
        loss = criterion(y_model, y)

        device = x_row.device
        reg = torch.tensor(0.).to(device, dtype=torch.float)
        for param in self.net.parameters():
            reg += torch.norm(param) ** 2  # 2021.5.10
        reg = torch.sqrt(reg)  # 2021.5.10
        loss += lambda_reg * reg

        return loss
