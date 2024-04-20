import torch
import torch.nn as nn
import torch.nn.functional as F

unit_conversions = {
    'kg': 1000,
    'g': 1,
    'mg': 0.001,
    'km': 1000,
    'm': 1,
    'cm': 0.01,
    'mm': 0.001,
    "l": 1,
    "ml": 0.001
}

class NumericOperationDataset(torch.utils.data.Dataset):
    def __init__(self, data, unit_to_idx, operation_to_idx):
        self.data = data
        self.unit_to_idx = unit_to_idx
        self.operation_to_idx = operation_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        value1, unit1, operation, value2, unit2 = self.data[idx]
        value1 = torch.tensor([value1], dtype=torch.float32)
        value2 = torch.tensor([value2], dtype=torch.float32)
        unit1 = torch.tensor([self.unit_to_idx[unit1]], dtype=torch.long)
        unit2 = torch.tensor([self.unit_to_idx[unit2]], dtype=torch.long)
        operation = torch.tensor([self.operation_to_idx[operation]], dtype=torch.long)

        return value1, unit2, value2, unit2, operation
    
class NumericalNet(nn.Module):
    def __init__(self, unit_to_idx, operation_to_idx, device, dim=200, num_layers=2, use_bias=True):
        super(NumericalNet, self).__init__()
        
        self.unit_to_idx = unit_to_idx
        self.idx_to_unit = {v: k for k, v in unit_to_idx.items()}
        self.operation_to_idx = operation_to_idx
        self.device = device
        self.unit_embeddings = nn.Embedding(len(self.unit_to_idx), dim)
        self.layer_norm = nn.LayerNorm(dim)

        # Build digit layers
        self.digit_layers = nn.ModuleList()
        self.digit_layers.append(nn.Linear(in_features=1, out_features=dim, bias=use_bias))
        for idx in range(num_layers):
            self.digit_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))
            if idx % 2 == 0:
                self.digit_layers.append(nn.LayerNorm(dim))
        self.digit_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))

        # Build unit layers
        self.unit_layers = nn.ModuleList()
        self.unit_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))
        for idx in range(num_layers):
            self.unit_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))
            if idx % 2 == 0:
                self.unit_layers.append(nn.LayerNorm(dim))
        self.unit_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))

        # Build joint layers
        self.joint_layers = nn.ModuleList()
        self.joint_layers.append(nn.Linear(in_features=dim*2, out_features=dim, bias=use_bias))
        for idx in range(num_layers):
            self.joint_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))
            if idx % 2 == 0:
                self.joint_layers.append(nn.LayerNorm(dim))
        self.joint_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))

    def forward(self, digit, unit):
        digit_embedding = self.get_digit_embedding(digit)
        unit_embedding = self.get_unit_embedding(unit)
        digit_unit = torch.concatenate([digit_embedding, unit_embedding], dim=-1)
        for layer in self.joint_layers:
            digit_unit = layer(digit_unit)
        return digit_unit

    def get_digit_embedding(self, digit):
        digit = torch.log(digit+1e-9)
        if not torch.is_tensor(digit):
            digit = torch.tensor([float(digit)], dtype=torch.float32).to(self.device)
        digit_embedding = digit
        for layer in self.digit_layers:
            digit_embedding = layer(digit_embedding)
        return digit_embedding

    def get_unit_embedding(self, unit):
        if not torch.is_tensor(unit):
            if isinstance(unit, str):
                unit = torch.tensor([self.unit_to_idx[unit]]).to(self.device)
            else:
                unit = torch.tensor([unit]).to(self.device)
        unit_embedding = self.unit_embeddings(unit).squeeze()
        for layer in self.unit_layers:
            unit_embedding = layer(unit_embedding)
        return unit_embedding 
    
class NumericalLoss(nn.Module):
    def __init__(self, alpha=1.2, gamma=0.8):
        super(NumericalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, embedding1, embedding2):
        # Compute the L2 norms of the original embeddings
        norm1 = torch.norm(embedding1, p=2, dim=1)
        norm2 = torch.norm(embedding2, p=2, dim=1)

        # Compute MSE Loss on the embeddings
        mse_loss = F.mse_loss(embedding1, embedding2)

        # Compute sophisticated reciprocal loss
        epsilon = 1e-8  # Small constant to avoid division by zero
        norm_ratio = torch.log(norm1 + epsilon) - torch.log(norm2 + epsilon)
        reciprocal_loss = torch.mean(torch.log(1 + torch.exp(torch.abs(norm_ratio))))

        # Calculate total loss
        total_loss = self.alpha * mse_loss + self.gamma * reciprocal_loss
        
        return total_loss