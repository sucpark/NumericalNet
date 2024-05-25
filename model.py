import torch
import torch.nn as nn
import torch.nn.functional as F
from deepctr_torch.layers.interaction import CrossNet

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
    def __init__(self, unit_to_idx, operation_to_idx, device, dim=200, num_layers=8, use_bias=True):
        super(NumericalNet, self).__init__()
        
        self.unit_to_idx = unit_to_idx
        self.idx_to_unit = {v: k for k, v in unit_to_idx.items()}
        self.operation_to_idx = operation_to_idx
        self.device = device
        self.unit_embeddings = nn.Embedding(len(self.unit_to_idx), dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.crossnet = CrossNet(in_features=2*dim, parameterization="matrix")

        # Initialize unit embeddings
        nn.init.xavier_uniform_(self.unit_embeddings.weight)
        
        # Build digit layers
        self.digit_layers = nn.ModuleList()
        self.digit_layers.append(nn.Linear(in_features=1, out_features=dim, bias=use_bias))
        nn.init.xavier_uniform_(self.digit_layers[0].weight)
        for idx in range(num_layers):
            self.digit_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))
            nn.init.xavier_uniform_(self.digit_layers[-1].weight)
            if idx % 2 == 0:
                self.digit_layers.append(nn.LayerNorm(dim))
        self.digit_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))

        # Build unit layers
        self.unit_layers = nn.ModuleList()
        self.unit_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))
        nn.init.xavier_uniform_(self.unit_layers[0].weight)
        for idx in range(num_layers):
            self.unit_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))
            nn.init.xavier_uniform_(self.unit_layers[0].weight)
            if idx % 2 == 0:
                self.unit_layers.append(nn.LayerNorm(dim))
        self.unit_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))

        # Build joint layers
        self.joint_layers = nn.ModuleList()
        self.joint_layers.append(nn.Linear(in_features=dim*2, out_features=dim, bias=use_bias))
        nn.init.xavier_uniform_(self.joint_layers[0].weight)
        for idx in range(num_layers):
            self.joint_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))
            nn.init.xavier_uniform_(self.joint_layers[-1].weight)
            if idx % 2 == 0:
                self.joint_layers.append(nn.LayerNorm(dim))
        self.joint_layers.append(nn.Linear(in_features=dim, out_features=dim, bias=use_bias))

    def forward(self, digit, unit):
        digit_embedding = self.get_digit_embedding(digit)
        unit_embedding = self.get_unit_embedding(unit)
        joint_embedding = self.get_joint_embedding(digit, unit)
        
        return digit_embedding, unit_embedding, joint_embedding

    def get_digit_embedding(self, digit):
        digit_embedding = digit
        for layer in self.digit_layers:
            digit_embedding = layer(digit_embedding)
        return digit_embedding

    def get_unit_embedding(self, unit):
        unit_embedding = self.unit_embeddings(unit).squeeze(1) # squeeze remove single-dimension
        for layer in self.unit_layers:
            unit_embedding = layer(unit_embedding)
        return unit_embedding
    
    def get_joint_embedding(self, digit, unit): # digit, unit: torch.Size([512, 1])
        digit_embedding = self.get_digit_embedding(digit)
        unit_embedding = self.get_unit_embedding(unit)
        digit_unit = torch.concatenate([digit_embedding, unit_embedding], dim=-1)
        digit_unit = self.crossnet(digit_unit)
        for layer in self.joint_layers:
            digit_unit = layer(digit_unit)
        return digit_unit

class NumericalLoss(nn.Module):
    def __init__(self, operation_to_idx, alpha=1.2, beta=0.7):
        super(NumericalLoss, self).__init__()
        self.operation_to_idx = operation_to_idx
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, joint1_embedding, joint2_embedding, operation):
        
        equal_to_mask = (operation == self.operation_to_idx['='])
        less_than_mask = (operation == self.operation_to_idx['<'])
        greater_than_mask = (operation == self.operation_to_idx['>'])
        
        equal_to_loss = F.mse_loss(torch.masked_select(joint1_embedding, equal_to_mask), torch.masked_select(joint2_embedding, equal_to_mask))

        joint1_norm = torch.norm(joint1_embedding, p=2, dim=1)
        joint2_norm = torch.norm(joint2_embedding, p=2, dim=1)
        
        less_than_loss = torch.mean(F.relu(torch.masked_select(joint1_norm - joint2_norm, less_than_mask)))
        greater_than_loss = torch.mean(F.relu(torch.masked_select(joint2_norm - joint1_norm, greater_than_mask)))
        
        if not less_than_mask.any() and not greater_than_mask.any():
            total_loss = self.alpha*equal_to_loss
        elif not less_than_mask.any():
            total_loss = self.alpha * equal_to_loss + self.beta * greater_than_loss
        elif not greater_than_mask.any():
            total_loss = self.alpha * equal_to_loss + self.beta * less_than_loss
        else:
            total_loss = self.alpha * equal_to_loss + self.beta * (less_than_loss + greater_than_loss)
        
        return total_loss