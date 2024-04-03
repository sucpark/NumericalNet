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
        value1, unit1, value2, unit2, operation = self.data[idx]
        value1 = torch.tensor([value1], dtype=torch.float32)
        value2 = torch.tensor([value2], dtype=torch.float32)
        unit1 = torch.tensor([self.unit_to_idx[unit1]], dtype=torch.long)
        unit2 = torch.tensor([self.unit_to_idx[unit2]], dtype=torch.long)
        operation = torch.tensor([self.operation_to_idx[operation]], dtype=torch.long)
        return value1, unit1, value2, unit2, operation
    
class NumericalNet(nn.Module):
    def __init__(self, unit_to_idx, operation_to_idx, device, dim=200, use_bias=False):
        super(NumericalNet, self).__init__()
        
        self.unit_to_idx = unit_to_idx
        self.operation_to_idx = operation_to_idx
        self.device = device
        self.unit_embeddings = nn.Embedding(len(self.unit_to_idx), dim)
        
        self.digit_fc1 = nn.Linear(in_features=1, out_features=dim, bias=use_bias)
        self.digit_fc2 = nn.Linear(in_features=dim, out_features=dim, bias=use_bias)
        self.digit_fc3 = nn.Linear(in_features=dim, out_features=dim, bias=use_bias)
        
        self.unit_fc1 = nn.Linear(in_features=dim, out_features=dim, bias=use_bias)
        self.unit_fc2 = nn.Linear(in_features=dim, out_features=dim, bias=use_bias)
        
        self.joint_fc1 = nn.Linear(in_features=dim, out_features=dim, bias=use_bias)
        self.joint_fc2 = nn.Linear(in_features=dim, out_features=dim, bias=use_bias)
        
    def forward(self, x):
        x = self.digit_fc1(x)
        x = F.leaky_relu(x)
        x = self.digit_fc2(x)
        x = F.leaky_relu(x)
        x = self.digit_fc3(x)
        return x
    
    def get_digit_embedding(self, digit):
        digit = torch.tensor([float(digit)], dtype=torch.float32).to(self.device)
        digit_embedding = self.forward(digit)
        return digit_embedding
    
    def get_unit_embedding(self, unit):
        x = self.unit_embeddings(unit).squeeze()
        x = self.unit_fc1(x)
        x = F.leaky_relu(x)
        x = self.unit_fc2(x)
        return x

    def joint_embedding(self, value_embedding, unit_embedding):       
        x = value_embedding + unit_embedding
        x = self.joint_fc1(x)
        x = F.leaky_relu(x)
        x = self.joint_fc2(x)
        return x
    
    def get_digit_unit_embedding(self, digit, unit):
        unit = torch.tensor([self.unit_to_idx[unit]]).to(self.device)
        digit_embedding = self.get_digit_embedding(digit).view(1,-1)
        unit_embedding = self.get_unit_embedding(unit).view(1,-1)
        
        return self.joint_embedding(digit_embedding, unit_embedding)

    def loss_fn(self, value1_embedding, value2_embedding, unit1, unit2, operation, alpha=1.5, beta=0.8):
        unit1_embedding = self.get_unit_embedding(unit1)
        unit2_embedding = self.get_unit_embedding(unit2)
        value1_unit1_embedding = self.joint_embedding(value1_embedding, unit1_embedding)
        value2_unit2_embedding = self.joint_embedding(value2_embedding, unit2_embedding)
        
        equality_mask = (operation == self.operation_to_idx['='])
        less_than_mask = (operation == self.operation_to_idx['<'])
        greater_than_mask = (operation == self.operation_to_idx['>'])

        # equality_loss = torch.mean(torch.masked_select(torch.abs(value1_unit1_embedding - value2_unit2_embedding), equality_mask.unsqueeze(-1)))
        # equality_loss = torch.mean(torch.masked_select((value1_unit1_embedding - value2_unit2_embedding)**2, equality_mask.unsqueeze(-1)))
    
        value1_unit1_norm = torch.norm(value1_unit1_embedding, p=2, dim=1)
        value2_unit2_norm = torch.norm(value2_unit2_embedding, p=2, dim=1)

        equality_loss = torch.mean(torch.masked_select((value1_unit1_norm - value2_unit2_norm)**2, equality_mask))

        less_than_target = less_than_mask.float().squeeze()
        greater_than_target = greater_than_mask.float().squeeze()

        less_than_pred = (value1_unit1_norm < value2_unit2_norm).float()
        greater_than_pred = (value1_unit1_norm > value2_unit2_norm).float()
        
        less_than_loss = F.binary_cross_entropy(less_than_pred, less_than_target)
        greater_than_loss = F.binary_cross_entropy(greater_than_pred, greater_than_target)

        inequality_loss = torch.mean(less_than_loss * less_than_mask.float() + greater_than_loss * greater_than_mask.float())
        output_loss = alpha*equality_loss + beta*inequality_loss

        return output_loss
    
    

    
    