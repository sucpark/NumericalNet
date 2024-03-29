import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn


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
    
class NumericNet(nn.Module):
    def __init__(self, use_bias=True):
        super(NumericNet, self).__init__()
        
        self.target_units = target_units
        self.num_units = len(target_units)
        # self.unit_to_idx = {unit: idx for idx, unit in enumerate(target_units)}
        
        self.fc1 = nn.Linear(in_features=1, out_features=256, bias=use_bias)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(in_features=256, out_features=512, bias=use_bias)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(in_features=512, out_features=1024, bias=use_bias)
        self.ln3 = nn.LayerNorm(1024)
        
        self.unit_embeddings = nn.Embedding(self.num_units, 1024)
        self.joint_fc1 = nn.Linear(in_features=2*1024, out_features=1024, bias=use_bias)
        self.ln4 = nn.LayerNorm(1024)
        
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.ln3(self.fc3(x))
        return x
    
    def get_unit_embedding(self, unit):
        unit_embedding = self.unit_embeddings(unit)
        unit_embedding = unit_embedding.squeeze()
        return unit_embedding

    def joint_embedding(self, value_embedding, unit_embedding):       
        joint_emb = torch.cat((value_embedding, unit_embedding), -1)
        joint_emb = self.ln4(self.joint_fc1(joint_emb))
        return joint_emb

    def loss_fn(self, value1_embedding, value2_embedding, unit1, unit2, alpha=1.2, beta=1.5):

        unit1_embedding = self.get_unit_embedding(unit1)
        unit2_embedding = self.get_unit_embedding(unit2)

        value1_unit1_embedding = self.joint_embedding(value1_embedding, unit1_embedding)
        value2_unit2_embedding = self.joint_embedding(value2_embedding, unit2_embedding)
    
        mse_loss = F.mse_loss(value1_unit1_embedding, value2_unit2_embedding)
        unit_loss = F.mse_loss(unit1_embedding, unit2_embedding)
        output_loss = alpha * mse_loss + beta * unit_loss

        return output_loss