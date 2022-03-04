import torch
import torch.nn as nn



class DynamicalFusionLayer(nn.Module):

    def __init__(self, dim_model, dim=-1):
        super(DynamicalFusionLayer, self).__init__()

        self.dim_model = dim_model
        self.dim = dim

        self.linear_layer = nn.Linear(dim_model * 2, dim_model * 2)ya
        self.glu_layer = nn.GLU(dim=dim)

    def forward(self, x0, x1, x2):
        assert x0.size() == x1.size()
	assert x0.size() == x2.size()
        fusion_input = torch.cat([x0, x1], self.dim)
        fusion_input = torch.cat([fusion_input, x2], self.dim)
        output = self.linear_layer(fusion_input)
        output = self.glu_layer(output)

        return output
