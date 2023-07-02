"""
Counter modules.
"""
from torch import nn

def get_regressor(cfg):
    counter_name = cfg.MODEL.counter
    if counter_name == 'density_x16':
        return DensityX16(counter_dim=cfg.MODEL.counter_dim)
    else:
        raise NotImplementedError
        

class DensityX16(nn.Module):
    def __init__(self, counter_dim):
        super().__init__()
        self.regressor =  nn.Sequential(
                                    nn.Conv2d(counter_dim, 196, 7, 2, padding=3),
                                    nn.ReLU(),
                                    nn.Conv2d(196, 128, 5, 2, padding=2),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 64, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 32, 2, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 1, 1),
                                    nn.ReLU()
                                )
        self.linear_layer = nn.Sequential(nn.Linear(25, 1), nn.ReLU())
        self._weight_init_()
        
    def forward(self, features):
        features = self.regressor(features)
        features = features.view(features.shape[0], -1)
        features = self.linear_layer(features)
        return features 

    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
           
