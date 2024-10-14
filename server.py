# server.py
import torch
import os

class Server:
    def __init__(self, num_classes, ipc, channel, im_size):
        self.noise = torch.randn((num_classes * ipc, channel, im_size[0], im_size[1]))
    
    def aggregate_noises(self, noises):
        # Simple mean aggregation
        self.noise = torch.mean(torch.stack(noises), dim=0)
    
    def save_final_noise(self, save_path='result'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, 'aggregated_noise.pt')
        torch.save(self.noise, save_name)
