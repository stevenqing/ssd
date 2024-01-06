import torch
import numpy as np
saving_path = '/scratch/prj/inf_du/shuqing/reward_model.pth'

loaded_model = torch.load(saving_path)
loaded_model.eval()

input_tensor = torch.randn(1,18)
print(np.shape(input_tensor))

a = loaded_model(input_tensor)
print(a)
