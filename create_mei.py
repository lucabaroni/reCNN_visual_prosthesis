import numpy as np
import torch 
import featurevis
import featurevis.ops as ops
from featurevis.utils import Compose
import torch.nn as nn


class SingleCellModel(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx
    
    def forward(self, x):
        return self.model(x)[:, self.idx].squeeze()


def create_mei(model, std, seed, img_res, pixel_min, pixel_max, gaussianblur, device, step_size, num_steps):
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed = seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    initial_image = torch.randn(1, 1, *img_res, dtype=torch.float32).to(device)*std
    model.eval()
    model.to(device)
    initial_image = initial_image.to(device)
    mean = (pixel_min + pixel_max)/2
    
    # TODO decide if we need to add it 
    post_update =Compose([ops.ChangeStats(std=std, mean=mean), ops.ClipRange(pixel_min, pixel_max)]) 
    opt_x, fevals, reg_values = featurevis.gradient_ascent(
        model,
        initial_image, 
        step_size=step_size,
        num_iterations=num_steps, 
        post_update=post_update,
        gradient_f = ops.GaussianBlur(gaussianblur),
        print_iters=1001,
    )
    mei = opt_x.detach().cpu().numpy().squeeze()
    mei_act = fevals[-1]
    return mei,  mei_act

def create_mei_of_n_neurons(model, model_idxs, std, seed, img_res, pixel_min, pixel_max, gaussianblur, device, step_size, num_steps): 
    meis = []
    mei_acts = []
    for idx in model_idxs:
        single_cell_model = SingleCellModel(model, idx)
        single_cell_model.eval()
        mei, mei_act = create_mei(single_cell_model, std, seed, img_res, pixel_min, pixel_max, gaussianblur, device, step_size, num_steps)
        meis.append(mei)
        mei_acts.append(mei_act)
    return meis, mei_acts
