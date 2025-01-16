#%%
from LSV1M_training.load_best_model import model
import torch
import featurevis
from featurevis.utils import Compose
import numpy as np
import featurevis.ops as ops
import torch.nn as nn
import matplotlib.pyplot as plt

# %%

class SingleCellModel(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx
    
    def forward(self, x):
        return self.model(x)[:, self.idx].squeeze()


def create_mei(model, std=0.05, seed=42, img_res = [93,93], pixel_min = -1.7876, pixel_max = 2.1919, gaussianblur=1., device=None, step_size=10, num_steps=1000):
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



pixel_min = -1.8347# (the lowest value encountered in the data that served to train the model, serves as the black reference)
pixel_max = 2.1465 # (the highest [...] serves at white reference )


def plot_img(img, pixel_min, pixel_max):
    plt.figure(figsize = (8,8))
    if type(img) != np.ndarray:
        img = img.cpu().detach().squeeze().numpy()
    plt.imshow(img, cmap="gray", vmax=pixel_max, vmin=pixel_min)
    plt.colorbar()
    plt.show()

# check mei changing neuron
for i in range(5):
    model_i = SingleCellModel(model, i)
    mei, act = create_mei(model_i.eval(), std=0.1, img_res=[55,55], num_steps=1000)
    print(i)
    plot_img(mei, pixel_min, pixel_max)


#%%
# check mei changing pos
for i in torch.linspace(0, 1, 5):
    model.readout['all_sessions'].mu.data[0, 0, 0, 0, 2]= i
    model_i = SingleCellModel(model, 0)
    mei, act = create_mei(model_i.eval(), std=0.1, img_res=[55,55], num_steps=1000)
    print(i)
    plot_img(mei, pixel_min, pixel_max)

#%%
# check mei changing ori
for i in torch.linspace(0, 0.5, 5):
    model.readout['all_sessions'].mu.data[0, 0, 0,0,0]= i
    model_i = SingleCellModel(model, 0)
    mei, act = create_mei(model_i.eval(), std=0.15, img_res=[55,55], num_steps=1000)
    print(i)
    plot_img(mei, pixel_min, pixel_max)

# %%


