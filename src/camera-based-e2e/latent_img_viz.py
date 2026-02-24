import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import os

from loader import WaymoE2E
from models.transfuser.team_code_transfuser import latentTF, latentTF_p2


def load_data(num_samples: int, data_root: str):
    dataset = WaymoE2E(
        batch_size=1,
        indexFile="index_train.pkl",
        data_dir=data_root,
        images=True,
        n_items=num_samples,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=4)
    return loader

def visualize_waymo_batch(loader, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, batch in enumerate(loader):
        if (i == 0):
            continue
        
        # [B, C, H, W]
        fl = batch["IMAGES"][1]   # Front Left
        frnt = batch["IMAGES"][2] # Front
        fr = batch["IMAGES"][3]   # Front Right

        print(f"Front Left shape:  {fl.shape}")
        print(f"Front shape:       {frnt.shape}")
        print(f"Front Right shape: {fr.shape}")

        fig1, axes1 = plt.subplots(4, 1, figsize=(15, 5))
        axes1[0].imshow(fl.squeeze(0).permute(1, 2, 0).cpu().numpy())
        axes1[0].set_title("Front Left")
        axes1[1].imshow(frnt.squeeze(0).permute(1, 2, 0).cpu().numpy())
        axes1[1].set_title("Front")
        axes1[2].imshow(fr.squeeze(0).permute(1, 2, 0).cpu().numpy())
        axes1[2].set_title("Front Right")
        axes1[3].imshow(np.concatenate([fl.squeeze(0).permute(1, 2, 0).cpu().numpy(), 
                                   frnt.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                                   fr.squeeze(0).permute(1, 2, 0).cpu().numpy()], dim=1))
        axes1[3].set_title("directly concatenated cameras")
        for ax in axes1: ax.axis('off')
        fig1.savefig("viz_plots/waymo_images.png")

        #res_transform() expects (front, front_left, front_right)
        ltf1_out = latentTF.res_transform(frnt, fl, fr)
        ltf2_out = latentTF_p2.res_transform(frnt, fl, fr)

        fig2, axes2 = plt.subplots(2, 1, figsize=(15, 8))
        axes2[0].imshow(ltf1_out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
        axes2[0].set_title("latentTF res_transform()")
        axes2[1].imshow(ltf2_out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
        axes2[1].set_title("latentTF_p2 res_transform()")
        for ax in axes2: ax.axis('off')
        fig2.savefig("viz_plots/latent_images.png")

        fig3, axes3 = plt.subplots(1, 1, figsize=(15, 8))
        axes3[0].imshow(torch.cat([fl, frnt, fr], dim=3))
        axes3[0].set_title("directly concatenated cameras")


LOADED_DATA = load_data(2, "/scratch/gilbreth/shar1159/waymo_open_dataset_end_to_end_camera_v_1_0_0")
visualize_waymo_batch(LOADED_DATA, "/scratch/gilbreth/shar1159/robotvision/src/camera-based-e2e/viz_plots")
