slurm-10221594 - Base deep Monocular
slurm-10221610 - AR giving the model the hist of prev generated points
slurm-10231561 - AR giving the model just the last currently generated value and time in one-hot

slurm-10231584 - ultra basic diffusion
slurm-10231588 - improved diffusion
slurm-10232531 - improved diffusion w/ clean

slurm-10232688 - improved AR with context + current state
slurm-10232706 - improved AR with context + current state + buffed transformer blocks

slurm-10233164/58 - querys for each timestep


Use current + past to make future state


10237922 - VAE LSTM 6 latent
10237924 - VAE LSTM 8 latent
10238390 - VAE LSTM 6 latent 256 h-dims (2x)

slurm-10242008 - Full AR
slurm-10242775 - Full AR w/ improved sequence
slurm-10243410 - Full AR w/ Teacher for faster training

slurm-10248865 - Diffusion fixed
slurm-10248974/9061 - VAE Dense model (Didn't fully freeze VAE weights, but loss 1.59 1.6)

slurm-10249582 - Better VAE (hidden_dim=512, latent_dim=8, num_layers=4)
slurm-10249621 - Better Diffuse using DDPM from diffusers + x0 pred (loss: 1.72)

slurm-10250570 - VAE Dense w/ 'Better VAE'

slurm-10250597 - Diffusion with timm/fastvit_t8.apple_dist_in1k
slurm-10252020 - VAE Dense w/ original VAE (loss: 1.75)

slurm-10252021 - VAE (hidden_dim=128, latent_dim=12, num_layers=3)
slurm-10252031 - Epsilon Diffusion w/ 1 layer

slurm-10252564 - DiffusionLTF 
slurm-10255646 - DiffusionLTF: Bigger Trajectory + Fixed input image scaling 225

slurm-10255648 - DDIM added

Failed: slurm-10255650 - Better normalization (z-score instead of constant scaling)

slurm-10255654 - added gradient clipping to try and cure unstable training

slurm-10256101 - Removed better normalization

slurm-10256101 - Longer run w/ more val -> 1.54

slurm-10263348 - Multi-Trajectories -> 1.860

slurm-10271602 - Discrete Gaussian noise anchor

slurm-10272278 - k-means clusters, min max scaling for datapoints -> 2.840

slurm-10272409 - ADE loss -> 4.180
slurm-10272419 - larger scorer (failed, removed) -> 4.810

slurm-10273343 - try to add score loss (w/o increased noise) -> 1.950
slurm-10273514 - reduced scaling factor (w/o increased noise) -> 1.98
slurm-10273205 - increase noise for ade loss -> 5.380
slurm-10273351 - add ade loss w/ increased noise -> 1.800

slurm-10275302 - add ade loss w/ increased noise (all correct) - 1.97


slurm-10275406 - Select closest anchor to cluster
slurm-10275409 - Select closest anchor to cluster ( reduced noise)


bycicle trajectory unwraping
try 3 cameras
reduce learning rate?