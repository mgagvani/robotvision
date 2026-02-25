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

// issue of double dividing anchor clusters - should rerun
slurm-10284430 - Select closest anchor to cluster - 1.64
slurm-10284429 - Select closest anchor to cluster ( reduced noise) - 2.370

slurm-10284823 - bycicle trajectory unwraping - 1.53
slurm-10284887 - bycicle trajectory unwraping (reduced noise) - 1.60

slurm-10284984 - fixed scorer selection - 1.59

slurm-10284991 - fixed masking - 1.48
slurm-10285071 - fixed masking (reduced noise) 1.53

slurm-10285072 - Reduced score_loss size (from /5k to /10k) - 1.47

slurm-10285296 - Fixed loss balencing issue for training - 4.2 (failed, reverted)

slurm-10285830 - Further reduced score_loss weight from /10k to /20k    - 1.51 - Failed, removed
slurm-10285834 - training scorerer on >800 instead of > 900             - 1.55 - Failed, removed
slurm-10285884 - Token-type based positional embeddings                 - 1.56

slurm-10286114 - Token-type based positional embeddings w/ /10k and > 900 - 2.3
    slurm-10286116 - tried adding RoPE - 1.65

slurm-10287306 - rope with reduced score loss - 1.54

slurm-10289070 - repeat Further reduced score_loss weight from /10k to /20k - 1.5
slurm-10289542 - changed score_loss weight from /20k to /15k - 1.78
slurm-10289544 - changed score_loss weight from /15k to /7k - 1.8

Score_loss on 10k for these:
    slurm-10289556 - get random anchor instead of get closest. - 1.57
    2slurm-10289566 - single prediction for all waypoints, instead of individual ones - 1.69

    slurm-10312828 - scale of noise is 0.8 - 1.66
    slurm-10312846 - scale of noise is 0.6 - 1.95
    slurm-10312847 - scale of noise is 0.4 - 1.96
    slurm-10312848 - scale of noise is 0.1 - 1.65


    slurm-10316118 - 50/50 chance of choosing best anchor or random anchor, so it can't rely on the best anchor
        slurm10316118 - scale of noise is 1 - 1.49
        slurm-10319654 - scale of noise is 0.6 - 1.58
        slurm-10319655 - scale of noise is 0.4 - 1.87
        slurm-10319656 - scale of noise is 0.2 - 1.85
        slurm-10319657 - scale of noise is 0.1 - 1.82
        slurm-10319658 - scale of noise is 0.05 - 2.04
        slurm-10319659 - scale of noise is 0.01 - 1.7

        scale of 0.2:
            10320688 - detach score loss - 1.59
            10320690 - timestep < 200 - 1.60
            10320692 - detached score loss with timestep < 200 - 1.60

        scale = 1:
            10321195 - timestep < 200 - 1.59
            10321194 - detached score loss with timestep < 200 - 1.57

        scale 0.1:
            10323090 - detached + timestep < 200 + more val
            10323095 - reduced scorer loss

10324950 - detached scoring so less cross training, timestep < 200, more val, fixed scoring scale entirely - 1.62
10324951 - 2 removed kinematic unwrap - 1.75


10326388 - Stability
10326391 - no val anchor noise
10326400 - < 400
10326403 - some change idk (but in copy)
10326404 - oracle

10328012 - larger scorer + intent onehot on end
10328031 -  5x score_loss weight

10328041 - 2 - revamp scoring context to generate past + intent token

10328790 - 2 - k train scorer 

Remove scoring !!! :)

10330130 - reworked scorer
10330135 - slightly smaller scorer

make scorer a transformer?
better way to incorperate scoring loss?
try more cameras
better noise (a multiplicitive?)
figure out if noisy trajectories are too noisy
