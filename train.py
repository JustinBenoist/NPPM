import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.color'] = "black"
import pyexr
import torch
import mitsuba as mi 

from integrators.DPMTrainer import DPMTrainer
from integrators.Model import DPMGaussianAniso, PhotonEncoder

print(mi.variants())
mi.set_variant('cuda_ad_rgb')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="scenes/dataset/", help='dataset file')
    parser.add_argument('--outfile', type=str, default="output/", help='output file')
    parser.add_argument('--encoder', type=str, default=None, help='encoder model if pretrained file')
    parser.add_argument('--ppi', type=int, default=100000, help='photons per iteration')
    parser.add_argument('--iter', type=int, default=100, help='number of iterations')
    parser.add_argument('--radius', type=float, default=0.05, help='inital radius size')
    parser.add_argument('--neighbors', type=int, default=200, help='number of max neighbors in radius search')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--batchsize', type=int, default=100, help='batch size')
    parser.add_argument('--dcv_size', type=int, default=32, help='Deep Context Vector size (power of 2)')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--large', action="store_true", default=False, help="indicates if large model is used")
    parser.add_argument('--classic', action="store_true", default=False, help="indicates if original DPM model is used")
    parser.add_argument('--isotropic', action="store_true", default=False, help="indicates if isotropic gaussian is used")
    parser.add_argument('--random_density', action="store_true", default=False, help="indicates if random density of photon is used")
    parser.add_argument('--all_caustics', action="store_true", default=False, help="indicates if we use indirect caustic illumination photons")
    parser.add_argument('--no_ratio_matches', action="store_true", default=False, help="indicates if we use ratio of matches as network input")
    parser.add_argument('--knn_mode', action="store_true", default=False, help="indicates if we use KNN search instead of radius search to gather photons")
    parser.add_argument('--opt', action="store_true", default=False, help="indicates if we use optimized model")
    # choice between l2, smape, l1, flip
    parser.add_argument('--loss', type=str, default='l2', help='loss function to use (l2, l1, smape, smape2, flip)')
    parser.add_argument('--no_freeze_encoder', action="store_true", default=False, help="indicates if we freeze the weights of the pretrained encoder during training, only use if encoder is not pretrained")
    
    opt = parser.parse_args()
    print(opt)
    
    PHOTONS_PER_ITER = opt.ppi
    N_ITER = opt.iter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Model running on {device}")
    MAX_PHOTONS = opt.neighbors
    OUT = opt.outfile
    EPOCHS = opt.epochs
    BATCHSIZE = opt.batchsize
    NEIGHBORS = opt.neighbors
    PHOTONS_ITER = opt.ppi
    RADIUS = opt.radius
    ALPHA = 2 / 3
    USE_MASK = True
    SIZE = "large" if opt.large else "small"
    GAUSSIAN = not opt.classic
    MC_SAMPLE = 20000
    APAM = True
    try:
        os.makedirs(OUT)
    except OSError:
        pass
    
    # Initialize scene
    scenes = []
    nb_scenes = 0
    for path in os.listdir(opt.dataset):
        if os.path.isfile(os.path.join(opt.dataset, path)) and path[-4:] == ".xml":
            nb_scenes += 1
    for i in range(nb_scenes):
        scenes.append(mi.load_file(os.path.join(opt.dataset, f"scene_{i}.xml")))
    print(f"Dataset composed of {len(scenes)} scenes")
    params = mi.traverse(scenes[0])
    camera: mi.ProjectiveCamera = scenes[0].sensors()[0]
    resolution = camera.film().size()
    N = resolution.x * resolution.y
    gt = torch.from_numpy(pyexr.read(os.path.join(opt.dataset, "gt_0.exr"))).reshape(N, 3).unsqueeze(0).to(device)
    masks = torch.from_numpy(pyexr.read(os.path.join(opt.dataset, "mask_scene_0.exr"))).reshape(N, 1).unsqueeze(0).to(device)
    for i in range(1, len(scenes) - 1):
        gt = torch.cat((gt, torch.from_numpy(pyexr.read(os.path.join(opt.dataset, f"gt_{i}.exr"))).reshape(N, 3).unsqueeze(0).to(device)), dim=0)
        masks = torch.cat((masks, torch.from_numpy(pyexr.read(os.path.join(opt.dataset, f"mask_scene_{i}.exr"))).reshape(N, 1).unsqueeze(0).to(device)), dim=0)
    gt_test = torch.from_numpy(pyexr.read(os.path.join(opt.dataset, f"gt_{len(scenes) - 1}.exr"))).reshape(N, 3).unsqueeze(0).to(device)
    masks_test = torch.from_numpy(pyexr.read(os.path.join(opt.dataset, f"mask_scene_{len(scenes) - 1}.exr"))).reshape(N, 1).unsqueeze(0).to(device)
    
    encoder = PhotonEncoder(NEIGHBORS, opt.dcv_size, 2, use_mask=USE_MASK, n_hidden_pred=3)
    if opt.encoder is not None:
        encoder.load_state_dict(torch.load(opt.encoder, map_location=device))
    encoder = encoder.to(device)

    network = DPMGaussianAniso(encoder, NEIGHBORS, opt.dcv_size, size=SIZE, ratio_matches=(not opt.no_ratio_matches)).to(device)
    if opt.opt:
        torch.set_float32_matmul_precision('high')
        network = torch.compile(network)
        
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    
    if opt.loss == "l2":
        loss_function = torch.nn.MSELoss()
    elif opt.loss == "l1":
        loss_function = torch.nn.L1Loss()
    elif opt.loss == "smape":
        loss_function = lambda x, y: torch.mean(torch.abs(x - y) / (torch.abs(x) + torch.abs(y) + 1e-8))
    elif opt.loss == "smape2":
        loss_function = lambda x, y: torch.mean(torch.square(x - y) / (torch.abs(x) + torch.abs(y) + 1e-8))
    elif opt.loss == "flip":
        raise "Not implemented yet"
        

    low = torch.ones(N, device=device) * RADIUS
    high = torch.ones(N, device=device) * RADIUS + 0.0001
    distribution = torch.distributions.Uniform(low=low, high=high)
    
    trainer = DPMTrainer(network, optimizer, scheduler, distribution, RADIUS, BATCHSIZE, device, loss_function, PHOTONS_ITER, 
                         random_density=opt.random_density, pure_caustic=(not opt.all_caustics),
                         ratio_matches=(not opt.no_ratio_matches), knn_mode=opt.knn_mode, gaussian=GAUSSIAN, 
                         freeze_encoder=(not opt.no_freeze_encoder))
    avg_train_loss, avg_test_loss, avg_box_loss = trainer.run(scenes[:-1], scenes[-1], gt, gt_test, masks, masks_test, EPOCHS, OUT)
    
    torch.save(network.state_dict(), os.path.join(OUT, "model.pth"))
    
    with plt.style.context("ggplot"):
        plt.title("Training DPM")
        plt.plot(avg_train_loss, label="train")
        plt.plot(avg_test_loss, label="test")
        plt.plot(avg_box_loss, label="box")
        plt.legend()
        plt.savefig(os.path.join(OUT, f"loss.png"))
        plt.show()
    