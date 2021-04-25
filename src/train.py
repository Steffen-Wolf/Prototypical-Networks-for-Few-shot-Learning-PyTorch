# coding=utf-8
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from skimage.io import imsave

from prototypical_batch_sampler import PrototypicalBatchSampler
from omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
from parser_util import get_parser
from itertools import islice

import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import os
import zarr
from dataset import FastDataset
from lisl.models.prototypical_network import PrototypicalNetwork, PrototypicalERFNetwork

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataloader(opt, mode):

    dataset = FastDataset(opt.dataset_root,
                          num_queries=opt.num_query_tr,
                          num_support=opt.num_support_tr,
                          num_class_per_iteration=opt.classes_per_it_tr,
                          lim_images=opt.lim_images,
                          lim_instances_per_image=opt.lim_instances_per_image,
                          lim_clicks_per_instance=opt.lim_clicks_per_instance)
    num_labeled_images, num_labeled_instances, num_labeled_pixels = dataset.labeled_images_instances_pixels
    print(
        f"Training with {num_labeled_pixels} clicks, {num_labeled_instances} instances, {num_labeled_images}/{opt.lim_images} images")

    # see https://github.com/pytorch/pytorch/issues/5059
    def wif(id):
        uint64_seed = torch.initial_seed()
        np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             sampler=None,
                                             num_workers=10,
                                             worker_init_fn=wif)
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    # TODO: make available as options
    in_channels = 512 + 32 + 1
    inst_out_channels = 4
    n_sem_classes = 2
    # model = PrototypicalNetwork(in_channels, inst_out_channels, n_sem_classes).to(device)
    model = PrototypicalERFNetwork(in_channels, inst_out_channels, n_sem_classes).to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def vis_embedding(output_file, coords, embedding):
    with torch.no_grad():
        coords = coords.detach().cpu().numpy()
        embedding = embedding.detach().cpu().numpy() - coords
        plt.quiver(coords[:, 0],
                    coords[:, 1],
                    embedding[:, 0],
                    embedding[:, 1], 
                    angles='xy',
                    scale_units='xy',
                    scale=1., color='#8fffdd')

    # plt.axis('off')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()
    

def train(opt, tr_dataloader, model, loss_fn, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss_log = [0]
    fgbg_loss_log = [0]
    inst_loss_log = [0]
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    accum_grad = 2

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    optim.zero_grad()

    for epoch in range(opt.epochs):
        c = 0
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()

        for batch_number, batch in enumerate(tqdm(tr_iter)):
            raw, inp, instance_coordinates, y, background_coordinates = batch
            inp, y = inp.to(device), y.to(device)
            model_output, sem_embedding = model(inp)

            inst_prediction = model_output[0, :, instance_coordinates[:, 1], instance_coordinates[:, 0]]
            fg_prediction = sem_embedding[0, :,
                                          instance_coordinates[:, 1], instance_coordinates[:, 0]]
            bg_prediction = sem_embedding[0, :,
                                          background_coordinates[:, 1], background_coordinates[:, 0]]

            inst_loss, acc = loss_fn(torch.transpose(inst_prediction, 0, 1),
                                target=y,
                                n_support=opt.num_support_tr)

            num_fg = fg_prediction.shape[1]
            num_bg = bg_prediction.shape[1]
            sem_target = torch.cat((y.new_ones(num_fg), y.new_zeros(num_bg)))
            sem_emb = torch.transpose(torch.cat((fg_prediction, bg_prediction), dim=1), 0, 1)
            sem_loss = F.cross_entropy(sem_emb,
                                       sem_target)

            loss = inst_loss + sem_loss.to(inst_loss.device)
            loss = loss / accum_grad
            loss.backward()

            inst_loss_log[-1] += inst_loss.item()
            fgbg_loss_log[-1] += sem_loss.item()
            train_loss_log[-1] += loss.item()

            if (batch_number + 1) % accum_grad == 0:
                optim.step()
                optim.zero_grad()

                inst_loss_log.append(0)
                fgbg_loss_log.append(0)
                train_loss_log.append(0)
        
            if batch_number % 100 == 0:
                eout = model_output.detach().cpu().numpy()[0]
                for c in range(len(eout)):
                    imsave(f"emb_{batch_number:08}_{c}.png", eout[c])

                clicks = np.zeros(eout[0].shape)
                clicks[instance_coordinates[:, 1], instance_coordinates[:, 0]] = 1
                clicks[background_coordinates[:, 1], background_coordinates[:, 0]] = -1
                print(clicks.shape)
                imsave(f"raw_{batch_number:08}_clicks.png", clicks)

                imsave(f"raw_{batch_number:08}.png",
                           raw.detach().cpu().numpy()[0])

                sout = sem_embedding.detach().softmax(dim=1).cpu().numpy()[0, 1]
                imsave(f"sem_{batch_number:08}.png", sout)

            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss_log[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {} Sem Loss {}'.format(avg_loss, avg_acc, float(sem_loss)))
        lr_scheduler.step()

        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        # TODO: remove islice with propper val loader
        for batch in islice(val_iter, 32):
            raw, inp, instance_coordinates, y, background_coordinates = batch
            inp, y = inp.to(device), y.to(device)
            model_output, _ = model(inp)

            inst_prediction = model_output[0, :, instance_coordinates[:, 1], instance_coordinates[:, 0]]
            inst_loss, acc = loss_fn(torch.transpose(inst_prediction, 0, 1),
                                target=y,
                                n_support=opt.num_support_val)

            val_loss.append(loss.item())
            val_acc.append(acc.item())

        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss_log', 'train_acc', 'val_loss', 'val_acc', 'inst_loss_log', 'fgbg_loss_log']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss_log, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model, loss_fn):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y, bg = batch
            x, y, bg = x.to(device), y.to(device), bg.to(device)
            model_output, _ = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if options.distance_fn == "euk":
        from prototypical_loss import prototypical_loss as loss_fn
    elif options.distance_fn == "rbf":
        print("using RBF distance")
        from prototypical_loss import prototypical_loss_rbf as loss_fn
    else:
        raise NotImplementedError("Unknown distance function")

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')
    # trainval_dataloader = init_dataloader(options, 'trainval')
    test_dataloader = init_dataloader(options, 'test')

    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                loss_fn=loss_fn,
                optim=optim,
                lr_scheduler=lr_scheduler)






if __name__ == '__main__':
    main()
