import numpy as np
import os
import torch
from utils.inferred_targets import initialise_prior

DEFAULT_G = np.array([1.0/32, 1.0/16, 1.0/8])

def extract_volunteers_helper(id_map, count, direct):
    for files in os.listdir(direct):
        with open(os.path.join(direct, files), 'r') as text:
            for line in text:
                for vol in line.split():
                    if vol not in id_map.keys():
                        id_map[vol] = count
                        count += 1
    return(id_map, count)


# this creates a dictionary for all volunteers
def extract_volunteers(dataDict):
    curdir = os.getcwd()
    id_map = {}
    count = 0
    direct = os.path.join(curdir, dataDict['train'], '..', '..', 'volunteers', 'train')
    direct2 = os.path.join(curdir, dataDict['val'], '..', '..', 'volunteers', 'val')

    id_map, count = extract_volunteers_helper(id_map, count, direct)
    id_map, count = extract_volunteers_helper(id_map, count, direct2)

    VOL_ID_MAP = id_map
    return (VOL_ID_MAP)

# Reads the "volunteers" file (generally located at train/volunteers/<im_name>.txt)
# as a filename:volunteertensor dictionary.
def get_file_volunteers_dict(data_dict, mode=['train'], vol_id_map=[]):
    file_vols_dict = {}
    for pth in mode:
        vol_path = os.path.join(data_dict[pth], '..', '..','volunteers')
        vol_mode_path = os.path.join(vol_path, pth)
        file_names = [x for x in os.listdir(vol_mode_path) if not x.startswith('.')]
        for fn in file_names:
            vol_file_path = os.path.join(vol_mode_path, fn)
            with open(vol_file_path) as f:
                vol_seq = [vol_id_map[x.strip()] for x in f.readlines()]
            file_vols_dict[fn] = torch.tensor(vol_seq).int()
    return file_vols_dict

def init_bcc_params(K, classes, diagPrior, cnvrgThresh):
    bcc_params = {'n_classes': classes,
                  'n_crowd_members': K,
                  'cm_diagonal_prior': diagPrior,
                  'convergence_threshold': cnvrgThresh}
    return bcc_params

def compute_param_confusion_matrices(bcc_params, torchMode=False):
    # set up variational parameters
    prior_param_confusion_matrices = initialise_prior(n_classes=bcc_params['n_classes'],
                                                      n_volunteers=bcc_params['n_crowd_members'],
                                                      alpha_diag_prior=bcc_params['cm_diagonal_prior'],
                                                      torchMode = torchMode)
    variational_param_confusion_matrices = prior_param_confusion_matrices.detach().clone() if torchMode else np.copy(prior_param_confusion_matrices)
    return {'prior': prior_param_confusion_matrices, 'variational': variational_param_confusion_matrices}

def convert_target_volunteers_yolo2bcc(target_volunteers, Na=3, G=DEFAULT_G, batch_size=None, vol_id_map=[], background_id=2):
    n_images = batch_size
    n_vols = len(vol_id_map)
    Ng = G.shape[0]

    vigcwh_list = [] # [v]olunteer, [i]mage, [g]rid choice, grid [c]ell id, [w]idth, [h]eight
    vigcwh_list1 = []
    vigcwh_list2 = []
    # grid cell id with deci, but why
    targets_per_i_bcc_list = []
    for i in range(n_images):
        target_vols_per_i = target_volunteers[target_volunteers[:, 0] == i][:, 1:]
        targets_per_ig_bcc_list = []

        for g in range(Ng):  # per grid choice
            g_frac = G[g]
            S_g = np.ceil(1/g_frac).astype(int)**2 # 6400, 1600, 400
            # Don't need a loop for anchor-boxes as we are simply repeating Na times below.
            targets_per_iv_bcc_list = []
            for v in range(n_vols):
                # volunteer did classify this image
                if v in target_vols_per_i[:, -1]:
                    targets_per_iv = target_vols_per_i[target_vols_per_i[:, -1] == v][:, :-1]
                    c, x, y, w, h = targets_per_iv.T  # w and h are ignored

                    x_cell_ids = torch.where(x < 1, x / g_frac, torch.ones(x.shape) * (np.ceil(1 / g_frac))).int()
                    y_cell_ids = torch.where(y < 1, y / g_frac, torch.ones(y.shape) * (np.ceil(1 / g_frac))).int()
                    gc_ids = ((y_cell_ids) * (np.ceil(1 / g_frac)) + x_cell_ids).long()
                    gc_ids1 = torch.add(gc_ids, S_g)
                    gc_ids2 = torch.add(gc_ids1, S_g)
                    gc_IDS = torch.cat((gc_ids, gc_ids1, gc_ids2))

                    vigcwh_list.append(torch.cat([torch.tensor([v, i, g]) * torch.ones(w.shape[0], 3), gc_ids.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)], axis=1))
                    vigcwh_list1.append(torch.cat([torch.tensor([v, i, g]) * torch.ones(w.shape[0], 3), gc_ids1.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)], axis=1))
                    vigcwh_list2.append(torch.cat([torch.tensor([v, i, g]) * torch.ones(w.shape[0], 3), gc_ids2.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)], axis=1))
                    whole_list = vigcwh_list + vigcwh_list1 + vigcwh_list2

                    targets_per_iv_bcc = background_id * torch.ones(S_g*Na)
                    targets_per_iv_bcc[gc_IDS] = c.repeat(Na)
                    targets_per_iv_bcc_list.append(targets_per_iv_bcc)
                # volunteer did not classify this image
                else:
                    targets_per_iv_bcc = -1 * torch.ones(S_g*Na)
                    targets_per_iv_bcc_list.append(targets_per_iv_bcc)
            targets_per_ig_bcc = torch.stack(tuple(targets_per_iv_bcc_list)).T
            targets_per_ig_bcc_list.append(targets_per_ig_bcc)
        targets_per_i_bcc = torch.cat(targets_per_ig_bcc_list)
        targets_per_i_bcc_list.append(targets_per_i_bcc)
    target_volunteers_bcc = torch.stack(tuple(targets_per_i_bcc_list))
    vigcwh = torch.cat(whole_list)
    return target_volunteers_bcc, vigcwh

# this function returns image id, x,y,w,h
def qt2yolo_soft(qt, G, Na, vigcwh, torchMode=False, device=None):
    Ng = G.shape[0]
    num_images = qt.shape[0]
    y_bcc = []
    for i in range(num_images):
        st = 0
        for g in range(Ng):
            g_frac = G[g]
            S_g = np.ceil(1 / g_frac).astype(int)
            n_cells = S_g * S_g

            ig_indices = torch.logical_and(vigcwh[:, 1] == i, vigcwh[:, 2] == g)
            vigcwh_ig = vigcwh[ig_indices, :]
            wh_ig = vigcwh_ig[:, -2:]
            wh_ig_mean = wh_ig.mean(axis=0)
            # wh_init_multiplier = wh_ig_mean if BKGD_WH_IS_MEAN and wh_ig.shape[0] > 0 else -1 #why -1 all the time
            # wh = wh_init_multiplier * torch.ones(n_cells, 2)
            wh = wh_ig_mean * torch.ones(n_cells*3, 2) #may need more complicated computation
            tagged_gc_ids = vigcwh_ig[:, 3].unique().int()
            for gc in tagged_gc_ids:
                igc_indices = torch.logical_and(ig_indices, vigcwh[:, 3] == gc)
                vigcwh_igc = vigcwh[igc_indices]
                wh_igc = vigcwh_igc[:, -2:]
                wh_igc_mean = wh_igc.mean(axis=0)
                wh[gc, :] = wh_ig_mean if wh_igc.shape[0] == 0 else wh_ig_mean
            # for a in range(Na):
            z = torch.linspace(g_frac / 2, 1 - g_frac / 2, S_g).repeat(S_g, 1).unsqueeze(-1)
            xy = torch.cat((z.permute(1, 0, 2), z), 2).permute(1, 0, 2).reshape(n_cells, 2)
            xy_a = torch.cat((xy, xy, xy), 0)
            icxywh = torch.cat(
                    ((i * torch.ones(n_cells*3, 1)).to(device), xy_a.to(device), wh.to(device)),1)
            y_bcc.append(icxywh)
            st += n_cells
    qt_yolo_soft = torch.cat(y_bcc)
    return qt_yolo_soft