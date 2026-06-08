import numpy as np

import torch


def to_cuda(data):
    for k in data:
        #print(k)
        if isinstance(data[k], np.ndarray):
            data[k] = torch.from_numpy(data[k])
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].cuda(non_blocking=True)
         
            data[k] = data[k].float()
        elif isinstance(data[k], dict):
            data[k] = to_cuda(data[k])
        else:
            continue
    return data

def to_cuda_batch(data):
    for k in data:
        #print(k)
        if isinstance(data[k], np.ndarray):
            data[k] = torch.from_numpy(data[k])
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].cuda(non_blocking=True)
         
            data[k] = data[k].float().half().unsqueeze(0)
        elif isinstance(data[k], dict):
            data[k] = to_cuda(data[k])
        else:
            raise TypeError('Unsupported data format {}'.format(type(data[k])))
    return data
color_set = ["shadow","direct_shading","direct_shadow_shading","shading","diffuse_direct_shading","specular_direct_shading",
             "albedo_mask","specular_mask","log1p_diffuse_direct_shading","log1p_specular_direct_shading","mask","albedo",
             "specular","instance_mask","light_albedo","log1p_diffuse_indirect_shading","log1p_specular_indirect_shading",
             "diffuse_indirect_shading","specular_indirect_shading","voxelDiffuse","voxelSpecular"]

def get_frame_data(data,frame_data,frame_idx,channel_cut,id_left,id_right):
    for k in data:
        if isinstance(data[k], dict):
            frame_data[k] = {}
            get_frame_data(data[k],frame_data[k],frame_idx,channel_cut,id_left,id_right)
        else:
            # print(k)
            # print(data[k].shape)
            # print(frame_idx)
            if k in color_set and channel_cut:
                if frame_idx == None:
                    frame_data[k] = data[k][:,...,id_left:id_right]
                else:
                    frame_data[k] = data[k][:,frame_idx,...,id_left:id_right]
            else:
                if frame_idx == None:
                    frame_data[k] = data[k][:,...]
                else:
                    frame_data[k] = data[k][:,frame_idx,...]
    #exit()
    return data


def preprocess_channel_cut(data):
    lightData = data["global"]
    localData = data["local"]
    light_shape = list(lightData["position"].shape)
    light_shape[0] = 3
    gbuffer_shape = list(localData["gbuffer"]["position"].shape)
    gbuffer_shape[0] = 3 
    lightData["radiance"] = lightData["radiance"].permute(-1, 1, 2, 3, 4, 0)
    lightData["position"] = lightData["position"].expand(light_shape)
    
    for key in localData["gbuffer"]:
        gbuffer_shape = list(localData["gbuffer"][key].shape)
        gbuffer_shape[0] = 3 
        localData["gbuffer"][key] = localData["gbuffer"][key].expand(gbuffer_shape)
    return data

def to_cpu(data):
    for k in data:
        if isinstance(data[k], np.ndarray):
            continue
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].cpu().numpy()
        elif isinstance(data[k], dict):
            data[k] = to_cpu(data[k])
        else:
            raise TypeError('Unsupported data format {}'.format(type(data[k])))
    return data


def compute_cubemap_axes(rot_mat=None):
    # rot_mat is the rotation matrix of camera 
    six_axis = [
        np.array([0., 0., 1.], np.float32),
        np.array([0., 0., -1.], np.float32),
        np.array([1., 0., 0.], np.float32),
        np.array([-1., 0., 0.], np.float32),
        np.array([0., 1., 0.], np.float32),
        np.array([0., -1., 0.], np.float32),
    ]
    ups = [np.array([0, 1, 0], np.float32)] * 4
    ups.append(np.array([0, 0, -1], np.float32))
    ups.append(np.array([0, 0, 1], np.float32))

    if rot_mat is None:
        return six_axis, ups
    else:
        axes = []
        for a in six_axis:
            axes.append(np.matmul(a, rot_mat[:3, :3]))
        ups[-2] = np.matmul(np.array([0, 0, -1]), rot_mat[:3, :3])
        ups[-1] = np.matmul(np.array([0, 0, 1]), rot_mat[:3, :3])
    return axes, ups

def get_lit_mask(data):
    e = data['emission']
    m = np.zeros_like(e[..., :1], np.float32)
    e = np.max(e, axis=-1, keepdims=True)
    m[e > 0.5] = 1
    return m    

def get_empty_mask(data):
    n = data['normal']
    bm = np.zeros_like(n[..., :1], np.float32)
    ns = np.max(np.abs(n), axis=-1, keepdims=True)
    bm[ns < 1e-2] = 1
    # bm = np.zeros_like(data['id'], np.float32)
    # bm[data['id'] < 0.2] = 1
    return bm

def get_valid_mask(data):
    lm = get_lit_mask(data)
    em = get_empty_mask(data)
    return (1 - lm) * (1 - em)

def get_mask(data, mode='S'):
    if mode == 'C':
        m = np.zeros_like(data['id'], np.float32)
        m[data['id'] > 0.999] = 1
    elif mode == 'A':
        m = np.zeros_like(data['id'], np.float32)
        m[data['id'] > 0.001] = 1
    else:
        m = data['id']
    return m

def transform_point_to_object_space(points, trans):
    assert(points.shape[-1] == 3)
    points = np.append(points, np.ones_like(points[..., :1]), axis=-1)
    points = np.matmul(points, trans[0])
    points = np.matmul(points, trans[1])
    points = points[..., :-1]
    return points

def transform_normal_to_object_space(normals, trans):
    assert(normals.shape[-1] == 3)
    normals = np.matmul(normals, trans[0][:3, :3])
    normals = np.matmul(normals, trans[1][:3, :3])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(min=1e-6)
    return normals

def generate_view_directions(cam_pos, pos_map):
    cam_pos = cam_pos.reshape(1, 1, -1)
    result = cam_pos - pos_map
    result /= np.linalg.norm(result, axis=-1, keepdims=True).clip(min=1e-6)
    return result

def log1p_np(x):
    return np.log(x.clip(min=0) + 1)

def expp1_np(x):
    return np.exp(x) - 1

def safe_divide_np(a, c, eps=1e-3):
    c = np.where(np.abs(c) < eps, 0, np.sign(c)*1.0/np.abs(c).clip(min=1e-5))
    return a * c

def stack_samples(samples: list): # list of dict of np.ndarray -> dict of np.ndarray
    ret_sample = {}
    assert isinstance(samples[0], dict)
    for k, v in samples[0].items():
        v_list = [samples[i][k] for i in range(len(samples))]
        if isinstance(v, np.ndarray):
            ret_sample[k] = np.stack(v_list, axis=0)
        elif isinstance(v, dict):
            ret_sample[k] = stack_samples(v_list)
        elif isinstance(v, list):
            v_list = [stack_samples(v_) for v_ in v_list] # list of list -> list of dict
            ret_sample[k] = stack_samples(v_list)
        else:
            raise ValueError('[stack_samples] Type {} is not supported. {}'.format(type(v), k))
    return ret_sample

def to_data_format(buffers, variable_params, gt, device=None, is_rgb=True):
    '''
    Convert data format to NeLT.
    # TODO: more items & pre-processing are needed for standard NeLT version (dataset.py).
    '''
    raise NotImplementedError

    # For initensity normalization of nelt
    # every scale is 1 in the case of AE
    I_scales = torch.stack([torch.ones([1, 1, 3 if is_rgb else 1], dtype=torch.float32)]*len(gt))
    I_scales *= 0.5 # AE use the exposure scale 0.5 to get pleasing visual results
    if device:
        buffers, variable_params, gt = buffers.to(device), variable_params.to(device), gt.to(device)
        I_scales = I_scales.to(device)
    data = {} # Convert data package format to NeLT style
    data['light_mask'] = None
    data['aggressive_mask'] = None
    data['empty_mask'] = None

    # data['emission'] = buffers[:, :, :, 0:3]
    # #TODO: set using variable names
    # data['gbuffer'] = {}
    # data['gbuffer']['normal'] = buffers[:, :, :, 3:6]
    # data['gbuffer']['position'] = buffers[:, :, :, 6:9]
    # data['gbuffer']['view_dir'] = buffers[:, :, :, 9:12]
    # data['gbuffer']['albedo'] = buffers[:, :, :, 12:15]
    # data['gbuffer']['alpha'] = buffers[:, :, :, 15:16]
    # data['gbuffer']['roughness'] = inputs[] # Only for NeLT
    # data['beauty'] = gt

    data['variable_params'] = variable_params

    # Tone-map here. Note that original AE code tonemap the results immediately in the variable renderer.
    data['log1p_emission'] = log1p_torch(data['emission'])
    data['log1p_beauty'] = log1p_torch(gt)

    # TODO: duplicate configs occurs in 'data_generator_configs' & overall configs
    data['I_scale'] = I_scales

    return data