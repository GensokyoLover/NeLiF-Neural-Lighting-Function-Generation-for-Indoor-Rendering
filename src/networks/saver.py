import numpy as np
import os.path as osp
import imageio
import os
from utils import image_utils, data_utils
import pyexr
import math
import pickle
import gzip

class Saver():
    def __init__(self, out_path, configs, test_mode=False):
        self.count = 0
        self.configs = configs
        self.out_path = out_path

        self.save_immediate = configs['save_during_training']['enable']
        self.immediate_num = configs['save_during_training']['num']

        self.proc_function = {
            'ILOG': lambda x, _: data_utils.expp1_np(x),
            'H2L': lambda x, _: image_utils.HDR2LDR(x),
            'N2L': lambda x, _: image_utils.normal2LDR(x),
            'S2L': lambda x, _: image_utils.float2uint8(x),
            'F2L': lambda x, _: image_utils.feature2uint8(x),
            'rescale': lambda x, info: x * info['I_scale'],
            'diffuse_weight': lambda x, info: x * info['diffuse_weight'],
            'mask': lambda x, m : x * m,
            'abs': lambda x, _ : np.abs(x),
            "l1": lambda x,_: x
        }

        self.out_features = list(self.configs['save_features'].keys())
        for i in self.configs['save_features']:
            if "indices" in self.configs['save_features'][i]:
                self.out_features.remove(i)
                for j in range(self.configs['save_features'][i]['indices']):
                    self.out_features.append(i+'_'+str(j))

        self.max_per_page = 100 if test_mode else self.configs['max_num_per_page']

        self.pages = []
        self.camera_dict = {}
        self.light_dict = {}
        self.reset()

    def reset(self):
        self.count = 0
        self.pages = []
        self.camera_dict = {}
        self.light_dict = {}
        self.reset_html()

    def reset_html(self):
        self.html_str = ''

        table_header_str = '<table align="center" rules=all frame=void> <tr> <th> ID </th>'
        for k in self.out_features:
            table_header_str += '<th> {} </th>'.format(k.replace('log_', ''))
        table_header_str += '</tr>'

        self.html_str += table_header_str
        self.html_empty = True

    def finish_html(self):
        self.html_str += '</table>'
        self.pages.append(self.html_str)
        self.reset_html()
    
    def output_pages(self, output_path=None):
        if not self.html_empty:
            self.finish_html()

        out_path = self.out_path if output_path is None else output_path

        for i, page_str in enumerate(self.pages):
            with open(osp.join(out_path, ('index.html' if len(self.pages) == 1 else 'main{}.html'.format(i))), 'w') as fout:
                print(page_str, file=fout)

    def save(self, preds, data, loss_map, during_training, output_path=None,now_epoch=0,name=None):
        if during_training and not self.save_immediate:
            return

        self.count += 1
        # rgb = self.configs['rgb']
        rgb = True

        # print('debug', loss_map['final_loss'])
        if len(loss_map['final_loss'].shape) == 0:
            loss_map['final_loss'] = loss_map['final_loss'].unsqueeze(0)
        n = loss_map['final_loss'].shape[0]
        for key in loss_map:
            if len(loss_map[key].shape) == 4:
                loss_map[key] = loss_map[key].permute(0,2,3,1)
        if not rgb:
            assert(n % 3 == 0)
            n //= 3
        
        is_intensity_norm = 'I_scale' in data
        if is_intensity_norm:
            I_scale = data['I_scale'].cpu().numpy()

        out_path = self.out_path if output_path is None else output_path

        for i in range(1):
            if name != None:
                out_folder = osp.join(out_path, 'data', f"{now_epoch:05d}" + "_" +name[0])
            else:
                out_folder = osp.join(out_path, 'data', f"{now_epoch:05d}" + "_" +str(self.count))
            os.makedirs(out_folder, exist_ok=True)

            self.html_str += '<tr>'

            self.html_str += '<td><p> {} </p></td>'.format(self.count)
            #print(self.out_features)
            for f_ in self.out_features:
                if f_[:5] == 'pred_':
                    d = preds
                    f = f_[5:]
                elif f_[:7] == 'global_':
                    d = data["global"]
                    f= f_[7:]
                elif f_[:5] == "loss_":
                    d = loss_map
                    f = f_[5:]
                elif f_[:6] == "light_":
                    d = data["local"]["lights"]["shadow"]
                    f = f_
                else:
                    d = data['local']
                    f = f_
                #print(f_,d.keys())
                try:
                    imgs = d[f].float().detach().cpu().numpy()
                except Exception as e:
                    print('[Warning] Feature {} not found!'.format(f), e)
                    continue
                print(f,imgs.shape)
                post_info = {}
                masks = set(['conserved_mask', 'aggressive_mask', 'light_mask', 'soft_mask', 'back_mask'])
                for mn in masks:
                    if mn in data:
                        if data[mn] is not None:
                            m = data[mn].cpu().numpy()
                            m = m[i] if rgb else m[i*3:i*3+3]
                            if not rgb:
                                m = m.squeeze(-1).transpose(1, 2, 0)
                            post_info[mn] = m


                if f_ not in self.configs['save_features']: # multiple feature buffers
                    f__ = '_'.join(f_.split('_')[:-1])
                    cfg = self.configs['save_features'][f__]
                else:
                    cfg = self.configs['save_features'][f_]
                if f == "radiance" or f == 'direction' or f == "clip_radiance":
                    imgs = imgs.transpose(-1,1,2,3,4,0)
                    B1,W1,H1,W2,H2,C = imgs.shape
                    B = B1 * W1 * H1
                    imgs= imgs.reshape(B,W2,W2,C)
                    B,W,H,C = imgs.shape
                    sqrtB = int(math.sqrt(B))
                    img = np.zeros((W*int(math.sqrt(B)),H*int(math.sqrt(B)),C))
                    for idx in range(B):
                        idx_x = idx // sqrtB *W2
                        idx_y = (idx - sqrtB*(idx//sqrtB)) *W2
                        img[idx_x:idx_x + W2,idx_y:idx_y+W2,...] = imgs[idx]
                else:
                    img = imgs[i] if rgb else imgs[i*3:i*3+3]
                
                if is_intensity_norm:
                    post_info['I_scale'] = I_scale[i] if rgb else I_scale[i*3:i*3+3]
                    # print(post_info['I_scale'])
                
                for subfix in cfg['outputs']:
                    out_img = img.copy()
                    if 'rescale' not in cfg['outputs'][subfix] and is_intensity_norm: # auto rescale
                        if f=='beauty' or f=='shading' or f=='direct_shading' or f=='indirect_shading' or f=='direct':
                            cfg['outputs'][subfix].insert(0, 'rescale')
                    for p in cfg['outputs'][subfix]:
                        #print(p)
                        if p in self.proc_function:
                            #print(self.proc_function.keys())
                            out_img = self.proc_function[p](out_img, post_info)
                        elif p in masks:
                            out_img = self.proc_function['mask'](out_img, post_info[p])
                        else:
                            raise KeyError('unknown output processing function')
                    out_name = osp.join(out_folder, '{}_{}.{}'.format(f_, str(self.count), subfix))
                    if subfix == "pkl":
                        with gzip.open(out_name + '.gz', 'wb') as f:
                            pickle.dump(out_img, f, pickle.HIGHEST_PROTOCOL)
                    elif subfix == 'exr':
                        pyexr.write(out_name, out_img)
                    else:
                        print("out_img shape",out_img.shape,)
                        #exit()
                        if out_img.shape[-1] == 1:
                            out_img = np.repeat(out_img, repeats=3, axis=-1)
                        imageio.imwrite(out_name, out_img)
                        

                        if subfix == 'png':
                            img_path = osp.join('data', str(self.count), osp.basename(out_name))
                            img_cfg = 'height={}'.format('"100%"' if self.configs['resize'] is None else self.configs['resize'])
                            if 'exr' in cfg['outputs']:
                                self.html_str += '<td> <div style="position: relative;"> <a href="{}" target="-blank"> <img src="{}" {}/> </a>'.format(img_path.replace('png', 'exr'), img_path, img_cfg)
                            else:
                                self.html_str += '<td> <div style="position: relative;"> <img src="{}" {}/>'.format(img_path, img_cfg)
                            t = 0
                            if 'visualize_loss' in cfg:
                                for lname in cfg['visualize_loss']:
                                    if lname is None:
                                        continue
                                    t += 1
                                    l = loss_map[lname][i].mean() if rgb else loss_map[lname][i*3:i*3+3].mean() # TODO: true metric instead of loss (log space / I_scale post weight)
                                    metric_name = lname[lname.rfind('_') + 1:]
                                    self.html_str += '<span style="position: absolute; top: {}px; left: 20px; color:red;"> {}:{:.3f} </span>'.format(20*t, metric_name, l)
                            self.html_str += '</div></td>'
            

            self.html_empty = False
            if self.count % self.max_per_page == 0:
                self.finish_html()
                
