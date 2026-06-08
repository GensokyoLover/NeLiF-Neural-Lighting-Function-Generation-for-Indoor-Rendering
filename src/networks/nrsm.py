import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import modules
from networks import utils
from networks.positional_pixel_generator import ShadingGenerator, MLPGenerator
from networks.unet import SUNet
from networks.loss_functions import Loss

class Repr_Encoders(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        assert len(configs) > 0

        if 'direct_vpls_encoder' in configs:
            self.light_encoder = self.build_direct_vpls_encoder(
                configs['direct_vpls_encoder'],
                utils.count_VPL_feature_dims,
                True
            )

        if 'indirect_vpls_encoder' in configs:
            self.inidrect_vpl_encoder = self.build_indirect_vpls_encoder(
                configs['indirect_vpls_encoder'],
                utils.count_VPL_feature_dims,
                True # rgb
            )

        if 'ssvpls_encoder' in configs:
            self.ssvpl_encoder = self.build_indirect_vpls_encoder(
                configs['ssvpls_encoder'],
                utils.count_VPL_feature_dims,
                True # rgb
            )

    def build_direct_vpls_encoder(self, configs, count_func, rgb):
        configs['dims'].insert(0, count_func(configs['input'], rgb, configs['pe']))
        configs['dims'].append(configs['repr_dim'])
         
        return modules.MLP(
            configs['dims'],
            'lrelu',
            'none'
        )

    def build_indirect_vpls_encoder(self, configs, count_func, rgb):
        configs['dims'].insert(0, count_func(configs['input'], rgb)) # Add input dimension
        configs['dims'].append(configs['repr_dim']) # Add output dimension
       
        return modules.SampleNet(
            configs['dims'],
            'lrelu',
            'none'
        )
        # return modules.MLP(
        #     configs['dims'],
        #     'lrelu',
        #     'none'
        # )
    
    def forward(self, data):
        light_repr = []

        if 'direct_vpls_encoder' in self.configs:
            light_features = utils.get_light_features(data['direct_vpls'], self.configs['direct_vpls_encoder']['input'], pe=self.configs['direct_vpls_encoder']['pe'])
            # print('[Direct]', light_features.shape)
            direct_light_repr = self.light_encoder(light_features)
            # TODO: self-attention here?
            direct_light_repr = torch.mean(direct_light_repr, dim=2)
            light_repr.append(direct_light_repr)

        if 'indirect_vpls_encoder' in self.configs:
            # Reduce to 64*64 per face. RSM is too large. reduce size for efficiency.
            vpl_features = utils.get_light_features(data['indirect_vpls'], self.configs['indirect_vpls_encoder']['input'])
            # print('[INDirect]', vpl_features.shape)

            # for idx, feature_name in enumerate(self.configs['light_encoder']['input']):
            #     for i in range(6):
            #         pyexr.write('before_{}_face{}.exr'.format(feature_name, i), vpl_features[0][i][..., 3*idx:3*(idx+1)].cpu().numpy())
            # Interpolate here for onnx
            # print('Interpolation:', vpl_features.shape, data['indirect_vpls']['position'].shape)
            bs, cs = vpl_features.shape[0], vpl_features.shape[-1]
            # vpl_features = vpl_features.view((-1, *vpl_features.shape[3:]))
            # vpl_features = F.interpolate(vpl_features.permute(0, 3, 1, 2), size=[64, 64], mode='nearest').permute(0, 2, 3, 1)
            vpl_features = vpl_features.view(bs, -1, cs) # drop spatial dims of indirect vpls
            # vpl_features = vpl_features.view(bs, vpl_features.shape[1], -1, cs) # drop spatial dims of indirect vpls

            # print('before:', vpl_features.shape)
            vpl_repr = self.inidrect_vpl_encoder(vpl_features)
            # print('after:', vpl_repr.shape)
            # TODO: self-attention here?
            # vpl_repr = torch.mean(vpl_repr.view(vpl_repr.shape[0], -1, vpl_repr.shape[-1]), dim=1)
            light_repr.append(vpl_repr)

        if 'ssvpls_encoder' in self.configs:
            vpl_features = utils.get_light_features(data['ssvpls'], self.configs['ssvpls_encoder']['input'])
            bs, cs = vpl_features.shape[0], vpl_features.shape[-1]
            vpl_features = F.interpolate(vpl_features.permute(0, 3, 1, 2), size=[128, 128], mode='nearest').permute(0, 2, 3, 1)
            # print('ssvpls!!', vpl_features.shape)
            vpl_repr = self.ssvpl_encoder(vpl_features)
            light_repr.append(vpl_repr)

        light_repr = torch.cat(light_repr, dim=-1)

        return light_repr
    
    # Test indirect VPLs culling. Deprecated.
    def forward_indirect(self, gbuffer, data):
        vpl_features = utils.get_light_features(data['indirect_vpls'], self.configs['indirect_vpls_encoder']['input'])

        # Encode all vpls
        bs, cs = vpl_features.shape[0], vpl_features.shape[-1]
        vpl_features = vpl_features.view(bs, -1, cs) # drop spatial dims of indirect vpls
        vpl_reprs = self.inidrect_vpl_encoder(vpl_features)

        # ** Generate indirect vpl mask for each pixel **
        # print(vpl_features.shape, vpl_features[..., :3].unsqueeze(1).unsqueeze(1).shape, gbuffer['position'].shape)
        dist_vec = vpl_features[..., :3].unsqueeze(1).unsqueeze(1) - gbuffer['position'].unsqueeze(-2)
        dist2 = torch.sum(torch.square(dist_vec)) # Use it?
        vpl_dir = dist_vec / torch.sqrt(dist2)
        is_valid_vpl = torch.sum(gbuffer['normal'].unsqueeze(-2) * vpl_dir, axis=-1) > 0
        is_valid_vpl &= (torch.sum(vpl_features[..., 3:6].unsqueeze(1).unsqueeze(1) * -vpl_dir, axis=-1) > 0)
        print(is_valid_vpl.shape)

        # Sample vpl_repr for each pixel
        vpl_repr = vpl_reprs[is_valid_vpl] # TODO: onxx compatiable
        return vpl_repr

    def split(self, data): # split after attention for feeding to different decoders
        out = {}
        i = 0
        for encoder_name, encoder_configs in self.configs.items():
            dim = encoder_configs['repr_dim']
            out['light_repr_'+encoder_name.split('_encoder')[0]] = data[..., i:i+dim]
            i += dim
        return out

    @property
    def repr_dim(self):
        dim = 0
        if 'direct_vpls_encoder' in self.configs:
            dim += self.configs['direct_vpls_encoder']['repr_dim']
        if 'indirect_vpls_encoder' in self.configs:
            dim += self.configs['indirect_vpls_encoder']['repr_dim']
        if 'ssvpls_encoder' in self.configs:
            dim += self.configs['ssvpls_encoder']['repr_dim']
        return dim

    @property
    def data_format(self):
        return {
            'direct_vpls': {
                'position': 3,
                'normal': 3,
                'intensity': 3
            },
            'indirect_vpls': {
                'position': 3,
                'normal': 3,
                'flux': 3
            }
        }

class Shading_Encoders(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.csm_shadow_input = False

        for encoder_name, encoder_configs in configs.items():
            if encoder_configs['type'] == 'mlp':
                encoder_configs['dims'].insert(0, utils.count_GB_feature_dims(encoder_configs['input'], True))
                encoder_configs['dims'].append(encoder_configs['repr_dim'])
                
                setattr(self, encoder_name, modules.MLP(
                    encoder_configs['dims'],
                    'lrelu',
                    'none'
                ))

            elif encoder_configs['type'] == 'unet':
                # setattr(self, encoder_name, SUNet(3, encoder_configs['repr_dim']))
                # setattr(self, encoder_name, SUNet(2, encoder_configs['repr_dim']))
                # setattr(self, encoder_name, SUNet(1, encoder_configs['repr_dim']))
                # setattr(self, encoder_name, SUNet(6, encoder_configs['repr_dim']))
                # setattr(self, encoder_name, SUNet(263, encoder_configs['repr_dim'])) # naive concatenation
                if 'input' in encoder_configs and encoder_configs['input'] == 'csm':
                    setattr(self, encoder_name, SUNet(8, encoder_configs['repr_dim']))
                    self.csm_shadow_input = True
                else:
                    setattr(self, encoder_name, SUNet(7, encoder_configs['repr_dim']))

            setattr(self, encoder_name+'_linear', nn.Linear(encoder_configs['repr_dim'], encoder_configs['repr_dim']))
    
    def forward(self, data, gbuffers, enable_timing_profile=False):

        # common used buffers
        c_c = torch.sum(gbuffers['normal'] * gbuffers['view_dir'], axis=-1)[..., None] # diffuse # TODO: Use this feature?

        timing_profile = {}
        if enable_timing_profile:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        shading_clue_repr = []
        for encoder_name in self.configs:
            feature_name = encoder_name.split('_')[0]
            encoder = getattr(self, encoder_name)

            input = None
            if feature_name == 'lightdir':
                # input = data['light_dir']
                input = data['specular']['light_dir']
            elif feature_name == 'halfvec':
                # h = data['light_dir'] + gbuffers['view_dir']
                # input = h / torch.sqrt(torch.sum(torch.square(h), dim=-1)).unsqueeze(-1) # normalization
                input = data['specular']['half_vec']
            elif feature_name == 'specular':
                input = torch.cat([data['specular']['light_dir'], data['specular']['half_vec']], dim=-1)
            elif feature_name == 'shadow':
                # ** Neural shadow mapping **
                z_f = data['shadow']['pixel_emitter_distance']
                z = data['shadow']['occluder_emitter_distance']
                # c_e = torch.clamp(torch.sum(gbuffers['normal'] * data['light_dir'], axis=-1)[..., None], min=0) # diffuse
                c_e = torch.sum(gbuffers['normal'][:, None, ...] * data['specular']['light_dir'], axis=-1)[..., None] # diffuse # TODO: Use this feature?
                # r_e = data['area'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(c_e.shape) # TODO: add this

                ## final selection
                if self.csm_shadow_input:
                    # envmap
                    # ablation1: old style
                    # input = torch.cat([z-z_f, z/z_f, c_e, c_c[:, None, ...].expand(z.shape), gbuffers['position'][:, None, ...].expand((*z.shape[:-1], 3))], dim=-1).permute(0, 1, 4, 2, 3)
                    # ablation2: shadowmap + distance map (global shadow space ndc's z)
                    # input = torch.cat([sm, z, z_f, c_e, c_c[:, None, ...].expand(z.shape), gbuffers['position'][:, None, ...].expand((*z.shape[:-1], 3))], dim=-1).permute(0, 1, 4, 2, 3) # envmap1-rotation-traj-512\~2023-12-09T19-05-44
                    # input = torch.cat([sm, z-z_f, z/z_f, c_e, c_c[:, None, ...].expand(z.shape), gbuffers['position'][:, None, ...].expand((*z.shape[:-1], 3))], dim=-1).permute(0, 1, 4, 2, 3) # envmap1-rotation-traj-512-ab_nsm_inputs~2023-12-10T16-16-59
                    sm = data['shadow']['shadowmap']
                    input = torch.cat([sm, (z_f-z)*10, (1-z/z_f)*10, c_e, c_c[:, None, ...].expand(z.shape), gbuffers['position'][:, None, ...].expand((*z.shape[:-1], 3))], dim=-1).permute(0, 1, 4, 2, 3) # envmap1-rotation-traj-512-ab_nsm_inputs2

                    # input = torch.cat([sm*(z_f-z), c_e, c_c[:, None, ...].expand(z.shape), gbuffers['position'][:, None, ...].expand((*z.shape[:-1], 3))], dim=-1).permute(0, 1, 4, 2, 3)
                    # input = torch.cat([(1-sm)*(z_f-z)*100, c_e, c_c[:, None, ...].expand(z.shape), gbuffers['position'][:, None, ...].expand((*z.shape[:-1], 3))], dim=-1).permute(0, 1, 4, 2, 3)
                    # input = torch.cat([(z_f-z)*100, c_e, c_c[:, None, ...].expand(z.shape), gbuffers['position'][:, None, ...].expand((*z.shape[:-1], 3))], dim=-1).permute(0, 1, 4, 2, 3)
                else:
                    # print('area:', data['area'])
                    r_e = data['area'].unsqueeze(-1).unsqueeze(-1).expand(c_e.shape) # TODO: add this # Disabled for envmap # Useful for changing envmap
                    # print('shadow buffer scale:', torch.min(z-z_f), torch.max(z-z_f))
                    # assert self.configs[encoder_name]['type'] == 'unet'
                    # input = torch.cat([z-z_f, z/z_f, c_e+r_e], dim=-1).permute(0, 3, 1, 2)
                    # input = torch.cat([z-z_f, z/z_f, c_e], dim=-1).permute(0, 3, 1, 2)
                    # input = torch.cat([z-z_f, z/z_f], dim=-1).permute(0, 3, 1, 2)

                    # subtraction_shadow = z-z_f
                    # self_occlusion_mask = torch.isclose(z, torch.zeros_like(z))
                    # subtraction_shadow[self_occlusion_mask] = 0
                    # input = torch.cat([subtraction_shadow, z/z_f], dim=-1).permute(0, 3, 1, 2)

                    # input = (z/z_f).permute(0, 3, 1, 2)
                    # input = torch.cat([z/z_f, r_e], dim=-1).permute(0, 3, 1, 2) # Add area info

                    # Full NSM version
                    # input = torch.cat([z-z_f, z/z_f, c_e+r_e*100, c_c, gbuffers['position']], dim=-1).permute(0, 3, 1, 2)

                    # falcor-livingroom version
                    input = torch.cat([z-z_f, z/z_f, c_e+r_e*100, c_c[:, None, ...].expand(z.shape), gbuffers['position'][:, None, ...].expand((*z.shape[:-1], 3))], dim=-1).permute(0, 1, 4, 2, 3)

                # input = input.flatten(0, 1)
                input = input.view(-1, *input.shape[2:])
                # print('Shadow feature', input.shape, torch.max(c_e), torch.min(c_e), torch.max(r_e), torch.min(r_e))

                # light_embedding = direct_vpls_embedding.unsqueeze(1).unsqueeze(1).expand(*c_c.shape[:3], direct_vpls_embedding.shape[-1])
                # input = torch.cat([z-z_f, z/z_f, c_e+r_e*100, c_c, gbuffers['position'], light_embedding], dim=-1).permute(0, 3, 1, 2)
                input[torch.isnan(input)] = 1.0 # Hack for invalid shading point in z_f

                # summary(encoder, input)
                # print(encoder)
            else:
                raise NotImplementedError

            ### RUN ###
            if enable_timing_profile:
                starter.record()
            encoded_feature = encoder(input)
            if enable_timing_profile:
                ender.record()
                torch.cuda.synchronize()
                timing_profile[feature_name] = starter.elapsed_time(ender)

            if feature_name == 'shadow':
                encoded_feature = encoded_feature.permute(0, 2, 3, 1)
                encoded_feature = encoded_feature.view((*z.shape[:2], *encoded_feature.shape[1:]))
            shading_clue_repr.append(encoded_feature)

        shading_clue_repr = torch.cat(shading_clue_repr, dim=-1)

        return shading_clue_repr, timing_profile
    
    def split(self, data):
        out = {}
        i = 0
        for encoder_name, encoder_configs in self.configs.items():
            dim = encoder_configs['repr_dim']
            out['shading_clue_repr_'+encoder_name.split('_encoder')[0]] = data[..., i:i+dim]
            # print('shading_clue_repr_'+encoder_name.split('_encoder')[0], out['shading_clue_repr_'+encoder_name.split('_encoder')[0]].shape)
            i += dim
        return out
    
    def linear(self, feature_vec):
        """ linear projection after attention """
        out = {}
        i = 0
        for encoder_name, encoder_configs in self.configs.items():
            dim = encoder_configs['repr_dim']
            linear_projector = getattr(self, encoder_name+'_linear')
            # out.append(linear_projector(feature_vec[..., i:i+dim])) # mlp is applied individually
            out['shading_clue_repr_'+encoder_name.split('_encoder')[0]] = linear_projector(feature_vec[..., i:i+dim]) # mlp is applied individually
            # print('shading_clue_repr_'+encoder_name.split('_encoder')[0], out['shading_clue_repr_'+encoder_name.split('_encoder')[0]].shape)
            i += dim
        
        # return torch.concatenate(out, dim=-1)
        return out

    @property
    def repr_dim(self):
        dim = 0
        for encoder_configs in self.configs.values():
            dim += encoder_configs['repr_dim']
        return dim

    @property
    def data_format(self):
        return {
                'shadow': {
                    'pixel_emitter_distance': 1,
                    'occluder_emitter_distance': 1
                },
                # 'light_dir': 3
                'specular': {
                    'half_vec': 3,
                    'light_dir': 3
                }
        }


class Shading_Decoders(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        for decoder_name, decoder_configs in configs.items():
            if decoder_configs['type'] == 'ngi':
                setattr(self, decoder_name, ShadingGenerator(decoder_configs))
                continue
            # if optional input/output dims are specified
            if 'input' in decoder_configs:
                decoder_configs['dims'].insert(0, utils.count_GB_feature_dims(decoder_configs['input'], True))
            if 'repr_dim' in decoder_configs:
                decoder_configs['dims'].append(decoder_configs['repr_dim'])
            if decoder_configs['type'] == 'mlp':
                setattr(self, decoder_name, modules.MLP(
                    decoder_configs['dims'],
                    'lrelu',
                    decoder_configs['out_act'] if 'out_act' in decoder_configs else 'none'
                ))
            elif decoder_configs['type'] == 'cnn':
                setattr(self, decoder_name, modules.CNN(
                    decoder_configs['dims'],
                    'lrelu',
                    # 'none'
                    decoder_configs['out_act'] if 'out_act' in decoder_configs else 'none'
                ))
            else:
                raise NotImplementedError
    
    def forward(self, data):
        result = {}

        # Direct
        # if hasattr(self, "direct_backbone"):
        if "backbone" in self.configs['direct_decoder']:
            decoder = getattr(self, 'direct_decoder')
            direct_embedding = decoder(data, out_flag='backbone')
            if self.configs['direct_decoder']['backbone'] == "hybrid":
                direct_shading_embedding = direct_embedding
                result['shadow'] = direct_embedding
            elif self.configs['direct_decoder']['backbone'] == "split":
                direct_shading_embedding = direct_embedding[..., :32]
                result['shadow'] = direct_embedding[..., 32:]
            else:
                raise NotImplementedError
            # Direct decoding
            decoder = getattr(self, 'direct_mlp_decoder')
            output = decoder(direct_shading_embedding)
            shading = utils.expp1_torch(output.detach())
            shading = torch.clamp(shading, 0)
            out_flag = 'direct'
            result.update({ 'log1p_{}_shading'.format(out_flag): output, '{}_shading'.format(out_flag): shading})
        else:
            decoder = getattr(self, 'direct_decoder')
            result = decoder(data, out_flag='direct')

        # Shadow
        if hasattr(self, "shadow_decoder"):
            decoder = getattr(self, 'shadow_decoder')
            # shadow = decoder(data['shading_clue_repr_shadow'])
            if 'shadow' in result:
                if "shadow_clues" in self.configs["shadow_decoder"] and self.configs["shadow_decoder"]["shadow_clues"]:
                    inputs = torch.cat([data['shading_clue_repr_shadow'], result['shadow']], dim=-1)
                else:
                    inputs = result['shadow']
            else:
                inputs = torch.cat([data['light_repr_direct_vpls'], data['shading_clue_repr_shadow']], dim=-1)
            shadow = decoder(inputs)
            # Normalize to [0, 1]
            if self.configs["shadow_decoder"]["out_act"] == "tanh":
                result['shadow'] = shadow * 0.5 + 0.5 # tanh
            elif self.configs["shadow_decoder"]["out_act"] == "sigmoid":
                result['shadow'] = shadow # sigmoid
            else:
                raise NotImplementedError
        direct_shading = result['direct_shading'] * result['shadow']
        # result['direct'] = direct_shading
        # result['log1p_direct'] = utils.log1p_torch(result['direct'])
        
        # Indirect.
        result.update(self.forward_indirect(data))

        # Compose
        # result['shading'] = result['direct_shading'] * result['shadow'] + data['emission'] + data['indirect_shading']
        result['shading'] = direct_shading + result['indirect_shading'] + data['emission']
        # result['shading'] = result['indirect_shading']
        # result['shading'] = direct_shading + data['emission']
        # result['shading'] = data['emission']
        result['log1p_shading'] = utils.log1p_torch(result['shading'])

        # if True:
        #     for i in range(3):
        #         print('Vis:', data['shadow_map_'+str(i)].shape)
        #         result['shading_clue_repr_shadow_'+str(i)] = data['shading_clue_repr_shadow_'+str(i)]
        #         result['shadow_map_'+str(i)] = data['shadow_map_'+str(i)]

        return result
    
    def forward_indirect(self, data):
        decoder = getattr(self, 'indirect_decoder')
        indirect = decoder(data, out_flag='indirect')
        # indirect = decoder(data, out_flag='indirect', direct_shading=direct_shading.detach())
        return indirect
    
    @property
    def gbuffer_dim(self):
        return self.direct_decoder.gbuffer_dim
    
    @property
    def light_repr_dim(self):
        dim = self.direct_decoder.light_repr_dim
        if hasattr(self, 'indirect_decoder'):
            dim += self.indirect_decoder.light_repr_dim
        return dim
    
    @property
    def data_format(self):
        return self.direct_decoder.data_format

    @property
    def inputs(self):
        return self.configs['direct_decoder']['input']

# Ablation
class Shading_Decoder_Unified(Shading_Decoders):

    def forward(self, data):
        decoder = getattr(self, 'unified_decoder')
        result = decoder(data)
        return result

    @property
    def gbuffer_dim(self):
        return self.unified_decoder.gbuffer_dim
    
    @property
    def light_repr_dim(self):
        dim = self.unified_decoder.light_repr_dim
        return dim
    
    @property
    def data_format(self):
        return self.unified_decoder.data_format

    @property
    def inputs(self):
        return self.configs['unified_decoder']['input']


class NeGL_Base(nn.Module):
    def __init__(self, configs, loss_configs=None, load_decoder=False):
        super().__init__()
        self.configs = configs

        if load_decoder:
            # Parse configs for shading net
            for decoder_name in configs['shading_decoders']:
                if 'shading_clues' in configs['shading_decoders'][decoder_name]:
                    shading_clue_dim = 0
                    for clue_name in configs['shading_decoders'][decoder_name]['shading_clues']:
                        shading_clue_dim += configs['shading_encoders'][clue_name+'_encoder']['repr_dim']
                    configs['shading_decoders'][decoder_name]['shading_clue_dim'] = shading_clue_dim

                if 'vpls' in configs['shading_decoders'][decoder_name]:
                    repr_dim = 0
                    for vpl_name in configs['shading_decoders'][decoder_name]['vpls']:
                        repr_dim += configs['light_encoders'][vpl_name+'_encoder']['repr_dim']
                    configs['shading_decoders'][decoder_name]['repr_dim'] = repr_dim

            if 'unified_decoder' in configs['shading_decoders']:
                self.shading_decoder = Shading_Decoder_Unified(configs['shading_decoders'])
            else:
                self.shading_decoder = Shading_Decoders(configs['shading_decoders'])
            self.loss_func = Loss(loss_configs) if loss_configs else None
    
    def forward(self, data):
        raise NotImplementedError

class NeGL_Direct(NeGL_Base):
    def __init__(self, configs, gbuffer_dim, encode_indirect=False):
        super().__init__(configs)
        cfg = {}
        cfg['direct_vpls_encoder'] = configs['light_encoders']['direct_vpls_encoder']
        if encode_indirect:
            cfg['indirect_vpls_encoder'] = configs['light_encoders']['indirect_vpls_encoder']
        self.direct_vpls_encoder = Repr_Encoders(cfg) # Encode all indirect vpls at *once*
        self.shading_clue_encoders = Shading_Encoders(configs['shading_encoders'])

        # Attention
        if 'attention' in configs:
            if 'cross' in configs['attention']:
                print(['Warnining'])
                self.cross_attention_lights = modules.CrossAttention(self.direct_vpls_encoder.repr_dim, gbuffer_dim, self.direct_vpls_encoder.repr_dim)
                # self.cross_attention_lights = modules.CrossAttention(self.direct_vpls_encoder.repr_dim, gbuffer_dim, 324)
            else:
                raise ValueError("Attention type {} is not supported.".format(configs['attention']))
        else:
            print('[Warning] No attention compositor is used!')

    def forward(self, data, enable_timing_profile=False):
        timing_profile = {}
        if enable_timing_profile:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        #### Encoding ####
        # Encode light representation & shading clues for each light
        if enable_timing_profile:
            starter.record()
        
        direct_light_reprs = self.direct_vpls_encoder(data['lights'])
        shading_clue_reprs, shading_clue_timing_profile = self.shading_clue_encoders(data['lights'], data['gbuffer'], enable_timing_profile)

        if enable_timing_profile:
            ender.record()
            torch.cuda.synchronize()
            timing_profile['direct_light_encoding'] = starter.elapsed_time(ender)
            for k, v in shading_clue_timing_profile.items():
                timing_profile['shading_clue ['+k+']'] = v
        
        return data, timing_profile, direct_light_reprs, shading_clue_reprs
    
    @property
    def data_format(self):
        data_channels = self.direct_vpls_encoder.data_format
        data_channels.update(self.shading_clue_encoders.data_format)
        return data_channels


class NeGL_Indirect(NeGL_Base):

    def __init__(self, configs):
        super().__init__(configs)

        ## Encoders ##
        cfg = {}
        cfg['indirect_vpls_encoder'] = configs['light_encoders']['indirect_vpls_encoder']
        if 'ssvpls_encoder' in configs['light_encoders']:
            cfg['ssvpls_encoder'] = configs['light_encoders']['ssvpls_encoder']
        self.indirect_vpls_encoder = Repr_Encoders(cfg) # Encode all indirect vpls at *once*

    def forward(self, data, enable_timing_profile=False):
        timing_profile = {}
        if enable_timing_profile:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
        
        indirect_data = {
            'indirect_vpls': {'position': data['lights']['indirect_vpls']['position'], 
                               'normal': data['lights']['indirect_vpls']['normal'],
                               'flux': data['lights']['indirect_vpls']['flux']}}
        if 'ssvpls_encoder' in self.configs['light_encoders']:
            indirect_data['ssvpls'] = {'position': data['gbuffer']['position'], 
                               'normal': data['gbuffer']['normal'], 
                               'flux': data['gbuffer']['albedo'] * data['lights']['shadow']['shadowmap'].squeeze(1)} # multiply light intensity?
        indirect_light_repr = self.indirect_vpls_encoder.forward(indirect_data)
        data.update(self.indirect_vpls_encoder.split(indirect_light_repr))
        # indirect_light_reprs = self.indirect_vpls_encoder.forward(indirect_data)
        # data.update(self.indirect_vpls_encoder.split(indirect_light_repr))

        if enable_timing_profile:
            ender.record()
            torch.cuda.synchronize()
            timing_profile['indirect_light_encoding'] = starter.elapsed_time(ender)

        # return data, timing_profile, indirect_light_reprs
        return data, timing_profile

    @property
    def data_format(self):
        return self.indirect_vpls_encoder.data_format


class NeGL_Full(NeGL_Base):
    def __init__(self, configs, loss_configs=None):
        super().__init__(configs, loss_configs, load_decoder=True)

        # Light encoders
        self.negl_direct = NeGL_Direct(configs, self.shading_decoder.gbuffer_dim)
        self.negl_indirect = NeGL_Indirect(configs)
    
    def forward(self, data, enable_timing_profile=False):
        data_direct, timing_profile, direct_light_reprs, shading_clue_reprs = self.negl_direct(data, enable_timing_profile)
        data, timing_indirect = self.negl_indirect(data_direct, enable_timing_profile)
        timing_profile.update(timing_indirect)

        if enable_timing_profile:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        #### Attention ####
        if enable_timing_profile:
            starter.record()
        if 'attention' in self.configs:
            if self.configs['attention'] == 'cross_pixel_direct': # direct embedding as K. (direct + shading clues) as V.
                gbuffers = utils.get_GB_features(data, self.shading_decoder.inputs)
                attention_lights, attention_weights = self.negl_direct.cross_attention_lights(direct_light_reprs, gbuffers.view(gbuffers.shape[0], -1, gbuffers.shape[-1])) # TODO: time complexity

                # Test pixel dims as batch
                # direct_light_reprs = direct_light_reprs.unsqueeze(1).unsqueeze(1).expand(*gbuffers.shape[:3], direct_light_reprs.shape[-2], direct_light_reprs.shape[-1])
                # # print('[Attention] shapes:', direct_light_reprs.shape, gbuffers.view(-1, 1, gbuffers.shape[-1]).shape)
                # attention_lights, attention_weights = self.negl_direct.cross_attention_lights(direct_light_reprs.view(-1, direct_light_reprs.shape[-2], direct_light_reprs.shape[-1]), gbuffers.view(-1, 1, gbuffers.shape[-1])) # TODO: time complexity
                # attention_lights = attention_lights.permute(1, 0, 2)
                # attention_weights = attention_weights.permute(1, 0, 2)
                # print(attention_lights.shape, attention_weights.shape)

                composed_light_repr = attention_lights.view(attention_lights.shape[0], gbuffers.shape[1], gbuffers.shape[2], attention_lights.shape[-1])
                data.update(self.negl_direct.direct_vpls_encoder.split(composed_light_repr))
                # *Share* attention weights with shading clues
                shading_clue_reprs = shading_clue_reprs.permute(0, 2, 3, 1, 4)
                attention_weights = attention_weights.view(shading_clue_reprs.shape[:-1])
                data.update(self.negl_direct.shading_clue_encoders.linear(torch.sum(attention_weights.unsqueeze(-1) * shading_clue_reprs, dim=-2))) # Weighted sum over all lights & mlp
            elif self.configs['attention'] == 'cross_pixel_full': # full light embedding as K & V. Weight sharing is not required. 
                gbuffers = utils.get_GB_features(data, self.shading_decoder.inputs)
                pass
            elif self.configs['attention'] == 'cross_pixel_direct2indirect': # direct embedding as K. full light embedding (direct + shading clues + indirect) as V.
                gbuffers = utils.get_GB_features(data, self.shading_decoder.inputs)
                attention_lights, attention_weights = self.negl_direct.cross_attention_lights(direct_light_reprs, gbuffers.view(gbuffers.shape[0], -1, gbuffers.shape[-1])) # TODO: time complexity
                composed_light_repr = attention_lights.view(attention_lights.shape[0], gbuffers.shape[1], gbuffers.shape[2], attention_lights.shape[-1])
                data.update(self.negl_direct.direct_vpls_encoder.split(composed_light_repr))
                # *Share* attention weights with indirect embeddings
                raise ValueError("Use nrsm-cross_pixel_direct2indirect.py")
                # indirect_light_reprs = indirect_light_reprs.unsqueeze(1).expand(indirect_light_reprs.shape[0], attention_weights.shape[1], indirect_light_reprs.shape[1], indirect_light_reprs.shape[-1])
                # composed_indirect_light_repr = torch.sum(attention_weights.unsqueeze(-1) * indirect_light_reprs, dim=-2).view(attention_lights.shape[0], gbuffers.shape[1], gbuffers.shape[2], indirect_light_reprs.shape[-1])
                # data.update(self.negl_indirect.indirect_vpls_encoder.split(composed_indirect_light_repr)) # Weighted sum over all lights & mlp
                # *Share* attention weights with shading clues
                shading_clue_reprs = shading_clue_reprs.permute(0, 2, 3, 1, 4)
                attention_weights = attention_weights.view(shading_clue_reprs.shape[:-1])
                data.update(self.negl_direct.shading_clue_encoders.linear(torch.sum(attention_weights.unsqueeze(-1) * shading_clue_reprs, dim=-2))) # Weighted sum over all lights & mlp
            elif self.configs['attention'] == 'cross_pixel_direct_full': # full *direct* light embedding as K & V. Weight sharing is not required.
                gbuffers = utils.get_GB_features(data, self.shading_decoder.inputs)
                direct_light_reprs = direct_light_reprs.unsqueeze(1).unsqueeze(1).expand(*gbuffers.shape[:3], direct_light_reprs.shape[-2], direct_light_reprs.shape[-1])
                shading_clue_reprs = shading_clue_reprs.permute(0, 2, 3, 1, 4)
                light_embeddings = torch.cat([direct_light_reprs, shading_clue_reprs], dim=-1)
                attention_lights, attention_weights = self.negl_direct.cross_attention_lights(light_embeddings.view(-1, light_embeddings.shape[-2], light_embeddings.shape[-1]), gbuffers.view(-1, 1, gbuffers.shape[-1]))
                attention_lights = attention_lights.view(gbuffers.shape[0], gbuffers.shape[1], gbuffers.shape[2], attention_lights.shape[-1])
                data.update(self.negl_direct.direct_vpls_encoder.split(attention_lights[...,0:direct_light_reprs.shape[-1]]))
                data.update(self.negl_direct.shading_clue_encoders.split(attention_lights[...,direct_light_reprs.shape[-1]:]))
                attention_weights = attention_weights.permute(1, 0, 2).view(*gbuffers.shape[:3], attention_weights.shape[-1])
            else:
                raise NotImplementedError('Attention type({}) is not supoprted.'.format(self.configs['attention']))
        else:
            data.update(self.negl_direct.direct_vpls_encoder.split(torch.mean(direct_light_reprs, dim=1)))
            data.update(self.negl_direct.shading_clue_encoders.split(torch.mean(shading_clue_reprs, dim=1)))

        if enable_timing_profile:
            ender.record()
            torch.cuda.synchronize()
            timing_profile['direct_light_attention'] = starter.elapsed_time(ender)

        #### Decoder ####
        if enable_timing_profile:
            starter.record()
        result = self.shading_decoder.forward(data)
        if enable_timing_profile:
            ender.record()
            torch.cuda.synchronize()
            timing_profile['deocder'] = starter.elapsed_time(ender)

        # Neural features visualization
        if True:
            # for i in range(self.shading_clue_encoders.configs['shadow_encoder']['repr_dim']):
            #     data['shading_clue_repr_shadow_'+str(i)] = data['shading_clue_repr_shadow'][..., i].unsqueeze(-1)
            if 'attention' in self.configs:
                # print('shape', attention_weights.shape)
                for i in range(attention_weights.shape[-1]):
                    result['attention_weights_'+str(i)] = attention_weights[..., i].unsqueeze(-1)

        # Calculate loss
        loss_map = None
        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)

        if enable_timing_profile:
            return result, loss_map, timing_profile
        else:
            return result, loss_map

    @property
    def data_format(self):
        data_channels = {}
        direct_data_channels = self.negl_direct.data_format
        indirect_data_channels = self.negl_indirect.data_format
        direct_data_channels.update(indirect_data_channels)
        data_channels['lights'] = direct_data_channels

        data_channels.update(self.shading_decoder.data_format)
        data_channels['lights']['shadow'] = {'clues': 2}
        data_channels.pop('light_repr')

        return data_channels
    
    def export_onnx(self, out_path, scene_info, fp16=True, fp32_input=True, res=300, outdoor=False):
        if outdoor:
            print('[Warning] Exporting outdoor version..')
            model = NeGL_ONNX_Outdoor(self, scene_info, fp16, internal_fp16_conversion=(fp16 and fp32_input))
        else:
            model = NeGL_ONNX(self, scene_info, fp16, internal_fp16_conversion=(fp16 and fp32_input))
        model.eval()
        if fp16:
            model = model.half()

        input_names = []
        dummy_input = []
        dynamic_axes = {}
        # Gbuffers
        for input_name, channel in self.data_format['gbuffer'].items():
            if input_name == 'specular':
                input_name = 'specular_roughness'
            if input_name == 'roughness':
                continue
            input_names.append(input_name)
            if channel == 3:
                channel += 1
            dummy_input.append(torch.randn(1, res, res, channel).cuda())
        input_names.append('emission')
        dummy_input.append(torch.randn(1, res, res, 4).cuda())
        input_names.append('view_dir') # TODO: only used for input of shadow
        dummy_input.append(torch.randn(1, res, res, 4).cuda())

        # Lights
        if outdoor:
            num_lights = 1
        else:
            # num_lights = 3 # TODO: the number of lights must match?? otherwise, result maybe incorrect. (I_scale ?)
            num_lights = 2
        for input_sub_name, channel in self.data_format['lights'].items():
            input_name = 'lights_' + input_sub_name
            if isinstance(channel, int):
                print('Deprecated', input_name)
                exit(0)
                input_names.append(input_name)
                if channel == 3:
                    channel += 1
                dummy_input.append(torch.randn(num_lights, 1, channel).cuda())
                dynamic_axes[input_name] = {1: "num_lights"}
            elif isinstance(channel, dict): # ob / shadow features
                for feature_name, c in channel.items():
                    if c == 3:
                        c += 1
                    n = input_name + '_' + feature_name
                    input_names.append(n)
                    if input_sub_name == 'direct_vpls' :
                        dummy_input.append(torch.randn(1, num_lights, 500, c).cuda())
                    elif input_sub_name == 'indirect_vpls':
                        if outdoor:
                            dummy_input.append(torch.randn(1, num_lights, 4, 64, 64, c).cuda())
                        else:
                            dummy_input.append(torch.randn(1, num_lights, 6, 64, 64, c).cuda())
                    elif 'shadow' == input_sub_name or 'specular' == input_sub_name:
                        dummy_input.append(torch.randn(1, num_lights, res, res, c).cuda())
                    else:
                        raise NotImplementedError(n)
                    dynamic_axes[n] = {1: "num_lights"}
            else:
                raise NotImplementedError(input_name)

        if fp16 and not fp32_input:
            for i in range(len(dummy_input)):
                dummy_input[i] = dummy_input[i].half()

        output_names = ['shading']
        # output_names = ['shading', 'direct_shading', 'shadow', 'indirect_shading'] # Debug

        for n, i in zip(input_names, dummy_input):
            print(n, i.shape, i.dtype)

        suffix = ''
        suffix += '_res'+str(res)
        if fp16:
            suffix += '_fp16'
        if fp32_input:
            suffix += '_fp32-input'
        out_fn = os.path.join(out_path, 'negl'+suffix+'.onnx')
        torch.onnx.export(
            model,
            tuple(dummy_input),
            out_fn,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11
        )
        print('Finish exporting. {}'.format(out_fn))

