import math
import torch.nn as nn
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import PatchEmbed, EncoderStack, MAEExtractor
from .modeling.models_conv import SimpleModulationStack
from .modeling.swin_transformer import SwinTransfomerSegHead
import torch
import time
from functools import partial

class DinoV2Extractor(nn.Module):
    def __init__(self, extractor_type, extraction_blocks=[2, 5, 8, 11]):
        super(DinoV2Extractor, self).__init__()
        self.extractor_type = extractor_type
        self.extraction_blocks = extraction_blocks
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', extractor_type)
        self.feature_map_dict = dict()

        for blk in self.dinov2.blocks:
            blk.register_forward_hook(partial(self.feature_hook))

    def feature_hook(self, module, input, output):
        self.feature_map_dict[str(output.device)].append(output)


    def forward(self, input):
        patch_size = self.dinov2.patch_embed.proj.stride
        grid_size = input.shape[2] // patch_size[0], input.shape[3] // patch_size[1]

        self.feature_map_dict[str(input.device)] = []
        self.dinov2(input)

        out_features = []
        for block_num in self.extraction_blocks:
            features = self.feature_map_dict[str(input.device)][block_num] # [batch_size, seq_len, channels]

            # At this point we may have seq_len = 1 + num_register_tokens + grid_size[0] * grid_size[1], since
            # we still have one class token and might have multiple register tokens. We will fix that in the next line.
            out_features.append(features[:, 1 + self.dinov2.num_register_tokens:])

        self.feature_map_dict = dict()

        return out_features

class DinoV3Extractor(nn.Module):
    def __init__(self, extractor_type, repo_path, weight_path, extraction_blocks=[2, 5, 8, 11]):
        super(DinoV3Extractor, self).__init__()
        self.extractor_type = extractor_type
        self.extraction_blocks = extraction_blocks
        self.dinov3 = torch.hub.load(repo_path, extractor_type, source='local', weights=weight_path)

    def forward(self, x_input):
        return list(
            self.dinov3.get_intermediate_layers(x_input, n=self.extraction_blocks)
        )


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024], skip_connections=False):
        super().__init__()
        self.skip_connections = skip_connections
        if skip_connections:
            in_dim = in_dim * 2
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x, skip_features=None):
        if self.skip_connections:
            assert skip_features is not None
            x_down_4 = self.down_4(torch.cat((x, skip_features[0]), dim=1))
            x_down_8 = self.down_8(torch.cat((x, skip_features[1]), dim=1))
            x_down_16 = self.down_16(torch.cat((x, skip_features[2]), dim=1))
            x_down_32 = self.down_32(torch.cat((x, skip_features[3]), dim=1))
        else:
            x_down_4 = self.down_4(x)
            x_down_8 = self.down_8(x)
            x_down_16 = self.down_16(x)
            x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class FeatureVitModel(ISModel):
    """
    Extractor types:
        'dinov2_vitb14_reg', 'dinov2_vitg14_reg',
        'mae_vitb16'
    """
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={}, 
        head_params={},
        random_split=False,
        extractor_type = 'dinov2_vitb14_reg',
        backbone_weight_path = None,
        backbone_repo_path = None,
        trained_extractor = False,
        use_intermediate = [11],
        skip_connections=False,
        use_conv_stack=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split
        self.extractor_type = extractor_type
        self.trained_extractor = trained_extractor

        self.use_intermediate = use_intermediate
        self.skip_connections = skip_connections
        self.use_conv_stack = use_conv_stack
        neck_params['skip_connections'] = skip_connections

        # The usage of skip connections requires the availability of intermediate features
        assert not self.skip_connections or self.use_intermediate

        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3 if self.with_prev_mask else 2, 
            embed_dim=backbone_params['embed_dim'],
        )

        if self.extractor_type[:6] == "dinov2":
            self.feature_extractor = DinoV2Extractor(extractor_type)
        elif self.extractor_type[:6] == "dinov3":
            self.feature_extractor = DinoV3Extractor(
                extractor_type=extractor_type,
                repo_path=backbone_repo_path,
                weight_path=backbone_weight_path,
            )

        if self.use_conv_stack:
            self.backbone_mix = SimpleModulationStack(**backbone_params)
        else: # Use a stack of transformer encoder blocks
            self.backbone_mix = EncoderStack(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

        if len(self.use_intermediate) > 1:
            # Use a linear layer to fuse the feature model
            self.fuse = nn.Linear(
                backbone_params["embed_dim"] * len(self.use_intermediate), backbone_params["embed_dim"]
            )

        self.dsum = 0.0
        self.dcount = 0

    def backbone_forward(self, image, coord_features=None):
        h, w = image.shape[-2], image.shape[-1]

        if self.trained_extractor:
            image_features = self.feature_extractor(image)
        else:
            with torch.no_grad():
                image_features = self.feature_extractor(image)
                image_features = [feat.detach() for feat in image_features]
        # List of features for skip connections
        skip_features = image_features

        if len(self.use_intermediate) > 1:
            image_features = torch.cat(image_features, dim=2) # [batch_size, seq_len, embed_dim * len(use_intermediate)]
            image_features = self.fuse(image_features) # [batch_size, seq_len, embed_dim]
        else:
            image_features = image_features[0] # Should be a single tensor instead of a list

        h_patch, w_patch = self.backbone_mix.patch_size
        grid_size = h // h_patch, w // w_patch

        start_time = time.time()

        coord_features = self.patch_embed_coords(coord_features)
        if self.use_conv_stack:
            backbone_features = self.backbone_mix(image_features, coord_features, grid_size)
        else: # Use a transformer encoder stack
            backbone_features = self.backbone_mix(image_features, h, w, coord_features, self.random_split)

        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape

        backbone_features = backbone_features.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])
        if self.skip_connections:
            assert len(skip_features) == 4
            skip_features = [feat.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1]) for feat in skip_features]
            multi_scale_features = self.neck(backbone_features, skip_features = skip_features)
        else:
            multi_scale_features = self.neck(backbone_features)
        instances = self.head(multi_scale_features)

        #self.dsum += (time.time() - start_time)
        #self.dcount += 1

        #print("Avg. duration: ", (self.dsum / self.dcount))
        return {'instances': instances, 'instances_aux': None}


    def set_image(self, image):
        """
        image : A [1xCxHxW] torch float tensor.
        """
        # 1. Normalize the image (origin: ISModel)
        self.image = self.normalization(image) # WILL BE USED BY apply_click

        # 2. Extract the features (origin: FeatureVitModel)
        self.h, self.w = image.shape[-2], image.shape[-1] # WILL BE USED BY apply_click

        if self.trained_extractor:
            image_features = self.feature_extractor(self.image)
        else:
            with torch.no_grad():
                image_features = self.feature_extractor(self.image)
                image_features = [feat.detach() for feat in image_features]

        self.skip_features = image_features # WILL BE USED BY apply_click


    def apply_click(self, prev_mask, points):
        if len(self.use_intermediate) > 1:
            print("SBE: feature fusion")
            image_features = torch.cat(self.skip_features,
                                       dim=2)  # [batch_size, seq_len, embed_dim * len(use_intermediate)]
            self.image_features = self.fuse(image_features)  # [batch_size, seq_len, embed_dim] # WILL BE USED BY apply_click


        h_patch, w_patch = self.backbone_mix.patch_size
        self.grid_size = self.h // h_patch, self.w // w_patch # WILL BE USED BY apply_click

        # 1. Transform features (origin: ISModel)
        coord_features = self.get_coord_features(self.image, prev_mask, points)
        coord_features = self.maps_transform(coord_features)

        # 2. Carry out the rest of SkipClick (origin: FeatureVitModel)
        coord_features_embedded = self.patch_embed_coords(coord_features)
        if self.use_conv_stack:
            backbone_features = self.backbone_mix(self.image_features, coord_features_embedded, self.grid_size)
        else:  # Use a transformer encoder stack
            backbone_features = self.backbone_mix(self.image_features, self.h, self.w, coord_features_embedded, self.random_split)

        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape

        backbone_features = backbone_features.transpose(-1, -2).view(B, C, self.grid_size[0], self.grid_size[1])
        if self.skip_connections:
            assert len(self.skip_features) == 4
            skip_features = [feat.transpose(-1, -2).view(B, C, self.grid_size[0], self.grid_size[1]) for feat in self.skip_features]
            multi_scale_features = self.neck(backbone_features, skip_features=skip_features)
        else:
            multi_scale_features = self.neck(backbone_features)
        instances = self.head(multi_scale_features)

        outputs = {'instances': instances, 'instances_aux': None}

        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=self.image.size()[2:],
                                                         mode='bilinear', align_corners=True)


        return outputs["instances"]