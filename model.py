import torch
import torch.nn as nn
from models.autoencoder_vgg19.vgg19_1 import vgg_normalised_conv1_1, feature_invertor_conv1_1
from models.autoencoder_vgg19.vgg19_2 import vgg_normalised_conv2_1, feature_invertor_conv2_1
from models.autoencoder_vgg19.vgg19_3 import vgg_normalised_conv3_1, feature_invertor_conv3_1
from models.autoencoder_vgg19.vgg19_4 import vgg_normalised_conv4_1, feature_invertor_conv4_1
from models.autoencoder_vgg19.vgg19_5 import vgg_normalised_conv5_1, feature_invertor_conv5_1
from wst import gaussian_transfer, gmm_transfer, sinkhorn_transfer, ot_emd_transfer

class Encoder(nn.Module):
    def __init__(self, depth):
        super(Encoder, self).__init__()
        assert(isinstance(depth, int) and 1 <= depth <= 5)
        self.depth = depth

        if depth == 1:
            self.model = vgg_normalised_conv1_1.vgg_normalised_conv1_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_1/vgg_normalised_conv1_1.pth", map_location='cpu', weights_only=True))
        elif depth == 2:
            self.model = vgg_normalised_conv2_1.vgg_normalised_conv2_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_2/vgg_normalised_conv2_1.pth", map_location='cpu', weights_only=True))
        elif depth == 3:
            self.model = vgg_normalised_conv3_1.vgg_normalised_conv3_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_3/vgg_normalised_conv3_1.pth", map_location='cpu', weights_only=True))
        elif depth == 4:
            self.model = vgg_normalised_conv4_1.vgg_normalised_conv4_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_4/vgg_normalised_conv4_1.pth", map_location='cpu', weights_only=True))
        elif depth == 5:
            self.model = vgg_normalised_conv5_1.vgg_normalised_conv5_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_5/vgg_normalised_conv5_1.pth", map_location='cpu', weights_only=True))

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, depth):
        super(Decoder, self).__init__()
        assert(isinstance(depth, int) and 1 <= depth <= 5)
        self.depth = depth

        if depth == 1:
            self.model = feature_invertor_conv1_1.feature_invertor_conv1_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_1/feature_invertor_conv1_1.pth", map_location='cpu', weights_only=True))
        elif depth == 2:
            self.model = feature_invertor_conv2_1.feature_invertor_conv2_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_2/feature_invertor_conv2_1.pth", map_location='cpu', weights_only=True))
        elif depth == 3:
            self.model = feature_invertor_conv3_1.feature_invertor_conv3_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_3/feature_invertor_conv3_1.pth", map_location='cpu', weights_only=True))
        elif depth == 4:
            self.model = feature_invertor_conv4_1.feature_invertor_conv4_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_4/feature_invertor_conv4_1.pth", map_location='cpu', weights_only=True))
        elif depth == 5:
            self.model = feature_invertor_conv5_1.feature_invertor_conv5_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_5/feature_invertor_conv5_1.pth", map_location='cpu', weights_only=True))

    def forward(self, x):
        return self.model(x)

def stylize(level, content, style, encoders, decoders, alpha, device):
    with torch.no_grad():
        content_features = encoders[level](content).squeeze(0)
        style_features = encoders[level](style).squeeze(0)
        
        # Apply Gaussian Wasserstein Style Transfer
        transformed_features = gaussian_transfer(alpha, content_features, style_features).to(device)
        
        return decoders[level](transformed_features)

class MultiLevelStyleTransfer(nn.Module):
    def __init__(self, alpha=0.5, style_weights=None, method='gaussian', K=5, epsilon=0.05, max_samples=10000, device='cpu'):
        super().__init__()
        self.alpha = alpha
        self.style_weights = style_weights
        self.method = method
        self.K = K
        self.epsilon = epsilon
        self.max_samples = max_samples
        self.device = device
        self.encoders = [Encoder(level).to(device) for level in range(5, 0, -1)]
        self.decoders = [Decoder(level).to(device) for level in range(5, 0, -1)]

    def forward(self, content_img, style_img):
        current_img = content_img
        for level, (encoder, decoder) in enumerate(zip(self.encoders, self.decoders)):
            
            with torch.no_grad():
                content_features = encoder(current_img).squeeze(0)
                
                # Handle single style or list of styles
                if isinstance(style_img, list):
                    style_features = [encoder(s).squeeze(0) for s in style_img]
                else:
                     style_features = encoder(style_img).squeeze(0)
                
                if self.method == 'gmm':
                    transformed_features = gmm_transfer(self.alpha, content_features, style_features, style_weights=self.style_weights, K=self.K).to(self.device)
                elif self.method == 'ot_emd':
                    transformed_features = ot_emd_transfer(self.alpha, content_features, style_features, style_weights=self.style_weights, max_samples=self.max_samples).to(self.device)
                else:
                    transformed_features = gaussian_transfer(self.alpha, content_features, style_features, style_weights=self.style_weights).to(self.device)
                
                current_img = decoder(transformed_features)
                
        return current_img
