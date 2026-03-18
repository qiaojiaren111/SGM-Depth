import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SCConv(nn.Module):
    """Simplified Spatial and Channel Reconstruction Convolution (SCConv)"""
    def __init__(self, in_channels, out_channels, reduction=4):
        super(SCConv, self).__init__()
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.spatial_bn = nn.BatchNorm2d(in_channels)
        self.channel_conv = nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1)
        self.channel_bn = nn.BatchNorm2d(out_channels // reduction)
        self.channel_restore = nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1)
        
    def forward(self, x):
        spatial = self.spatial_bn(self.spatial_conv(x))
        spatial = F.relu(spatial)
        channel = self.channel_bn(self.channel_conv(spatial))
        channel = F.relu(channel)
        channel = self.channel_restore(channel)
        return channel + x

class GlobalAttention(nn.Module):
    """Global Attention Mechanism using SCConv"""
    def __init__(self, in_channels, reduction=4):
        super(GlobalAttention, self).__init__()
        self.scconv = SCConv(in_channels, in_channels, reduction)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.scconv(x)
        features = self.conv(features)
        max_pooled = self.max_pool(features).view(features.size(0), -1)
        avg_pooled = self.avg_pool(features).view(features.size(0), -1)
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)
        attention = self.mlp(pooled).view(features.size(0), -1, 1, 1)
        return features * attention

class LocalAttention(nn.Module):
    """Local Attention Mechanism with Multi-Head Self-Attention"""
    def __init__(self, in_channels, num_instances=8, num_heads=4):
        super(LocalAttention, self).__init__()
        self.in_channels = in_channels
        self.num_instances = num_instances
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.instance_conv = nn.Conv2d(in_channels, num_instances, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        instance_maps = self.instance_conv(x)
        instance_maps = F.softmax(instance_maps, dim=1)
        
        local_attention = torch.zeros_like(x)
        
        for i in range(self.num_instances):
            mask = instance_maps[:, i:i+1, :, :]
            instance_features = x * mask
            
            query = self.query_conv(instance_features).view(batch_size, self.num_heads, self.head_dim, H * W)
            key = self.key_conv(instance_features).view(batch_size, self.num_heads, self.head_dim, H * W)
            value = self.value_conv(instance_features).view(batch_size, self.num_heads, self.head_dim, H * W)
            
            query = F.normalize(query, dim=-1)
            key = F.normalize(key, dim=-1)
            
            attention_scores = torch.einsum('bhci,bhcj->bhij', query, key) / (self.head_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            attended = torch.einsum('bhij,bhcj->bhci', attention_weights, value)
            attended = attended.view(batch_size, C, H, W)
            
            local_attention += attended * mask
        
        return local_attention

class GlobalLocalAttentionModule(nn.Module):
    """Global-Local Attention Module (GLAM)"""
    def __init__(self, in_channels, num_instances=8, num_heads=4, reduction=4):
        super(GlobalLocalAttentionModule, self).__init__()
        self.global_attention = GlobalAttention(in_channels, reduction)
        self.local_attention = LocalAttention(in_channels, num_instances, num_heads)
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
    def forward(self, x):
        global_features = self.global_attention(x)
        local_features = self.local_attention(x)
        combined = torch.cat([global_features, local_features], dim=1)
        fused = self.fusion_conv(combined)
        return fused + x

class ResNetEncoder(nn.Module):
    """ResNet-based Encoder with Global-Local Attention"""
    def __init__(self, backbone='resnet50', pretrained=True):
        super(ResNetEncoder, self).__init__()
        # Load pretrained ResNet
        resnet = getattr(models, backbone)(pretrained=pretrained)
        
        # Encoder layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 1/4 scale
        self.layer2 = resnet.layer2  # 1/8 scale
        self.layer3 = resnet.layer3  # 1/16 scale
        self.layer4 = resnet.layer4  # 1/32 scale
        
        # Channel sizes (for ResNet50)
        self.channel_sizes = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }
        
        # Add Global-Local Attention Modules after layer3 and layer4
        self.glam3 = GlobalLocalAttentionModule(
            in_channels=self.channel_sizes['layer3'],
            num_instances=8,
            num_heads=4,
            reduction=4
        )
        self.glam4 = GlobalLocalAttentionModule(
            in_channels=self.channel_sizes['layer4'],
            num_instances=8,
            num_heads=4,
            reduction=4
        )
        
    def forward(self, x):
        features = []
        
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # L1: 1/2 scale
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)  # L2: 1/4 scale
        features.append(x)
        x = self.layer2(x)  # L3: 1/8 scale
        features.append(x)
        x = self.layer3(x)  # L4: 1/16 scale
        x = self.glam3(x)   # Apply GLAM after layer3
        features.append(x)
        x = self.layer4(x)  # L5: 1/32 scale
        x = self.glam4(x)   # Apply GLAM after layer4
        features.append(x)
        
        return features  # Return multi-scale features: [L1, L2, L3, L4, L5]

# Example usage
if __name__ == "__main__":
    # Sample input tensor: [batch_size, channels, height, width]
    input_tensor = torch.randn(1, 3, 256, 256)
    # Initialize encoder
    encoder = ResNetEncoder(backbone='resnet50', pretrained=False)
    # Forward pass
    features = encoder(input_tensor)
    for i, feat in enumerate(features):
        print(f"Feature L{i+1} shape: {feat.shape}")