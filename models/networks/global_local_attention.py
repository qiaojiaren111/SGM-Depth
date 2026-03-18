import torch
import torch.nn as nn
import torch.nn.functional as F

class SCConv(nn.Module):
    """Simplified Spatial and Channel Reconstruction Convolution (SCConv)"""
    def __init__(self, in_channels, out_channels, reduction=4):
        super(SCConv, self).__init__()
        # Spatial Reconstruction Unit (SRU): Enhance spatial dependencies
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.spatial_bn = nn.BatchNorm2d(in_channels)
        # Channel Reconstruction Unit (CRU): Enhance channel-wise dependencies
        self.channel_conv = nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1)
        self.channel_bn = nn.BatchNorm2d(out_channels // reduction)
        self.channel_restore = nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Spatial enhancement
        spatial = self.spatial_bn(self.spatial_conv(x))
        spatial = F.relu(spatial)
        # Channel enhancement
        channel = self.channel_bn(self.channel_conv(spatial))
        channel = F.relu(channel)
        channel = self.channel_restore(channel)
        return channel + x  # Residual connection

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
        # Apply SCConv for feature enhancement
        features = self.scconv(x)
        features = self.conv(features)
        # Pooling to capture global context
        max_pooled = self.max_pool(features).view(features.size(0), -1)
        avg_pooled = self.avg_pool(features).view(features.size(0), -1)
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)
        # Generate attention weights
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
        
        # Simplified instance segmentation (placeholder for actual instance detection)
        self.instance_conv = nn.Conv2d(in_channels, num_instances, kernel_size=1)
        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        # Simulate instance masks (in practice, use a proper instance segmentation model)
        instance_maps = self.instance_conv(x)
        instance_maps = F.softmax(instance_maps, dim=1)  # [B, num_instances, H, W]
        
        # Initialize output
        local_attention = torch.zeros_like(x)
        
        for i in range(self.num_instances):
            # Extract instance mask
            mask = instance_maps[:, i:i+1, :, :]  # [B, 1, H, W]
            instance_features = x * mask  # [B, C, H, W]
            
            # Multi-head self-attention
            query = self.query_conv(instance_features).view(batch_size, self.num_heads, self.head_dim, H * W)
            key = self.key_conv(instance_features).view(batch_size, self.num_heads, self.head_dim, H * W)
            value = self.value_conv(instance_features).view(batch_size, self.num_heads, self.head_dim, H * W)
            
            # Normalize key and query
            query = F.normalize(query, dim=-1)
            key = F.normalize(key, dim=-1)
            
            # Attention scores
            attention_scores = torch.einsum('bhci,bhcj->bhij', query, key) / (self.head_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply attention
            attended = torch.einsum('bhij,bhcj->bhci', attention_weights, value)
            attended = attended.view(batch_size, C, H, W)
            
            # Add to output with instance mask
            local_attention += attended * mask
        
        return local_attention

class GlobalLocalAttentionModule(nn.Module):
    """Global-Local Attention Module (GLAM)"""
    def __init__(self, in_channels, num_instances=8, num_heads=4, reduction=4):
        super(GlobalLocalAttentionModule, self).__init__()
        self.global_attention = GlobalAttention(in_channels, reduction)
        self.local_attention = LocalAttention(in_channels, num_instances, num_heads)
        # Cross-attention fusion
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
    def forward(self, x):
        # Compute global attention
        global_features = self.global_attention(x)
        # Compute local attention
        local_features = self.local_attention(x)
        # Concatenate global and local features
        combined = torch.cat([global_features, local_features], dim=1)
        # Fuse features
        fused = self.fusion_conv(combined)
        # Residual connection
        return fused + x

# Example usage
if __name__ == "__main__":
    # Sample input tensor: [batch_size, channels, height, width]
    input_tensor = torch.randn(1, 64, 128, 128)
    # Initialize GLAM
    glam = GlobalLocalAttentionModule(in_channels=64, num_instances=8, num_heads=4, reduction=4)
    # Forward pass
    output = glam(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")