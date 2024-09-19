import torch
import torch.nn as nn

    
class AttentionBasedPatchClassifier(nn.Module):
    def __init__(self, cfg, embed_dim=512, num_classes=75):
        super(AttentionBasedPatchClassifier, self).__init__()
        # self.num_heads = cfg.num_heads
        self.num_heads = 8
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)  # Fully Connected Layer
    
    def forward(self, x):
        # Patch Embedding
        
        # 두 feature map의 patch tokens을 concat
        tokens = x.permute(1,0,2) # Shape: (num_patches, batch_size, d_model)
        # tokens = self.positional_encoding(tokens)  # Apply positional encoding

        # Self-Attention 적용
        attended_tokens, _ = self.self_attention(tokens, tokens, tokens)
        # attended_tokens의 shape: (num_patches*2, Batch, embed_dim)

        # Flatten 후 FC Layer에 전달
        attended_tokens = attended_tokens.mean(dim=0)  # 평균을 사용한 Global Pooling
        # attended_tokens의 shape: (Batch, embed_dim)

        output = self.fc(attended_tokens)
        # output의 shape: (Batch, num_classes)

        return output
    


if __name__=='__main__':
    net = AttentionBasedPatchClassifier(0).cuda()
    # print(net)
    from thop import profile
    input = torch.zeros(1, 1, 512)
    input = input.type(torch.LongTensor).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
