# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.layers.attention import CrossAttention


class SD_VSum(nn.Module):
    def __init__(self, input_size=512, text_size=512, output_size=512, pos_enc=True,
                 heads=8):
        """
        Class wrapping the SD-VSum model; its key modules and parameters.
        :param int input_size: The expected input feature size for video features
        :param int text_size: The expected input feature size for text features.
        :param int output_size: The hidden feature size of the attention mechanisms.
        :param bool pos_enc: Whether to apply sinusoidal positional encoding.
        :param int heads: The number of attention heads.
        """
        super(SD_VSum, self).__init__()

        self.cross_attention = CrossAttention(input_size=input_size, text_size=text_size, output_size=output_size,
                                        pos_enc=pos_enc, heads=heads)

        self.linear_layer = nn.Linear(in_features=input_size, out_features=1)
        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.sigmoid = nn.Sigmoid()

        # =========== Transformer ===========
        transformer_kwargs = {
            'batch_first': False,
        }
        self.transformer = nn.Transformer(input_size, **transformer_kwargs)


    def forward(self, frame_features, text_features):
        """
        Produce frame importance scores using the SD_VSum model.
        :param torch.Tensor frame_features: Tensor of shape [N, input_size] containing frame features, where N is the number of frames.
        :param torch.Tensor text_features: Tensor of shape [M, text_size] containing text features, where M is the number of sentences.
        :return torch.Tensor: Tensor of shape [1, N] containing the frame importance scores in [0, 1].
        """
        # ========= Cross-Attention =============
        attended_values = self.cross_attention(frame_features, text_features)
        y = self.drop(attended_values)
        y = self.norm_y(y)

        # ========= Transformer =============
        y = self.transformer(y, y)
        y = self.linear_layer(y)
        y = self.sigmoid(y)
        y = y.view(1, -1)

        return y


if __name__ == '__main__':
    pass
    # Uncomment for a quick proof of concept
    # model = SD_VSum().cuda()
    # visual_featues = torch.randn(500, 512).cuda()  # [num_frames, hidden_size]
    # text_featues = torch.randn(7, 512).cuda() # [num_sentences, hidden_size]
    # output = model(visual_featues, text_featues)
    # print(f"Output shape: {output.shape}")
