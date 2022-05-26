import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

class Dueling_DQN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(Dueling_DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_outputs
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).reshape(1, -1).size(1)


class EgoAttetion(nn.Module):
    def __init__(self, feature_size, heads):
        super().__init__()
        self.feature_size = feature_size
        self.heads = heads
        self.features_per_head = int(feature_size / heads)

        self.value_all = nn.Linear(feature_size, feature_size, bias=False)
        self.key_all = nn.Linear(feature_size, feature_size, bias=False)
        self.query_ego = nn.Linear(feature_size, feature_size, bias=False)
        self.attention_combine = nn.Linear(feature_size, feature_size, bias=False)

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.feature_size), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities, self.heads, self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities, self.heads, self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1, self.heads, self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.heads, 1, 1))
        value, attention_matrix = attention(query_ego, key_all, value_all, mask,
                                            nn.Dropout(0))
        result = (self.attention_combine(value.reshape((batch_size, self.feature_size))) + ego.squeeze(1)) / 2
        return result, attention_matrix


def attention(query, key, value, mask=None, dropout=None):
    """
        Compute a Scaled Dot Product Attention.
    :param query: size: batch, head, 1 (ego-entity), features
    :param key:  size: batch, head, entities, features
    :param value: size: batch, head, entities, features
    :param mask: size: batch,  head, 1 (absence feature), 1 (ego-entity)
    :param dropout:
    :return: the attention softmax(QK^T/sqrt(dk))V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


class EgoAttentionNetwork(nn.Module):
    def __init__(self, input_shape=7, num_outputs=5):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.heads = 2

        # self.ego_embedding = nn.Sequential(
        #     nn.Linear(self.input_shape, 512),
        #     nn.ReLU(),
        #     # nn.Linear(512, 512),
        #     # nn.ReLU(),
        # )
        #
        # self.others_embedding = nn.Sequential(
        #     nn.Linear(self.input_shape, 512),
        #     nn.ReLU(),
        # )

        self.ego_embedding = nn.Sequential(
            nn.Linear(self.input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.others_embedding = nn.Sequential(
            nn.Linear(self.input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.attention_layer = EgoAttetion(64, self.heads)

        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # self.advantage = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_outputs)
        # )
        #
        # self.value = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)
        # )
        self.advantage = nn.Linear(64, self.num_actions)

        self.value = nn.Linear(64, 1)

    def forward(self, x):
        ego_embedded_att, _ = self.forward_attention(x)
        x = self.decoder(ego_embedded_att)
        advantage = self.advantage(x)
        value = self.value(x).expand(-1,  self.num_actions)
        return value + advantage - advantage.mean(1).unsqueeze(1).expand(-1,  self.num_actions)

    def split_input(self, x, mask=None):
        # Dims: batch, entities, features
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            mask = x[:, :, 0:1] < 0.5
        return ego, others, mask

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego, others = self.ego_embedding(ego), self.others_embedding(others)
        return self.attention_layer(ego, others, mask)

    def get_attention_matrix(self, x):
        _, attention_matrix = self.forward_attention(x)
        return attention_matrix



class Dueling_DQN_vector(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(Dueling_DQN_vector, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_outputs
        
        self.features = nn.Sequential(
            nn.Linear(self.input_shape, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()


class attention_Dueling_DQN(nn.Module):
    def __init__(self, im_shape=[4,150,600], num_outputs=5):
        super(attention_Dueling_DQN, self).__init__()
        
        self.input_shape = im_shape
        self.num_actions = num_outputs

        self.features = nn.Sequential(
            nn.Conv2d(im_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        ##加入nonloacl-attention部分
        self.attention = NONLocalBlock2D(64, sub_sample=False, bn_layer=False)


        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = x.reshape(x.size(0), -1)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage - advantage.mean()
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).reshape(1, -1).size(1)


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        #不同的卷积，对应不同格式的数据需求
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        #权重g
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        #输出层w
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        #theat
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        #phi
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class MultiAttentionNetwork(EgoAttentionNetwork, attention_Dueling_DQN):
    def __init__(self, input_shape, img_shape, num_outputs):
        super(MultiAttentionNetwork, self).__init__()
        #EgoAttentionNetwork.__init__(self, input_shape=input_shape, num_outputs=num_outputs)
        #attention_Dueling_DQN.__init__(self, img_shape, num_outputs)
        self.vec_shape = input_shape
        self.img_shape = img_shape
        self.num_actions = num_outputs
        self.heads = 2

        # 向量attention部分
        self.ego_embedding = nn.Sequential(
            nn.Linear(self.vec_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.others_embedding = nn.Sequential(
            nn.Linear(self.vec_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.attention_layer = EgoAttetion(64, self.heads)

        # 图像attention部分
        self.features = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.attention = NONLocalBlock2D(64, sub_sample=False, bn_layer=False)

        self.decoder = nn.Sequential(
            nn.Linear(68224, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.advantage = nn.Linear(128, self.num_actions)

        self.value = nn.Linear(128, 1)

    def forward(self, vec_x, img_x):
        ego_embedded_att, _ = self.forward_attention(vec_x)
        feature = self.features(img_x)
        img_attention = self.attention(feature)
        img_attention = img_attention.reshape(img_attention.size(0), -1)
        attention = torch.cat([ego_embedded_att, img_attention], dim=1)
        x = self.decoder(attention)
        advantage = self.advantage(x)
        value = self.value(x).expand(-1, self.num_actions)
        return value + advantage - advantage.mean(1).unsqueeze(1).expand(-1, self.num_actions)
