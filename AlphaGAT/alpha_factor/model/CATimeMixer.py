import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from indicator.model.Norm import Normalize


class DFT_series_decomp(nn.Module):
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class BottomUPConv1D(nn.Module):
    def __init__(self, configs):
        super(BottomUPConv1D, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )
        self.use_cov1d = configs.use_cov1d
        if self.use_cov1d:
            self.up_cov1d_layers = torch.nn.ModuleList(
                [
                    nn.Conv1d(in_channels=configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                              out_channels=configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                              kernel_size=(i+2))
                    for i in range(configs.down_sampling_layers)
                ])

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
        out_high_res_list = [out_high]
        BS = out_low.shape[0]
        D = out_low.shape[1]
        for i in range(len(season_list) - 1):
            if self.use_cov1d:
                for j in range(len(out_high_res_list)):
                    out_high_res_list[j] = self.down_sampling_layers[i]((out_high_res_list[j]))

                out_high_res_list.append(out_low)
                stack_tensor = torch.stack(out_high_res_list, dim=-1)
                stack_tensor = rearrange(stack_tensor, 'b s l i -> (b s) l i')
                out_low = self.up_cov1d_layers[i](stack_tensor)
                out_low = out_low.reshape(BS, D, -1)
                out_high_res_list[-1] = out_low
                out_season_list.append(out_low.permute(0, 2, 1))
                if i + 2 <= len(season_list) - 1:
                    out_low = season_list[i + 2]
            else:
                out_low_res = self.down_sampling_layers[i](out_high)
                out_low = out_low + out_low_res
                out_high = out_low
                if i + 2 <= len(season_list) - 1:
                    out_low = season_list[i + 2]
                out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class TopDownConv1D(nn.Module):
    def __init__(self, configs):
        super(TopDownConv1D, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

        self.use_cov1d  =configs.use_cov1d
        if self.use_cov1d:
            self.up_cov1d_layers = torch.nn.ModuleList(
                [
                    ##Conv1d
                    nn.Conv1d(in_channels=configs.seq_len // (configs.down_sampling_window ** i), out_channels=configs.seq_len // (configs.down_sampling_window ** i), kernel_size=(configs.down_sampling_layers - i)+1)
                    for i in reversed(range(configs.down_sampling_layers))
                ])

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]
        out_high_res_list = [out_low]
        BS = out_low.shape[0]
        D = out_low.shape[1]
        for i in range(len(trend_list_reverse) - 1):
            if self.use_cov1d:
                for j in range(len(out_high_res_list)):
                    out_high_res_list[j] = self.up_sampling_layers[i]((out_high_res_list[j]))
                out_high_res_list.append(out_high)
                stack_tensor = torch.stack(out_high_res_list, dim=-1)
                stack_tensor = rearrange(stack_tensor, 'b s l i -> (b s) l i')
                out_high = self.up_cov1d_layers[i](stack_tensor)
                out_high = out_high.reshape(BS,D, -1 )
                out_high_res_list[-1] = out_high
                out_trend_list.append(out_high.permute(0, 2, 1))
                if i + 2 <= len(trend_list_reverse) - 1:
                    out_high = trend_list_reverse[i + 2]

            else:
                out_high_res = self.up_sampling_layers[i](out_low)
                out_high = out_high + out_high_res
                out_low = out_high
                if i + 2 <= len(trend_list_reverse) - 1:
                    out_high = trend_list_reverse[i + 2]
                out_trend_list.append(out_low.permute(0, 2, 1))
        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence
        self.decompsition = series_decomp(configs.moving_avg)


        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = BottomUPConv1D(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = TopDownConv1D(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class CrossAssetAttention(nn.Module):
    def __init__(self, configs):
        super(CrossAssetAttention, self).__init__()
        self.d_model = configs.d_model
        self.stock_num = configs.stock_num
        self.enc_in = configs.enc_in
        self.head_num = configs.head_num
        self.d_k = self.d_model // self.head_num

        self.query_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                self.d_model,
                self.d_model,
            )
            for _ in range(configs.down_sampling_layers + 1)
        ])
        self.key_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                self.d_model,
                self.d_model,
            )
            for _ in range(configs.down_sampling_layers + 1)
        ])
        self.value_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                self.d_model,
                self.d_model,
            )
            for _ in range(configs.down_sampling_layers + 1)
        ])

        self.out_layer = torch.nn.ModuleList([
            torch.nn.Linear(
                self.d_model,
                configs.seq_len // (configs.down_sampling_window ** i) * self.d_model * self.enc_in
            )
            for i in range(configs.down_sampling_layers + 1)
        ])

        self.scale_list = [torch.sqrt(torch.FloatTensor([self.d_k])).to(configs.device)
                           for _ in range(configs.down_sampling_layers + 1)]

    def forward(self, x_list, B):
        for i, x in zip(range(len(x_list)), x_list):
            l = x.shape[1]
            d = x.shape[2]

            # Reshape x
            x = rearrange(x, '(b s) l d -> b s l d', b=B, s=self.stock_num)
            x = x.permute(0, 2, 1, 3).contiguous()
            x = rearrange(x, 'b l s d -> (b l) s d')

            # Apply linear layers and split into multiple heads
            query_i = self.query_layers[i](x).view(-1, x.shape[1], self.heads, self.d_k)
            key_i = self.key_layers[i](x).view(-1, x.shape[1], self.heads, self.d_k)
            value_i = self.value_layers[i](x).view(-1, x.shape[1], self.heads, self.d_k)

            # Transpose to get dimensions (batch, heads, seq_len, d_k)
            query_i = query_i.permute(0, 2, 1, 3)
            key_i = key_i.permute(0, 2, 1, 3)
            value_i = value_i.permute(0, 2, 1, 3)

            # Calculate attention scores
            attn_scores = torch.matmul(query_i, key_i.transpose(-2, -1)) / self.scale_list[i]
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Apply attention weights to value
            attn_output = torch.matmul(attn_weights, value_i)

            # Concatenate heads
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
            attn_output = attn_output.view(-1, attn_output.shape[1], self.d_model)

            # Reshape back to original dimensions
            attn_output = rearrange(attn_output, '(b l) s d -> b l s d', b=B, l=l)
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
            attn_output = rearrange(attn_output, 'b s l d -> (b s) l d')

            x_list[i] += attn_output

        return x_list





class CATimeMixer(nn.Module):
    def __init__(self, configs):
        super(CATimeMixer, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])
        self.use_attention = configs.use_attention
        self.attention_blocks = nn.ModuleList([CrossAssetAttention(configs)
                                               for _ in range(configs.e_layers)])
        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.enc_embedding_linear = nn.Linear(configs.enc_in, configs.d_model)
        self.enc_embedding_drop = nn.Dropout(p=configs.dropout)
        self.layer = configs.e_layers
        if self.configs.down_sampling_method == 'max':
            self.down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            self.down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                       kernel_size=3, padding=padding,
                                       stride=self.configs.down_sampling_window,
                                       padding_mode='circular',
                                       bias=False)
        else:
            raise ValueError('Downsampling method is error,only supporting the max, avg, conv1D')
        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        self.alpha_generator = nn.Linear(configs.d_model, configs.alphas)
        self.stock_num = configs.stock_num
        self.alphas = configs.alphas

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc):
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = self.down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        return x_enc

    def forecast(self, x_enc):

        # x_enc:  [batch,catch_len, stock_num, input_dim]
        batch_size = x_enc.shape[0];
        x_enc = x_enc.permute(0, 2, 1, 3);  # x: [B,L,  Z, M] -> [B, Z, M, L]
        x_enc = rearrange(x_enc, 'b s l i -> (b s) l i')  ##[B, Z, M, L] -> [B * Z, M, L]
        #####x_enc  batch  Len channels
        x_enc = self.__multi_scale_process_inputs(x_enc)
        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        for i, x in zip(range(len(x_list[0])), x_list[0]):
            enc_out = self.enc_embedding_drop(self.enc_embedding_linear(x))
            enc_out_list.append(enc_out)
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
            if self.use_attention:
                enc_out_list = self.attention_blocks[i](enc_out_list, batch_size)
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.alpha_generator(dec_out)
                ####dec_out  [B*S, 1, alphafactors ]
                dec_out = dec_out.squeeze(1)
                dec_out = dec_out.reshape(-1, self.stock_num, self.alphas).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc = None, x_dec = None, x_mark_dec = None, mask=None):
        alpha_factors = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        ##dec_out_list  batch , stocknum alphafactors
        alpha_factors = alpha_factors.permute(0, 2, 1).contiguous()
        return alpha_factors

