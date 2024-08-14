import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from indicator.model.layers.Autoformer_EncDec import series_decomp
from indicator.model.layers.Embed import DataEmbedding_wo_pos
from indicator.model.layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

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


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

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

        ###leosc  addd

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

        # mixing high->low
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

                ###out_high shape[B * S * D, win ,1]
                out_low = self.up_cov1d_layers[i](stack_tensor)

                ###out_high shape[B * S , D, len]
                out_low = out_low.reshape(BS, D, -1)

                out_high_res_list[-1] = out_low

                ###out_trend_list item [B*S, win, dim]
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


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

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
                    nn.Conv1d(in_channels=configs.seq_len // (configs.down_sampling_window ** i), out_channels=configs.seq_len // (configs.down_sampling_window ** i), kernel_size=(configs.down_sampling_layers - i)+1)
                    for i in reversed(range(configs.down_sampling_layers))
                ])

    def forward(self, trend_list):

        # mixing low->high
        ####shallow copy
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        ####lsc changge   not add  cov 1d replaces
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

                ###out_high shape[B * S * D, win ,1]
                out_high = self.up_cov1d_layers[i](stack_tensor)

                ###out_high shape[B * S , D, len]
                out_high = out_high.reshape(BS,D, -1 )

                out_high_res_list[-1] = out_high

                ###out_trend_list item [B*S, win, dim]
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

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

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


class AttentionLayer(nn.Module):
    def __init__(self, configs):
        super(AttentionLayer, self).__init__()
        self.d_model = configs.d_model
        self.stock_num = configs.stock_num
        self.enc_in = configs.enc_in

        # self.embedding_layer =  torch.nn.ModuleList([
        #         torch.nn.Linear(
        #             configs.seq_len // (configs.down_sampling_window ** i)  * configs.d_model * configs.enc_in,
        #             configs.d_model
        #         )
        #         for i in range(configs.down_sampling_layers + 1)
        #     ])
        # self.dropout = nn.Dropout(configs.drop)

        # self.query_layes = torch.nn.ModuleList([
        #         torch.nn.Linear(
        #             configs.seq_len // (configs.down_sampling_window ** i)  * configs.d_model * configs.enc_in,
        #             configs.seq_len // (configs.down_sampling_window ** i)  * configs.d_model * configs.enc_in,
        #         )
        #         for i in range(configs.down_sampling_layers + 1)
        #     ])

        self.query_layes = torch.nn.ModuleList([
            torch.nn.Linear(
                configs.d_model,
                configs.d_model,
            )
            for i in range(configs.down_sampling_layers + 1)
        ])
        self.key_layes = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.d_model,
                    configs.d_model,
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

        self.value_layes = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.d_model,
                    configs.d_model,
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

        self.out_layer = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.d_model,
                    configs.seq_len // (configs.down_sampling_window ** i)  * configs.d_model * configs.enc_in
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

        self.scale_list = [torch.sqrt(torch.FloatTensor([configs.seq_len // (configs.down_sampling_window ** i)  * configs.d_model])).to(configs.device)
                      for i in range(configs.down_sampling_layers + 1)]

    def forward(self, x_list, B):

        ##x_list   []


        for i, x in zip(range(len(x_list)), x_list):
            # x = x.reshape(B,  self.stock_num, -1, )
            l = x.shape[1]
            d  =x.shape[2]
            ##x  (B*S, daywin, dim) -> (B, S, daywin, dim)
            x = rearrange(x , '(b s) l d -> b s l d', b=B, s=self.stock_num)

            ###x(B * S, daywin, dim) -> (B, day win,  S, dim)
            x = x.permute(0,2,1,3).contiguous()
            ###x(B * S, daywin, dim) -> (B* day win,  S, dim)
            x =rearrange(x , 'b l s d -> (b l) s d')




            # x =  self.dropout( self.embedding_layer[i](x))
            query_i = self.query_layes[i](x)
            key_i = self.key_layes[i](x)
            value_i = self.value_layes[i](x)

            # Calculate attention scores
            attn_scores = torch.bmm(query_i, key_i.transpose(1, 2)) / self.scale_list[i]  # Shape: (batch, len, len)

            # Apply softmax to get attention weights
            attn_weights = F.softmax(attn_scores, dim=2)  # Shape: (batch, len, len)


            ####attn_output  (B* day win,  S, dim)
            # Apply attention weights to value
            attn_output = torch.bmm(attn_weights, value_i)  # Shape: (batch, len, dim)


            ###x(B  * daywin,S, dim) -> (B , day win,  S, dim)
            attn_output = rearrange(attn_output , '(b l) s d -> b l s d', b=B, l=l)

            ###x(B , S, daywin, dim) -> (B, S ,day win,  , dim)
            attn_output = attn_output.permute(0,2,1,3).contiguous()
            ###x(B  , S, daywin, dim) -> (B*  S,DAYWIN,  dim)
            attn_output =rearrange(attn_output , 'b s l d -> (b s) l d')

            # # attn_output = self.out_layer[i](attn_output)
            # #
            # # attn_output = rearrange(attn_output , '(b l) s d -> (b s) l d', b=B, s=self.stock_num, l=l, d = d)
            #
            # attn_output = attn_output.reshape(B*self.stock_num * self.enc_in, -1 , self.d_model)


            x_list[i] += attn_output


        return x_list



class AttentionAlphaLayer(nn.Module):
    def __init__(self, configs):
        super(AttentionAlphaLayer, self).__init__()
        self.stock_num = configs.stock_num
        self.enc_in = configs.enc_in
        self.attention_dim = configs.alphas * 2
        self.embedlayer = nn.Linear(self.enc_in, self.attention_dim)
        # self.dropout = nn.Dropout(configs.drop)

        self.query = nn.Linear(self.attention_dim, self.attention_dim)
        self.key = nn.Linear(self.attention_dim, self.attention_dim)
        self.value = nn.Linear(self.attention_dim, self.attention_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(configs.device)
        self.outputlayer = nn.Linear(self.attention_dim, configs.alphas)



    def forward(self, x):
        batch_size, stock_Num, indicator = x.size()

        x = self.embedlayer(x)
        # x = self.dropout(x)
        # Compute query, key, value
        query = self.query(x)  # Shape: (batch, stock_Num, dim)
        key = self.key(x)  # Shape: (batch, stock_Num, dim)
        value = self.value(x)  # Shape: (batch, stock_Num, dim)

        # Calculate attention scores
        attn_scores = torch.bmm(query, key.transpose(1, 2)) / self.scale  # Shape: (batch, stock_Num, stock_Num)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=2)  # Shape: (batch, stock_Num, stock_Num)

        # Apply attention weights to value
        attn_output = torch.bmm(attn_weights, value)  # Shape: (batch, stock_Num, stock_Num)

        alphas = self.outputlayer(attn_output)

        return alphas




class TimeMix(nn.Module):

    def __init__(self, configs):
        super(TimeMix, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        ####add  attention
        self.use_attention = configs.use_attention
        self.attention_blocks = nn.ModuleList([AttentionLayer(configs)
                                               for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        ###change the embedding layer  indicator -> embeddings

        self.enc_embedding_linear = nn.Linear(configs.enc_in, configs.d_model)

        self.enc_embedding_drop = nn.Dropout(p=configs.dropout)



        # if self.channel_independence == 1:
        #     self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
        #                                               configs.dropout)
        # else:
        #     self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                               configs.dropout)

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

        ####预测layer
        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        ####channel_independence  默认就为1  直接将所有的channel  dim 输出为一个维度
        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(
                configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
        ####   正则化层
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )


        ###alpha generator
        # self.alpha_generator = AttentionAlphaLayer(configs)
        self.alpha_generator = nn.Linear(configs.d_model, configs.alphas)


        ####parameter for alpha
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

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = self.down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # x_enc:  [batch,catch_len, stock_num, input_dim]
        batch_size = x_enc.shape[0];
        x_enc = x_enc.permute(0, 2, 1, 3);  # x: [B,L,  Z, M] -> [B, Z, M, L]
        x_enc = rearrange(x_enc, 'b s l i -> (b s) l i')  ##[B, Z, M, L] -> [B * Z, M, L]
        #####x_enc  batch  Len channels
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)



        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                # if self.channel_independence == 1:
                #     x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                ####change
                # enc_out = self.enc_embedding(x)  # [B,T,C]
                enc_out = self.enc_embedding_drop(self.enc_embedding_linear(x))
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
            if self.use_attention:
                enc_out_list = self.attention_blocks[i](enc_out_list, batch_size)




        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)

        ####no need denorm   leosc
        # dec_out = self.normalize_layers[0](dec_out, 'denorm')


        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension

                ###dec_out = self.projection_layer(dec_out)
                ###change lesc    d_model int to alpha factoors
                ####dec_out  [B*S, 1, alphafactors ]
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

        # data:  [batch,catch_len, stock_num, input_dim]
        B, L, S, I = x_enc.shape;


        # x_enc = x_enc.permute(0, 2, 1, 3);  # x: [B,L,  Z, M] -> [B, Z, M, L]
        # x_enc = rearrange(x_enc, 'b s l i -> (b s) l i')  ##[B, Z, M, L] -> [B * Z, M, L]


        # dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        #
        # ##dec_out_list  batch * stocknum 1 indicator
        #
        #
        #
        # dec_out_list = rearrange(dec_out_list, '(b s) l i -> b s (l i)', b=B,  s=S)
        #
        # alpha_factors = self.alpha_generator(dec_out_list)
        # alpha_factors = alpha_factors.permute(0,2,1)



        alpha_factors = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        ##dec_out_list  batch , stocknum alphafactors

        alpha_factors = alpha_factors.permute(0, 2, 1).contiguous()



        return alpha_factors

