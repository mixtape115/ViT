# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

print("===========2-3 Input Layer==========")

class VitInputLayer(nn.Module):
    def __init__(self, in_channels:int=3, emb_dim:int=384, num_patch_row:int=2, image_size:int=32):
        """
        引数:
            in_channels : 入力画像のチャネル数
            emb_dims : 埋め込み後のベクトルの長さ.  = クラストークンの長さ
            num_patch_row : 高さ方向のパッチ数. 例は2*2であるため、2をデフォルト値とした.
            image_size : 入力画像の1辺の大きさ. 入力画像の大きさと幅は同じであると仮定
        """
        super(VitInputLayer, self).__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        # パッチの数
        ## 例 : 入力画像を2*2 のパッチに分ける場合. num_patchは4
        self.num_patch = self.num_patch_row**2

        # パッチの大きさ

        ## 例: 入力画像の一変の大きさが32の場合, patch_sizeは16
        self.patch_size = int(self.image_size//self.num_patch_row)

        # 入力画像のパッチへの分割 & パッチの埋め込みを一気に行う層
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # クラストークン ---　学習可能なパラメーター
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_dim)
        )

        # 位置埋め込み --- パッチの位置情報を学ぶ
        ## クラストークンが先頭に結合されているため,
        ## 長さemb_dimの位置埋め込みベクトルを(バッチ数+1)個用意
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch+1, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: 入力画像. 形状は(B, C, H, W). 式1
                B: バッチサイズ, C: チャネル数, H: 高さ, W: 幅
        
        返り値: 
            z_0: ViTへの入力. 形状は(B, N, D)
            B: バッチサイズ. N: トークン数, D: 埋め込みベクトルの長さ
        """
        # パッチの埋め込み & flatten 式3
        ## パッチの埋め込み (B, C, H, W) -> (B, D, H/P, W/P)
        ## ここで, Pはパッチ1辺の大きさ
        z_0 = self.patch_emb_layer(x)

        ## パッチのflatten(B, D, H/P, W/P) -> (B, D, Np)
        ## ここで, Npはパッチの数(=H*W/P**2)
        print('-----------------')
        # torch.Size([2, 384, 2, 2])
        print(z_0.shape)
        # z_0.flatten(2)はz_0の3番目以降をflattenしますといういみ.たぶん.
        z_0 = z_0.flatten(2)
        # torch.Size([2, 384, 4])
        print(z_0.shape)
        print('-----------------')


        ## 軸の入れ替え (B, D, Np) -> (B, Np, D)
        z_0 = z_0.transpose(1, 2)

        # パッチの埋め込みの先頭にクラストークンを結合 式4
        ## (B, Np, D) -> (B, N, D)
        ## N = (Np+1)
        ## また, cls_tokenの形状は(1, 1, D)
        ## respectメソッドによって(B, 1, D)に変換してからパッチの埋め込みとの結合をおこまう
        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1)

        
        # 位置埋め込みのかさん 式5
        ## (B, N, D) -> (B, N, D)
        z_0 = z_0 + self.pos_emb
        return z_0

batch_size, channel, height, width = 2, 3, 32, 32
x = torch.randn(batch_size, channel, height, width)
input_layer = VitInputLayer(num_patch_row=2)
z_0=input_layer(x)

# (2, 5, 384)(=(B, N, D))になっていることを確認
print(z_0.shape)

