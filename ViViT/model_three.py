import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.ops import StochasticDepth
from einops.layers.torch import Rearrange
import numpy as np
from torch.utils.data import Dataset

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import numpy as np
import pandas as pd
from pathlib import Path
import os
import os.path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score
from numbers import Number
import sys
from PIL import Image
import matplotlib.pyplot as plt
import imutils
from scipy import stats
import csv
from scipy.spatial import distance as dist
from imutils import perspective
import cv2
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import glob
import imageio
import math
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
      while True:
          ret, frame = cap.read()
          if not ret:
              break
          #frame = crop_center_square(frame)
          frame = cv2.resize(frame, resize)
          frame = frame[:, :, [2, 1, 0]]
          frames.append(frame)
          if len(frames) == max_frames:
              break
  finally:
      cap.release()
  return np.array(frames)

def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        y = y.unsqueeze(1)
        # 1. Forward pass
        y_pred = model(X.float())
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred.float(), y.float())
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    return train_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.float().to(device)

            y = y.unsqueeze(1)
            # 1. Forward pass
            test_pred_logits = model(X.float())

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()


    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    
    return test_loss


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "test_loss": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"test_loss: {test_loss:.4f} | "

        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    # Return the filled results at the end of the epochs
    return results

class CustomPandasDataset(Dataset):
  def __init__(self, dataframe, transform=None):
    self.data = dataframe
    self.labels = self.data['weights']
    self.directories = "/home/amun/ViViT_v2/ResizedVideo/" + self.data['file_names']
    
  def __getitem__(self, index):
    directory = self.directories[index]
    video = load_video(directory)
    video_tensor = torch.tensor(video)
    video_tensor = video_tensor.permute(0,3,1,2)
    label = self.labels[index]
    return video_tensor, label
  
  def __len__(self):
    return len(self.labels)

def pair(t):
    """
    Parameters
    ----------
    t: tuple[int] or int
    """
    return t if isinstance(t, tuple) else (t, t)

class BaseClassificationModel(nn.Module):
    """

    Parameters
    -----------
    img_size: int
        Size of the image
    patch_size: int or tuple(int)
        Size of the patch
    in_channels: int
        Number of channels in input image
    pool: str
        Feature pooling type, must be one of {``mean``, ``cls``}
    """

    def __init__(self, img_size, patch_size, in_channels=3, pool="cls"):
        super(BaseClassificationModel, self).__init__()

        img_height, img_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            img_height % patch_height == 0 and img_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (img_height // patch_height) * (img_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patches = num_patches
        self.patch_dim = patch_dim

        assert pool in {
            "cls",
            "mean",
        }, "Feature pooling type must be either cls (cls token) or mean (mean pooling)"
        self.pool = pool

class MLPDecoder(nn.Module):
    """
    Parameters
    ----------
    config : int or tuple or list
        Configuration of the hidden layer(s)
    n_classes : int
        Number of classes for classification
    """

    def __init__(self, config=(1024,), n_classes=10):
        super(MLPDecoder, self).__init__()

        self.decoder = nn.ModuleList()

        if not isinstance(config, list) and not isinstance(config, tuple):
            config = [config]

        if len(config) > 1:
            for i in range(len(config) - 1):

                self.decoder.append(nn.LayerNorm(config[i]))
                self.decoder.append(nn.Linear(config[i], config[i + 1]))

        self.decoder.append(nn.LayerNorm(config[-1]))
        self.decoder.append(nn.Linear(config[-1], n_classes))

        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns output tensor of size `n_classes`, Note that `torch.nn.Softmax` is not applied to the output tensor.

        """
        return self.decoder(x)

class LinearVideoEmbedding(nn.Module):
    """

    Parameters
    -----------
    embedding_dim: int
        Dimension of the resultant embedding
    patch_height: int
        Height of the patch
    patch_width: int
        Width of the patch
    patch_dim: int
        patch_dimension

    """

    def __init__(
        self,
        embedding_dim,
        patch_height,
        patch_width,
        patch_dim,
    ):

        super().__init__()
        self.patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h ph) (w pw) -> b t (h w) (ph pw c)",
                ph=patch_height,
                pw=patch_width,
            ),
            nn.Linear(patch_dim, embedding_dim),
        )

    def forward(self, x):
        """

        Parameters
        -----------
        x: torch.Tensor
            Input tensor

        Returns
        ----------
        torch.Tensor
            Returns patch embeddings of size `embedding_dim`

        """

        return self.patch_embedding(x)


#
class TubeletEmbedding(nn.Module):
    """

    Parameters
    ----------
    embedding_dim: int
        Dimension of the resultant embedding
    tubelet_t: int
        Temporal length of single tube/patch
    tubelet_h: int
        Heigth  of single tube/patch
    tubelet_w: int
        Width of single tube/patch
    in_channels: int
        Number of channels
    """

    def __init__(self, embedding_dim, tubelet_t, tubelet_h, tubelet_w, in_channels):
        super(TubeletEmbedding, self).__init__()
        tubelet_dim = in_channels * tubelet_h * tubelet_w * tubelet_t
        self.tubelet_embedding = nn.Sequential(
            Rearrange(
                "b  (t pt) c (h ph) (w pw) -> b t (h w) (pt ph pw c)",
                pt=tubelet_t,
                ph=tubelet_h,
                pw=tubelet_w,
            ),
            nn.Linear(tubelet_dim, embedding_dim),
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x: Torch.tensor
            Input tensor

        """
        return self.tubelet_embedding(x)
    
class PosEmbedding(nn.Module):
    """
    Generalised Positional Embedding class
    """

    def __init__(self, shape, dim, drop=None, sinusoidal=False, std=0.02):
        super(PosEmbedding, self).__init__()

        if not sinusoidal:
            if isinstance(shape, int):
                shape = [1, shape, dim]
            else:
                shape = [1] + list(shape) + [dim]
            self.pos_embed = nn.Parameter(torch.zeros(shape))

        else:
            pe = torch.FloatTensor(
                [
                    [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                    for p in range(shape)
                ]
            )
            pe[:, 0::2] = torch.sin(pe[:, 0::2])
            pe[:, 1::2] = torch.cos(pe[:, 1::2])
            self.pos_embed = pe
            self.pos_embed.requires_grad = False
        nn.init.trunc_normal_(self.pos_embed, std=std)
        self.pos_drop = nn.Dropout(drop) if drop is not None else nn.Identity()

    def forward(self, x):
        x = x + self.pos_embed
        return self.pos_drop(x)
    
class VanillaSelfAttention(nn.Module):
    """
    Vanilla O(:math:`n^2`) Self attention introduced in `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Parameters
    -----------
    dim: int
        Dimension of the embedding
    num_heads: int
        Number of the attention heads
    head_dim: int
        Dimension of each head
    p_dropout: float
        Dropout Probability

    """

    def __init__(self, dim, num_heads=8, head_dim=64, p_dropout=0.0):
        super().__init__()

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == dim)

        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying self-attention on input tensor

        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)
    
class PreNorm(nn.Module):
    """
    Parameters
    ----------
    dim: int
        Dimension of the embedding
    fn:nn.Module
        Attention class
    context_dim: int
        Dimension of the context array used in cross attention
    """

    def __init__(self, dim, fn, context_dim=None):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.context_norm = (
            nn.LayerNorm(context_dim) if context_dim is not None else None
        )
        self.fn = fn

    def forward(self, x, **kwargs):
        if "context" in kwargs.keys() and kwargs["context"] is not None:
            normed_context = self.context_norm(kwargs["context"])
            kwargs.update(context=normed_context)
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """

    Parameters
    ----------
    dim: int
        Dimension of the input tensor
    hidden_dim: int, optional
        Dimension of hidden layer
    out_dim: int, optional
        Dimension of the output tensor
    p_dropout: float
        Dropout probability, default=0.0

    """

    def __init__(self, dim, hidden_dim=None, out_dim=None, p_dropout=0.0):
        super().__init__()

        out_dim = out_dim if out_dim is not None else dim
        hidden_dim = hidden_dim if hidden_dim is not None else dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------

        torch.Tensor
            Returns output tensor by performing linear operations and activation on input tensor

        """

        return self.net(x)
    
class VanillaEncoder(nn.Module):
    """

    Parameters
    ----------
    embedding_dim: int
        Dimension of the embedding
    depth: int
        Number of self-attention layers
    num_heads: int
        Number of the attention heads
    head_dim: int
        Dimension of each head
    mlp_dim: int
        Dimension of the hidden layer in the feed-forward layer
    p_dropout: float
        Dropout Probability
    attn_dropout: float
        Dropout Probability
    drop_path_rate: float
        Stochastic drop path rate
    """

    def __init__(
        self,
        embedding_dim,
        depth,
        num_heads,
        head_dim,
        mlp_dim,
        p_dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.0,
        drop_path_mode="batch",
    ):
        super().__init__()

        self.encoder = nn.ModuleList([])
        for _ in range(depth):
            self.encoder.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim=embedding_dim,
                            fn=VanillaSelfAttention(
                                dim=embedding_dim,
                                num_heads=num_heads,
                                head_dim=head_dim,
                                p_dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(
                            dim=embedding_dim,
                            fn=FeedForward(
                                dim=embedding_dim,
                                hidden_dim=mlp_dim,
                                p_dropout=p_dropout,
                            ),
                        ),
                    ]
                )
            )
        self.drop_path = (
            StochasticDepth(p=drop_path_rate, mode=drop_path_mode)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor

        Returns
        ----------
        torch.Tensor
            Returns output tensor
        """
        for attn, ff in self.encoder:
            x = attn(x) + x
            x = self.drop_path(ff(x)) + x

        return x

class ViViTEncoderBlock(nn.Module):
    """For model 3 only"""

    def __init__(
        self, dim, num_heads, head_dim, p_dropout, out_dim=None, hidden_dim=None
    ):
        super(ViViTEncoderBlock, self).__init__()

        self.temporal_attention = PreNorm(
            dim=dim, fn=VanillaSelfAttention(dim, num_heads, head_dim, p_dropout)
        )
        self.spatial_attention = PreNorm(
            dim=dim, fn=VanillaSelfAttention(dim, num_heads, head_dim, p_dropout)
        )

        self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim, out_dim=out_dim)

    def forward(self, x):

        b, n, s, d = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=1)  # 1×nt·nh·nw·d --> nt×nh·nw·d

        x = self.spatial_attention(x) + x

        x = x.reshape(b, n, s, d).transpose(1, 2)
        x = torch.flatten(x, start_dim=0, end_dim=1)  # nt×nh·nw·d --> nh·nw×nt·d

        x = self.temporal_attention(x) + x

        x = self.mlp(x) + x

        x = x.reshape(
            b, n, s, d
        )  # reshaping because this block is used for several depths in ViViTEncoder class and Next layer will expect the x in proper shape

        return x



class ViViTEncoder(nn.Module):
    """model 3 only"""

    def __init__(
        self, dim, num_heads, head_dim, p_dropout, depth, out_dim=None, hidden_dim=None
    ):
        super(ViViTEncoder, self).__init__()
        self.encoder = nn.ModuleList()

        for _ in range(depth):
            self.encoder.append(
                ViViTEncoderBlock(
                    dim, num_heads, head_dim, p_dropout, out_dim, hidden_dim
                )
            )

    def forward(self, x):

        b = x.shape[0]

        for blk in self.encoder:
            x = blk(x)

        x = x.reshape(b, -1, x.shape[-1])

        return x
    
class ViViTModel2(BaseClassificationModel):
    """
    Model 2 implementation of: `ViViT: A Video Vision Transformer <https://arxiv.org/abs/2103.15691>`_

    Parameters
    -----------
    img_size:int
        Size of single frame/ image in video
    in_channels:int
        Number of channels
    patch_size: int
        Patch size
    embedding_dim: int
        Embedding dimension of a patch
    num_frames:int
        Number of seconds in each Video
    depth:int
        Number of encoder layers
    num_heads:int
        Number of attention heads
    head_dim:int
        Dimension of head
    n_classes:int
        Number of classes
    mlp_dim: int
        Dimension of hidden layer
    pool: str
        Pooling operation,must be one of {"cls","mean"},default is "cls"
    p_dropout:float
        Dropout probability
    attn_dropout:float
        Dropout probability
    drop_path_rate:float
        Stochastic drop path rate
    """

    def __init__(
        self,
        img_size,
        in_channels,
        patch_size,
        embedding_dim,
        num_frames,
        depth,
        num_heads,
        head_dim,
        n_classes,
        mlp_dim=None,
        pool="cls",
        p_dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.02,
    ):
        super(ViViTModel2, self).__init__(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            pool=pool,
        )

        patch_dim = in_channels * patch_size**2
        self.patch_embedding = LinearVideoEmbedding(
            embedding_dim=embedding_dim,
            patch_height=patch_size,
            patch_width=patch_size,
            patch_dim=patch_dim,
        )

        self.pos_embedding = PosEmbedding(
            shape=[num_frames, self.num_patches + 1], dim=embedding_dim, drop=p_dropout
        )

        self.space_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )  # this is similar to using cls token in vanilla vision transformer
        self.spatial_transformer = VanillaEncoder(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            p_dropout=p_dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
        )

        self.time_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.temporal_transformer = VanillaEncoder(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            p_dropout=p_dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
        )

        self.decoder = MLPDecoder(
            config=[
                embedding_dim,
            ],
            n_classes=n_classes,
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        (
            b,
            t,
            n,
            d,
        ) = x.shape  # shape of x will be number of videos,time,num_frames,embedding dim
        cls_space_tokens = repeat(self.space_token, "() n d -> b t n d", b=b, t=t)

        x = nn.Parameter(torch.cat((cls_space_tokens, x), dim=2))
        x = self.pos_embedding(x)

        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.spatial_transformer(x)
        x = rearrange(x[:, 0], "(b t) ... -> b t ...", b=b)

        cls_temporal_tokens = repeat(self.time_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.decoder(x)

        return x


# model 3
class ViViTModel3(BaseClassificationModel):
    """
    Model 3 Implementation from : `ViViT: A Video Vision Transformer <https://arxiv.org/abs/2103.15691>`_

    Parameters
    ----------
    img_size:int or tuple[int]
        size of a frame
    patch_t:int
        Temporal length of single tube/patch in tubelet embedding
    patch_h:int
        Height  of single tube/patch in tubelet embedding
    patch_w:int
        Width  of single tube/patch in tubelet embedding
    in_channels: int
        Number of input channels, default is 3
    n_classes:int
        Number of classes
    num_frames :int
        Number of seconds in each Video
    embedding_dim:int
        Embedding dimension of a patch
    depth:int
        Number of Encoder layers
    num_heads: int
        Number of attention heads
    head_dim:int
        Dimension of attention head
    p_dropout:float
        Dropout rate/probability, default is 0.0
    mlp_dim: int
        Hidden dimension, optional
    """

    def __init__(
        self,
        img_size,
        patch_t,
        patch_h,
        patch_w,
        in_channels,
        n_classes,
        num_frames,
        embedding_dim,
        depth,
        num_heads,
        head_dim,
        p_dropout,
        mlp_dim=None,
    ):

        super(ViViTModel3, self).__init__(
            in_channels=in_channels,
            patch_size=(patch_h, patch_w),
            pool="mean",
            img_size=img_size,
        )
        h, w = pair(img_size)
        self.tubelet_embedding = TubeletEmbedding(
            embedding_dim=embedding_dim,
            tubelet_t=patch_t,
            tubelet_h=patch_h,
            tubelet_w=patch_w,
            in_channels=in_channels,
        )

        self.pos_embbedding = PosEmbedding(
            shape=[num_frames // patch_t, (h * w) // (patch_w * patch_h)],
            dim=embedding_dim,
        )
        self.encoder = ViViTEncoder(
            dim=embedding_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            p_dropout=p_dropout,
            depth=depth,
            hidden_dim=mlp_dim,
        )

        self.decoder = MLPDecoder(
            config=[
                embedding_dim,
            ],
            n_classes=n_classes,
        )

    def forward(self, x):

        x = self.tubelet_embedding(x)
        x = self.pos_embbedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.decoder(x)

        return x
    
if __name__ == "__main__":
    main_directory = "/home/amun/ViViT_v2/"
    video_directory = main_directory + "ResizedVideo/"
    csv_path = main_directory + "resized.csv"
    img_size = 224
    patch_t = 4
    patch_h = 16
    patch_w = 16
    in_channels = 3
    n_classes = 1
    num_frames = 16
    embedding_dim = 192
    num_frames = 16
    depth = 4
    num_heads = 3
    head_dim = 64
    p_dropout = 0.0

    batch_size = 16
    learning_rate = 3e-3
    weight_decay=0.3

    dataframe = pd.read_csv(csv_path).sample(frac=1.0, random_state = 1).reset_index(drop=True)
    convert_dict = {'weights': int}
    dataframe = dataframe.astype(convert_dict)
    #print(dataframe)
    loaded_data = CustomPandasDataset(dataframe)
    
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    dataset_size = len(loaded_data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(loaded_data, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(loaded_data, batch_size=batch_size,
                                                    sampler=valid_sampler)


    model = ViViTModel3(img_size = img_size,
                        patch_t = patch_t,
                        patch_h = patch_h,
                        patch_w = patch_w,
                        in_channels = in_channels,
                        n_classes = n_classes,
                        num_frames = num_frames,
                        embedding_dim = embedding_dim,
                        depth = depth,
                        num_heads = num_heads,
                        head_dim = head_dim,
                        p_dropout = p_dropout)
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    #print('Trainable Parameters: %.3fM' % parameters)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper 
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                lr=learning_rate, # Base LR from Table 3 for ViT-* ImageNet-1k
                                betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                                weight_decay=weight_decay) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

    # Setup the loss function for multi-class classification
    loss_fn = torch.nn.MSELoss()
    results = train(model=model,
                    train_dataloader=train_loader,
                    test_dataloader=validation_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=74,
                    device=device)