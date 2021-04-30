#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import inspect
from typing import List
from typing import Sequence
from typing import Union

import numpy as np

import torch_optimizer
from matplotlib import pyplot as plt

from disent.data.groundtruth import Cars3dData
from disent.data.groundtruth import Shapes3dData
from disent.data.groundtruth import XYSquaresData
import torch
import torch.nn.functional as F

from disent.dataset.groundtruth import GroundTruthDataset
from disent.transform import ToStandardisedTensor


# ========================================================================= #
# optimizer                                                                 #
# ========================================================================= #


def make_optimizer(model: torch.nn.Module, name: str = 'sgd', lr=1e-3):
    if isinstance(model, torch.nn.Module):
        params = model.parameters()
    elif isinstance(model, torch.Tensor):
        assert model.requires_grad
        params = [model]
    else:
        raise TypeError(f'cannot optimize type: {type(model)}')
    # make optimizer
    if name == 'sgd': return torch.optim.SGD(params, lr=lr)
    elif name == 'sgd_m': return torch.optim.SGD(params, lr=lr, momentum=0.1)
    elif name == 'adam': return torch.optim.Adam(params, lr=lr)
    elif name == 'radam': return torch_optimizer.RAdam(params, lr=lr)
    else: raise KeyError(f'invalid optimizer name: {repr(name)}')


def step_optimizer(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ========================================================================= #
# dataset                                                                   #
# ========================================================================= #


def make_dataset(name: str = 'xysquares', dataloader=False):
    if name == 'xysquares':  dataset = GroundTruthDataset(XYSquaresData(), transform=ToStandardisedTensor())
    elif name == 'xysquares_1x1':  dataset = GroundTruthDataset(XYSquaresData(square_size=1), transform=ToStandardisedTensor())
    elif name == 'xysquares_2x2':  dataset = GroundTruthDataset(XYSquaresData(square_size=2), transform=ToStandardisedTensor())
    elif name == 'xysquares_4x4':  dataset = GroundTruthDataset(XYSquaresData(square_size=4), transform=ToStandardisedTensor())
    elif name == 'cars3d':   dataset = GroundTruthDataset(Cars3dData(),    transform=ToStandardisedTensor(size=64))
    elif name == 'shapes3d': dataset = GroundTruthDataset(Shapes3dData(),  transform=ToStandardisedTensor())
    else: raise KeyError(f'invalid data name: {repr(name)}')
    return dataset


def get_single_batch(dataloader, cuda=True):
    for batch in dataloader:
        (x_targ,) = batch['x_targ']
        break
    if cuda:
        x_targ = x_targ.cuda()
    return x_targ


# ========================================================================= #
# images                                                                    #
# ========================================================================= #


def to_img(x: torch.Tensor, scale=False):
    assert x.dtype in {torch.float16, torch.float32, torch.float64, torch.complex32, torch.complex64}, f'unsupported dtype: {x.dtype}'
    x = x.detach().cpu()
    x = torch.abs(x)
    if scale:
        m, M = torch.min(x), torch.max(x)
        x = (x - m) / (M - m)
    x = torch.moveaxis(x, 0, -1)
    x = torch.clamp(x, 0, 1)
    x = (x * 255).to(torch.uint8)
    return x


def show_img(x: torch.Tensor, scale=False, i=None, step=None, show=True):
    if show:
        if (i is None) or (step is None) or (i % step == 0):
            plt.imshow(to_img(x, scale=scale))
            plt.axis('off')
            plt.tight_layout()
            plt.show()


def show_imgs(xs: Sequence[torch.Tensor], scale=False, i=None, step=None, show=True):
    if show:
        if (i is None) or (step is None) or (i % step == 0):
            n = int(np.ceil(np.sqrt(len(xs))))
            fig, axs = plt.subplots(n, n)
            for ax, im in zip(np.array(axs).flatten(), xs):
                ax.imshow(to_img(im, scale=scale))
                ax.set_axis_off()
            plt.tight_layout()
            plt.show()


# ========================================================================= #
# LOSS                                                                      #
# ========================================================================= #


def _unreduced_mse_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, targ, reduction='none')


def _unreduced_mae_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - targ)


def _unreduced_msae_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - targ) + F.mse_loss(pred, targ, reduction='none')


def unreduced_loss(pred: torch.Tensor, targ: torch.Tensor, mode='mse') -> torch.Tensor:
    return _LOSS_FNS[mode](pred, targ)


_LOSS_FNS = {
    'mse': _unreduced_mse_loss,
    'mae': _unreduced_mae_loss,
    'msae': _unreduced_msae_loss,
}


# ========================================================================= #
# LOSS                                                                      #
# ========================================================================= #


def unreduced_overlap(pred: torch.Tensor, targ: torch.Tensor, mode='mse') -> torch.Tensor:
    # -ve loss
    return -unreduced_loss(pred=pred, targ=targ, mode=mode)


def pairwise_overlap(pred: torch.Tensor, targ: torch.Tensor, mode='mse') -> torch.Tensor:
    # check input
    assert pred.shape == targ.shape
    assert pred.ndim >= 1
    # mean over final dims
    overlap = unreduced_overlap(pred=pred, targ=targ, mode=mode)
    if overlap.ndim >= 2:
        overlap = overlap.mean(dim=tuple(range(1, overlap.ndim)))
    # check result
    assert overlap.ndim == 1
    assert overlap.shape == pred.shape[:1]
    # done
    return overlap


# ========================================================================= #
# sampling helper                                                           #
# ========================================================================= #


def normalise_factor_idx(dataset, factor: Union[int, str]) -> int:
    if isinstance(factor, str):
        try:
            f_idx = dataset.factor_names.index(factor)
        except:
            raise KeyError(f'{repr(factor)} is not one of: {dataset.factor_names}')
    else:
        f_idx = factor
    assert isinstance(f_idx, int)
    assert 0 <= f_idx < dataset.num_factors
    return f_idx


def sample_factors(dataset, num_obs: int = 1024, factor_mode: str = 'sample_random', factor: Union[int, str] = None):
    # sample multiple random factor traversals
    if factor_mode == 'sample_traversals':
        assert factor is not None, f'factor cannot be None when factor_mode=={repr(factor_mode)}'
        # get traversal
        f_idx = normalise_factor_idx(dataset, factor)
        # generate traversals
        factors = []
        for i in range((num_obs + dataset.factor_sizes[f_idx] - 1) // dataset.factor_sizes[f_idx]):
            factors.append(dataset.sample_random_traversal_factors(f_idx=f_idx))
        factors = np.concatenate(factors, axis=0)
    elif factor_mode == 'sample_random':
        factors = dataset.sample_factors(num_obs)
    else:
        raise KeyError
    return factors


def sample_batch_and_factors(dataset, num_samples: int, factor_mode: str = 'sample_random', factor: Union[int, str] = None, device=None):
    factors = sample_factors(dataset, num_obs=num_samples, factor_mode=factor_mode, factor=factor)
    batch = dataset.dataset_batch_from_factors(factors, mode='target').to(device=device)
    factors = torch.from_numpy(factors).to(dtype=torch.float32, device=device)
    return batch, factors


# ========================================================================= #
# mask helper                                                               #
# ========================================================================= #


def make_changed_mask(batch, masked=True):
    if masked:
        mask = torch.zeros_like(batch[0], dtype=torch.bool)
        for i in range(len(batch)):
            mask |= (batch[0] != batch[i])
    else:
        mask = torch.ones_like(batch[0], dtype=torch.bool)
    return mask


# ========================================================================= #
# fn args helper                                                            #
# ========================================================================= #


def _get_fn_from_stack(fn_name: str, stack):
    # -- do we actually need all of this?
    fn = None
    for s in stack:
        if fn_name in s.frame.f_locals:
            fn = s.frame.f_locals[fn_name]
            break
    if fn is None:
        raise RuntimeError(f'could not retrieve function: {repr(fn_name)} from call stack.')
    return fn


def get_caller_params(sort: bool = False, exclude: Sequence[str] = None) -> dict:
    stack = inspect.stack()
    fn_name = stack[1].function
    fn_locals = stack[1].frame.f_locals
    # get function and params
    fn = _get_fn_from_stack(fn_name, stack)
    fn_params = inspect.getfullargspec(fn).args
    # check excluded
    exclude = set() if (exclude is None) else set(exclude)
    fn_params = [p for p in fn_params if (p not in exclude)]
    # sort values
    if sort:
        fn_params = sorted(fn_params)
    # return dict
    return {
        k: fn_locals[k] for k in fn_params
    }


def params_as_string(params: dict, sep: str = '_', names: bool = False):
    # get strings
    if names:
        return sep.join(f"{k}={v}" for k, v in params.items())
    else:
        return sep.join(f"{v}" for k, v in params.items())


# ========================================================================= #
# dataset indices                                                           #
# ========================================================================= #


def sample_unique_batch_indices(num_obs, num_samples) -> np.ndarray:
    assert num_obs >= num_samples, 'not enough values to sample'
    assert (num_obs - num_samples) / num_obs > 0.5, 'this method might be inefficient'
    # get random sample
    indices = set()
    while len(indices) < num_samples:
        indices.update(np.random.randint(low=0, high=num_obs, size=num_samples - len(indices)))
    # make sure indices are randomly ordered
    indices = np.fromiter(indices, dtype=int)
    # indices = np.array(list(indices), dtype=int)
    np.random.shuffle(indices)
    # return values
    return indices


def generate_epoch_batch_idxs(num_obs: int, num_batches: int, mode: str = 'shuffle') -> List[np.ndarray]:
    """
    Generate `num_batches` batches of indices.
    - Each index is in the range [0, num_obs).
    - If num_obs is not divisible by num_batches, then batches may not all be the same size.

    eg. [0, 1, 2, 3, 4] -> [[0, 1], [2, 3], [4]] -- num_obs=5, num_batches=3, sample_mode='range'
    eg. [0, 1, 2, 3, 4] -> [[1, 4], [2, 0], [3]] -- num_obs=5, num_batches=3, sample_mode='shuffle'
    eg. [0, 1, 0, 3, 2] -> [[0, 1], [0, 3], [2]] -- num_obs=5, num_batches=3, sample_mode='random'
    """
    # generate indices
    if mode == 'range':
        idxs = np.arange(num_obs)
    elif mode == 'shuffle':
        idxs = np.arange(num_obs)
        np.random.shuffle(idxs)
    elif mode == 'random':
        idxs = np.random.randint(0, num_obs, size=(num_obs,))
    else:
        raise KeyError(f'invalid mode={repr(mode)}')
    # return batches
    return np.array_split(idxs, num_batches)


def generate_epochs_batch_idxs(num_obs: int, num_epochs: int, num_epoch_batches: int, mode: str = 'shuffle') -> List[np.ndarray]:
    """
    Like generate_epoch_batch_idxs, but concatenate the batches of calling the function `num_epochs` times.
    - The total number of batches returned is: `num_epochs * num_epoch_batches`
    """
    batches = []
    for i in range(num_epochs):
        batches.extend(generate_epoch_batch_idxs(num_obs=num_obs, num_batches=num_epoch_batches, mode=mode))
    return batches


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
