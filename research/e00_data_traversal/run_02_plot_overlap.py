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

import os
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

import research.util as H
from disent.dataset import DisentDataset
from disent.dataset.data import Cars3dData
from disent.dataset.data import DSpritesData
from disent.dataset.data import DSpritesImagenetData
from disent.dataset.data import GroundTruthData
from disent.dataset.data import SelfContainedHdf5GroundTruthData
from disent.dataset.data import Shapes3dData
from disent.dataset.data import SmallNorbData
from disent.dataset.data import XYBlocksData
from disent.dataset.data import XYObjectData
from disent.dataset.data import XYObjectShadedData
from disent.dataset.data import XYSquaresData
from disent.util.seeds import TempNumpySeed


# ========================================================================= #
# core                                                                      #
# ========================================================================= #


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.shape[-1] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    assert img.shape[-1] == 3, f'last channel of array is not of size 3 for RGB, got shape: {tuple(img.shape)}'
    return img


def plot_dataset_traversals(
    gt_data: GroundTruthData,
    f_idxs=None,
    num_cols: Optional[int] = 8,
    base_factors=None,
    add_random_traversal=True,
    pad=8,
    bg_color=127,
    border=False,
    rel_path=None,
    save=True,
    seed=777,
    plt_scale=4.5,
    offset=0.75
):
    # convert
    dataset = DisentDataset(gt_data)
    f_idxs = gt_data.normalise_factor_idxs(f_idxs)
    num_cols = num_cols if (num_cols is not None) else min(max(gt_data.factor_sizes), 32)
    # get traversal grid
    row_labels = [gt_data.factor_names[i] for i in f_idxs]
    grid, _, _ = H.visualize_dataset_traversal(
        dataset=dataset,
        data_mode='raw',
        factor_names=f_idxs,
        num_frames=num_cols,
        seed=seed,
        base_factors=base_factors,
        traverse_mode='interval',
        pad=pad,
        bg_color=bg_color,
        border=border,
    )
    # add random traversal
    if add_random_traversal:
        with TempNumpySeed(seed):
            row_labels = ['random'] + row_labels
            row = dataset.dataset_sample_batch(num_samples=num_cols, mode='raw')[None, ...]  # torch.Tensor
            grid = np.concatenate([ensure_rgb(row), grid])
    # make figure
    factors, frames, _, _, c = grid.shape
    assert c == 3
    fig, axs = H.plt_subplots_imshow(grid, label_size=18, title_size=24, title=gt_data.name, row_labels=row_labels, subplot_padding=None, figsize=(offset + (1/2.54)*frames*plt_scale, (1/2.54)*(factors+0.45)*plt_scale))
    # save figure
    if save and (rel_path is not None):
        path = H.make_rel_path_add_ext(rel_path, ext='.png')
        plt.savefig(path)
        print(f'saved: {repr(path)}')
    plt.show()
    # done!
    return fig, axs


# ========================================================================= #
# entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../gadfly.mplstyle'))

    # options
    all_squares = True
    add_random_traversal = True
    num_cols = 7
    seed = 47

    # get name
    prefix = 'traversal' if add_random_traversal else 'traversal-noran'

    # save images
    for i in ([1, 2, 4, 8] if all_squares else [1, 8]):
        data = XYSquaresData(grid_spacing=i, grid_size=8, no_warnings=True)
        plot_dataset_traversals(data, rel_path=f'plots/{prefix}__xy-squares__spacing{i}', seed=seed-40, add_random_traversal=add_random_traversal, num_cols=num_cols)
        plot_dataset_traversals(data, rel_path=f'plots/{prefix}__xy-squares__spacing{i}__some', seed=seed-40, add_random_traversal=add_random_traversal, num_cols=num_cols, f_idxs=[0, 3])

    plot_dataset_traversals(XYObjectData(),                  rel_path=f'plots/{prefix}__xy-object',                seed=seed, add_random_traversal=add_random_traversal, num_cols=num_cols)
    plot_dataset_traversals(XYObjectShadedData(),            rel_path=f'plots/{prefix}__xy-object-shaded',         seed=seed, add_random_traversal=add_random_traversal, num_cols=num_cols)
    plot_dataset_traversals(XYBlocksData(),                  rel_path=f'plots/{prefix}__xy-blocks',                seed=seed, add_random_traversal=add_random_traversal, num_cols=num_cols)
    plot_dataset_traversals(Shapes3dData(),                  rel_path=f'plots/{prefix}__shapes3d',                 seed=seed, add_random_traversal=add_random_traversal, num_cols=num_cols)
    plot_dataset_traversals(DSpritesData(),                  rel_path=f'plots/{prefix}__dsprites',                 seed=seed, add_random_traversal=add_random_traversal, num_cols=num_cols)
    plot_dataset_traversals(DSpritesImagenetData(100, 'bg'), rel_path=f'plots/{prefix}__dsprites-imagenet-bg-100', seed=seed-6, add_random_traversal=add_random_traversal, num_cols=num_cols)
    plot_dataset_traversals(DSpritesImagenetData( 50, 'bg'), rel_path=f'plots/{prefix}__dsprites-imagenet-bg-50',  seed=seed-6, add_random_traversal=add_random_traversal, num_cols=num_cols)
    plot_dataset_traversals(DSpritesImagenetData(100, 'fg'), rel_path=f'plots/{prefix}__dsprites-imagenet-fg-100', seed=seed-6, add_random_traversal=add_random_traversal, num_cols=num_cols)
    plot_dataset_traversals(DSpritesImagenetData( 50, 'fg'), rel_path=f'plots/{prefix}__dsprites-imagenet-fg-50',  seed=seed-6, add_random_traversal=add_random_traversal, num_cols=num_cols)
    plot_dataset_traversals(SmallNorbData(),                 rel_path=f'plots/{prefix}__smallnorb',                seed=seed, add_random_traversal=add_random_traversal, num_cols=num_cols)
    plot_dataset_traversals(Cars3dData(),                    rel_path=f'plots/{prefix}__cars3d',                   seed=seed, add_random_traversal=add_random_traversal, num_cols=num_cols)

    BASE = os.path.abspath(os.path.join(__file__, '../../../out/adversarial_data_approx'))

    for folder in [
        # 'const' datasets
        ('2021-08-18--00-58-22_FINAL-dsprites_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('2021-08-18--01-33-47_FINAL-shapes3d_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('2021-08-18--02-20-13_FINAL-cars3d_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('2021-08-18--03-10-53_FINAL-smallnorb_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        # 'invert' datasets
        ('2021-08-18--03-52-31_FINAL-dsprites_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('2021-08-18--04-29-25_FINAL-shapes3d_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('2021-08-18--05-13-15_FINAL-cars3d_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('2021-08-18--06-03-32_FINAL-smallnorb_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        # stronger 'invert' datasets
        ('2021-09-06--00-29-23_INVERT-VSTRONG-shapes3d_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06'),
        ('2021-09-06--03-17-28_INVERT-VSTRONG-dsprites_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06'),
        ('2021-09-06--05-42-06_INVERT-VSTRONG-cars3d_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06'),
        ('2021-09-06--09-10-59_INVERT-VSTRONG-smallnorb_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06'),
    ]:
        plot_dataset_traversals(SelfContainedHdf5GroundTruthData(f'{BASE}/{folder}/data.h5'), rel_path=f'plots/{prefix}__{folder}.png', seed=seed, add_random_traversal=add_random_traversal, num_cols=num_cols)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
