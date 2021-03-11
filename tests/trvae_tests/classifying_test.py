import scanpy as sc
import torch
import os
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import numpy as np


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

test_nr = 4
condition_key = "condition"
cell_type_key = "final_annotation"
n_labelled_samples_per_class = 500

if test_nr == 1:
    reference = ['10X']
    query = ['Oetjen', 'Sun', 'Freytag']
elif test_nr == 2:
    reference = ['10X', 'Oetjen']
    query = ['Sun', 'Freytag']
elif test_nr == 3:
    reference = ['10X', 'Oetjen', 'Sun']
    query = ['Freytag']
elif test_nr == 4:
    reference = ['10X', 'Oetjen', 'Sun','Freytag']
    query = []

adata_all = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/Immune_ALL_human_wo_villani_rqr_normalized_hvg.h5ad'))
adata = adata_all.raw.to_adata()
adata = remove_sparsity(adata)
source_adata = adata[adata.obs.study.isin(reference)]
target_adata = adata[adata.obs.study.isin(query)]

labeled_ind = []
unlabeled_ind = []
un_labels_r = source_adata[source_adata.obs.study != '10X'].obs[cell_type_key].unique().tolist()
print(un_labels_r)
for label in un_labels_r:
    mask = source_adata.obs[cell_type_key] == label
    mask = mask.tolist()
    idx = np.where(mask)[0]
    np.random.shuffle(idx)
    labeled_ind += idx[:n_labelled_samples_per_class].tolist()
    unlabeled_ind += idx[n_labelled_samples_per_class:].tolist()

print(len(labeled_ind))
print(len(unlabeled_ind))

tranvae = sca.models.TRANVAE.load(
    dir_path=os.path.expanduser(f'~/Documents/reference_model'),
    adata=source_adata
)

preds, probs = tranvae.classify(version='dist')
print('Distance Classifier:', np.mean(preds == source_adata.obs[cell_type_key]))
preds, probs = tranvae.classify()
print('Probability Classifier:', np.mean(preds == source_adata.obs[cell_type_key]))
correct_probs = probs[preds == source_adata.obs[cell_type_key]]
incorrect_probs = probs[preds != source_adata.obs[cell_type_key]]
data = [correct_probs, incorrect_probs]


fig, ax = plt.subplots()
ax.set_title('Default violin plot')
ax.set_ylabel('Observed values')
ax.violinplot(data)
labels = ['Correct', 'Incorrect']
set_axis_style(ax, labels)
plt.show()

x,y,c,p = tranvae.get_landmarks_info()
print(p)
print(y)
y_l = np.unique(y).tolist()
c_l = np.unique(c).tolist()
y_uniq = source_adata.obs[cell_type_key].unique().tolist()
y_uniq_m = tranvae.cell_types_


data_latent = tranvae.get_latent()
data_extended = np.concatenate((data_latent, x))
adata_latent = sc.AnnData(data_extended)
adata_latent.obs['celltype'] = source_adata.obs[cell_type_key].tolist() + y.tolist()
adata_latent.obs['batch'] = source_adata.obs[condition_key].tolist() + c.tolist()
adata_latent.obs['predictions'] = preds.tolist() + y.tolist()

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent,
           color=['batch'],
           groups=c_l,
           frameon=False,
           wspace=0.6,
           size=50,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/umap_ref_tranvae_batch_l.png'), bbox_inches='tight')

sc.pl.umap(adata_latent,
           color=['celltype'],
           groups=y_l,
           frameon=False,
           wspace=0.6,
           size=50,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/umap_ref_tranvae_ct_l.png'), bbox_inches='tight')

sc.pl.umap(adata_latent,
           color=['celltype'],
           groups=y_uniq,
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/umap_ref_tranvae_ct.png'), bbox_inches='tight')

sc.pl.umap(adata_latent,
           color=['predictions'],
           groups=y_uniq_m,
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/umap_ref_tranvae_pred.png'), bbox_inches='tight')
