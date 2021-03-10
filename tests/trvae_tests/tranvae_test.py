import scanpy as sc
import torch
import os
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import numpy as np

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

test_nr = 4
condition_key = "condition"
cell_type_key = "final_annotation"
n_labelled_samples_per_class = 500


tranvae_epochs = 500

early_stopping_kwargs = {
    "early_stopping_metric": "val_classifier_loss",
    "threshold": 0.2,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
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


tranvae = sca.models.TRANVAE(
    adata=source_adata,
    condition_key=condition_key,
    cell_type_key=cell_type_key,
    labeled_indices=labeled_ind,
    hidden_layer_sizes=[128, 128],
    use_mmd=False,
    clustering='louvain',
)
tranvae.model.load_state_dict(torch.load(os.path.expanduser(f'~/Documents/reference_model_state_dict')))

tranvae.train(
    n_epochs=tranvae_epochs,
    early_stopping_kwargs=early_stopping_kwargs,
    eta=10,
    tau=0,
)
ref_path = os.path.expanduser(f'~/Documents/reference_model')
tranvae.save(ref_path, overwrite=True)

adata_latent = sc.AnnData(tranvae.get_latent())
adata_latent.obs['cell_type'] = source_adata.obs[cell_type_key].tolist()
adata_latent.obs['batch'] = source_adata.obs[condition_key].tolist()

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent,
           color=['batch', 'cell_type'],
           frameon=False,
           wspace=0.6,
           show=False
           )
plt.savefig(os.path.expanduser(f'~/Documents/umap_ref_tranvae.png'), bbox_inches='tight')