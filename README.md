# Physics-Informed-Probabilistic-Forecasting-of-Lithium-Ion-Battery-Degradation

## Layout

- `data_pipeline/`: dataset loaders (Severson, Kirkaldy), locked train/val/test splits (`splits/*.json`) and evaluation metrics.
- `baselines/`: the four baseline forecasters (naive slope, empirical fade fit, Wiener process, plain GP) and the evaluation script.
- `tests/`: tests for the metrics and the split lock (`python -m pytest tests -q`).
- `results/baselines/`: baseline results on the train and validation cells.
- `hpc/`: PyBaMM simulation sweep scripts for the Imperial CX3 cluster.

## Data

The raw datasets are not stored in this repository. They are public and can be downloaded again. Put them under `data/` (gitignored):

| Dataset | Source | Put in |
|---|---|---|
| Severson et al. (2019), 124 LFP/graphite fast-charge cells (BEEP structured data, `FastCharge.zip`) | https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 | `data/severson/` |
| Kirkaldy et al. (2024), 40 LG M50T cells, five experiment zips | https://doi.org/10.5281/zenodo.10637534 | `data/kirkaldy/` |

Then build the local caches:

```bash
python -m data_pipeline.severson     # reads the zip directly, writes data/severson/cache/
python -c "from data_pipeline import kirkaldy; kirkaldy.extract()"   # summary CSVs only
```

## Data credits

Files in `results/` and `data_pipeline/splits/` contain values derived from the two datasets below, which are used under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). The derived files include cell identifiers, capacity-based metrics and forecast errors. The data have been reformatted, filtered and analysed as described in the loader docstrings; the original data are unchanged at source.

- **Severson et al. dataset.** K. A. Severson, P. M. Attia, N. Jin, N. Perkins, B. Jiang, Z. Yang, M. H. Chen, M. Aykol, P. K. Herring, D. Fraggedakis, M. Z. Bazant, S. J. Harris, W. C. Chueh and R. D. Braatz, "Data-driven prediction of battery cycle life before capacity degradation," *Nature Energy* 4, 383–391 (2019), https://doi.org/10.1038/s41560-019-0356-8. Data from the Toyota Research Institute Experimental Data Platform, https://data.matr.io/1/ (CC BY 4.0).
- **Kirkaldy et al. dataset.** N. Kirkaldy, M. A. Samieian, G. J. Offer, M. Marinescu and Y. Patel, "Lithium-ion battery degradation: Comprehensive cycle ageing data and analysis for commercial 21700 cells," *Journal of Power Sources* 603, 234185 (2024), https://doi.org/10.1016/j.jpowsour.2024.234185. Data: Kirkaldy, Samieian, Offer, Marinescu & Patel (2024), Zenodo, https://doi.org/10.5281/zenodo.10637534 (CC BY 4.0).
