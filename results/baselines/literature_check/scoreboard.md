| Target | Published | Best variant (picked on train) | Train | Val | Meets? | All variants (val) |
|---|---|---|---|---|---|---|
| Fade fit: RUL RMSE (cycles) | 147.4 | fadefit_prior_recal | 35.9 | 64.2 | yes | fadefit_gp_s16* 115, fadefit_gp_s4* 115, fadefit_gp_s8* 115, fadefit_gp* 115, fadefit_prior_recal 64.2, fadefit_prior* 94.6, fadefit* 116 |
| Fade fit: calibration error | 0.06 | fadefit_prior_recal | 0.152 | 0.182 | no | fadefit_gp_s16* 0.233, fadefit_gp_s4* 0.251, fadefit_gp_s8* 0.231, fadefit_gp* 0.31, fadefit_prior_recal 0.182, fadefit_prior* 0.337, fadefit* 0.34 |
| Fade fit: alpha-accuracy (%) | 32.9 | fadefit_prior_recal | 31.1 | 21.3 | no | fadefit_gp_s16* 18.4, fadefit_gp_s4* 18.4, fadefit_gp_s8* 18.4, fadefit_gp* 18.4, fadefit_prior_recal 21.3, fadefit_prior* 13.5, fadefit* 14.2 |
| Plain GP: MAPE last 5 pts, 50% context | 5.45 | plain_gp_bayes* | 11 | 9.82 | no | gp_linear_mean* 12.6, plain_gp_bayes* 9.82, plain_gp* 16.8, plain_gp_zeromean* 12 |
| Plain GP: MAPE last 5 pts, 60% context | 6.08 | plain_gp_bayes* | 10.5 | 9.54 | no | gp_linear_mean* 11.5, plain_gp_bayes* 9.54, plain_gp* 15.5, plain_gp_zeromean* 11.7 |
| Plain GP: MAPE last 5 pts, 70% context | 5.86 | plain_gp_zeromean* | 8.92 | 8.5 | no | gp_linear_mean* 9.85, plain_gp_bayes* 8.54, plain_gp* 12.4, plain_gp_zeromean* 8.5 |
| Plain GP: MAPE last 5 pts, 80% context | 3.67 | plain_gp_zeromean* | 4.2 | 3.91 | no | gp_linear_mean* 5.76, plain_gp_bayes* 5.02, plain_gp* 6.25, plain_gp_zeromean* 3.91 |
| Naive: EOL MAE (cycles), first 50 cycles | 1636.56 | linear_persistence* | nan | 1.04e+03 | yes | linear_persistence* 1.04e+03 |
| Naive: EOL MAE (cycles), first 100 cycles | 1562.07 | linear_persistence* | nan | 1.01e+03 | yes | linear_persistence* 1.01e+03 |
| GP: RMSE SOH pts, train 100 -> forecast 68 | 13.03 | our_plain_gp* | nan | 0.434 | yes | liu_basic_gpr* 25.6, our_plain_gp* 0.434 |

`*` = result made by an older code version (re-run with `run`). Sources: Kenney 2024 Table 4, GP without pre-training; Liu 2013 Tables 1-3, basic GPR, best of 13.03 / 22.51 / 20.70; Thelen 2022 Table 5, exponential model-based, primary test (secondary 105.0); Thelen 2022 Table 5, primary test (secondary 0.20); Thelen 2022 Table 5, primary test (secondary 15.4); Zou 2026 Sec. 3, linear persistence (cap rule not stated).
