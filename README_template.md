# Optimal Scheduling of Mixed Energy Sources

Day-ahead load forecasting -> economic dispatch -> constraint-breach flagging, for a mixed solar / wind / hydro / coal / gas grid.
Python: statsmodels (ARIMA, SARIMA), scikit-learn (K-means, Random Forest), SciPy (HiGHS LP).

**Data is synthetic** (2023-2024 hourly load, solar, wind with weekday, Sunday, festival, temperature and EV effects). Prices and plant sizes are illustrative.

## Results (held-out Sep-Dec 2024, 122 days)
| | ARIMA | SARIMA (weekly) | K-means + ARIMA |
|---|---|---|---|
| MAPE, all days | {{mape_a}}% | {{sar}}% | **{{mape_h}}%** |
| MAPE, Sundays | {{sun_a}}% | | **{{sun_h}}%** |
| MAPE, holidays | {{hol_a}}% | | **{{hol_h}}%** |

* Random Forest breach flagging: AUC {{auc}}, recall {{rec}}%, precision {{prec}}%, {{flag}}% of hours flagged.
* RF-targeted reserve vs ARIMA baseline: unserved energy {{uns_ca}}%, operating cost {{cost_ca}}%. Blanket reserve costs more than targeted reserve.
* The cost saving is small. Fuel cost dominates; the gain is security of supply per rupee of reserve.

![forecast](results/fig_forecast.png) ![policy](results/fig_policy.png)

## Run
```
pip install -r requirements.txt
python run_pipeline.py     # ~2 min, writes results/
python build_page.py       # builds index.html and README.md
```

## Limitations
LP relaxation (no unit commitment, min stable load or start-up costs); real-time stage is hour-by-hour without inter-hour ramp coupling; single seed and split; synthetic data. Next: public load data, a Simulink model of the dispatch, and unit commitment as a MILP.

Built with AI assistance (Claude), then run, checked and explained by me.
