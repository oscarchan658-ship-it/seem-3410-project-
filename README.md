# CUHK Cafe (Tolo) cashier staffing simulation

Requirements: Python 3.10+ (tested on 3.14.0), numpy 2.4.6, scipy 1.16.2, pandas 3.0.3, matplotlib 3.10.6.
Install: `pip install numpy scipy pandas matplotlib`

Data
- Collected on-site at the CUHK Cafe (Tolo) ordering counter on 23, 24 and 25 November 2026 (Mon-Wed), lunch peak 11:45-13:30.
- 3 days, 310 customers. Each row of `data/customers.csv` is one customer: arrival, service start/end, order type (small/large), outcome (served/abandoned/balked). Times are minutes after 11:45.

Files
- `simulation.py`: discrete-event model, policies, common-random-number stream generator
- `run_experiments.py`: validation vs Erlang C, main experiment, sensitivity, figures -> `results/`
- `fit_inputs.py`: fits `inputs.json` from `data/customers.csv` (format: `data/customers_template.csv`)
- `inputs.json`: model inputs fitted from the observations in `data/customers.csv`.

Run
```
python fit_inputs.py            # fits inputs.json from data/customers.csv
python run_experiments.py --reps 300
```
Seeds: `SeedSequence(3410)` for the main experiment (777 sensitivity, 99 validation). Default 300 replications (150 for each sensitivity scenario).
Every policy within a replication uses the same random stream (common random numbers).
