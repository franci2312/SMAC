import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=float, default=0.05)
    return parser.parse_args()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    input_folder = 'data'
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)

    args = parse_arguments()
    k = args.k
    n3 = 4000
    cp = 50
    maxrl = 4000 + cp 
    stat_type = 'cusum'
    B = 1000
    n_levels = 6

    methods = ['gl', 'isomap', 'lb']
    ranks = [101, 3, 101]
    label = 'loc'

    df_loc, results_loc = run_oc(input_folder=input_folder, methods=methods, ranks=ranks, n3=n3, label=label, 
                                 stat_type=stat_type, k=k, B=B, maxrl=maxrl, nrep=30, cp=cp, levels=n_levels, seed=4321)

    df_loc.to_csv(os.path.join(results_folder, f'df_{label}_{stat_type}_k{k}.csv'))
    df_loc.index = [('SMAC' if m == 'LB' else m) for m in df_loc.index]
    latex_str = df_loc.to_latex(escape=False, column_format='l' + 'c'*n_levels)
    print("\nOC Results for shape-only scenario:")
    print("=" * 40)
    print(latex_str)
    print("=" * 40)
    print(f"Results saved to results/ with name df_{label}_{stat_type}_k{k}.csv")

    ic_csv_path = os.path.join(results_folder, f'ic_n3_sensitivity_{stat_type}_k{k}.csv')
    if not os.path.exists(ic_csv_path):
        raise FileNotFoundError(f"IC results not found: {ic_csv_path}. Please run Table_1.py first.")

    plot_arl_from_csv(
        csv_oc_path=os.path.join(results_folder, f'df_{label}_{stat_type}_k{k}.csv'),
        csv_ic_path=ic_csv_path,
        ic_column=str(n3),
        shifts=[1.0, 1.15, 1.25, 1.75, 2.25, 2.75, 3.25],
        title='', x_label='SNR',
        y_lim=(0, 120),
        path_output=os.path.join(results_folder, f'figure_{label}_{stat_type}_k{k}.png'),
        n_std=1
    )
    print(f"Figure saved to results/ with name figure_{label}_{stat_type}_k{k}.png")