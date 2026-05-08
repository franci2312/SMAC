import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=float, default=0.05)
    parser.add_argument('--panel', type=int, default=1, choices=[1, 2, 3, 4], help="Panel number (Figure 7)")
    return parser.parse_args()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    input_folder = 'data'
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)

    args = parse_arguments()
    k = args.k
    panel = args.panel
    n3 = 4000
    cp = 50
    maxrl = 4000 + cp 
    stat_type = 'cusum'
    B = 1000
    n_levels = 5

    methods = ['gl', 'isomap', 'lb']
    ranks = [101, 3, 101]
    label = f'cspot{panel}'

    df_loc_spot, results_loc_spot = run_oc(input_folder=input_folder, methods=methods, ranks=ranks, n3=n3, label=label, 
                                 stat_type=stat_type, k=k, B=B, maxrl=maxrl, nrep=30, cp=cp, levels=n_levels, seed=4321)

    df_loc_spot.to_csv(os.path.join(results_folder, f'df_{label}_{stat_type}_k{k}.csv'))
    df_loc_spot.index = [('SMAC' if m == 'LB' else m) for m in df_loc_spot.index]
    latex_str = df_loc_spot.to_latex(escape=False, column_format='l' + 'c'*n_levels)
    print(f"\nOC Results for shape and color scenario (panel {panel}):")
    print("=" * 40)
    print(latex_str)
    print("=" * 40)
    print(f"Results saved to results/ with name df_{label}_{stat_type}_k{k}.csv")

    loc_path = os.path.join(results_folder, f'df_loc_{stat_type}_k{k}.csv')
    if not os.path.exists(loc_path):
        raise FileNotFoundError(f"Shape-only results not found: {loc_path}. Please run Figure_6a.py first.")

    plot_arl_from_csv(
        csv_oc_path=os.path.join(results_folder, f'df_{label}_{stat_type}_k{k}.csv'),
        csv_ic_path=loc_path,
        ic_column=f'Level {panel}',  
        shifts=[0, 0.01, 0.05, 0.1, 0.15, 0.2],
        title='', x_label='Shift',
        y_lim=(0, 120),
        path_output=os.path.join(results_folder, f'figure_{label}_{stat_type}_k{k}.png'), 
        n_std = 1
    )
    print(f"Figure saved to results/ with name figure_{label}_{stat_type}_k{k}.png")