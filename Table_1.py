import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=float, default=0.05, help='CUSUM/EWMA parameter k')
    return parser.parse_args()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    input_folder = 'data'
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    args = parse_arguments()
    methods = ['gl', 'lle','isomap', 'lb']
    ranks = [101, 3, 3, 101]

    n1 = 600
    n2 = 300
    n3_list = [500, 1000, 2000, 3000, 4000]
    n4 = 5000
    cp = 50
    maxrl = 4000 + cp 
    stat_type = 'cusum'
    k = args.k
    df, results = run_sensitivity_n3(input_folder=input_folder, methods=methods, ranks=ranks, n1=n1, n2=n2, n3_list=n3_list, n4=n4,
                                    stat_type=stat_type, k=k, B=1000, maxrl=maxrl, arl0=100.0, maxiter=100, nrep=30, nsims=1000, verbose=False, ss=cp, seed=1142)
    
    df.to_csv(os.path.join(results_folder, f'ic_n3_sensitivity_{stat_type}_k{k}.csv'))
    print(f"IC sensitivity results saved to results/ with name ic_n3_sensitivity_{stat_type}_k{k}.csv")

    latex_str = df.to_latex(escape=False, column_format='l' + 'c'*len(n3_list))
    print("\nTable 1: Sensitivity analysis for varying sample sizes m3")
    print("=" * 40)
    print(latex_str)

