#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--samples", default=100)
parser.add_argument("--samples_condition", default=50)
parser.add_argument("--genes", default=1000)
parser.add_argument("--conditions", default=4)
parser.add_argument("--output_expr",default="../data/gene_expression.tsv")
parser.add_argument("--output_condition", default="../data/sample_conditions.tsv")
parser.add_argument("--seed", default=0)
args = parser.parse_args()
args.samples=int(args.samples)
args.samples_condition =int(args.samples_condition)
args.genes=int(args.genes)
args.seed=int(args.seed)
np.random.seed(args.seed)

def main():
	### params
	# n=theta
	# p= theta / (theta + mu)
	dispersion_housekeeping=10
	mean_housekeeping = 100
	p = dispersion_housekeeping / (dispersion_housekeeping + mean_housekeeping)
	X_housekeeping = np.random.negative_binomial(dispersion_housekeeping, p, size = (args.samples, args.genes // 2))
	dispersion = 50
	mean = 10
	p = dispersion / (dispersion + mean)
	X = np.random.negative_binomial(dispersion, p, size = (args.samples, args.genes //2))
	X = np.hstack((X, X_housekeeping))
	# shuffle columns
	X = np.random.permutation(X.T).T
	df = pd.DataFrame(X)
	columns = [f'gene_{i}' for i in range(args.genes)]
	df.columns = columns
	samples = [f'sample_{i}' for i in range(args.samples)]
	df['sample'] = samples
	order = ['sample'] + (df.columns.tolist()[:-1])
	df = df[order]
	df.to_csv(args.output_expr,sep='\t',index=False)
	samples_w_cond_data = np.random.choice(np.arange(args.samples),size=args.samples_condition,replace=False)
	samples = [f"sample_{i}" for i in samples_w_cond_data]
	conditions = np.random.choice([0,1],size=(args.samples_condition, args.conditions))
	df = pd.DataFrame(conditions)
	df.columns = [f"condition_{i}" for i in range(args.conditions)]
	df['sample'] = samples
	order = ['sample'] + (df.columns.tolist()[:-1])
	df = df[order]
	df.to_csv(args.output_condition,sep='\t',index=False)
	breakpoint()

if __name__ == "__main__":
    main()