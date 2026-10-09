#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("--input",default="../data/pkg_stars.tsv")
parser.add_argument("--output", default="../results/pkg_stars.bar.png")
args = parser.parse_args()

def main():
	df = pd.read_csv(args.input,sep="\t")
	fig,ax = plt.subplots(figsize=(6,4))
	ax.bar(
		x=df['package'],
		height=df['stars'],
		color='black'
	)
	ax.set_title("Python Package popularity", loc="left")
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_xlabel("Package")
	ax.set_ylabel("GitHub stars")
	plt.tight_layout()
	fig.savefig(args.output)
 
    

if __name__ == "__main__":
    main()