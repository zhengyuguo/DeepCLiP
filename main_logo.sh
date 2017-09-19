#!/usr/bin/env bash
# vim: set noexpandtab tabstop=2:

datadir=/Users/zhengyuguo/Downloads/datasets
datadir=./new_data/datasets
outdir=$(pwd)/res_logo

export PYTHONPATH="$(pwd):$PYTHONPATH"

function cmd {
script_name=$1
while read -r tn
do
	dname=$(dirname "$tn")
	dname=$outdir/$script_name/${dname##$datadir}
	mkdir -p "$dname"
 	echo "./run_scripts/$script_name.py" "$tn/down_sampled.txt.gz" "aaa" "$dname"
 	"./run_scripts/$script_name.py" "$tn/down_sampled.txt.gz" "aaa" "$dname"
	break
done < <(find.sh -d "$datadir" -n tn_merge | sort.sh)
}

#cmd ideep_weblogo
cmd cnn_auto_weblogo
#cmd ideep_tt
#cmd cnn_glob_tt
#cmd cnn_auto_tt
#cmd ideep_32_tt
#cmd ideep_16_tt
#cmd cnn_glob_cv
#cmd ideep_cv
#cmd cnn_auto_cv
