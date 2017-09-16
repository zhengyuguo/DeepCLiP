#!/usr/bin/env bash
# vim: set noexpandtab tabstop=2:

datadir=/Users/zhengyuguo/Downloads/datasets
datadir=./new_data/datasets
outdir=$(pwd)/res_cv

export PYTHONPATH="$(pwd):$PYTHONPATH"

function cmd {
script_name=$1
while read -r tn tt
do
	dname=$(dirname "$tn")
	dname=$outdir/${dname##$datadir}
	mkdir -p "$dname"
	obname=$dname/$script_name
	echo "./run_scripts/$script_name.py" "$tn/down_sampled.txt.gz" "$tt/down_sampled.txt.gz" "$obname"
	"./run_scripts/$script_name.py" "$tn/down_sampled.txt.gz" "$tt/down_sampled.txt.gz" "$obname"
#done < <(paste.sh <(find.sh -d "$datadir" -n training_sample_0 | sort.sh) <(find.sh -d "$datadir" -n test_sample_0 | sort.sh))
done < <(paste.sh <(find.sh -d "$datadir" -n tn_merge | sort.sh) <(find.sh -d "$datadir" -n tt_merge | sort.sh))
}

#cmd ideep_tt
#cmd cnn_glob_tt
#cmd cnn_auto_tt
#cmd ideep_32_tt
#cmd ideep_16_tt
#cmd cnn_glob_cv
#cmd ideep_cv
cmd cnn_auto_cv
