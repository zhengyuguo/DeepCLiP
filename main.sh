#!/usr/bin/env bash
# vim: set noexpandtab tabstop=2:

datadir=/Users/zhengyuguo/Downloads/datasets
outdir=/tmp/res
script_name=ideep_tt
script_name=ideep_cv

export PYTHONPATH="$(pwd):$PYTHONPATH"

while read -r tn tt
do
	dname=$(dirname "$tn")
	dname=$outdir/${dname##$datadir}
	mkdir -p "$dname"
	obname=$dname/$script_name
	echo "./run_scripts/$script_name.py" "$tn/folded.txt.gz" "$tt/folded.txt.gz" "$obname"
	"./run_scripts/$script_name.py" "$tn/folded.txt.gz" "$tt/folded.txt.gz" "$obname"
done < <(paste.sh <(find.sh -d "$datadir" -n training_sample_0 | sort.sh) <(find.sh -d "$datadir" -n test_sample_0 | sort.sh))
