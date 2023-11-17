for n in 64 256 512
do
    for r in 1 0.1 0.01 0.001 0.0001 0
    do
	    for d in 5 10 15 20
	    do
            python run_interp-voxelize+ann-density-cut.py -o test-result -n $n -r $r -d $d
	    done
    done
done
