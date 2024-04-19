
conda activate parallel
nbuffers="$((10**7)) $((10**6)) $((10**5)) $((10**4)) $((10**3))"
scripts="parallel_optimized.py"
for script in $scripts
do
    for nbuffer in $nbuffers
    do
        time mpiexec -n 8 python $script -N 500 -M 250 -b $nbuffer -f
    done
done