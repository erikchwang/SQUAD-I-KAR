ROOT=$(dirname $(realpath $0))

if [ "$#" -eq 0 ] || [ "$1" -eq 0 ]
then
    python $ROOT/preprocess.py 0
    python $ROOT/preprocess.py 1
    python $ROOT/preprocess.py 2
fi

if [ "$#" -eq 0 ] || [ "$1" -eq 1 ]
then
    python $ROOT/construct.py
    python $ROOT/optimize.py
fi

if [ "$#" -eq 0 ] || [ "$1" -eq 2 ]
then
    python $ROOT/execute.py $ROOT/data/develop_dataset $ROOT/data/develop_solution
    python $ROOT/data/evaluate_script $ROOT/data/develop_dataset $ROOT/data/develop_solution
fi

if [ "$#" -eq 0 ] || [ "$1" -eq 3 ]
then
    python $ROOT/execute.py $ROOT/data/addsent_dataset $ROOT/data/addsent_solution
    python $ROOT/data/evaluate_script $ROOT/data/addsent_dataset $ROOT/data/addsent_solution
fi

if [ "$#" -eq 0 ] || [ "$1" -eq 4 ]
then
    python $ROOT/execute.py $ROOT/data/addonesent_dataset $ROOT/data/addonesent_solution
    python $ROOT/data/evaluate_script $ROOT/data/addonesent_dataset $ROOT/data/addonesent_solution
fi