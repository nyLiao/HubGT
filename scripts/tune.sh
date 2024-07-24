DEV=${1:-0}
SEED_P=20
SEED_S="20,21,22"
ARGS_P=(
    "--dev" "$DEV"
    "--seed" "$SEED_P"
    "--n_trials" "50"
    "--loglevel" "30"
    "--epoch" "200"
    "--patience" "50"
)
ARGS_S=(
    "--dev" "$DEV"
    "--seed" "$SEED_S"
    "--seed_param" "$SEED_P"
    "--loglevel" "25"
    "--epoch" "500"
    "--patience" "-1"
)

# DATAS=("cora" "citeseer" "pubmed" )
DATAS=("citeseer")
for data in ${DATAS[@]}; do
    python main_tune.py --data $data "${ARGS_P[@]}"
done
