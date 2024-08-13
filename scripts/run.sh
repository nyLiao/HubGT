DEV=${1:-0}
SEED_P=21
SEED_S="20,21,22"
ARGS_S=(
    "--dev" "$DEV"
    "--seed" "$SEED_S"
    "--seed_tune" "$SEED_P"
    "--loglevel" "25"
    "--epoch" "200"
    "--patience" "100"
)

# main: ogbn-mag, reddit, ogbn-products, penn94, genius, twitch-gamer,
# appendix: cora, citeseer, pubmed, cs, physic, chameleon_filter, squirrel_filter
DATAS=("physics")
for data in ${DATAS[@]}; do
    python main.py --data $data "${ARGS_S[@]}" "--aggr_output" "0" "--var_vfeat" "0" "--suffix" "a0v0"
    python main.py --data $data "${ARGS_S[@]}" "--aggr_output" "1" "--var_vfeat" "0" "--suffix" "a1v0"
    python main.py --data $data "${ARGS_S[@]}" "--aggr_output" "0" "--var_vfeat" "1" "--suffix" "a0v1"
    # python main.py --data $data "${ARGS_S[@]}" "--aggr_output" "1" "--var_vfeat" "1" "--suffix" "a1v1"
done
