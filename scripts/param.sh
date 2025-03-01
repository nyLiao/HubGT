DEV=${1:-0}
SEED_P=21
SEED_S="42"
ARGS_S=(
    "--dev" "$DEV"
    "--seed" "$SEED_S"
    # "--seed_tune" "$SEED_P"
    "--loglevel" "25"
    "--epoch" "200"
    "--patience" "30"
    "--aggr_output" "0"
    "-quiet"
)
PARAM="s0"
VALS=("6" "12" "18" "24" "30" "36" "42")

# main: ogbn-mag, reddit, ogbn-products, penn94, genius, twitch-gamer,
# appendix: cora, citeseer, pubmed, cs, physic, chameleon_filter, squirrel_filter
DATAS=("citeseer")
for data in ${DATAS[@]}; do
    for val in ${VALS[@]}; do
        python main_param.py --data $data "${ARGS_S[@]}" \
            --suffix $PARAM -"${PARAM}" $val
    done
done
