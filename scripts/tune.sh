DEV=${1:-0}
SEED_P="1,2"
ARGS_P=(
    "--dev" "$DEV"
    "--seed" "$SEED_P"
    "--n_trials" "60"
    "--loglevel" "30"
    "--epoch" "200"
    "--patience" "30"
)

# main: physics, ogbn-mag, reddit, ogbn-arxiv, penn94, genius, twitch-gamer, pokec
# appendix: cora, citeseer, pubmed, cs, chameleon_filtered, squirrel_filtered, tolokers, flickr
DATAS=("pubmed" "physics")
for data in ${DATAS[@]}; do
    python main_tune.py --data $data "${ARGS_P[@]}"
done
