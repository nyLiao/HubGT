DEV=${1:-0}
SEED_P=21
SEED_S="20,21,22"
ARGS_S=(
    "--dev" "$DEV"
    "--seed" "$SEED_S"
    "--seed_tune" "$SEED_P"
    "--loglevel" "25"
    "--epoch" "300"
    "--patience" "50"
)

# main: physics, ogbn-mag, reddit, ogbn-arxiv, penn94, genius, twitch-gamer, pokec
# appendix: cora, citeseer, pubmed, cs, chameleon_filtered, squirrel_filtered, tolokers, flickr
DATAS=("physics")
for data in ${DATAS[@]}; do
    python main.py --data $data "${ARGS_S[@]}"
done
