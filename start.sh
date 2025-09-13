git pull
docker compose -f $1.yaml down
docker compose -f $1.yaml up -d
docker logs -f trade-$1