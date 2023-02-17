xhost +

docker run -it \
    --rm \
    --env="DISPLAY=${DISPLAY}" \
    -p 8889:8889 \
    --gpus all \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="${PWD}/ch2/:/home/user/ch2/" \
    --volume="${PWD}/ch3/:/home/user/ch3/" \
    --name=vit \
    vit \
    bash
echo "done"
