WORK=$(pwd)

image_name="stmixer:v1"
container_name="stmixer-train" 

#docker rm "$container_name" 
#docker run  -it --rm --name "$container_name" --gpus=all -p 8886:8888  -v $WORK:/work/mmpose/files "$image_name"
docker create  -it  --name "$container_name" --gpus=all -p 8876:8888  --shm-size="3G" -v $WORK:/work  -v /data/ava/frames:/work/ava/frames   "$image_name"