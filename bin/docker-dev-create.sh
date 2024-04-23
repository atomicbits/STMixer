WORK=$(pwd)

image_name="stmixer:v1"
container_name="stmixer" 

#docker rm "$container_name" 
#docker run  -it --rm --name "$container_name" --gpus=all -p 8886:8888  -v $WORK:/work/mmpose/files "$image_name"
docker create  -it  --name "$container_name" --gpus=all -p 8879:8888  -v $WORK:/work "$image_name"