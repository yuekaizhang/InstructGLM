### GLM


```sh
# docker build -f Dockerfile -t soar97/torch-glm:22.12 .
docker pull soar97/torch-glm:22.12

docker run -it --name "glm-lora-finetune" --net host --gpus all -v /mnt:/mnt soar97/torch-glm:22.12
```


### SFT command



