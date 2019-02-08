test.sh: takes less 20 hours on p2.xlarge (not sure how much exactly)
train.sh: takes ~6days with 4 titan xp (designed to use exactly such system with 4 gpu).

On every start docker will download ~8gb zip file with models weights and extract to /wdata. If models already downloaded or you want to check train, you could add /bin/bash to the end of nvidia-docker run command

It's necessary that "sudo nvidia-persistenced" command executed before docker run on p2.xlarge instance - without it I've faced with a problem to use GPU inside docker.

It's necessary to add '--ipc=host' option to run docker to test train.sh. Otherwise multithreaded pytorch's dataloader will crash. Example:
nvidia-docker run -v /spacenet_data_path:/data:ro -v /wdata_path:/wdata --ipc=host -it cannab (/bin/bash - if don't need to download models again)
