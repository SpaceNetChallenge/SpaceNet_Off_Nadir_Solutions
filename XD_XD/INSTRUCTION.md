1. Build docker image first.

```
$ unzip -d XD_XD_solution xdxd_solution_181229_rev1.zip
$ cd XD_XD_solution
$ tree
.
├── Dockerfile
├── main.py
├── test.sh
├── train.sh
└── working
    ├── cv.txt
    └── models
        ├── vgg16-397923af.pth
        ├── v12_f0
        │   └── v12_f0_best
        ├── v12_f1
        │   └── v12_f1_best
        └── v12_f2
            └── v12_f2_best

$ docker build -t xdxd-solution .
Sending build context to Docker daemon  913MB
Step 1/9 : FROM nvidia/cuda:8.0-devel-ubuntu16.04
.
.
(snip)
.
.
Successfully built 691ea5072dd3
Successfully tagged xdxd-solution:latest
```

2. Inference test images. Mount test image directory with `-v` option.

* My home built models are stored on `/root/working`.
* `--rm` option automatically remove the container exists.
* Requied disk space my solution needs:
    * For testing:
        * preprocessed files: 1.3GB (/wdata/dataset/test_rgb)
    * For training:
        * intermediate models: 60GB (/wdata/models)
        * preprocessed files: 39GB (/wdata/dataset/train_rgb) & 932M (/wdata/dataset/masks)

```
$ nvidia-docker run \
    --rm \
    -t \
    -i \
    -v /data/spacenet4/SpaceNet-Off-Nadir_Test_Public:/data/test \
    -v /opt/tmpdir:/wdata \
    xdxd-solution \
    test.sh /wdata/out.txt
```

3. Train models from scratch. Mount train data directory with `-v` option.

* **WARNINGS**: train.sh updates my home built models.
* Use `--shm-size` option since pytorch's DataLoader utilizes shared memory (shm).
* Trained models will be stored on `/root/working/models` in the container.

```
$ nvidia-docker run \
    --cpuset-cpus 0-15 \
    --shm-size=6g \
    -t \
    -i \
    -v /data/spacenet4/SpaceNet-Off-Nadir_Train:/data/training \
    -v /opt/tmpdir:/wdata \
    xdxd-solution \
    train.sh
```
