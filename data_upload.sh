gcloud compute scp /jet/workspace/2019Tencent/train2 --recurse\
    tensorflow-python-cuda-minilab-2-vm:/mnt/tencent \
    --project thermal-circle-206807 --zone asia-east1-a

gcloud compute scp /jet/workspace/2019Tencent/test2 --recurse\
    tensorflow-python-cuda-minilab-2-vm:/mnt/tencent \
    --project thermal-circle-206807 --zone asia-east1-a