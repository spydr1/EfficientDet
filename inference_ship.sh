python inference_ship.py --model_path=checkpoints/test5/ship_30_1.1624_1.7138.h5 --image_dir=/home/minjun/Jupyter/Ship_Detection/Data/test/img  --dst_path=./test2 --patch_size=1280 --overlay_size=420 --score_threshold=0.9 --model_nms_threshold=0.7 --nms_threshold=0.7 --save_img

#--test_one
#--save_img


#/home/minjun/Jupyter/Ship_Detection/Data/test/img 
#/home/minjun/Data/Ship/test_v2

# checkpoints/2020-03-16/ship_88_0.4634_0.5052.h5  최고점수 ? 
# 03-19 21:49 ~ ship_23_0.5987_0.6748.h5 테스트중 nms = 0.5 score_threshold = 0.3 overlay=480

# 03-19 22:33 ~ ship_30_0.5331_0.5167.h5 --patch_size=1280 --overlay_size=100 --score_threshold=0.5 --nms_threshold=0.7

# /train/images
# /test/img 
# 


#--score_threshold=0.5 -> 학습이 더 잘되면 값을 올려도 될듯. 
#--model_nms_threshold=0.6 -> 최소 0.5이상 .. 일단 0.6으로 설정하자. ~0.7 정도가 좋을듯
#--nms_threshold=0.2 -> ~0.3 정도가 좋아보인다.
# nms는 모델의 정확도가 높을수록 같이 값을 올리자. 

# 7_6_2