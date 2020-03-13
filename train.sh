python train.py --snapshot=checkpoints/2020-03-13/ship_29_0.7065_1.7293.h5 --freeze-backbone --detect-quadrangle --detect-ship --phi=5 --gpu=0,1,2,4 --random-transform --batch-size=16 --steps=1000 --init-epoch=30 --compute-val-loss --workers=4 --max-queue-size=8 ship '/home/minjun/Jupyter/Ship_Detection/Data/train_tfrecorder/train_data2.tfrecords' 0.8
#--freeze-backbone
#--multiprocessing
# mAP callback 제거
# imagenet
# 'checkpoints/2020-03-10/ship_08_1.3620_2.1948.h5' - all layer 

# 03-11 16:26 ~ val cls_loss가 너무 안떨어진다 .. train cls_loss 는 대략 0.2~ 정도이지만 val cls_loss 는 거의 0.5 (all , freeze backbone 모두 ..)
# -> ratio를 줄였다. 0.25,0.5 2, 4 -> 0.5 ,1 ,2  phi = 1->5 
# 유의미한 학습이 안되고 있었다. (stride가 너무 커서 일수도 ?) -> 그래서 phi=1->5로 변경.
# ratio가 이상했다 ? 1:1의 종횡비를 갖는 배는 없다고 생각해서 0.25, 0.5 , 2 ,4 라고 정하긴 했는데 .. 흠 ? 


# checkpoints/2020-03-12/ship_24_0.9185_1.9215.h5 까지 freeze ~
# freeze 빼고 leanring rate = 0.001 -> 0.0001 로 수정한다.


# weight regularization 추가 : L2 (0.001)