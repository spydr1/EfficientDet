python train.py --snapshot=checkpoints/reanchor/ship_99_0.3717_0.3662.h5 --snapshot-path=checkpoints/reanchor --detect-quadrangle --detect-ship --phi=1 --gpu=0,1,2,4 --random-transform --batch-size=16 --steps=1600 --compute-val-loss --workers=8 --max-queue-size=8 --epochs=200 ship None 0.8 
#--freeze-backbone
#--multiprocessing
# mAP callback 제거
# imagenet
# 'checkpoints/2020-03-17/ship_14_5.2078_5.1929.h5' - all layer 
# snapshot-path='checkpoints/resizing


# 03-11 16:26 ~ val cls_loss가 너무 안떨어진다 .. train cls_loss 는 대략 0.2~ 정도이지만 val cls_loss 는 거의 0.5 (all , freeze backbone 모두 ..)
# -> ratio를 줄였다. 0.25,0.5 2, 4 -> 0.5 ,1 ,2  phi = 1->5 
# 유의미한 학습이 안되고 있었다. (stride가 너무 커서 일수도 ?) -> 그래서 phi=1->5로 변경.
# ratio가 이상했다 ? 1:1의 종횡비를 갖는 배는 없다고 생각해서 0.25, 0.5 , 2 ,4 라고 정하긴 했는데 .. 흠 ? 


# checkpoints/2020-03-12/ship_24_0.9185_1.9215.h5 까지 freeze ~
# freeze 빼고 leanring rate = 0.001 -> 0.0001 로 수정한다.


# weight regularization 추가 : L2 (0.001)

# 0319 22:22 ~ 정규화 빼고, 최대한 많은 레이어 학습 (--freeze-backbone빼기)

# 정규화로 쭉 .. 
# 전체 데이터 로 학습 .. 