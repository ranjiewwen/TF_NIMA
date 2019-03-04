
# NIMA: Neural Image Assessment base on tensorflow

- In this project , I use the vgg mdoel to complete nima on tid2013 datasets through native tensorflow api tools.

## Get started

- first you need download the vgg pre-train model weight from :[vgg16_weights.npz](http://www.cs.toronto.edu/~frossard/post/vgg16/)
- then put it into the directory: `data/vgg_models/`
- then you shold prepare the datasets list throgh `data/*` scripts
- then you need modity the `BASE_PATH` in the code which is your iqa dataset root path
- final you can run the code, make sure you have follow env : `ubuntu16.04+tensorflow1.8+cuda9.0`
- by the way, you can modify the parameters in `tools/train_nima.py` funtions `process_command_args`

## Run the code

- first train the models: run the `tools/train_nima.py` scripts.
```

2019-03-04 16:45:08,248 TF_NIMA_training INFO: step 8850/9000, the emd loss is 0.115794,l2_loss is 25.657944,total loss is 0.115794, time 3823.410413,learning rate: 0.000001
2019-03-04 16:45:08,370 TF_NIMA_training INFO: evaluate train batch SROCC_v: 0.818	 KROCC: 0.650	 PLCC_v: 0.864	 RMSE_v: 1.034	 mse: 1.069

2019-03-04 16:45:30,609 TF_NIMA_training INFO: step 8900/9000, the emd loss is 0.089719,l2_loss is 21.096500,total loss is 0.089719, time 3845.770660,learning rate: 0.000001
2019-03-04 16:45:30,727 TF_NIMA_training INFO: evaluate train batch SROCC_v: 0.944	 KROCC: 0.798	 PLCC_v: 0.953	 RMSE_v: 0.938	 mse: 0.879

2019-03-04 16:45:53,219 TF_NIMA_training INFO: step 8950/9000, the emd loss is 0.116848,l2_loss is 28.304466,total loss is 0.116848, time 3868.381484,learning rate: 0.000001
2019-03-04 16:45:53,342 TF_NIMA_training INFO: evaluate train batch SROCC_v: 0.891	 KROCC: 0.769	 PLCC_v: 0.910	 RMSE_v: 1.086	 mse: 1.179

2019-03-04 16:46:07,030 TF_NIMA_training INFO: Optimization finish!

```
- second test the models: run the `tools/evaluate.py` scripts.

```
2019-03-04 17:14:59,458 TF_NIMA_evaluating INFO: test image:500/600, true_mean_mos/predict_mos is [6.11765]/1.9778872739322422,the emd loss: 0.5347248911857605.
2019-03-04 17:14:59,459 TF_NIMA_evaluating INFO: image score_:[[0.    0.    0.    0.    0.    0.995 0.005 0.    0.    0.   ]]
2019-03-04 17:14:59,459 TF_NIMA_evaluating INFO: image score_hat:[[4.5310372e-07 1.0060684e-01 8.2096982e-01 7.8351051e-02 7.1739130e-05
  7.9168867e-09 1.0901127e-08 1.4305775e-08 2.6052367e-08 3.0621173e-08]]
2019-03-04 17:14:59,913 TF_NIMA_evaluating INFO: test image:550/600, true_mean_mos/predict_mos is [3.4]/1.889601976246836,the emd loss: 0.0557439923286438.
2019-03-04 17:14:59,913 TF_NIMA_evaluating INFO: image score_:[[0.     0.     0.8615 0.1385 0.     0.     0.     0.     0.     0.    ]]
2019-03-04 17:14:59,914 TF_NIMA_evaluating INFO: image score_hat:[[6.4253740e-08 1.3145407e-01 8.4749258e-01 2.1050246e-02 2.9920257e-06
  9.3558952e-11 8.3277246e-10 1.1079678e-09 2.0081541e-09 2.5365026e-09]]
2019-03-04 17:15:00,302 TF_NIMA_evaluating INFO: SROCC_v: 0.422	 KROCC: 0.299	 PLCC_v: 0.453	 RMSE_v: 2.712	 mse: 7.356

2019-03-04 17:15:00,302 TF_NIMA_evaluating INFO: Test finish!

```
- last predict images: run the `demo/predict.py` scripts.

```
test:
distorted_images/I16_01_1.bmp 5.615380 0.106440
distorted_images/i16_03_3.bmp 3.692310 0.089950
distorted_images/i16_23_5.bmp 1.526320 0.081000

```
![I16_01_1](https://github.com/ranjiewwen/TF_NIMA/blob/master/demo/img/I16_01_1.bmp.png)
![i16_03_3](https://github.com/ranjiewwen/TF_NIMA/blob/master/demo/img/i16_03_3.bmp.png)
![i16_23_5](https://github.com/ranjiewwen/TF_NIMA/blob/master/demo/img/i16_23_5.bmp.png)


## Experiments result

- you can see train/test log from directory `experiments/datasets/`
- tensorboard logs in `experiments/datasets/logs`

![nima_tensorboard](https://github.com/ranjiewwen/TF_NIMA/blob/master/demo/img/nima_tensorboard.png)

- save train modes in `experiments/datasets/experiment_name`

## Coming soon optimization

- from tensorboard curve it maybe overfit ,but predict scores it looks normal, i am confused.
- maybe there has some bug in the code ,but i will contimue optimazation this project.
- if you have any questions or some adivise!  please make issue on this project. thanks!

