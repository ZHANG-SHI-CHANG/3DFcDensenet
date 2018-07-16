输入网络的数据尺寸是一个很重要的参数，从数据集生成到训练到测试，这个参数要保持一致，否则会出错

1、生成数据集
python DataProcessing.py 32
32是输入网络的数据尺寸，即32*32*32，生成的数据集保存在processed_dataset文件夹中

2、训练
python Train.py 32 16 1000
32是输入网络的数据尺寸，16是batch size，根据运行环境改变，1000是迭代次数，根据精度和iou情况自行停止训练或加大迭代次数

3、测试
python Test.py 32 test_dataset/1000_3.nrrd test_result
32是输入网络的数据尺寸，test_dataset/1000_3.nrrd是测试的数据，测试数据集存放在test_dataset中，test_result是测试保存结果的路径