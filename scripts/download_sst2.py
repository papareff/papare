import torchtext.datasets as datasets

# 这行代码会自动下载并处理数据集
# root参数指定了下载位置，你可以根据需要修改
train_iter, dev_iter, test_iter = datasets.SST2(root='./data')

# 打印一条成功信息，验证是否下载成功
print("SST-2 dataset downloaded successfully!")