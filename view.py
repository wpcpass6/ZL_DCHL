import pickle

# 打开 pkl 文件
with open(r"E:\code\lwCode\ZL_DCHL\datasets\TKY\test_samples.pkl", "rb") as f:
    data = pickle.load(f)

# 打印数据类型和内容
print("数据类型:", type(data))
print("数据内容:\n", data)