import matplotlib.pyplot as plt

with open("mnist_log.txt", "r") as f:
    data_ = f.readlines()
    data = []
    for item in data_:
        data.append(eval(item))

computation_time = []
for item in data:
    computation_time.append(item["Dual_time"])

print(sum(computation_time) / len(computation_time))