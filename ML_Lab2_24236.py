import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

# A1

def load_shop_data(path):
    data = pd.read_excel(path, sheet_name=0)

    c = data["Candies (#)"].values
    m = data["Mangoes (Kg)"].values
    milk = data["Milk Packets (#)"].values

    features = np.column_stack((c, m, milk))
    target = data["Payment (Rs)"].values

    return features, target


def find_rank(matrix):
    return np.linalg.matrix_rank(matrix)


def find_costs(features, payment):
    pinv = np.linalg.pinv(features)
    return np.dot(pinv, payment)


# A2

def group_customers(payments):
    groups = []
    for amount in payments:
        if amount > 200:
            groups.append("RICH")
        else:
            groups.append("POOR")
    return groups


# A3

def load_market_data(path):
    return pd.read_excel(path, sheet_name=1)


def mean_manual(arr):
    total = 0
    for x in arr:
        total += x
    return total / len(arr)


def variance_manual(arr):
    mean_val = mean_manual(arr)
    s = 0
    for x in arr:
        s += (x - mean_val) ** 2
    return s / len(arr)   # population variance


def time_taken(func, arr):
    times = []
    for _ in range(10):
        start = time.time()
        func(arr)
        times.append(time.time() - start)
    return sum(times) / len(times)


def loss_probability(changes):
    cnt = 0
    for x in changes:
        if x < 0:
            cnt += 1
    return cnt / len(changes)


def wednesday_profit_prob(df):
    wed = df[df["Day"] == "Wednesday"]
    prof = wed[wed["Chg%"] > 0]
    return len(prof) / len(wed)


# A4

def load_thyroid(path):
    return pd.read_excel(path, sheet_name=2)


def check_missing(df):
    res = {}
    for col in df.columns:
        res[col] = df[col].isnull().sum()
    return res


# A5

def jaccard(v1, v2):
    a = b = c = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            a += 1
        elif v1[i] == 1 and v2[i] == 0:
            b += 1
        elif v1[i] == 0 and v2[i] == 1:
            c += 1
    if (a + b + c) == 0:
        return 0
    return a / (a + b + c)


def smc(v1, v2):
    a = b = c = d = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            a += 1
        elif v1[i] == 0 and v2[i] == 0:
            d += 1
        elif v1[i] == 1 and v2[i] == 0:
            b += 1
        else:
            c += 1
    return (a + d) / (a + b + c + d)


# A6

def cosine_sim(v1, v2):
    dot = 0
    s1 = 0
    s2 = 0

    for i in range(len(v1)):
        dot += v1[i] * v2[i]
        s1 += v1[i] ** 2
        s2 += v2[i] ** 2

    if s1 == 0 or s2 == 0:
        return 0

    return dot / (np.sqrt(s1) * np.sqrt(s2))


# A7

def show_heatmap(data):
    n = len(data)
    sim_mat = []

    for i in range(n):
        row = []
        for j in range(n):
            row.append(cosine_sim(data[i], data[j]))
        sim_mat.append(row)

    sns.heatmap(sim_mat)
    plt.title("Cosine Similarity Heatmap")
    plt.show()


# MAIN

def main():
    path = "Lab Session Data.xlsx"

    # A1
    X, y = load_shop_data(path)
    print("Rank of Feature Matrix:", find_rank(X))
    print("Cost of Products:", find_costs(X, y))

    # A2
    print("Customer Classification:", group_customers(y))

    # A3
    stock = load_market_data(path)
    price = stock["Price"].values
    print("Mean (Manual):", mean_manual(price))
    print("Variance (Manual):", variance_manual(price))
    print("Mean Time:", time_taken(mean_manual, price))

    # A4
    thyroid = load_thyroid(path)
    print("Missing Value Analysis:", check_missing(thyroid))

    # A5 & A6
    v1 = np.array([1, 0, 1, 1])
    v2 = np.array([1, 1, 0, 1])
    print("Jaccard:", jaccard(v1, v2))
    print("SMC:", smc(v1, v2))
    print("Cosine:", cosine_sim(v1, v2))

    # A7
    show_heatmap(X)


main()
