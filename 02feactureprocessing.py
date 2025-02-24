import pandas as pd

pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_csv("pre.csv")
print(df.head())

# 对学校进行处理
df1 = pd.DataFrame(df.groupby("school").mean()["posttest"])
df1.plot(kind="bar")
plt.title("学校与分数的关系")
plt.xlabel("学校")
plt.ylabel("分数")
plt.show()

# 将学校按照均分进行处理
school_0 = df1[df1["posttest"] < 60].index
school_1 = df1[(df1["posttest"] >= 60) & (df1["posttest"] < 70)].index
school_2 = df1[(df1["posttest"] >= 70) & (df1["posttest"] < 80)].index
school_3 = df1[df1["posttest"] >= 80].index


# 分组函数
def fn_school(x):
    if x in school_0:
        return 0
    elif x in school_1:
        return 1
    elif x in school_2:
        return 2
    elif x in school_3:
        return 3


df["school"] = df["school"].map(fn_school)
print(df.head())

# 针对学校位置进行处理、分析
df1 = pd.DataFrame(df.groupby("school_setting").mean()["posttest"])
df1.plot(kind="bar")
plt.title("学校位置与分数的关系")
plt.xlabel("学校位置")
plt.ylabel("均分")
plt.show()


# 对学校位置标签化处理
def fn_school_setting(x):
    if x == "Rural":
        return 0
    elif x == "Suburban":
        return 1
    else:
        return 2


df["school_setting"] = df["school_setting"].map(fn_school_setting)
print(df.head())

# 针对学校类型进行处理、分析
df1 = pd.DataFrame(df.groupby("school_type").mean()["posttest"])
df1.plot(kind="bar")
plt.title("学校类型与分数的关系")
plt.xlabel("学校类型")
plt.ylabel("均分")
plt.show()


# 对学校类型进行标签化处理
def fn_school_type(x):
    if x == "Public":
        return 0
    else:
        return 1


df["school_type"] = df["school_type"].map(fn_school_type)
print(df.head())

# 针对教室类型进行处理、分析
df1 = pd.DataFrame(df.groupby("classroom").mean()["posttest"])
df1.plot(kind="bar")
plt.title("教室类型与分数的关系")
plt.xlabel("教室类型")
plt.ylabel("均分")
plt.show()

# 从40分开始，往上每10分划分一个档次
class_0 = df1[(df1["posttest"] >= 40) & (df1["posttest"] < 50)]
class_1 = df1[(df1["posttest"] >= 50) & (df1["posttest"] < 60)]
class_2 = df1[(df1["posttest"] >= 60) & (df1["posttest"] < 70)]
class_3 = df1[(df1["posttest"] >= 70) & (df1["posttest"] < 80)]
class_4 = df1[(df1["posttest"] >= 80) & (df1["posttest"] < 90)]
class_5 = df1[(df1["posttest"] >= 90) & (df1["posttest"] < 100)]


# 教室类型的标签化处理
def fn_class(x):
    if x in class_0:
        return 0
    elif x in class_1:
        return 1
    elif x in class_2:
        return 2
    elif x in class_3:
        return 3
    elif x in class_4:
        return 4
    elif x in class_5:
        return 5
    else:
        return 6


df["classroom"] = df["classroom"].map(fn_class)
print(df.head())

# 针对教学方式进行处理、分析
df1 = pd.DataFrame(df.groupby("teaching_method").mean()["posttest"])
df1.plot(kind="bar")
plt.title("教学方式与分数的关系")
plt.xlabel("教学方式")
plt.ylabel("均分")
plt.show()


def fn_method(x):
    if x == "Standard":
        return 0
    else:
        return 1


df["teaching_method"] = df["teaching_method"].map(fn_method)

# 针对性别进行处理、分析
df1 = pd.DataFrame(df.groupby("gender").mean()["posttest"])
df1.plot(kind="bar")
plt.title("性别与分数的关系")
plt.xlabel("性别")
plt.ylabel("均分")
plt.show()

del df["gender"]

# 针对午餐方式进行处理、分析
df1 = pd.DataFrame(df.groupby("lunch").mean()["posttest"])
df1.plot(kind="bar")
plt.title("午餐与分数的关系")
plt.xlabel("午餐")
plt.ylabel("均分")
plt.show()


# 对午餐进行标签化处理
def fn_lunch(x):
    if x == "Does not qualify":
        return 0
    else:
        return 1


df["lunch"] = df["lunch"].map(fn_lunch)

# pretest字段，直接删除
del df["pretest"]

# 保存处理后的数据
df.to_csv("pre1.csv", index=None)
