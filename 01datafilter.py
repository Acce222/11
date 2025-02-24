# 数据查看
import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv("test_scores.csv")
print("前五行的数据为：")
print(df.head())
print("描述信息为：")
print(df.describe())
print("数据的信息为：")
print(df.info())

# 初步离散值筛选
print(df.columns)
obj_columns = ['school', 'school_setting', 'school_type', 'classroom',
               'teaching_method', 'n_student', 'student_id', 'gender', 'lunch',
               'pretest', 'posttest']

# 遍历离散分布
print("离散值的分布：")
for i in obj_columns:
    print(i)
    print(df[i].value_counts())

del df["student_id"]


# 步骤三：班级人数分组，小于20,20-25，大于25的
def fn(x):
    if x < 20:
        return 0
    elif 20 <= x <= 25:
        return 1
    else:
        return 2
df["n_student"] = df["n_student"].map(fn)
print(df.head())

# 保存代码
df.to_csv("pre.csv",index=None)