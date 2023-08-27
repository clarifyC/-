import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style
info_filename1 = r"E:\pycharm files\housing_pricing_model\data\train_data.csv"
info_filename2 = r"E:\pycharm files\housing_pricing_model\data\jw.csv"
df_data = pd.read_csv(info_filename1,encoding="gbk")
df_jw = pd.read_csv(info_filename2,encoding="utf-8")
df_data = df_data.iloc[:,1:53]
df = pd.concat([df_data,df_jw],axis=1)


def draw(minprice, maxprice, df,color):
    idx = df["单价"].between(minprice, maxprice)
    xy = df.loc[idx, :]
    x = xy["经度"]
    y = xy["纬度"]

    colors = df.loc[idx, "单价"]
    style.use('ggplot')
    plt.xlim(120.1, 120.5)
    plt.ylim(31.45, 31.7)
    # 绘制散点图
    plt.scatter(x, y, c=colors, cmap=color)

    # 添加颜色图例
    cbar = plt.colorbar()
    cbar.set_label('Color')

    # 添加标题和坐标轴标签
    plt.title(f"{minprice}--{maxprice}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.figure(figsize=(50, 40))
    # 显示图形
    plt.show()


# %%

draw(0, 10000, df,"Blues")
draw(15000, 18000, df,"Greens")
draw(18000, 20000, df,"Oranges")
draw(20000, 25000, df,"Reds")
draw(25000, 50000, df,"Purples_r")
