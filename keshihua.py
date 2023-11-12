import matplotlib.pyplot as plt

X1, Y1 = [], []
X2, Y2 = [], []
X3, Y3 = [], []
X4, Y4 = [], []

# filename = '/data/SwinRestormer/zlbd/bqsquare/HEVC_BQSquare_416x240_600.txt'
# with open(filename, 'r') as f:#1
#     lines = f.readlines()#2
#     strs = 'POC'
#     for line in lines:#3
#         if line.startswith(strs):
#             value = [str(s) for s in line.split()]#4
#             # print(value)
#             X1.append(float(value[1]))#5
#             Y1.append(float(value[14]))
#
#
# filename = '/data/SwinRestormer/zlbd/bqsquare/STDF_BQSquare_416x240_600.txt'
# with open(filename, 'r') as f:#1
#     lines = f.readlines()#2
#     for line in lines:#3
#         value = [str(s) for s in line.split()]#4
#         # print(value)
#         X2.append(float(value[0]))#5
#         Y2.append(float(value[3]))

filename = '/data/cws/swinrestormer/zlbd/basketballpass/HEVC_BasketballPass_416x240_500.txt'
with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    strs = 'POC'
    for line in lines:#3
        if line.startswith(strs):
            value = [str(s) for s in line.split()]#4
            # print(value)
            X3.append(float(value[1]))#5
            Y3.append(float(value[14]))

filename = '/data/cws/swinrestormer/zlbd/basketballpass/OURS_BasketballPass_416x240_500.txt'
with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        value = [str(s) for s in line.split()]#4
        # print(value)
        X4.append(float(value[0]))#5
        Y4.append(float(value[3]))

list =[Y4[i] - Y3[i] for i in range(len(Y4))]
print('最大差值为：',max(list),'最大差值索引为：',list.index(max(list))+1)

sorted_id = sorted(range(len(list)), key=lambda k: list[k], reverse=True)
list1 = [sorted_id[i]+1 for i in range(len(sorted_id))]
print('元素索引序列：', list1)

#
# # x、y坐标以及标题
# plt.xlabel('Frames', fontsize=26)
# plt.ylabel('PSNR(dB)', fontsize=26)
#
# # P1 = plt.plot(X1, Y1,linewidth=3,marker='o',markersize=6,markeredgecolor='white',color="#2878B5",  label="HEVC")
# # P2 = plt.plot(X2, Y2,linewidth=3,marker='o',markersize=6,markeredgecolor='white',color="#057748",  label="STDF")
# # P3 = plt.plot(X3, Y3,linewidth=1,marker='o',markersize=6,markeredgecolor='white',color="#c89b40",  label="RFDA")
# P4 = plt.plot(X4, list,linewidth=1,marker='o',markersize=6,markeredgecolor='white',color="#C82423",  label="OURS")
# # plt.grid(linestyle='-.')
# plt.legend(loc="upper left",ncol=4,fontsize=26,shadow=True) # 把标签加载到图中哪个位置
# plt.tick_params(labelsize=24)
# # plt.xlim(300, 350)
# # plt.ylim(30, 36)
# # plt.legend(loc='center', fontsize=40)  # 标签位置
# # change x internal size
# plt.gca().margins(x=0)
# plt.gcf().canvas.draw()
# tl = plt.gca().get_xticklabels()
# # maxsize = max([t.get_window_extent().width for t in tl])
# # maxsize = 4
# # N= len(X1)
# # m = 0.2  # inch margin
# # s = maxsize / plt.gcf().dpi * N + 2 * m
# # margin = m / plt.gcf().get_size_inches()[0]
# #
# # plt.gcf().subplots_adjust(left=margin, right=1. - margin)
# # plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
# # plt.grid(axis="y")
# plt.show()
