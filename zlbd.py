import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
X1, Y1 = [], []
X2, Y2 = [], []
X3, Y3 = [], []
X4, Y4 = [], []

filename = '/data/cws/STDF/zlbd/bqsquare/HEVC_BQSquare_416x240_600.txt'
with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    strs = 'POC'
    for line in lines:#3
        if line.startswith(strs):
            value = [str(s) for s in line.split()]#4
            # print(value)
            X1.append(float(value[1]))#5
            Y1.append(float(value[14]))


filename = '/data/cws/STDF/zlbd/bqsquare/STDF_BQSquare_416x240_600.txt'
with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        value = [str(s) for s in line.split()]#4
        # print(value)
        X2.append(float(value[0]))#5
        Y2.append(float(value[3]))

filename = '/data/cws/STDF/zlbd/bqsquare/RFDA_BQSquare_416x240_600.txt'
with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        value = [str(s) for s in line.split()]#4
        # print(value)
        X3.append(float(value[0]))#5
        Y3.append(float(value[3]))

filename = '/data/cws/STDF/zlbd/bqsquare/OURS_BQSquare_416x240_600.txt'
with open(filename, 'r') as f:#1
    lines = f.readlines()#2
    for line in lines:#3
        value = [str(s) for s in line.split()]#4
        # print(value)
        X4.append(float(value[0]))#5
        Y4.append(float(value[3]))


# x、y坐标以及标题
plt.xlabel('Frames', fontsize=30)
plt.ylabel('PSNR(dB)', fontsize=30)

P1 = plt.plot(X1, Y1,linewidth=3,marker='o',markersize=6,markeredgecolor='white',color="#2878B5",  label="HEVC")
P2 = plt.plot(X2, Y2,linewidth=3,marker='o',markersize=6,markeredgecolor='white',color="#057748",  label="STDF")
P3 = plt.plot(X3, Y3,linewidth=3,marker='o',markersize=6,markeredgecolor='white',color="#c89b40",  label="RFDA")
P4 = plt.plot(X4, Y4,linewidth=3,marker='o',markersize=6,markeredgecolor='white',color="#C82423",  label="Ours")
# plt.grid(linestyle='-.')
plt.legend(loc="upper left",ncol=4,fontsize=30,shadow=True) # 把标签加载到图中哪个位置
plt.tick_params(labelsize=28)
plt.xlim(50, 100)
plt.ylim(27, 31)
# plt.legend(loc='center', fontsize=40)  # 标签位置
# change x internal size
plt.gca().margins(x=0)
plt.gcf().canvas.draw()
tl = plt.gca().get_xticklabels()
# maxsize = max([t.get_window_extent().width for t in tl])
maxsize = 4
N= len(X1)
m = 0.2  # inch margin
s = maxsize / plt.gcf().dpi * N + 2 * m
margin = m / plt.gcf().get_size_inches()[0]

plt.gcf().subplots_adjust(left=margin, right=1. - margin)
plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
plt.grid(axis="y")
plt.show()
