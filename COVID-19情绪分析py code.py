from snownlp import SnowNLP
import jieba
import matplotlib.pyplot as plt

#用GB18030 因为py在打开csv的时候utf-8可能会报错：
file=open("nCov_10k_test.csv","rt",encoding='gb18030',errors='ignore')
#打开：
text=file.readlines()


#设立数据存放的字典：
counts={}
#逐行分析：
for line in text:
    words=jieba.lcut(line)
    index = SnowNLP(line)
    values = index.sentiments
    if 1 > values > 0.6:
        values = 1
        counts[1]=counts.get(1,0)+1
    if 0 < values < 0.4:
        values = -1
        counts[-1]=counts.get(-1,0)+1
    if 0.4 <= values <= 0.6:
        values = 0
        counts[0]=counts.get(0,0)+1
    print(words[0],"情绪分析值为",values)
    
#数据统计分析:
a=counts[1]
b=counts[0]
c=counts[-1]
total=a+b+c
print("\n分析统计:",counts,"总计数据为{}条".format(total))
print("备注:其中1表示乐观，0表示中立，-1表示消极")

#可视化数据:
#设置坐标：
x=1
#设置长度：
w=2
#并列柱状图
plt.bar(x,a,w,label='positive')
plt.bar(x+w,b,w,label='objective')
plt.bar(x+w*2,c,w,label='negative')
plt.legend()
plt.show()

