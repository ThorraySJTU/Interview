# Interview

## 待投递

Check Check:

- [x] 快手

- 字节跳动

- 微软 9月16日截至

- [x] 中国银行 9月17日截至

投递：

- [ ] Google 
- [ ] Microsoft 
- [x] 中金技术所 
- [x] 小红书 
- [x] 中国移动 
- [ ] 中国联通 
- [ ] 中国电信

## 8月27日更新

1. Python中 \_\_init\_\_ 和 \_\_new\_\_的区别？
```
Ans: 

__new__作为构造器，起构建一个类实例的作用，而__init__作为初始化器，起初始化一个已被创建的实例的作用。

对于两个函数的调用顺序，先通过__new__构造实例，__init__在__new__返回一个实例的时候调用，实例作为self参数被传入__init__函数。

__new__返回一个已经存在的实例，则__init__不会被调用
```

## 8月26日更新

## 算法题

1. Leetcode 76 最小覆盖子串问题 - 滑动窗口

## 面经

1. CNN能否使用在NLP中？

   CNN的应用条件是要求卷积对象有局部相关性，文本符合这个条件。

   - input shuffle之后不影响结果，就不可以。

   用一个滑动窗口（宽度是2）滑过一个句子，可以提取出句子中的所有2-gram

2. 关于RNN的梯度消失问题 、 LSTM如何优化

   RNN总的梯度是不会消失的，梯度越传越弱，只是远距离的梯度消失，近距离的梯度不会消失，梯度被近距离的梯度主导，模型难以学到远距离的依赖关系

3. Hinge loss的理解

4. Dropout怎么实现？

   针对每一层的np.adarray生成一个同样大小的0,1分布的np.adarray，用来当做mask与当前np.adarray相乘，以此来进行选择。

5. 介绍GBDT

## 蚂蚁金服 电话面经

1. 如何解决overfitting

   1. BatchNorm
   2. 增加训练集
   3. 正则化
   4. Dropout

2. 交叉熵损失与均方差损失的差别

   1. 在线性回归问题中，均方差误差的梯度是线性的
   2. 在逻辑回归问题中（二分类），交叉熵损失可以表示成类线性的

3. Gradient descent的方向和大小如何确定

4. Adam优化器的工作原理

5. EM算法

6. Deep learning 和Machine learning的差别

7. Matrix正定矩阵如何判定

8. 对大数定理的理解

   样本数量越多，则其算术平均值就有越高的概率接近期望值

## 旷视面经 8.25-网

1. 了解的神经网络
2. 介绍PCA，特征值分解、奇异值分解
3. 算法题：
   1. 十进制转二进制
   2. numpy完成average pooling

网络权重初始化的方法
## 笔试题知识点

- [x] 霍夫曼带权树
- [ ] 拓扑排序算法
```
从中选出一个入度为0的顶点作为序列的下一顶点。

从N网中删除所选顶点及其所有的出边。

反复执行上面两步操作，直到已经选出了图中的所有顶点，或者再也找不到入度非0的顶点时算法结束。
```
- [x] 读取带空格的字符串 getline(cin, s)

- [ ] 正态分布

```
P(\mu - \sigma <= X <= \mu + \sigma) = 0.6828
P(\mu - 2\sigma <= X <= \mu + 2\sigma) = 0.9544
P(\mu - 3\sigma <= X <= \mu + 3\sigma) = 0.9974
```
- [ ] 牛顿法求解

- [ ] KMP算法

- [ ] map / multimap的底层实现是红黑树，查找的复杂度为O（logN），unordered_map的底层实现是哈希表，查找的复杂度为O（1）

- [ ] BatchNorm是对一个Batch下的多张图片的同一通道做Normalization

## 字节跳动

算法题目：

- [x] Leetcode 951 
- [ ] Leetcode 786 （AI Lab）
- [ ] 圆上三个点成锐角三角形的概率。
- [ ] 已知单链表，要求奇数位置降序，偶数位置升序
- [ ] 一个整数数组A（有正有负），A[i]可选可不选，但必须确保相邻的两个数不同时选，问可选的最大和？
- [ ] 接雨水
- [ ] 二叉树子路径和为k的路径个数

## 百度

算法题目：

- [ ] 给出一个区间的集合，请合并所有重叠的区间

- [ ] 最小编辑距离

## 腾讯

C++题目：

Vector怎么分配内存？

vector容器erase时迭代器失效怎么解决？

vector不指定一块内存大小的数组的连续储存，访问随机，节省空间 / 在内部进行插入删除效率低，只能在vector的最后进行push和pop / 当动态添加的数据超过vector默认分配的大小时，要重新分配、拷贝和释放。

算法题目：

## 寒武纪

基础问题：

描述空洞卷积

多核CPU上矩阵乘法怎么加速
 
从头实现Conv2D

## 网易互娱提前批

样本方差和总体方差的区别

BN和LN的区别

## NVIDIA

NMS的优化

## 蚂蚁金服

HMM

## 快手

### C++开发

C++中static怎么用？存放在内存哪个位置？

进程的内存分布

虚表跟纯虚类

算法题目：

- [ ] Leetcode 540

## 科大讯飞

聚类方法有哪些？

PCA和LDA的区别

## 华为

- [ ] Leetcode 1162
