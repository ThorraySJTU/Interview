# MCMC采样

马尔科夫链蒙特卡罗

利用蒙特卡洛方法求解 求和或积分问题：f(x)原函数较为复杂，通过建立蒙特卡洛方法模拟求解近似值，在区间[a,b]中随机采样一个点计算(b-a)f(x0)，一个点代表所有值太粗糙，采用n个值再求平均的方法（假设前提x在区间[a,b]是均匀分布的）

如果可以得到x在[a,b]区间的概率分布函数p(x)就可以进行用f(x)/p(x)来计算。

问题转化为如何求出x的分布p(x)

### 概率分布采样

通过uniform(0,1)得到的采样样本转化得到，F分布、t分布、Beta分布、gamma分布

### 接受-拒绝采样

设定一个程序可采样的分布q(x)比如高斯分布，然后按照一定的方法拒绝默写杨蓓，以达到接近p(x)分布的目的。

设定一个常用概率分布函数q(x)，以及常量k，使得p(x)总在kq(x)的下方，采样一个样本z0，从均匀分布(0,kq(z0))中采样得到一个样本u，如果u落在了灰色区域，则拒绝采样，否则接受样本，重复以上过程得到n个接受的样本。

——对于一些二维分布p(x,y)，大部分时候只能得到条件分布，但是很难得到二维分布的一般形式

——对于一些高维的复杂非常见分布，很难找到一个合适的q(x)和k

### MCMC马尔科夫链

某一时刻状态转移的概率只依赖于前一个状态。

马尔科夫链模型的状态转移矩阵和蒙特克罗方法需要的概率分布样本集的关系。

马尔科夫链模型的状态转移矩阵收敛到的稳定概率分布与初始状态概率分布无关

#### 基于马尔科夫链采样

基于初始任意简单概率分布（如高斯分布）采样得到状态x0，基于条件分布采样状态值x1，迭代n次。（前提是需要采样样本的平稳分布所对应的马尔科夫链状态转移矩阵）->MCMC通过迂回的方法解决了这个问题

**MCMC采样和M-H采样**

马尔科夫链的细致平稳条件，pi(i)P(i,j) = pi(j)P(j,i)

构造alpha(. , .)函数使得pi(i)P(i,j)alpha(i,j) = pi(j)P(j,i)alpha(j,i)

对于目标矩阵P可以通过一个马尔科夫链状态转移矩阵Q乘alpha得到——alpha(i,j)被称为接受率

如果接受率很小，则大部分采样值都被拒绝转移，效率低，可能进行了很多次马尔科夫链还未收敛。

**M-H采样**

将接受率在等式两边同时放大，提升采样率后，采样的效率会提升。

```python
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt


def norm_distribute(theta):
	y = norm.pdf(theta, 3, 2)
	return y

T = 5000
pi = [0 for i in range(T)]
sigma = 1
t = 0

while t < T-1:
	t = t+1
	pi_s = norm.rvs(loc = pi[t-1], scale = sigma, size = 1, random_state = None)
    # 相对于MCMC采样的改进
	alpha = min(1, (norm_distribute(pi_s[0]) / (norm_distribute(pi[t-1]))))

	u = random.uniform(0, 1)
	if(u < alpha):
		pi[t] = pi_s[0]
	else:
		pi[t] = pi[t-1]


num_bins = 50
plt.hist(pi, num_bins, facecolor = 'red', alpha = 0.7)
plt.scatter(pi, norm.pdf(pi, 3, 2), c='blue')
plt.show()

```

M-H采样的问题：

- 数据特征太多，计算接受率的时间消耗过大，而且接受率小于1，总会有拒绝采样
- 特征唯独太大，很难求出目标的各特征唯独的联合分布

**Gibbs采样**

平面上任意两点E，F满足细致平稳条件 pi(E)P(E->F) = pi(E)P(F->E)

