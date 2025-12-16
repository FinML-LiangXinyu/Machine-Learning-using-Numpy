# Machine-Learning-using-Numpy
## 一、多层感知机神经网络
对于实例 ${(x,y)}$ ，有 $x=\left[x^1,x^2,x^3,\ldots,x^m\right]^T\in\chi\subseteq R^m$ ， $\chi$ 表示 $m$ 维特征子空间，若 $y$ 为离散标签，有 $y\in\mathcal{V}={\{c_k\}}_{k=1,2,...,K}$ ，若 $y$ 为连续标签，有 $y\in R$ 。

对于 $T$ 层 $MLP$ 神经网络，用 $h$ 表示 $MLP$ 神经网络的隐层输入，假设第 $t$ 层隐层有 $m$ 个神经元，则有 $h^{(t)}=[{h_1}^{(t)},{h_2}^{(t)},...,{h_m}^{(t)}]$ 代表第 $t$ 层隐层的输出和第 $t+1$ 层隐层的输出。其中， $h^{(0)}=x$ ， $h^{(T)}$ 为 $MLP$ 神经网络的最终输出。利用矩阵 $W^{(t)}=[w_1^{(t)},w_2^{(t)},...,w_m^{(t)}]$ 和偏置 $b^{(t)}$ 将第 $t-1$ 层的输出 $h^{(t-1)}$ 转换为 $z^{(t)}=[{z_1}^{(t)},{z_2}^{(t)},...,{z_m}^{(t)}]$ ，再对 $z^{(t)}$ 进行非线性变换 $\sigma(·)$ 后，成为第 $t$ 层隐层的输出 $h^{(t)}$ 。上述关系用公式表示为：

$z^{\left(t\right)}={W^{\left(t\right)}}^Th^{\left(t-1\right)}+b^{\left(t\right)}$

$h^{\left(t\right)}=\sigma\left(z^{\left(t\right)}\right)=\sigma\left({W^{\left(t\right)}}^Th^{\left(t-1\right)}+b^{\left(t\right)}\right)$

非输出层的非线性变换 $\sigma(·)$ 又称为激活函数，输出层的非线性变换 $\sigma(·)$ 通常采用 $softmax$ 函数：

$h^{\left(T\right)}=softmax\left(z^{\left(T\right)}\right)$

$h_k^{\left(T\right)}=softmax\left(z_k^{\left(T\right)}\right)=\frac{e^{{z_k}^{(T)}}}{\sum_{k=1}^{K}e^{{z_k}^{(T)}}}$

令 ${\delta}^{\left(t\right)}$ 表示 $\frac{\partial L}{\partial z^{\left(t\right)}}$ ，利用矩阵微分公式：

$dL=tr\left({\frac{\partial L}{\partial z^{\left(t\right)}}}^Tdz^{\left(t\right)}\right)=tr\left({\frac{\partial L}{\partial z^{\left(t\right)}}}^Td{W^{\left(t\right)}}^Th^{\left(t-1\right)}\right)=tr\left(h^{\left(t-1\right)}{\frac{\partial L}{\partial z^{\left(t\right)}}}^Td{W^{\left(t\right)}}^T\right)=tr\left(h^{\left(t-1\right)}{\delta^{\left(t\right)}}^Td{W^{\left(t\right)}}^T\right)$

可得：

$\frac{\partial L}{\partial {W^{\left(t\right)}}^T}=\delta^{\left(t\right)}{h^{\left(t-1\right)}}^T$

对于第 $t-1$ 层隐层，存在：

$h^{\left(t-1\right)}=\sigma\left(z^{\left(t-1\right)}\right)=\sigma\left({W^{\left(t-1\right)}}^Th^{\left(t-2\right)}+b^{\left(t-1\right)}\right)$

利用矩阵微分公式：

$dL=tr\left({\frac{\partial L}{\partial z^{\left(t\right)}}}^Tdz^{\left(t\right)}\right)=tr\left({\delta^{\left(t\right)}}^T{W^{\left(t\right)}}^T\sigma^\prime\left(z^{\left(t-1\right)}\right)\odot d z^{\left(t-1\right)}\right)=tr\left(\left(W^{\left(t\right)}\delta^{\left(t\right)}\right)^T\sigma^\prime\left(z^{\left(t-1\right)}\right)\odot d z^{\left(t-1\right)}\right)=tr\left(\left(W^{\left(t\right)}\delta^{\left(t\right)}\odot\sigma^\prime\left(z^{\left(t-1\right)}\right)\right)^Tdz^{\left(t-1\right)}\right)$

可得第 $t-1$ 层隐层的 ${\delta}^{\left(t-1\right)}$ 和第 $t$ 层隐层的 ${\delta}^{\left(t\right)}$ 之间存在如下关系式：

$\delta^{\left(t-1\right)}=\frac{\partial L}{\partial z^{\left(t-1\right)}}=W^{\left(t\right)}\delta^{\left(t\right)}\odot\sigma^\prime\left(z^{\left(t-1\right)}\right)$

对于离散实例，有该实例预测为 $c_k$ 的概率为： $\pi_k=p\left(y=c_k\middle| x\right)=\frac{e^{{z_k}^Tx}}{\sum_{k=1}^{K}e^{{z_k}^Tx}}$ 。其中， $z_k$ 是对应类别 $c_k$ 的权重向量： $z_k=\left[z_k^1,z_k^2,z_k^3,\ldots,z_k^m\right]^T\in R^m$ 。此时， $MLP$ 神经网络的输出层 $h^{(T)}$ 为一向量，向量中的元素 $h_k^{(T)}$ 表示为： ${{h}_k}^{\left(T\right)}=\frac{e^{{z_k}^{(T)}}}{\sum_{c=1}^{C}e^{{z_c}^{(T)}}}$ 。

输出层 $h^{(T)}$ 中元素对向量中第 $k$ 个元素的中间变量 $z_k^{(T)}$ 的偏导数为：

$\frac{\partial{{h}_k}^{\left(T\right)}}{\partial{z_k}^{\left(T\right)}}=h_k^{\left(T\right)}\left(1-h_k^{\left(T\right)}\right)$

$\frac{\partial{{h}_{c\neq k}}^{(T)}}{\partial{z_k}^{\left(T\right)}}=-{{h}_c}^{(T)}{{h}_k}^{(T)}$

对于离散实例，损失函数为如下交叉熵损失函数：

$L=-\sum_{c=1}^{C}{y_cln\left({{h}_c}^{(T)}\right)}$

则输出层对应的 ${\delta_k}^{\left(T\right)}$ 和 ${\delta}^{\left(T\right)}$ 如下：


${\delta_k}^{\left(T\right)}=\frac{\partial L}{\partial{z_k}^{\left(T\right)}}=-\sum_{c=1}^{C}{\frac{y_c}{{{h}_c}^{(T)}}\frac{\partial{{h}_c}^{(T)}}{\partial{z_k}^{\left(T\right)}}}=-\frac{y_k}{{{h}_k}^{(T)}}{{h}_k}^{(T)}(1-h_k)+\sum_{c\neq k}{\frac{y_c}{{{h}_c}^{\left(T\right)}}{{h}_c}^{\left(T\right)}{{h}_k}^{(T)}}=-y_k(1-{{h}_k}^{(T)})+\sum_{c\neq k}{y_c{{h}_k}^{\left(T\right)}}=-y_k(1-{{h}_k}^{(T)})+{{h}_k}^{\left(T\right)}(1-y_k)={{h}_k}^{\left(T\right)}-y_k$ 

$\delta^{(T)}=\frac{\partial L}{\partial z^{\left(T\right)}}={h}^{\left(T\right)}-y$

针对不同的激活函数 $\sigma(·)$ ， $\sigma^\prime\left(z^{\left(t\right)}\right)$ 有着不同的表达形式：

激活函数为： $y=sigmoid(x)=\frac{1}{1+e^{-x}}$。则： $\sigma^\prime\left(z^{\left(t\right)}\right)=h^{\left(t\right)}\odot(1-h^{\left(t\right)})$ 

激活函数为： $y=tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$。则： $\sigma^\prime\left(z^{\left(t\right)}\right)=1-h^{\left(t\right)}\odot h^{\left(t\right)}$ 

激活函数为： $y=relu(x)=max(0,x)$ 。则：

$\sigma^\prime\left({z_k}^{\left(t\right)}\right)=\left\{\begin{array}{}1,if\quad {z_k}^{(t)}>0\\0,if\quad {z_k}^{(t)}≤0\end{array}
\right.$

对于连续实例，损失函数为如下均方误损失函数：

$L=\frac{1}{2}\left(y-h^{\left(T\right)}\right)^2$

回归算法输出层 $h^{(T)}$ 为一标量，输出层对应的 ${\delta}^{\left(T\right)}$ 同样为一标量：

$\delta^{\left(T\right)}=\frac{\partial L}{\partial z^{\left(T\right)}}=h^{\left(T\right)}-y$
## 二、简单循环神经网络
对于长度为 $T$ 的时间序列数据 $x=\left[x_1,x_2,\ldots,x_t,\ldots,x_T\right]$ ， $x_t$ 为时刻 $t$ 的输入向量。简单循环神经网络 $S-RNN$ 算法的结构单元如下：

$s_t=Uh_{t-1}+Wx_t+b$

$h_t=tanh\left(s_t\right)$

$z_t=Vh_t+c$

$\widehat{y_t}=softmax\left(z_t\right)$

其中， $h_{t-1}$ 代表 $t-1$ 时刻的隐状态， $x_t$ 为时刻 $t$ 的输入，时刻 $t$ 的净输入 $s_t$ 经过 $tanh(·)$ 激活函数转换为 $t$ 时刻的隐状态，时刻 t 的净输入 $z_t$ 经过 $softmax(·)$ 转换为时刻 $t$ 的最终输出 $\widehat{y_t}$ ， $U$ 、 $W$ 、 $V$ 为神经网络的权重矩阵， $b$ 、 $c$ 为神经网络净输入的偏置向量。

 $S-RNN$ 结构单元表明：在时刻 $t$ ，时刻 $t$ 的输入 $x_t$ 和上一个时刻的隐状态 $h_{t-1}$ 共同决定了时刻 $t$ 的隐状态 $h_t$ 以及输出 $\widehat{y_t}$ 。对于长度为 $T$ 的时间序列数据 $x$ ，循环使用该结构 $T$ 次可以得到 $S-RNN$ 算法在各个时刻的输出 $\hat{y}=\left[\widehat{y_1},\widehat{y_2},\ldots,\widehat{y_t},\ldots,\widehat{y_T}\right]$ 。上述关系用数学公式表示为：

$\widehat{y_t}=P(y_t|x_t,h_{t-1})=P(y_t|x_t,x_{t-1},h_{t-2})=P(y_t|x_t,x_{t-1},...,x_1)$

 $S-RNN$ 算法的反向传播过程如下：

矩阵 $V$ 在每个时刻 $t$ 都会影响该时刻的损失 $L_t$ 进而影响总损失 $L$ ，对应前向链式传播路径： $V\rightarrow z_1,z_2,\ldots,z_t,\ldots,z_T\rightarrow L_1,L_2,\ldots,L_t,\ldots,L_T\rightarrow L$ ，可推：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial z_t}\right)^Tdz_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial z_t}\right)^TdVh_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial z_t}{h_t}^T\right)^TdV}\right)$

$\frac{\partial L}{\partial V}=\sum_{t=1}^{T}{\frac{\partial L}{\partial z_t}{h_t}^T}=\sum_{t=1}^{T}{\frac{\partial L}{\partial z_t}\otimes h_t}=\sum_{t=1}^{T}{\left(\widehat{y_t}-y_t\right)\otimes h_t}$

 $t$ 时刻的向量 $s_t$ 通过影响损失 $L_t$ 和 $t+1$ 时刻的向量 $s_{t+1}$ 进而影响总损失 $L$ ，对应前向链式传播路径： $s_t\rightarrow L_t\rightarrow L；s_t\rightarrow s_{t+1}\rightarrow L_{t+1}\rightarrow L$ ，可推：

$dL=tr\left(\left(\frac{\partial L}{\partial s_{t+1}}\right)^Tds_{t+1}\right)+dL_t=tr\left(\left(\frac{\partial L}{\partial s_{t+1}}\right)^TU\left(1-h_t\odot h_t\right)\odot d s_t\right)+tr\left(\left(V^T\frac{\partial L_t}{\partial z_t}\odot\left(1-h_t\odot h_t\right)\right)^Tds_t\right)=tr\left(\left(U^T\frac{\partial L}{\partial s_{t+1}}\odot\left(1-h_t\odot h_t\right)\right)^Tds_t+\left(V^T\frac{\partial L_t}{\partial z_t}\odot\left(1-h_t\odot h_t\right)\right)^Tds_t\right)=tr\left(\left(U^T\frac{\partial L}{\partial s_{t+1}}\odot\left(1-h_t\odot h_t\right)+V^T\frac{\partial L_t}{\partial z_t}\odot\left(1-h_t\odot h_t\right)\right)^Tds_t\right)$

$\frac{\partial L}{\partial s_t}=U^T\frac{\partial L}{\partial s_{t+1}}\odot\left(1-h_t\odot h_t\right)+V^T\frac{\partial L_t}{\partial z_t}\odot\left(1-h_t\odot h_t\right)$

 $T$ 时刻的向量 $s_T$ 通过影响损失 $L_T$ 进而影响总损失 $L$ ，对应前向链式传播路径： $s_T\rightarrow L_T\rightarrow L$ ，可推：

$dL=tr\left(dL_T\right)=tr\left(\left(\widehat{y_T}-y_T\right)^Tdz_T\right)=tr\left(\left(\widehat{y_T}-y_T\right)^TVdh_T\right)=tr\left(\left(\widehat{y_T}-y_T\right)^TV\left(1-h_T\odot h_T\right)\odot d s_T\right)=tr\left(\left(V^T\left(\widehat{y_T}-y_T\right)\right)^T\left(1-h_T\odot h_T\right)\odot d s_T\right)=tr\left(\left(V^T\left(\widehat{y_T}-y_T\right)\odot\left(1-h_T\odot h_T\right)\right)^Tds_T\right)$

$\frac{\partial L}{\partial s_T}=V^T\left(\widehat{y_T}-y_T\right)\odot\left(1-h_T\odot h_T\right)$

矩阵 $U$ 在每个时刻 $t$ 都会影响该时刻的损失 $L_t$ 进而影响总损失 $L$ ，对应前向链式传播路径： $U\rightarrow s_1,s_2,\ldots,s_t,\ldots,s_T\rightarrow L_1,L_2,\ldots,L_t,...,L_T\rightarrow L$ ，可推：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial s_t}\right)^Tds_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial s_t}\right)^TdUh_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial s_t}{h_{t-1}}^T\right)^TdU}\right)=tr\left(\left(\sum_{t=1}^{T}{\frac{\partial L}{\partial s_t}{h_{t-1}}^T}\right)^TdU\right)$

$\frac{\partial L}{\partial U}=\sum_{t=1}^{T}{\frac{\partial L}{\partial s_t}{h_{t-1}}^T}=\sum_{t=1}^{T}{\frac{\partial L}{\partial s_t}\otimes h_{t-1}}$

矩阵 $W$ 在每个时刻 $t$ 都会影响该时刻的损失 $L_t$ 进而影响总损失 $L$ ，对应前向链式传播路径： $W\rightarrow s_1,s_2,\ldots,s_t,\ldots,s_T\rightarrow L_1,L_2,\ldots,L_t,...,L_T\rightarrow L$ ，可推：

$\frac{\partial L}{\partial W}=\sum_{t=1}^{T}{\frac{\partial L}{\partial s_t}{x_t}^T}=\sum_{t=1}^{T}{\frac{\partial L}{\partial s_t}\otimes x_t}$

向量 $b$ 在每个时刻 $t$ 都会影响该时刻的损失 $L_t$ 进而影响总损失 $L$ ，对应前向链式传播路径： $b\rightarrow s_1,s_2,\ldots,s_t,\ldots,s_T\rightarrow L_1,L_2,\ldots,L_t,...,L_T\rightarrow L$ ，可推：

$\frac{\partial L}{\partial b}=\sum_{t=1}^{T}\frac{\partial L}{\partial s_t}$

向量 $c$ 在每个时刻 $t$ 都会影响该时刻的损失 $L_t$ 进而影响总损失 $L$ ，对应前向链式传播路径： $c\rightarrow z_1,z_2,\ldots,z_t,\ldots,z_T\rightarrow L_1,L_2,\ldots,L_t,...,L_T\rightarrow L$ ，可推：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial z_t}\right)^Tdz_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial z_t}\right)^Tdc}\right)=tr\left(\left(\sum_{t=1}^{T}\frac{\partial L}{\partial z_t}\right)^Tdc\right)$

$\frac{\partial L}{\partial c}=\sum_{t=1}^{T}\frac{\partial L}{\partial z_t}$
