---
layout: page
permalink: /ideas/2025/Matlab编程笔记/index.html
title: Matlab编程笔记
---
## Chap 1 
1.Matlab只是一个功能强大的计算器
2.基本数学运算：
(1)两个数相加：a+b，a-b (2)相乘 a*b (3)相除：a/b表示a是被除数
(4)幂运算：a^b(其中a,b都是实数)
3.变量：
- 变量定义：r=4;(直接写就可以了，有；表示不会有相应的输出，没有的话会输出r的值)
- 命名规则：同C语言
- 变量查询：who:显示工作空间中的所有变量，whos：查看所有变量的详细属性
- 变量清除：clear r(清除r), clear(清除所有变量)
4.基本数学函数：三角，双曲，log(就是ln)，log19,l0g2,exp,
    随机数rand 取整：floor,ceil round分别表示下取整，上取整和四舍五入
5.数值类型：int8,double,char, cell array(单元数组)，struct array(结构数组)
6.复数：
- 定义：a=3+2i;
- 相应的运算：abs,angle(辐角)，real(实部)，imag(虚部)，complex(3，2)，conj(a)//a的共轭

以上所有的数据类型的存储格式都是矩阵，标量->1*1,一位的称为数组
7.数组的定义：
- 定义：x1=[1 2 3]or[1,2,3]
- 可以使用冒号来生成间隔相等的数组，如a=[1:5](1 2 3 4 5),b=[-2:3:8] (-2,1,4,7),fs=10;x=[0:1/fs:2*pi]  (a:b:c)表示从a到c每隔开b取出一个数
8.矩阵的定义
- 定义：a=[1 2 3;4 5 6] (用;来换行)，比如a=[1;2;3]为一个三行的列向量
- 矩阵的元素运算(数组也是矩阵)：x=[1 2 3;4 5 6],sin(x),exp(x)表示对于矩阵的每一个元素进行运算，比如y=sin(2*pi*3*x);
- **plot(x,y)表示以x的值为横轴，y的值为纵轴绘图**
9.随机数：
- x=rand//表示从0-1之间均匀分布的随机数
- x=randn(1,10)表示产生均值为0,方差为1的恶搞四分布随机数
10.逻辑运算:除了不等于~=其余都和C语言一样，&& || ~(非)
11.clc
## Chap 2
### Part 1 数组与矩阵
1.常见的矩阵运算
- 矩阵生成函数：(1)a=ones(5)生成5*5的全为1的矩阵或数组 (2)zeros(全0) (3)eye (单位矩阵) (4)rand(随机数的矩阵) (5) magic(行，列，对角线元素之和相等的方阵)
- 数组寻址：已有x=[10 20 30 40 50 60] (符号从1还是)
  (1)x(4)=40 (2)x(3:8)将3号~8号提取出来成为一个新的数组 (3)x(end) (4)x([2,4,5])(根据数组里面的数对应的东西去寻址)
- 矩阵寻址:若已有x=magic(5)
  (1)x(3,2)=7//直接提取元素 (2)x(2,:),x(:,1)直接提取某一行或列中的元素
- find函数：ind=find(abs(x>0.8)),plot(t(ind),x(ind))其中ind是存储位置的东西
- 矩阵常见函数：max,min,mean(a)都是统计矩阵中的最大，最小，中位数的东西，sort排序，repmat(A,m,n)表示将A分别在列上复制n次，行上复制m次
2.矩阵常见操作：
- 转置: A'即为A的转置，而对于复数的矩阵而言直接A'是共轭，而A.'才是转置
- 矩阵运算：直接A*B,A^2即可，只要矩阵是满足要求的
- 行列式det(A)，秩rank(A)表示矩阵中线性无关的行数或列数， 迹，特征值之和，trace(A)
- 矩阵的逆inv(A)
- 对于矩阵的运算的时候，如果是对于元素运算，就要加上点A.^2，否则会有问题，常见的如果是对于矩阵的元素进行运算的话就要使用.*,./,.^之类的东西
### Part 2 脚本与函数
1.输入与输出：
(1)a=input('prompt'//提示词,'s'//默认如果没有输入的话就是实数，否则就是字符串)
(2)disp(a//表示输出的东西是一个变量，or 'prompt'//表示输出的东西是一个字符串)
2.基本逻辑语法
(1)循环：for 循环变量=表达式 比如 k=1:n
            循环体
        end//表示循环变量依次赋予向量中每一个元素的值
   以及 while 表达式
            循环体
        end//只要表达式的值不为0程序继续进行
(2)判断：if 
        elseif
        else
        end
以及有break,continue的使用
3.关于脚本文件与自定义函数
(1)脚本文件存储自定义函数
存储位置必须与要调用的东西在同一个目录下面，而且对于脚本文件内部是来写一个叫abc的文件，那么文件名只可以叫abc.m
(2)格式：
```
function y=my1stfunc(x)//函数文件第一行是函数名字，my1stfunc是函数名，表示输入变量是x,同时也可以有多个输入变量，y为输出变量，也可以有多个
    z=x.^2;
    y=sin(z);
```
(3)子函数：一个函数文件可以包含多个函数，但是子函数只可以在函数文件内部的主函数和其余的子函数调用，但是不可以在外部调用
(4)函数句柄@
变量名=@(输入参数列表)运算表达式
比如sqr=@(x)x.^2,则a=sqr(3)之后为9，以及ln=@(x) log(x),a=ln(3),从而可以提高函数的调用速度，但是当工作区清空的时候，用函数句柄创建的函数失效。
(5)内联函数inline类似于C中的define的功能
4.程序优化：
(1)循环向量化：比如for t=1:10000,y(t)=sin(t),与t=1:10000; y=sin(t)时间差了好多
(2)为数组预先分配内存：比如在for k=2:1e8 y(k)=y(k-1)+1; 操作之前加上y=zeros(1,1e8)预先把内存开好会加快10倍的速度。

### Part 3 常用信号的产生
1.指数衰减的正弦信号：
```
f0=1;
fs=20;
t=0:1/fs:5;
x=exp(-t).*sin(2*pi*t);
plot(t,x)//plot运算的时候前面的是横轴，右边的是纵轴的元素
```
2.线性调频信号：满足x(t)=sin(2pif(t)t),f(t)=f0+beta*t,beta=(f1-f0)*t/(t1-0)
```
t1 = 2;
fs = 200;
t = 0:1/fs:t1; 
f0 = 1;
f1 = 10;
beta = (f1-f0)/(t1-0)*t;
x = sin(2*pi*beta.*t);
plot(t, x)
```
3.sinc信号：sinx/x,注意axis tight是将图像集中到中央的东西，若有函数t=-(10*pi):0.01*pi:10*pi;length(t)给出的是t的总长度
4.连续周期型号的产生
(1)几个波的连接：已经有len1,len0;
- 连接两个东西：sig1=[len1,len0];
- 重复5次：sig=repmat(sig1,1,5);//一定要注意对于重复的次数
**特别注意，如果我们希望频率是200Hz,fs=200,t=0:1/fs:?,对于?的东西而言是如果希望重复T秒，那么就是T-1/fs，因为第一次0也算在里面**
(2)直线的产生：len1 = ones(1,floor(T*fs*0.5));
len0 = zeros(1,floor(T*fs*(1-0.5)));,同理对于锯齿波，使用比例来计算
(3)几个常见的图像函数
- axis([a b,c d])表示横轴的范围为a~b,纵轴的范围是c~d
- plot(x,y)表示横轴是写的是x向量中的值，纵轴写的是y向量中的值
- axis tight 表示的是图像集中
- grid on表示在图像中加上网格，hold on表示要画多张图像的时候原来的图像不会消失
- shg表示将该窗口始终放到最前面，pause(t)t表示停顿t秒。

> 同时可以注意以下，如果在特别的优化下面，即为使用向量化循环的时候，每秒可以进行2.5亿次加法运算(我的计算机)。

## Chap 3 关于绘图
### Part 1 二维绘图
1.基本绘图函数：
(1)plot(x,y)分别表示的是横坐标和纵坐标，要求**x,y向量的长度必须要一样**
(2)如果要绘制多条曲线的话，如y1=sin(x),y2=cos(x),y3=sin(x)+cos(x),则为plot(x,y1,x,y2,x,y3)横坐标与纵坐标依次排列,或者令z=[y1;y2;y3]则plot(x,z)依次对于每一个行向量画图
(3)plot(x,y,'*-'),plot(x,y,'k:*');
- 对于'*'表示曲线的点形，包括o,+,x,*,.,^,v,>,<,square,diamond,pentagram,hexagram,None(圆形，加号，叉号，星，点，上三角，下三角，右三角，左三角，方形，菱形，五角星，六角形)
- 对于'-'设置曲线的线型，-实现，--虚线，：点线，-.点虚线
- k:表示设置颜色，b,c,g,k,m,r,w,y(blur,cyan,green,black,magenta,red,white,yellow)
2.加入说明文字
(1)xlabel('t=0 to 2\pi);ylabel('values of sin(t) and e^{-x})
表示在x轴和y轴的边上分别写上这样的话，而且满足latex格式
(2)title('Function grams')表示在图的上方写上标题function grams
(3)legend('sin(t)','e^{-x}')对于plot中画了多个函数，对应的在右上角的框中表示每一种限行表示那种东西
(4)text(x,y,'内容')表示在(x,y)为起始位置加入以下的内容
3.其余函数
(1)保持：hold on命令可以让图像保持原来的样子，如果要清除状态，使用hold off(在一张的图上面同时绘制多个曲线)
(2)网格：grid函数，grid on为加上网格，grid off为去除网格，grid minor为细网格
(3)双坐标轴 plotyy(x,y1,x,y2,'plot),yyaxis left设置位置
(4)坐标轴控制：axis([xmin,xmax,ymin,ymax])，axis nomal(默认的长宽比)，axis square(长宽比为1)axis tight图轴紧靠图形
(5)在一个图中绘制多个图形：subplot(m,n,p)表示绘制m*n个图形，p表示第p个图，从左往右，从上往下计数，如果要把几个东西合起来就是 subplot(2,2,[1,3]);
subplot(2,2,4); plot(x, x.^2);即为一个例子

### 三维绘图
1.三维曲线
x=x(t),y=y(t),z=z(t)绘图的时候使用hp=plot3(x,y,z)
设置线型set(hp,'linewidth',2,'color','b')对于轴，以及网格也可以画
2.空间曲面
对于z=f(x,y)
(1)基本绘图函数
mesh(X,Y,Z)，X,Y表示横纵坐标的一维向量，Z=f(X,Y)为二维的矩阵
- 生成矩阵：
  1.[X,Y]=meshgrid(-3:0.1:3)(简写)
  2.[X,Y]=meshgrid(-3:0.1:3,-5:0.5:5);(将两个一维向量拼起来)
  3.X=-3:0.1:3,Y=-1:0.01:1,[GX,GY]=meshgrid(X,Y)这样的东西也可以
  对于绘制Z=f(x,y)的时候，对于z=sin(r)/r,使用
  Z1=sin(sqrt(X1.^2+Y1.^2))./(sqrt(X1.^2+Y1.^2));
  Z1(isnan(Z1))=1;(没有定义的地方补上)
(2)一些特殊的曲线
- meshc：绘制等高线的空间曲面
- meshz:绘制含有0平面的空间曲面
- waterfall:只绘制x方向的网格线
- surf(X,Y,Z)绘制着色的三维表面图(mesh为网格图)
- 设置颜色：
  1.命令：colorbar; 表示在图表的右侧的一个条子，显示的是颜色与值的大小的对应关系
  2.命令：colormap('颜色' or 一个64*3的矩阵)而常见的有颜色：hsv,hot,cool,summer,gray,copper,autumn,winter,spring,bone,pink,flag
  3.按照不同要求设定颜色：surf(x,y,z,gradient(z))表示按照曲面梯度设置颜色，surf(x,y,z,del2(z))表示按照曲率的大小设置颜色
  4.使得表面的颜色连续变化：shading interp
  5.固定视角命令：view(az,el)分别表示方位角和仰角
- 绘制等高线：
  1.contour：平面上等高线，contour3空间上的等高线:C=contour(X,Y,Z,12)(表示总共有12条)；
  2.clabel在每一个等高线上标注相应的高度：clabel(C);
- 绘制切片:需要同时给出来x,y,z切片(即使我们不需要)
  xslice=[-1.2,0.8,2];yslice=[2],zslice=[-2,0];
  slice(x,y,z,v,xslice,yslice,zslice);(v为待绘制的函数)

### 动画显示
1.一个点在曲线上面移动
(1)设置参考点的格式:hp=plot(t(1),y(1),'marker','o','markersize',10,'markerfacecolor','r');
(2)使用循环来绘制点：
已经有了一个N，t=0：0.05：4*pi,y为t对应的曲线
for k=2:N
  set(hp,'xdata',t(k),'ydata',y(k));
  pause(0.05);
end
2.将动画存储为gif
```
for k=1:N
    set(hp,'xdata',t(k),'ydata',y(k));
    pause(0.01);

    F=getframe(gcf);
    im=frame2im(F); 
    [I,map]=rgb2ind(im,256);
    if(k==1)
        imwrite(I,map,'test.gif','gif','Loopcount', inf,'DelayTime',0.1);
    else
        imwrite(I,map,'test.gif','gif','WriteMode','append','DelayTime',0.1);
    end
end
```
表示以256色写入该目录下面的叫做test.gif的文件里面