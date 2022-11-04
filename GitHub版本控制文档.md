# GitHub版本控制文档

#### 第一步创建一个代码仓库

- 若没有GitHub账号，则去官网上注册一个
- 然后接下来的操作都可以在浏览器上完成
- 首先我们要创建一个项目的代码仓库(Repository)， 点击下图new按钮即可。

![截屏2022-11-04 14.53.54](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 14.53.54.png)

- 然后就会进入到仓库创建页面

![截屏2022-11-04 14.58.20](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 14.58.20.png)

- 添加一些自己需要的信息(例如private和public等)之后，点击Create按钮即可创建。
- 当然也可以通过

#### 二 如何查看和对比仓库中的不同版本的代码

- 为了测试如何管理一份代码的版本迭代，我先upload一份本地文件

![截屏2022-11-04 15.08.58](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 15.08.58.png)

- 这是我们点击上图中的commits， 可以看到如下页面

![截屏2022-11-04 15.11.31](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 15.11.31.png)

- 可以看到该仓库一共有两个版本，第一个initial commit是创建时产生的版本，第二个add files via upload是刚刚通过上传文件得到的版本，点击Add files via upload可以查看该版本的内容如下：

![截屏2022-11-04 15.15.00](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 15.15.00.png)

- 点击上图的split按钮可以查看该版本相比于修改之前做了哪些修改
- “+”：增加了一行
- “-”：删除了一行

![截屏2022-11-04 15.21.22](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 15.21.22.png)

- 上图的右上角的‘1parent：1b06935’是该版本的父版本的版本号，通过点击该版本号可以查看此次提交之前的仓库版本如下：

![截屏2022-11-04 15.18.54](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 15.18.54.png)

- 由于父版本即是最开始创建时产生的，所以父版本是最初的版本，对应‘0parent’，

####  三 创建本地仓库

- 回到最开始的仓库页面， 复制Code下面ssh的内容git@github.com:lyh250-666/learn_how_to_use_git.git

![截屏2022-11-04 15.29.43](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 15.29.43.png)

- 在本地创建一个用于clone该仓库的新的文件夹，本人是在桌面上新建了一个test文件夹，Desktop/test
- 然后在终端中进入test文件夹并输入命令 git clone git@github.com:lyh250-666/learn_how_to_use_git.git即可将该仓库复制到本地，如下图：

![截屏2022-11-04 15.35.24](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 15.35.24.png)

- 然后在本地修改了main-Copy1.py之后，要提交该文件到远程仓库

- 将工作区内容添加到缓冲区注意要在learn_how_to_use_git这个文件夹下使用命令：`git add .`

- 将缓冲区提交到本地git仓库：`git commit -m "本次提交说明(该说明也可以不加)"`

![image-20221104154531173](/Users/bupt_lyh/Library/Application Support/typora-user-images/image-20221104154531173.png)

- 然后`git push`即可
- 此次修改我修改了一下README.md

![截屏2022-11-04 15.55.34](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 15.55.34.png)

- 之后我又修改了一下main-Copy1.py文件，

![image-20221104155718815](/Users/bupt_lyh/Library/Application Support/typora-user-images/image-20221104155718815.png)

- 提交之后的对比结果如下：

![截屏2022-11-04 15.58.15](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 15.58.15.png)

- 整个过程新产生了两个迭代版本。

![截屏2022-11-04 15.58.53](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 15.58.53.png)

#### 四 关于如何使用不同的branch

- 由于我们创建仓库的时候是只有一个branch的（该例子中原始branch的名称叫做main），由于只有一个branch所以上述的commit和push操作都是默认提交到main这个默认branch中的。

- 一般开新分支的目的是：有新的想法需要开发新的代码，但是又不想污染到原始分支上的很重要的代码。原则上是原始分支上代码都应该是随时可以放到服务器上去运行的代码，测试性的代码不能放在原始分支上。
- 这里我新建了一名为master(名字可以随便起)的分支， 操作如下图：点击branch

![截屏2022-11-04 16.33.58](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 16.33.58.png)

- 之后new branch，填写branch名称之后点击create branch，创建之后如下图所示：

![截屏2022-11-04 16.36.07](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 16.36.07.png)

- 然后本地仓库上也要建立一个新的分支，在本地终端下使用命令`git checkout -b master(这个名称可以随意)`
- 或者直接在本地从远程仓库里拉取远程仓库的指定分支：`git checkout -b master(这个名称可以随意) origin/master(这个名称要看远程仓库你创建了哪些分支)`， 若本地没有名为master的分支该命令会自动创建名为master的本地分支。
- 然后查看自己的本地分支有哪些：`git branch -l`

![截屏2022-11-04 17.06.10](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 17.06.10.png)

- 其中带*的是当前本地工作所在的本地分支(该new分支创建的目的是为了测试本地分支名字是否可以和想要提交的远程分支名不同)
- 然后在new分支下提交本地代码到远程master分支：
- `git add .`
- `git commit "master branch"`
- `git push origin master`
- 然后可以看到远程仓库里面的master分支如下图所示：

![截屏2022-11-04 17.14.27](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 17.14.27.png)

- master分支的版本迭代信息如下：

![截屏2022-11-04 17.16.28](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 17.16.28.png)

- 由于远程master branch是从main分支复制过来的，所以前四次的commit信息和main分支一样
- 第五次commit的信息如下：

![截屏2022-11-04 17.18.21](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 17.18.21.png)

- 修改了main-Copy1.py的一行，新增了一个main.py文件
- 但是可以看到main分支没有任何变化：

![截屏2022-11-04 17.15.15](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 17.15.15.png)

- 最后如果master分支的代码经过测试没有问题，就可以将master分支和main分支合并。

- 点击上图的Pull requests

![image-20221104173259827](/Users/bupt_lyh/Library/Application Support/typora-user-images/image-20221104173259827.png)

- 然后New pull request，并选择要合并的源分支和目的分支

![截屏2022-11-04 17.33.36](/Users/bupt_lyh/Library/Application Support/typora-user-images/截屏2022-11-04 17.33.36.png)

- 之后create pull request

![image-20221104173453671](/Users/bupt_lyh/Library/Application Support/typora-user-images/image-20221104173453671.png)

- 然后点击Create pull request，跳转页面之后点击Merge pull request，然后点击commit Merge即可合并。

#### 五 从本地仓库到远程仓库

- 上述的整个过程是从远程仓库克隆到本地然后在本地进行开发，下面介绍如何从本地现有的项目创建远程github仓库。
- 