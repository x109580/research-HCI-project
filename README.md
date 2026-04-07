# research-HCI-project

在你放着 app.py 的那个目录里启动。

先在终端进入项目目录：

cd "D:\python project code\Hci model"

然后启动后端：

uvicorn app:app --reload

这里的意思是：

前一个 app = 文件名 app.py
后一个 app = 文件里的 FastAPI 实例 app = FastAPI(...)

如果启动成功，你会看到类似：

Uvicorn running on http://127.0.0.1:8000

然后你就可以打开：

http://127.0.0.1:8000/docs

如果你已经在虚拟环境里，直接运行上面的命令就行。


前端直接在pycharm或者vscode中直接打开就可以了！！！
