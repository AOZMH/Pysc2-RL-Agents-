# 星际争霸作业 MoveToBeacon 及 FindAndDefeatZerglings 实现

Github 仓库：https://github.com/AOZMH/Pysc2-RL-Agents-

参考：https://github.com/xhujoy/pysc2-agents

## MoveToBeacon 复现

### 评估已有的模型

> python -m main --map=MoveToBeacon --training=False

该命令会加载训练好的模型，执行20次重复试验，输出类似于：
```
...
INFO:tensorflow:Restoring parameters from ./snapshot/MoveToBeacon/fcn\model.pkl-4121
I1206 17:21:53.575125  2588 saver.py:1270] Restoring parameters from ./snapshot/MoveToBeacon/fcn\model.pkl-4121
...
I1206 17:20:28.835436 30048 sc2_env.py:507] Starting episode 20: [terran, zerg] on FindAndDefeatZerglings
I1206 17:20:49.223704 30048 sc2_env.py:725] Episode 20 finished after 2880 game steps. Outcome: [1], reward: [0], score: [26]
...
Max/Min/Avg score: 30 / 22 / 25.3
```

### 训练新模型
> python -m main --map=MoveToBeacon

该命令会从头开始训练 MoveToBeacon 的模型，同时打印一些log，每20 episode会进行一次评估。

设置continuation参数以从checkpoint开始训练，设置teaching以实现脚本辅助训练。完整用法见 main.py。


## FindAndDefeatZerglings 复现

评估已有的模型及训练新模型的方法与 MoveToBeacon 类似，只不过将map参数换为 FindAndDefeatZerglings 即可。


## 训练好的模型下载
将 https://disk.pku.edu.cn:443/link/516E3EF7498129ECD0E610ED35407B06 中的**snapshot**文件夹放在本目录下或替换本目录下的该文件夹，即可下载已有的模型参数。
