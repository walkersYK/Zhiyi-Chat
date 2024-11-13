				HTML


​					
​				
​				
​						
​				

# [🥳 Chat Zhiyi](https://github.com/walkersYK/Zhiyi-Chat)



<a href="https://github.com/walkersYK/zhiyichat/stargazers"><img src="https://img.shields.io/github/stars/walkersYK/zhiyichat" alt="Stars Badge"/></a>
<a href="https://github.com/walkersYK/zhiyichat/network/members"><img src="https://img.shields.io/github/forks/walkersYK/zhiyichat" alt="Forks Badge"/></a>
<a href="https://github.com/walkersYK/zhiyichatt/pulls"><img src="https://img.shields.io/github/issues-pr/walkersYK/zhiyichat" alt="Pull Requests Badge"/></a>
<a href="https://github.com/walkersYK/zhiyichat/issues"><img src="https://img.shields.io/github/issues/walkersYK/zhiyichat" alt="Issues Badge"/></a>
<a href="https://github.com/walkersYK/zhiyichat/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/walkersYK/zhiyichat?color=2b9348"></a>
<a href="https://github.com/walkersYK/zhiyichat/blob/master/LICENSE"><img src="https://img.shields.io/github/license/walkersYK/zhiyichat?color=2b9348" alt="License Badge"/></a>

<a href="https://github.com/walkersYK/zhiyichat/blob/main/enREADME.md"><img src="https://img.shields.io/static/v1?label=&labelColor=505050&message=English README 英文自述文件&color=%230076D6&style=flat&logo=google-chrome&logoColor=green" alt="website"/></a>

[🥳 chat- zhiyi](https://chatnio.com)

🚀 下一代 医疗教育行业AIGC 一站式商业解决方案

*“ Chat Zhiyi > [Next Web](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web) + [One API](https://github.com/songquanpeng/one-api) ”*

<!-- <img src="http://hits.dwyl.com/peng-zhihui/ElectronBot.svg" alt="Hits Badge"/> -->

<i>喜欢这个项目吗？请考虑给 Star ⭐️ 以帮助改进！</i>

</div>

![image-20241110164256624](C:\Users\YunJin\AppData\Roaming\Typora\typora-user-images\image-20241110164256624.png)

> 本项目是一个整合ASR，multi features, AIGC, image identification, semantic segmentation可学习门的前端框架，后端模型采用多模型嵌套，以追求不同场景的运用需求
>
> 本项目提供了配套的全套开发资料和对应SDK以供二次开发，SDK使用说明见后文。
>
> **视频介绍**：
>
> **Video** :



**注意：Issues里面是讨论项目开发相关话题的，不要在里面发无意义的消息，不然watch了仓库的人都会收到通知邮件会给别人造成困扰的！！！灌水可以在仓库的Discuss里讨论！**

📝 功能

- ✅ 美观商业级 UI, 漂亮的前端界面与后台管理
- ✅ 支持 TTS & STT, 插件市场, RAG 知识库等丰富功能与模块
- ✅ 更多支付供应商, 更多计费模式和高级订单管理
- ✅ 支持更多鉴权方式，包括短信登录、OAuth 登录等
- ✅ 支持模型监控，渠道健康检测，故障告警自动渠道切换
- ✅ 支持多租户 API Key 分发系统, 企业级令牌权限管理与访问者限制
- ✅ 支持安全审核, 日志记录, 模型限速, API Gateway 等高级功能
- ✅ 支持推广奖励，专业数据统计，用户画像分析等商业分析能力
- ✅ 支持Discord/Telegram/飞书等机器人对接集成能力 (扩展模块)

## 📦 部署方式

> [!TIP]
> **部署成功后, 管理员账号为 `root`, 密码默认为 `chat123456`**

### ✨ Zeabur (一键部署)

[![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/templates/M86XJI)

> Zeabur 提供一定的免费额度, 可以使用非付费区域进行一键部署，同时也支持计划订阅和弹性计费等方式弹性扩展。
>
> 1. 点击 `Deploy` 进行部署, 并输入你希望绑定的域名，等待部署完成。
> 2. 部署完成后, 请访问你的域名, 并使用用户名 `root` 密码 `chatnio123456` 登录后台管理，请按照提示在 chatnio 后台及时修改密码。


### ⚡ Docker Compose 安装 (推荐)

> [!NOTE]
> 运行成功后, 宿主机映射地址为 `http://localhost:8000`

 ```shell
 git clone --depth=1 --branch=main --single-branch https://github.com/Deeptrain-Community/chatnio.git
 cd chatnio
 docker-compose up -d # 运行服务
# 如需使用 stable 版本, 请使用 docker-compose -f docker-compose.stable.yaml up -d 替代
# 如需使用 watchtower 自动更新, 请使用 docker-compose -f docker-compose.watch.yaml up -d 替代
 ```

版本更新（_开启 Watchtower 自动更新的情况下, 无需手动更新_）：

```shell
docker-compose down 
docker-compose pull
docker-compose up -d
```

> - MySQL 数据库挂载目录项目 ~/**db**
> - Redis 数据库挂载目录项目 ~/**redis**
> - 配置文件挂载目录项目 ~/**config**

### ⚡ Docker 安装 (轻量运行时, 常用于外置 _MYSQL/RDS_ 服务)

> [!NOTE]
> 运行成功后, 宿主机地址为 `http://localhost:8094`。
>
> 如需使用 stable 版本, 请使用 `programzmh/chatnio:stable` 替代 `programzmh/chatnio:latest`  

```shell
docker run -d --name chatnio \
   --network host \
   -v ~/config:/config \
   -v ~/logs:/logs \
   -v ~/storage:/storage \
   -e MYSQL_HOST=localhost \
   -e MYSQL_PORT=3306 \
   -e MYSQL_DB=chatnio \
   -e MYSQL_USER=root \
   -e MYSQL_PASSWORD=chatnio123456 \
   -e REDIS_HOST=localhost \
   -e REDIS_PORT=6379 \
   -e SECRET=secret \
   -e SERVE_STATIC=true \
   programzmh/chatnio:latest
```

> - *--network host* 指使用宿主机网络, 使 Docker 容器使用宿主机的网络, 可自行修改
> - SECRET: JWT 密钥, 自行生成随机字符串修改
> - SERVE_STATIC: 是否启用静态文件服务 (正常情况下不需要更改此项, 详见下方常见问题解答)
> - *-v ~/config:/config* 挂载配置文件, *-v ~/logs:/logs* 挂载日志文件的宿主机目录, *-v ~/storage:/storage* 挂载附加功能的生成文件
> - 需配置 MySQL 和 Redis 服务, 请自行参考上方信息修改环境变量

 版本更新 （_开启 Watchtower 后无需手动更新, 执行后按照上述步骤重新运行即可_）：

 ```shell
docker stop chatnio
docker rm chatnio
docker pull programzmh/chatnio:latest
 ```

### ⚒ 编译安装

> [!NOTE]
> 部署成功后, 默认端口为 **8094**, 访问地址为 `http://localhost:8094`
>
> Config 配置项 (~/config/**config.yaml**) 可以使用环境变量进行覆盖, 如 `MYSQL_HOST` 环境变量可覆盖 `mysql.host` 配置项

```shell
git clone https://github.com/Deeptrain-Community/chatnio.git
cd chatnio

cd app
npm install -g pnpm
pnpm install
pnpm build

cd ..
go build -o chatnio

# e.g. using nohup (you can also use systemd or other service manager)
nohup ./chatnio > output.log & # using nohup to run in background
```

## ❓ 常见问题 Q&A

⚡ Docker 安装 (轻量运行时, 常用于外置 _MYSQL/RDS_ 服务)

> 感谢以下项目：
>
> [opencv/opencv: Open Source Computer Vision Library (github.com)](https://github.com/opencv/opencv)
>
> https://github.com/CMU-Perceptual-Computing-Lab/openpose
>
> [Lexikos/AutoHotkey_L: AutoHotkey - macro-creation and automation-oriented scripting utility for Windows. (github.com)](https://github.com/Lexikos/AutoHotkey_L)
>
> https://blog.csdn.net/pq8888168/article/details/85781908

## ❤ 捐助

如果您觉得这个项目对您有所帮助, 您可以点个 Star 支持一下！
