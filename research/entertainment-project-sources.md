# 娱乐类项目来源核查

核查日期：2026-10-26。以下项目结构是基于资料提出的设计建议，不是原论文的直接复现，也不保证课程评分。课程中 LSH、一般聚类、图聚类、频繁项集各算一个主题；单主题方案要有足够深度。

## 1. Last.fm：你的音乐圈子是否等于你的朋友圈？

**趣味问题**：听同类艺人的人会成为朋友吗？能否从音乐圈子中发现用户还没听过的艺人？

- 数据：HetRec 2011 Last.fm 2K 官方页面提供 2.5MB zip，无需账号或爬虫。官方 README 已实际读取：1,892 用户、17,632 艺人、92,834 用户-艺人听歌计数、12,717 条双向朋友关系；包含艺人名称与标签。[数据页](https://grouplens.org/datasets/hetrec-2011/)、[官方 README](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-readme.txt)、[直接下载](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip)。
- 课程方法：MinHash/LSH 基于听过的艺人集合建相似用户图；Louvain 发现音乐口味社区，和原始朋友图对照。两个主题均有明确作用。
- 课外方法候选：Personalized PageRank，从用户偏好出发在用户-艺人图中传播并生成艺人推荐。[NetworkX 官方 API](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html)。PageRank 在 Intro 被提及，因此应明确研究 personalized random walk 的机制及其区别；如老师认为不够新，可换矩阵分解推荐算法。
- 评估：LSH 对精确 Jaccard 邻居的查全率/时间；社区内部与外部的音乐相似度；留出部分已听艺人，用 Recall/NDCG 与热门榜比较。
- 约束：listeningCount 没有逐次听歌时间戳，不能声称做时间序列预测；tag 的时间戳不能当听歌时间。朋友和口味关系只能做关联分析，不能声称因果。非商业使用。普通 CPU 足以处理该规模，实际运行时间需测量。
- 最小范围：只使用艺人集合、好友边和艺人名，先不做音频、Spotify 接口、真实音乐播放、网页。

## 2. Steam：找口味相近的玩家，发现热门榜之外的游戏

**趣味问题**：推荐系统能否找到“和你一样玩这些游戏的人”，并推荐尚未玩过的游戏？

- 原始研究数据：UCSD McAuley 实验室公开 Steam 用户交互、评论、游戏信息和游戏捆绑包。数据页列有 Version 1 评论约 6.7MB、用户物品约 71MB；Version 2 评论约 1.3GB，第一版不需要后者。[实验室数据页](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data)。
- 原研究先例：研究者用 Steam 数据研究个性化游戏包，将用户偏好与游戏兼容性结合；项目可借鉴数据场景，不必复现完整模型。[作者论文](https://cseweb.ucsd.edu/~jmcauley/pdfs/sigir17.pdf)、[出版页](https://doi.org/10.1145/3077136.3080724)。
- 课程方法：MinHash/LSH 搜索游戏集合相似用户；可选 Louvain 对相似用户图找游戏圈，比较圈内召回与全局召回。第二主题不应只是展示图。
- 课外方法：矩阵分解/隐式反馈排序，或个性化随机游走。将其与相似用户打分比较，保持候选范围相同。
- 评估：推荐查全率、长尾物品覆盖率、热门程度、召回时间和 LSH 近似误差。将训练交互和测试交互先分离再建立索引。
- 约束：购买不等于喜欢，拥有游戏不等于玩过；playtime 是快照累计值，不能凭它评估未来游玩。将“有游玩时间”定义为一种隐式偏好需要说明。71MB 压缩数据解压并转换 Python 对象会更大，可限定用户子集和活跃物品。不同文件的物品 ID 与元数据覆盖要先实测，不能假设全部可连接。
- 最小范围：游戏名、用户交互、游玩时间即可；不接 Steam 实时账号、不做实时价格和促销、不做完整游戏包优化。

## 3. MovieLens：四个人一起看什么，怎样避免总有人不满意？

**趣味问题**：为小组推荐电影时，是追求平均满意，还是照顾最不满意的人？

- 数据：MovieLens 100K 官方 README 已实际读取，有 943 用户、1,682 电影、100,000 评分，提供电影名、类型和评分时间；zip 约 5MB。[数据页](https://grouplens.org/datasets/movielens/100k/)、[README](https://files.grouplens.org/datasets/movielens/ml-100k/README)、[下载](https://files.grouplens.org/datasets/movielens/ml-100k.zip)。
- 课程方法：基于喜欢的电影集合用 MinHash/LSH 找邻居，为每个组员生成候选并估计个人偏好；可选对用户聚类，用来构造“口味相近”和“口味冲突”的模拟小组并分层评价，这样第二主题直接服务实验。
- 课外方法：群体推荐的 Average、Least Misery、Average Without Misery 聚合。Least Misery 最大化组员中的最低满意分；可以进一步学习社会选择排序。[原始实验论文](https://ceur-ws.org/Vol-811/paper7.pdf)、[原始群体偏好研究](https://doi.org/10.1016/j.ins.2011.11.037)。
- 评估：平均成员满意度、最低成员满意度、组员间分歧，和热门电影、单纯平均策略对照。模拟 2/4/6 人且偏好一致/冲突的组。
- 约束：个体历史评分不等于实际群体决策；模拟组指标是离线代理，需明确局限。数据老旧但任务仍可做。评分矩阵稀疏，缺失评分不能当成 0；可先限制共同评分充分的测试子集并报告覆盖率。
- 许可：官方条款要求引用，禁止未获许可再分发数据；项目仓库提供官方下载脚本，提交方式遵守数据条款。CPU 优先，无需 LLM。
- 最小范围：两个聚合策略加一套个人预测方法；先做 Notebook，不做聊天、多轮协商、电影简介爬取。

## 工作量判断

若用户偏音乐，Last.fm 2K 最容易拿到体积小且语义清楚的数据；若四人喜欢游戏，Steam 题材更直观，但清洗和规模控制稍多；群体电影推荐最能现场体验，评价设计比普通单人推荐更绕。以上工作量判断是方案分析，不是来源对完成时间的承诺。
