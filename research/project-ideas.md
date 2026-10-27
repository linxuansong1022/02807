# 02807：有趣且可控制范围的项目候选

检索日期：2026-10-26。目标：四人组，至少一位成员零基础且时间有限。本文是选题设计建议；没有安装或运行引用的代码，也没有下载完整数据验证。课程约束以根目录 project.pdf 为准：至少一个、最好两个课内主题，另需至少一项课外方法；单主题需要充分深度。

## 1. Steam：跳出热门榜的游戏推荐

- 问题：基于玩家历史，推荐其可能喜欢但不是热门榜重复出现的游戏。
- 一手数据：UCSD McAuley Lab 的 Steam 用户物品记录、评论、游戏元数据。V1用户物品记录压缩约71 MB，评论约6.6 MB；不要直接选1.2 GB的V2评论文件。
- 来源：https://cseweb.ucsd.edu/~jmcauley/datasets.html
- 文件目录：https://mcauleylab.ucsd.edu/public_datasets/data/steam/
- 课内：MinHash/LSH 对喜欢/游玩物品集合检索相似用户；可选图聚类作为扩展，非第一版硬塞。
- 课外：矩阵分解排序；LLM重排序可作为替代扩展，不必两个都做。
- 评价：固定时间切分、明确正反馈定义；召回效果、NDCG/Recall、热门物品占比与覆盖、耗时。只对有时间信息的交互进行时间切分，游玩时长不自动等于喜欢。
- 最小范围：活跃用户的受控子集，CPU，Notebook和报告。
- 风险：多个文件版本和ID需要核对，历史数据不能当作今天的Steam目录。只含一个课内主题时实现LSH核心、做参数/速度/召回分析。

## 2. 音乐口味和朋友圈是否一致

- 更推荐的轻量版本：HetRec LastFM-2K，官方README经curl读取，1,892用户、17,632艺人、92,834用户艺人听歌计数、12,717条无向好友关系，有艺人名字与标签；非商业使用。课内LSH建口味相似图+Louvain发现社区，课外Personalized PageRank在用户艺人图上推荐。先留出部分用户艺人边，再重建相似图和社区，避免泄漏。听歌计数没有交互时间戳，只能描述静态留出评价，不能声称预测下一首歌曲。
- 官方入口：https://grouplens.org/datasets/hetrec-2011/
- 直接README：https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-readme.txt
- Personalized PageRank实现文档：https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
- 以下LastFM Asia是另一种备选，不能与HetRec的字段和统计混用。
- 数据：LastFM Asia，7,624节点、27,806条相互关注边；节点特征来自喜欢的艺人，标签是国家。
- 来源：https://snap.stanford.edu/data/feather-lastfm-social.html
- 课内：Louvain图社区，与用户艺人特征上的k-means分群比较。
- 课外：node2vec图嵌入，再做k-means，比较结构表示的影响。
- 方法来源与作者实现：https://snap.stanford.edu/node2vec/ 、https://github.com/aditya-grover/node2vec
- 评价：社区内外艺人特征相似度；不同分群的一致性；模块度；规模平衡的随机对照。国家标签不能当成音乐口味的真值。
- 最小范围：离线分析，CPU，2—3种有清晰比较目的的流程。
- 风险：数据页面没有保证原始可读艺人名称，不能预先承诺展示艺人名或歌单；先查下载文件。

## 3. Reddit圈子地图：谁互相提及，哪些关系带负面情绪

- 数据：Stanford SNAP Reddit Hyperlinks，55,863个subreddit、858,490条链接，含方向、时间和正负标签；时间范围2014—2017。
- 来源：https://snap.stanford.edu/data/soc-RedditHyperlinks.html
- 课内：在明确构造的正向/无符号聚合图上运行Louvain；在node2vec表示上运行k-means作为第二主题。
- 课外：node2vec，来源同上。
- 问题：社区内部和跨社区的负面链接比例是否不同？结构相近的社区是否真的互相友好？
- 最小范围：固定时间窗、受控节点数，保留边界统计口径；CPU。
- 风险：不能直接把负权重塞进普通Louvain并假设语义正确；情绪标签不是人工验证的仇恨真值。社区链接也不是用户级社交图。中等偏高解释工作量。

## 4. LLM电影推荐：真的懂口味，还是受热门/位置影响

- 数据：MovieLens，小数据可用于原型；报告保留具体版本、下载时间，或选固定的100K/1M版本。
- 来源：https://grouplens.org/datasets/movielens/ 、https://files.grouplens.org/datasets/movielens/
- 课内：MinHash/LSH近邻用户候选召回。
- 课外：LLM listwise重排序。
- 论文：https://arxiv.org/abs/2305.08845
- 作者实现：https://github.com/RUCAIBox/LLMRank
- 原始依赖：https://github.com/RUCAIBox/LLMRank/blob/master/requirements.txt
- 研究设计：固定候选集，比较传统排序与LLM；重复打乱候选顺序；测排序质量、热门偏向、非法ID率、耗时和API成本。检索和prompt只用训练期可得数据；禁止强行把正确目标注入召回并当成端到端质量。
- 风险：作者实现固定较老的PyTorch、RecBole和OpenAI客户端版本，参考思路不等于直接可运行；MovieLens不附带完整剧情简介。使用云API可免本地GPU但有费用。

## 5. 冰箱救援：现有食材能做什么，还缺哪些材料

- 数据：Food.com Recipes and Interactions，由研究作者发布；作者仓库链接到Kaggle。可只读取食谱与食材字段，抽取子集，不需要训练生成模型。
- 作者代码和数据入口：https://github.com/majumderb/recipe-personalization
- 数据：https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
- 原始论文：https://aclanthology.org/D19-1613/
- 课内：LSH找食材集合相似的食谱；Apriori挖掘经常共现的食材。
- 课外：FP-Growth，和Apriori在相同阈值、相同最大项集长度下比较。
- 研究设计：从已有食谱隐藏部分食材，测试能否检索回原食谱或可接受替代；比较精确Jaccard与LSH速度/召回；评价共现推荐的缺失材料补全。该隐藏实验只证明模拟任务能力，不能声称真人满意度。
- 风险：食材名称规范化是主要工作；Jaccard与“尽量少缺食材”不是相同目标，需要明确最终排序规则；Kaggle可能要求登录。作者原始生成模型旧且需要GPU，不属于本方案。

## 建议

优先把Steam游戏发现与音乐圈子比较带去组内讨论：前者产品感强且接近推荐岗位，后者两个课内主题加一个课外算法衔接自然。LLM电影排序仅在团队接受API费用与额外实验工作时考虑。所有选题都需明确自己的实现或非平凡组合，不能用数据集和库本身代替贡献。
