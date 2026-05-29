# 项目笔记

这个项目的核心目标是把知识库维护流程拆成可测试的模块。第一阶段先证明 CLI、状态机、artifact 和 validator 能跑通。

简化 Ingest 不直接写入正式 wiki，而是先生成 source 和 concept 草稿。这样可以先验证工程骨架，再逐步接入真实模型。

