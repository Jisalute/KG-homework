# 实体抽取功能说明

## 功能概述

本功能实现了对大模型（LLM）回答进行实体抽取，并与知识图谱中的三元组进行比对验证。

## 实现文件

1. **`entity_extractor.py`** - 实体抽取模块
   - 从知识图谱加载所有已知实体（歌曲、专辑、人物）
   - 从文本中提取匹配的实体

2. **`two_stage.py`** - 两阶段问答系统（已更新）
   - 集成了实体抽取功能
   - 实现了实体级别的比对逻辑

## 工作流程

1. **LLM生成答案**：调用大模型生成初始答案
2. **实体抽取**：从LLM回答中提取音乐相关实体（歌曲名、专辑名、人物名）
3. **KG查询**：查询知识图谱获取正确答案
4. **实体比对**：将抽取的实体与KG结果进行比对
5. **答案修正**：根据比对结果决定使用LLM答案或KG答案

## 匹配策略

系统使用多种匹配策略（按优先级）：

1. **字符串完全匹配**：LLM答案与KG结果完全一致
2. **实体匹配**：LLM回答中的实体与KG结果中的实体匹配
3. **实体部分匹配**：LLM回答中的实体与KG结果相关但不完全一致
4. **实体不匹配**：LLM回答中的实体与KG结果不匹配（可能是幻觉）
5. **未提取到实体**：无法从LLM回答中提取已知实体

## 返回结果格式

```python
{
    "final_answer": "最终答案",
    "source": "答案来源（verified_by_kg_entity/corrected_by_kg_entity等）",
    "llm_answer": "LLM原始回答",
    "kg_answers": ["KG查询结果列表"],
    "llm_entities": {
        "songs": ["歌曲列表"],
        "albums": ["专辑列表"],
        "persons": ["人物列表"]
    },
    "matched_entities": ["匹配的实体列表"],
    "is_hallucination": True/False/None,
    "match_type": "匹配类型（entity_match/entity_mismatch等）"
}
```

## 使用方法

在 `two_stage_qa()` 函数中会自动调用实体抽取功能，无需额外配置。

API调用示例：
```python
from two_stage import two_stage_qa

result = two_stage_qa("歌曲兰亭序的作词人是")
print(result)
```

## 注意事项

1. 实体抽取器会在首次使用时从知识图谱加载实体列表
2. 如果数据库连接失败，实体抽取功能会降级使用空实体列表
3. 实体匹配不区分大小写
4. 优先匹配较长的实体名，避免部分匹配问题

