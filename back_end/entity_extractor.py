# coding=utf-8
"""
实体抽取模块 - 专门用于从LLM回答中提取音乐相关实体（歌曲、专辑、人物）
通过查询知识图谱获取已知实体列表，然后在文本中匹配
"""
from db import get_db, close_db
import re
from typing import List, Dict


class MusicEntityExtractor:
    """音乐领域实体抽取器"""
    
    def __init__(self):
        """初始化实体抽取器，从KG加载实体列表"""
        self.songs = set()  # 歌曲/作品名
        self.albums = set()  # 专辑名
        self.persons = set()  # 人物名
        self._load_entities_from_kg()
    
    def _load_entities_from_kg(self):
        """从知识图谱加载所有实体"""
        try:
            db = get_db()
            try:
                with db.session() as session:
                    # 加载所有歌曲/作品
                    result = session.run("MATCH (n:作品) RETURN n.name AS name")
                    self.songs = {record["name"] for record in result}
                    print(f"加载了 {len(self.songs)} 首歌曲")
                    
                    # 加载所有专辑
                    result = session.run("MATCH (n:专辑) RETURN n.name AS name")
                    self.albums = {record["name"] for record in result}
                    print(f"加载了 {len(self.albums)} 个专辑")
                    
                    # 加载所有人物
                    result = session.run("MATCH (n:人物) RETURN n.name AS name")
                    self.persons = {record["name"] for record in result}
                    print(f"加载了 {len(self.persons)} 个人物")
            finally:
                close_db(db)
        except Exception as e:
            print(f"警告: 加载实体列表时出错: {e}")
            print("实体抽取器将使用空实体列表，可能影响实体匹配功能")
            # 保持空集合，避免后续出错
            self.songs = set()
            self.albums = set()
            self.persons = set()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        从文本中提取音乐相关实体
        
        Args:
            text: 输入文本
            
        Returns:
            包含实体类型的字典: {
                "songs": [歌曲名列表],
                "albums": [专辑名列表],
                "persons": [人物名列表]
            }
        """
        entities = {
            "songs": [],
            "albums": [],
            "persons": []
        }
        
        # 先按长度降序排序，优先匹配较长的实体名（避免部分匹配问题）
        all_entities = {
            "songs": sorted(self.songs, key=len, reverse=True),
            "albums": sorted(self.albums, key=len, reverse=True),
            "persons": sorted(self.persons, key=len, reverse=True)
        }
        
        # 记录已匹配的位置，避免重复匹配
        matched_positions = set()
        
        # 对每种实体类型进行匹配
        for entity_type in ["songs", "albums", "persons"]:
            entity_set = all_entities[entity_type]
            found_entities = []
            
            for entity in entity_set:
                # 在文本中查找实体（不区分大小写）
                pattern = re.escape(entity)
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                for match in matches:
                    start, end = match.span()
                    # 检查是否已经被其他更长的实体匹配过
                    is_overlapped = False
                    for matched_start, matched_end in matched_positions:
                        if not (end <= matched_start or start >= matched_end):
                            is_overlapped = True
                            break
                    
                    if not is_overlapped:
                        found_entities.append(entity)
                        matched_positions.add((start, end))
                        break  # 每个实体只匹配一次
            
            # 去重
            entities[entity_type] = list(set(found_entities))
        
        return entities
    
    def extract_all_entities(self, text: str) -> List[str]:
        """
        提取所有实体（不分类）
        
        Args:
            text: 输入文本
            
        Returns:
            实体名称列表
        """
        entities_dict = self.extract_entities(text)
        all_entities = entities_dict["songs"] + entities_dict["albums"] + entities_dict["persons"]
        return list(set(all_entities))  # 去重


# 全局单例实例
_extractor_instance = None


def get_entity_extractor() -> MusicEntityExtractor:
    """获取实体抽取器单例"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = MusicEntityExtractor()
    return _extractor_instance

