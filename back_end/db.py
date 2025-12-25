# coding=utf-8
# 数据库相关操作
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Aqweasd123."


def get_db():
    """建立数据库的连接"""
    db = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    db.verify_connectivity()
    return db


def close_db(db):
    """关闭数据库连接"""
    db.close()
