# -*- encoding: utf-8 -*-
'''
@File    : data_query.py
@Time    : 2024/06/05 14:46:07
@Author  : Kofi Wang
@Contact : wonkefei@gmail.com
'''
# import pymysql
from sqlalchemy import create_engine, text
import pandas as pd


class DBOperate:
    def __init__(self, db):
        """
        db: database infos, including user, password, host, port, database_name
        """
        self.db = db
        self.db_engine = create_engine(
            'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
                db.user, db.passwd, db.host, db.port, db.name
            ),
            pool_pre_ping=True,     # ⭐ 防止死连接
            pool_recycle=3600,      # ⭐ 小于 MySQL wait_timeout
            pool_size=5,
            max_overflow=10
        )

    def read_sql(self, sql):
        """
        查询数据库，并返回 dataframe
        """
        with self.db_engine.connect() as conn:
            df = pd.read_sql(text(sql), con=conn)
        return df

    def df_to_sql(self, df, tb_name, if_exists='append'):
        with self.db_engine.begin() as conn:
            df.to_sql(
                name=tb_name,
                con=conn,
                if_exists=if_exists,
                index=False
            )

    def execute(self, sql):
        """
        执行 DML / DDL
        """
        with self.db_engine.begin() as conn:
            conn.execute(text(sql))

if __name__ == "__main__":
    print("This is a library file and should be imported, not run directly.")
