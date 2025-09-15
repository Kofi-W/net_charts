# -*- encoding: utf-8 -*-
'''
@File    : data_query.py
@Time    : 2024/06/05 14:46:07
@Author  : Kofi Wang
@Contact : wonkefei@gmail.com
'''
import pymysql
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
                db.user, db.passwd, db.host, db.port, db.name))
        self.db_connect = self.db_engine.connect()

    def read_sql(self, sql):
        """
        查询数据库，并返回dataframe
        :param sql:
        :return:
        """
        df = pd.read_sql(text(sql), con=self.db_connect)
        return df

    def df_to_sql(self, df, tb_name, if_exists='append'):
        df.to_sql(name=tb_name,
                  con=self.db_engine,
                  if_exists=if_exists,
                  index=False,
                  index_label=False)

    def execute(self, sql):
        conn = pymysql.connect(host=self.db.host,
                               port=self.db.port,
                               user=self.db.user,
                               passwd=self.db.passwd,
                               db=self.db.name,
                               charset='utf8')
        cur = conn.cursor()
        try:
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cur.close()
            conn.close()

if __name__ == "__main__":
    print("This is a library file and should be imported, not run directly.")
