# -*- coding: utf-8 -*-
# @File : DBUtil.py 
# @Description :
class SQLiteUtil():
    def __init__(self):
        self.conn = None
    def get_conn(self):
        return self.conn

    def connect_to_db(self, dbname):
        import sqlite3
        self.conn = sqlite3.connect(dbname)
        print ("数据库打开成功")

    def create_pet_table(self):
        print("数据库打开成功")
        c = self.conn.cursor()
        c.execute('''CREATE TABLE pet_table
           (Breed         TEXT  NOT NULL,
            Classification         TEXT  NOT NULL,
            height_low_inches         INT  NOT NULL,
            height_high_inches          INT NOT NULL);''')
        print("数据表创建成功")
        self.conn.commit()

    def create_table_user(self, tb_name, **kwargs):
        print("数据库打开成功")
        c = self.conn.cursor()
        c.execute('''CREATE TABLE {}
                       (USERNAME         TEXT  NOT NULL,
                       PASSWORD          TEXT NOT NULL);'''.format(tb_name))
        print("数据表创建成功")
        self.conn.commit()

    def create_table(self, tb_name, **kwargs):
        print("数据库打开成功")
        c = self.conn.cursor()
        c.execute('''CREATE TABLE {}
               (PRODUCT_TYPE        TEXT  NOT NULL,
               PRODUCT_NAME         TEXT    NOT NULL,
               ENTERPRISE_NAME      TEXT     NOT NULL,
               REPORT_INST          TEXT NOT NULL,
               REPORT_TIME          TEXT);'''.format(tb_name))
        print("数据表创建成功")
        self.conn.commit()

    def clear_one_table(self, tb_name):
        c = self.conn.cursor()
        c.execute("DELETE from {};".format(tb_name))
        self.conn.commit()
        print("数据操作成功")

    def drop_one_table(self, db_name, tb_name):
        self.connect_to_db(db_name)
        c = self.conn.cursor()
        c.execute("DROP TABLE IF EXISTS {};".format(tb_name))
        self.conn.commit()

    def insert_one_record_miniapp_finish_seconds(self, nickname, finish_seconds):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO miniapp_finish_seconds (nickname, finish_seconds) VALUES (?, ?)",
            (nickname, finish_seconds))
        self.conn.commit()
        print("数据插入成功")

    def create_miniapp_finish_seconds_table(self):
        print("数据库打开成功")
        c = self.conn.cursor()
        c.execute('''CREATE TABLE {}
               (nickname  TEXT  NOT NULL,
               finish_seconds         FLOAT    NOT NULL);'''.format("miniapp_finish_seconds"))
        print("数据表创建成功")
        self.conn.commit()

    def get_fastest_K_user(self, K=10):
        c = self.conn.cursor()
        cursor = c.execute("SELECT nickname, finish_seconds_1 as finish_seconds FROM (SELECT nickname, min(finish_seconds) as finish_seconds_1 from miniapp_finish_seconds GROUP BY nickname) ORDER BY finish_seconds_1 LIMIT {};".format(K))
        print("数据操作成功")
        ret = [{
            "nickname": row[0],
            "finish_seconds": row[1],
        } for row in cursor]
        return ret

    def get_all_finish_seconds_records(self):
        c = self.conn.cursor()
        cursor = c.execute("SELECT * from miniapp_finish_seconds")
        print("数据操作成功")
        ret = [{
            "nickname": row[0],
            "finish_seconds": row[1],
        } for row in cursor]
        return ret

    def insert_one_pet_info(self, d: dict):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO pet_table (Breed, Classification, height_low_inches, height_high_inches) VALUES (?, ?, ?, ?)",
            (d["breed"], d["classification"], d["height_low_inches"], d["height_high_inches"]))
        self.conn.commit()
        print("数据插入成功")

    def insert_one_record(self, tb_name, PRODUCT_TYPE, PRODUCT_NAME, ENTERPRISE_NAME, REPORT_INST, REPORT_TIME):
        c = self.conn.cursor()
        # c.execute("INSERT INTO {} (PRODUCT_TYPE, PRODUCT_NAME, ENTERPRISE_NAME, REPORT_INST, REPORT_TIME) \
        #       VALUES ({}, {}, {}, {}, {});".format(
        #     tb_name,
        #     PRODUCT_TYPE, PRODUCT_NAME, ENTERPRISE_NAME, REPORT_INST, REPORT_TIME
        # ))
        c.execute("INSERT INTO {} (PRODUCT_TYPE, PRODUCT_NAME, ENTERPRISE_NAME, REPORT_INST, REPORT_TIME) VALUES (?, ?, ?, ?, ?)".format(tb_name),
                  (PRODUCT_TYPE, PRODUCT_NAME, ENTERPRISE_NAME, REPORT_INST, REPORT_TIME))
        self.conn.commit()
        print("数据插入成功")

    def select_all_pet(self):
        c = self.conn.cursor()
        cursor = c.execute("SELECT * from pet_table")
        print("数据操作成功")
        ret = [{
            "breed": row[0],
            "classification": row[1],
            "height_low_inches": row[2],
            "height_high_inches": row[3]
        } for row in cursor]
        return ret

    def calculate_pie(self):
        c = self.conn.cursor()
        cursor = c.execute("SELECT Classification, count(*) from pet_table group by Classification")
        print("数据操作成功")
        ret = [{
            "classification": row[0],
            "count": row[1],
        } for row in cursor]
        ret = [
            [item['classification'], item['count']] for item in ret
        ]
        return ret

    def calculate_pie(self):
        c = self.conn.cursor()
        cursor = c.execute("SELECT Classification, count(*) from pet_table group by Classification")
        print("数据操作成功")
        ret = [{
            "classification": row[0],
            "count": row[1],
        } for row in cursor]
        ret = [
            [item['classification'], item['count']] for item in ret
        ]
        return ret

    def select_by_username(self, tb_name, un):
        c = self.conn.cursor()
        cursor = c.execute("SELECT * from {} where USERNAME = ?;".format(tb_name), (un,))
        print("数据操作成功")
        ret = [{
            "USERNAME": row[0],
            "PASSWORD": row[1]
        } for row in cursor]
        return ret

    def select_all(self, tb_name):
        c = self.conn.cursor()
        cursor = c.execute("SELECT * from {};".format(tb_name))
        print("数据操作成功")
        ret = [{
            "PRODUCT_TYPE": row[0],
            "PRODUCT_NAME": row[1],
            "ENTERPRISE_NAME": row[2],
            "REPORT_INST": row[3],
            "REPORT_TIME": row[4]
        } for row in cursor]
        print(ret)
        return ret

    def close_connection(self):
        self.conn.close()

    def create_table_train_result_info(self, tb_name, **kwargs):
        print("数据库打开成功")
        c = self.conn.cursor()
        c.execute('''CREATE TABLE {}
                               (TRAIN_RESULT_ID         TEXT  NOT NULL,
                               TRAIN_PLAN_NAME          TEXT NOT NULL,
                               FINISH_COUNT          TEXT NOT NULL,
                               SECONDS_LAST          TEXT NOT NULL,
                               FINISH_DATETIME          TEXT NOT NULL
                               );'''.format(tb_name))
        print("数据表创建成功")
        self.conn.commit()

    def select_one_train_result_by_id(self, train_result_id):
        c = self.conn.cursor()
        cursor = c.execute(
            "SELECT * from train_result_info where TRAIN_RESULT_ID={}".format(train_result_id)
        )
        print("数据操作成功")
        ret = [{
            "train_result_id": row[0],
            "train_plan_name": row[1],
            "finish_count": row[2],
            "seconds_last": row[3],
            "finish_datetime": row[4]
        } for row in cursor]
        return ret

    def select_all_train_result(self):
        c = self.conn.cursor()
        cursor = c.execute("SELECT * from train_result_info")
        print("数据操作成功")
        ret = [{
            "train_result_id": row[0],
            "train_plan_name": row[1],
            "finish_count": row[2],
            "seconds_last": row[3],
            "finish_datetime": row[4]
        } for row in cursor]
        return ret

    def insert_one_train_result_info(self, tbname, train_result_id, train_plan_name, finish_count, seconds_last, finish_datetime):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO {} (TRAIN_RESULT_ID, TRAIN_PLAN_NAME, FINISH_COUNT, SECONDS_LAST, FINISH_DATETIME) VALUES (?, ?, ?, ?, ?)".format(
                tbname),
            (train_result_id, train_plan_name, finish_count, seconds_last, finish_datetime))
        self.conn.commit()
        print("数据插入成功")

if __name__ == "__main__":
    import requests
    # requests.post(
    #     url="http://127.0.0.1:5000/submit_train_result",
    #     data={
    #             'train_result_id': "idididididididi__1",
    #             'name': "n1",
    #             'finish_count': "fc1",
    #             'seconds_last': "sl1",
    #             'finish_datetime': "fd1111"
    #     }
    # )
    s = SQLiteUtil()
    db_name = "../FitnessGuidanceSystem.db"
    s.connect_to_db(db_name)
    s.drop_one_table(db_name, "miniapp_finish_seconds")
    s.create_miniapp_finish_seconds_table()
    s.insert_one_record_miniapp_finish_seconds("name42", 164512)
    s.insert_one_record_miniapp_finish_seconds("name41", 453424)
    s.insert_one_record_miniapp_finish_seconds("name24", 12324)
    s.insert_one_record_miniapp_finish_seconds("name14", 123444)
    s.insert_one_record_miniapp_finish_seconds("name24", 2523524)
    s.insert_one_record_miniapp_finish_seconds("name34", 13214)
    s.insert_one_record_miniapp_finish_seconds("name44", 423234)
    s.insert_one_record_miniapp_finish_seconds("name54", 31234)
    s.insert_one_record_miniapp_finish_seconds("name64", 64424)
    s.insert_one_record_miniapp_finish_seconds("name71", 223144)
    s.insert_one_record_miniapp_finish_seconds("name11", 12344)
    s.insert_one_record_miniapp_finish_seconds("name11", 12324)
    s.insert_one_record_miniapp_finish_seconds("name11", 4214)
    s.insert_one_record_miniapp_finish_seconds("name12", 4444)
    # s.insert_one_record_miniapp_finish_seconds("name12", 4)    s.insert_one_record_miniapp_finish_seconds("name42", 112)
    # s.insert_one_record_miniapp_finish_seconds("name41", 424)
    # s.insert_one_record_miniapp_finish_seconds("name24", 124)
    # s.insert_one_record_miniapp_finish_seconds("name14", 144)
    # s.insert_one_record_miniapp_finish_seconds("name24", 2524)
    # s.insert_one_record_miniapp_finish_seconds("name34", 114)
    # s.insert_one_record_miniapp_finish_seconds("name44", 4234)
    # s.insert_one_record_miniapp_finish_seconds("name54", 134)
    # s.insert_one_record_miniapp_finish_seconds("name64", 624)
    # s.insert_one_record_miniapp_finish_seconds("name71", 244)
    # s.insert_one_record_miniapp_finish_seconds("name11", 44)
    # s.insert_one_record_miniapp_finish_seconds("name11", 24)
    # s.insert_one_record_miniapp_finish_seconds("name11", 14)
    # s.insert_one_record_miniapp_finish_seconds("name12", 44)
    # s.insert_one_record_miniapp_finish_seconds("name12", 4)
    # r = s.get_all_finish_seconds_records()
    r = s.get_fastest_K_user()
    print(r)
