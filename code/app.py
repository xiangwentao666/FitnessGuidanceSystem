import json
import random

from flask import Flask, render_template, request, session
import sqlite3
app = Flask(__name__)
app.secret_key = 'xxxxxxx'
from util.DBUtil import SQLiteUtil
@app.route('/get_one_train_result_by_id', methods=['GET'])
def get_one_train_result_by_id():
    train_result_id = request.args.get("train_result_id")
    s = SQLiteUtil()
    db_name = "FitnessGuidanceSystem.db"
    s.connect_to_db(db_name)
    data = s.select_one_train_result_by_id(train_result_id)
    data_list = data
    ret = {
        'data': data_list
    }
    return json.dumps(ret)

@app.route('/get_all_train_result')
def get_all_train_result():
    s = SQLiteUtil()
    db_name = "FitnessGuidanceSystem.db"
    s.connect_to_db(db_name)
    data = s.select_all_train_result()
    data_list = []
    for item in data:
        data_list.append(item)
    data_list = [item for item in data_list]
    print(data_list)
    return json.dumps({"data": data_list})

@app.route('/submit_train_result', methods=['POST'])
def submit_train_result():  # put application's code here
    # data_to_send = {
    #     'name': self.train_plan_comboBox.currentText(),
    #     'finish_count': self.finish_percentage_value_label.text(),
    #     'seconds_last': self.current_count_down_value,
    #     'finish_datetime': self.current_datetime_value_label.text()
    #     'train_result_id': ""
    # }
    train_result_id = request.values.get("train_result_id").strip()
    train_plan_name = request.values.get("name").strip()
    finish_count = request.values.get("finish_count").strip()
    seconds_last = request.values.get("seconds_last").strip()
    finish_datetime = request.values.get("finish_datetime").strip()
    print(train_plan_name, finish_count, seconds_last, finish_datetime)
    sqlutil = SQLiteUtil()
    db_name = "FitnessGuidanceSystem.db"
    sqlutil.connect_to_db("{}".format(db_name))
    tb_name = "train_result_info"
    try:
        # sqlutil.drop_one_table(db_name, tb_name)
        sqlutil.create_table_train_result_info(tb_name)
    except Exception as e:
        print(e)
    # sqlutil.clear_one_table(tb_name)
    sqlutil.insert_one_train_result_info(tb_name, train_result_id, train_plan_name, finish_count, seconds_last, finish_datetime)
    return 'ok'

@app.route('/submit_miniapp_duration')
def submit_miniapp_duration():
    nickname = request.args.get("nickname")
    miniapp_finish_seconds = request.args.get("miniapp_finish_seconds")

    s = SQLiteUtil()
    db_name = "FitnessGuidanceSystem.db"
    s.connect_to_db(db_name)
    try:
        s.create_miniapp_finish_seconds_table()
    except:
        pass
    s.insert_one_record_miniapp_finish_seconds(nickname, miniapp_finish_seconds)
    ret = s.get_all_finish_seconds_records()
    print(ret)
    return json.dumps({"data": "ok"})

@app.route('/get_fastest_5_user')
def get_fastest_5_user():
    s = SQLiteUtil()
    db_name = "FitnessGuidanceSystem.db"
    s.connect_to_db(db_name)
    data = s.get_fastest_K_user()
    data_list = []
    for item in data:
        data_list.append(item)
    data_list = [item for item in data_list]
    print(data_list)
    return json.dumps({"data": data_list})

if __name__ == '__main__':
    app.run()
