# -*- coding: utf-8 -*-
# @File : config.py 
# @Description : 
# @Author : Xiang Wentao, Software College of NEU
# @Contact : neu_xiangwentao@163.com
# @Date : 2023/3/13 9:29

'''
    中文名，id，倒计时，目标完成数
'''
configuration = {
    'train_plan_list': [
        {
            'name': "自由训练",
            "id": 'zyxl',
            'count_down': 99999,
            'target_finish_count': 99999
        },
        {
            'name': "肱三头肌训练",
            "id": 'gstjxl',
            'count_down': 600,
            'target_finish_count': 3
        },
        {
            'name': "肱二头肌训练",
            "id": 'getjxl',
            'count_down': 60,
            'target_finish_count': 4
        },
        {
            'name': "三角肌前束训练",
            "id": 'sjjqsxl',
            'count_down': 30,
            'target_finish_count': 2
        },
    ],

}
