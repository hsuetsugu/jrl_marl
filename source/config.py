import os
import numpy as np

USE_PRIVATE = True

if USE_PRIVATE:
    UseLineNotification = True
    UploadToGCP = True
    from setting_private import *
else:
    UploadToGCP = False
    UseLineNotification = False

''' Exmerimental Setting '''
LT_RET = 3
LT_WH  = 3

def check_and_create_dir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)
        print('create directory : {}'.format(path))

def get_ex_setting(scenario):
    inv_cap = 0
    trans_cap = 20

    if scenario =='ex1_nocap':
        num_products = 2
        season = False
        myu = [2, 2]
        lot_retailer, lot_wh = [4, 4], [10, 10]
        trans_cap = 200

    if scenario =='ex1':
        num_products = 2
        season = False
        myu = [2, 2]
        lot_retailer, lot_wh = [4, 4], [10, 10]
        trans_cap = 20

    if scenario =='ex1_step':
        num_products = 2
        season = False
        myu = [15, 15]
        lot_retailer, lot_wh = [10, 10], [10, 10]
        trans_cap = 20

    if scenario =='ex1_invcap':
        num_products = 2
        season = False
        myu = [2, 2]
        lot_retailer, lot_wh = [4, 4], [10, 10]
        trans_cap = 200
        inv_cap = 20

    if scenario =='ex1_step2':
        num_products = 2
        season = False
        myu = [15, 15]
        lot_retailer, lot_wh = [10, 10], [10, 10]
        trans_cap = 20
        inv_cap = 50

    if scenario =='ex5_nocap':
        num_products = 5
        season = False
        myu = [0.3, 0.4, 0.5, 0.5, 0.7]
        lot_retailer, lot_wh = [1, 1, 1, 1, 2], [5, 5, 5, 5, 5]
        trans_cap = 2000

    if scenario =='ex5':
        num_products = 5
        season = False
        myu = [0.3, 0.4, 0.5, 0.5, 0.7]
        lot_retailer, lot_wh = [1, 1, 1, 1, 2], [5, 5, 5, 5, 5]
        trans_cap = 20

    if scenario =='ex5_step':
        num_products = 5
        season = False
        myu = [3, 4, 5, 5, 7]
        lot_retailer, lot_wh = [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]
        trans_cap = 20

    if scenario =='ex5_invcap':
        num_products = 5
        season = False
        myu = [0.3, 0.4, 0.5, 0.5, 0.7]
        lot_retailer, lot_wh = [1, 1, 1, 1, 2], [5, 5, 5, 5, 5]
        trans_cap = 200
        inv_cap = 20

    if scenario =='ex6_nocap':
        num_products = 10
        season = False
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2]
        lot_retailer, lot_wh = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        trans_cap = 200

    if scenario =='ex6':
        num_products = 10
        season = False
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2]
        lot_retailer, lot_wh = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        trans_cap = 20

    if scenario =='ex6_step':
        num_products = 10
        season = False
        myu = [1.5, 2, 2.5, 2.5, 3.5, 4.5, 5, 5, 6, 6]
        lot_retailer, lot_wh = [3, 3, 3, 3, 5, 5, 5, 7, 10, 10], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        trans_cap = 20

    if scenario =='ex6_invcap':
        num_products = 10
        season = False
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2]
        lot_retailer, lot_wh = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        inv_cap = 20
        trans_cap = 200

    # 在庫キャパシティあり＋階段コスト
    if scenario =='ex8_step':
        num_products = 10
        season = False
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2]
        lot_retailer, lot_wh = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        trans_cap = 20
        inv_cap = 0

    if scenario =='ex7':
        # コンテナ数が常に週あたり1.5くらいになるように
        num_products = 10
        season = False
        myu = [1.2, 1.6, 2.0, 2.0, 2.8, 3.6, 4.0, 4.0, 4.8, 4.8]
        lot_retailer, lot_wh = [4, 4, 4, 4, 8, 8, 8, 8, 8, 8], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    if scenario =='ex50':
        num_products = 50
        season = False
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2] * 5
        lot_retailer, lot_wh = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3] * 5, [5] * 50

    if scenario =='ex50s':
        num_products = 50
        season = True
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2] * 5
        lot_retailer, lot_wh = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3] * 5, [5] * 50

    if scenario =='ex2':
        num_products = 2
        season = True
        myu = [2, 2]
        lot_retailer, lot_wh = [4, 4], [10, 10]

    if scenario =='ex3':
        num_products = 10
        season = True
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2]
        lot_retailer, lot_wh = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    if scenario =='ex3_high':
        num_products = 10
        season = True
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2]
        myu = list(np.array(myu) * 2)
        lot_retailer, lot_wh = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    if scenario =='ex3_low':
        num_products = 10
        season = True
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2]
        myu = list(np.array(myu) * .5)
        lot_retailer, lot_wh = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    if scenario =='ex3_lot2':
        num_products = 10
        season = True
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2]
        lot_retailer, lot_wh = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    if scenario =='ex3_lot3':
        num_products = 10
        season = True
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 0.9, 1.0, 1.0, 1.2, 1.2]
        lot_retailer, lot_wh = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    if scenario =='ex4':
        num_products = 10
        season = True
        myu = [0.3, 0.4, 0.5, 0.5, 0.7, 1.0, 2.5, 3.0, 3.0, 4.5]
        lot_retailer, lot_wh = [1, 1, 1, 1, 2, 2, 3, 3, 3, 5], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]


    return num_products, season, myu, lot_retailer, lot_wh, inv_cap, trans_cap

check_and_create_dir('model')
check_and_create_dir('log')
check_and_create_dir('result')
check_and_create_dir('result_val')
check_and_create_dir('scm')
