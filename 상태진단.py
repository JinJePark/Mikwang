#%%
import pandas as pd
import pymssql 
import warnings
import numpy as np
import time
import joblib
import json
import xgboost as xgb
import tensorflow as tf

# 그래픽카드 성능 제한
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=0.25 #최대치의 25%까지
    )
)
sess = tf.compat.v1.Session(config=config)

warnings.filterwarnings('ignore') # 경고메시지 무시


model1_path = r"C:\Users\202304\Desktop\condition_monitoring\heat"
model1_1 = joblib.load(model1_path+'\Heat1.pkl')
model1_2 = joblib.load(model1_path+'\Heat2.pkl')
model1_3 = joblib.load(model1_path+'\Heat3.pkl')
model1_4 = joblib.load(model1_path+'\Heat4.pkl')
model1_5 = joblib.load(model1_path+'\Heat5.pkl')
model1_6 = joblib.load(model1_path+'\Heat6.pkl')
model1_7 = joblib.load(model1_path+'\Heat7.pkl')
model1_8 = joblib.load(model1_path+'\Heat8.pkl')
model1_9 = joblib.load(model1_path+'\Heat9.pkl')
model1_10 = joblib.load(model1_path+'\Heat10.pkl')
scaler1_1 = joblib.load(model1_path+'\scaler1.pkl')
scaler1_2 = joblib.load(model1_path+'\scaler2.pkl')
scaler1_3 = joblib.load(model1_path+'\scaler3.pkl')
scaler1_4 = joblib.load(model1_path+'\scaler4.pkl')
scaler1_5 = joblib.load(model1_path+'\scaler5.pkl')
scaler1_6 = joblib.load(model1_path+'\scaler6.pkl')
scaler1_7 = joblib.load(model1_path+'\scaler7.pkl')
scaler1_8 = joblib.load(model1_path+'\scaler8.pkl')
scaler1_9 = joblib.load(model1_path+'\scaler9.pkl')
scaler1_10 = joblib.load(model1_path+'\scaler10.pkl')

model3_path = r"C:\Users\202304\Desktop\condition_monitoring\conveyor"
model3 = tf.keras.models.load_model(model3_path+'\Conveyor.h5')
pca3 = joblib.load(model3_path+'\Conveyor(pca).pkl')
scaler3 = joblib.load(model3_path+'\Conveyor(scaler).pkl')


#===========================================================================
#main coding
def main_run():
    #상태진단 데이터 LOAD
    conn= pymssql.connect(host='211.197.86.57', port='1433', user='sa', password='mk12#$', database='MG_IOT_ENERGY', charset='utf8')
    sql_statement='''SELECT TOP 2
                *
                FROM dbo.MG_IOT_NEW
                ORDER BY Time DESC
                ;'''

    data = pd.read_sql(sql=sql_statement, con=conn)


    Time = data.loc[0, ['TIME']]
    Time = Time[0].strftime("%Y-%m-%d %H:%M:%S")
    
    json_file = 'time.json'
    with open(json_file, 'r') as json_file:
        config_data = json.load(json_file)
    recent_time = config_data["recent_date"]
    if Time == recent_time:
        print("대기 중...")
        
    else:
        config_data["recent_date"] = Time
        print(f"\n예측 데이터 수집 시간은"+config_data["recent_date"]+"입니다.")
        # 갱신된 데이터를 JSON 파일에 쓰기
        json_file = 'time.json'
        with open(json_file, 'w') as json_file:
            json.dump(config_data, json_file, indent=4)
        json_file.close()
    
        
    #===========================================================================
    #히팅시스템1
        
        model1_input = data.loc[0,['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128']]
        
        if int(data.loc[0, 'ITEM128']) == 0:
            model1_1_pre_output = 0
            model1_1_real_output = float(data.loc[0, 'ITEM001'])
            predict_state1_1 = 'Not working'
            different1_1 = 0
        elif int(data.loc[0, 'ITEM128']) != 0:  
            model1_1_real_output = data.loc[0,['ITEM001']]
            model1_1_input1 = data.loc[1,['ITEM001']]
            model1_1_input = pd.concat([model1_input, model1_1_input1])
            model1_1_input_numpy = np.reshape(model1_1_input,(1, 5))

            model1_1_input_scaler = scaler1_1.transform(model1_1_input_numpy)
            model1_1_input_scaler = pd.DataFrame(model1_1_input_scaler, columns=['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128', 'alpha'])
            dtest = xgb.DMatrix(model1_1_input_scaler)
            model1_1_pre_output = model1_1.predict(dtest)
            model1_1_pre_output = model1_1_pre_output.item()
            different1_1 = model1_1_real_output - model1_1_pre_output
            different1_1 = different1_1.item()
            threshold1_1 = model1_1_real_output*0.1
            if different1_1.item() > threshold1_1.item(): 
                predict_state1_1 = 'error'
            else:
                predict_state1_1 = 'Good'
            model1_1_real_output = float(model1_1_real_output)
        if int(data.loc[0,['ITEM128']]) == 0:
            model1_2_pre_output = 0
            model1_2_real_output = float(data.loc[0, 'ITEM005'])
            predict_state1_2 = 'Not working'
            different1_2 = 0
        elif int(data.loc[0,['ITEM128']]) != 0:  
            model1_2_real_output = data.loc[0,['ITEM005']]
            model1_2_input1 = data.loc[1,['ITEM005']]
            model1_2_input = pd.concat([model1_input, model1_2_input1])
            model1_2_input_numpy = np.reshape(model1_2_input,(1, 5))

            model1_2_input_scaler = scaler1_2.transform(model1_2_input_numpy)
            model1_2_input_scaler = pd.DataFrame(model1_2_input_scaler, columns=['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128', 'alpha'])
            dtest = xgb.DMatrix(model1_2_input_scaler)
            model1_2_pre_output = model1_2.predict(dtest)
            model1_2_pre_output = model1_2_pre_output.item()

            different1_2 = model1_2_real_output - model1_2_pre_output
            
            different1_2 = different1_2.item()

            threshold1_2 = model1_2_real_output*0.1
            if different1_2.item() > threshold1_2.item():
                predict_state1_2 = 'error'
            else:
                predict_state1_2 = 'Good'
            model1_2_real_output = float(model1_2_real_output)

        if int(data.loc[0,['ITEM128']]) == 0:
            model1_3_pre_output = 0
            model1_3_real_output = float(data.loc[0, 'ITEM009'])
            predict_state1_3 = 'Not working'
            different1_3 = 0
        elif int(data.loc[0,['ITEM128']]) != 0:  
            model1_3_real_output = data.loc[0,['ITEM009']]
            model1_3_input1 = data.loc[1,['ITEM009']]
            model1_3_input = pd.concat([model1_input, model1_3_input1])
            model1_3_input_numpy = np.reshape(model1_3_input,(1, 5))

            model1_3_input_scaler = scaler1_3.transform(model1_3_input_numpy)
            model1_3_input_scaler = pd.DataFrame(model1_3_input_scaler, columns=['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128', 'alpha'])
            dtest = xgb.DMatrix(model1_3_input_scaler)
            model1_3_pre_output = model1_3.predict(dtest)
            model1_3_pre_output = model1_3_pre_output.item()
            different1_3 = model1_3_real_output - model1_3_pre_output
            different1_3 = different1_3.item()

            threshold1_3 = model1_3_real_output*0.1
            if different1_3.item() > threshold1_3.item():
                predict_state1_3 = 'error'
            else:
                predict_state1_3 = 'Good'
            model1_3_real_output = float(model1_3_real_output)

        if int(data.loc[0,['ITEM128']]) == 0:
            model1_4_pre_output = 0
            model1_4_real_output = float(data.loc[0, 'ITEM013'])
            predict_state1_4 = 'Not working'
            different1_4 = 0
        elif int(data.loc[0,['ITEM128']]) != 0:  
            model1_4_real_output = data.loc[0,['ITEM013']]
            model1_4_input1 = data.loc[1,['ITEM013']]
            model1_4_input = pd.concat([model1_input, model1_4_input1])
            model1_4_input_numpy = np.reshape(model1_4_input,(1, 5))

            model1_4_input_scaler = scaler1_4.transform(model1_4_input_numpy)
            model1_4_input_scaler = pd.DataFrame(model1_4_input_scaler, columns=['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128', 'alpha'])
            dtest = xgb.DMatrix(model1_4_input_scaler)
            model1_4_pre_output = model1_4.predict(dtest)
            model1_4_pre_output = model1_4_pre_output.item()
            different1_4 = model1_4_real_output - model1_4_pre_output
            different1_4 = different1_4.item()

            threshold1_4 = model1_4_real_output*0.1
            if different1_4.item() > threshold1_4.item():
                predict_state1_4 = 'error'
            else:
                predict_state1_4 = 'Good'
            model1_4_real_output = float(model1_4_real_output)
            
        if int(data.loc[0,['ITEM128']]) == 0:
            model1_5_pre_output = 0
            model1_5_real_output = float(data.loc[0, 'ITEM017'])
            predict_state1_5 = 'Not working'
            different1_5 = 0
        elif int(data.loc[0,['ITEM128']]) != 0:  
            model1_5_real_output = data.loc[0,['ITEM017']]
            model1_5_input1 = data.loc[1,['ITEM017']]
            model1_5_input = pd.concat([model1_input, model1_5_input1])
            model1_5_input_numpy = np.reshape(model1_5_input,(1, 5))

            model1_5_input_scaler = scaler1_5.transform(model1_5_input_numpy)
            model1_5_input_scaler = pd.DataFrame(model1_5_input_scaler, columns=['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128', 'alpha'])
            dtest = xgb.DMatrix(model1_5_input_scaler)
            model1_5_pre_output = model1_5.predict(dtest)
            model1_5_pre_output = model1_5_pre_output.item()
            different1_5 = model1_5_real_output - model1_5_pre_output
            different1_5 = different1_5.item()

            threshold1_5 = model1_5_real_output*0.1
            if different1_5.item() > threshold1_5.item():
                predict_state1_5 = 'error'
            else:
                predict_state1_5 = 'Good'
            model1_5_real_output = float(model1_5_real_output)

        if int(data.loc[0,['ITEM128']]) == 0:
            model1_6_pre_output = 0
            model1_6_real_output = float(data.loc[0, 'ITEM021'])
            predict_state1_6 = 'Not working'
            different1_6 = 0
        elif int(data.loc[0,['ITEM128']]) != 0:  
            model1_6_real_output = data.loc[0,['ITEM021']]
            model1_6_input1 = data.loc[1,['ITEM021']]
            model1_6_input = pd.concat([model1_input, model1_6_input1])
            model1_6_input_numpy = np.reshape(model1_6_input,(1, 5))

            model1_6_input_scaler = scaler1_6.transform(model1_6_input_numpy)
            model1_6_input_scaler = pd.DataFrame(model1_6_input_scaler, columns=['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128', 'alpha'])
            dtest = xgb.DMatrix(model1_6_input_scaler)
            model1_6_pre_output = model1_6.predict(dtest)
            model1_6_pre_output = model1_6_pre_output.item()
            different1_6 = model1_6_real_output - model1_6_pre_output
            different1_6 = different1_6.item()

            threshold1_6 = model1_6_real_output*0.1
            if different1_6.item() > threshold1_6.item():
                predict_state1_6 = 'error'
            else:
                predict_state1_6 = 'Good'
            model1_6_real_output = float(model1_6_real_output)

        if int(data.loc[0,['ITEM128']]) == 0:
            model1_7_pre_output = 0
            model1_7_real_output = float(data.loc[0, 'ITEM025'])
            predict_state1_7 = 'Not working'
            different1_7 = 0
        elif int(data.loc[0,['ITEM128']]) != 0:  
            model1_7_real_output = data.loc[0,['ITEM025']]
            model1_7_input1 = data.loc[1,['ITEM025']]
            model1_7_input = pd.concat([model1_input, model1_7_input1])
            model1_7_input_numpy = np.reshape(model1_7_input,(1, 5))

            model1_7_input_scaler = scaler1_7.transform(model1_7_input_numpy)
            model1_7_input_scaler = pd.DataFrame(model1_7_input_scaler, columns=['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128', 'alpha'])
            dtest = xgb.DMatrix(model1_7_input_scaler)
            model1_7_pre_output = model1_7.predict(dtest)
            model1_7_pre_output = model1_7_pre_output.item()
            different1_7 = model1_7_real_output - model1_7_pre_output
            different1_7 = different1_7.item()

            threshold1_7 = model1_7_real_output*0.1
            if different1_7.item() > threshold1_7.item():
                predict_state1_7 = 'error'
            else:
                predict_state1_7 = 'Good'
            model1_7_real_output = float(model1_7_real_output)

        if int(data.loc[0,['ITEM128']]) == 0:
            model1_8_pre_output = 0
            model1_8_real_output = float(data.loc[0, 'ITEM029'])
            predict_state1_8 = 'Not working'
            different1_8 = 0
        elif int(data.loc[0,['ITEM128']]) != 0:  
            model1_8_real_output = data.loc[0,['ITEM029']]
            model1_8_input1 = data.loc[1,['ITEM029']]
            model1_8_input = pd.concat([model1_input, model1_8_input1])
            model1_8_input_numpy = np.reshape(model1_8_input,(1, 5))

            model1_8_input_scaler = scaler1_8.transform(model1_8_input_numpy)
            model1_8_input_scaler = pd.DataFrame(model1_8_input_scaler, columns=['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128', 'alpha'])
            dtest = xgb.DMatrix(model1_8_input_scaler)
            model1_8_pre_output = model1_8.predict(dtest)
            model1_8_pre_output = model1_8_pre_output.item()
            different1_8 = model1_8_real_output - model1_8_pre_output
            different1_8 = different1_8.item()

            threshold1_8 = model1_8_real_output*0.1
            if different1_8.item() > threshold1_8.item():
                predict_state1_8 = 'error'
            else:
                predict_state1_8 = 'Good'
            model1_8_real_output = float(model1_8_real_output)

        if int(data.loc[0,['ITEM128']]) == 0:
            model1_9_pre_output = 0
            model1_9_real_output = float(data.loc[0, 'ITEM033'])
            predict_state1_9 = 'Not working'
            different1_9 = 0
        elif int(data.loc[0,['ITEM128']]) != 0:  
            model1_9_real_output = data.loc[0,['ITEM033']]
            model1_9_input1 = data.loc[1,['ITEM033']]
            model1_9_input = pd.concat([model1_input, model1_9_input1])
            model1_9_input_numpy = np.reshape(model1_9_input,(1, 5))

            model1_9_input_scaler = scaler1_9.transform(model1_9_input_numpy)
            model1_9_input_scaler = pd.DataFrame(model1_9_input_scaler, columns=['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128', 'alpha'])
            dtest = xgb.DMatrix(model1_9_input_scaler)
            model1_9_pre_output = model1_9.predict(dtest)
            model1_9_pre_output = model1_9_pre_output.item()
            different1_9 = model1_9_real_output - model1_9_pre_output
            different1_9 = different1_9.item()

            threshold1_9 = model1_9_real_output*0.1
            if different1_9.item() > threshold1_9.item():
                predict_state1_9 = 'error'
            else:
                predict_state1_9 = 'Good'
            model1_9_real_output = float(model1_9_real_output)
            
        if int(data.loc[0,['ITEM128']]) == 0:
            model1_10_pre_output = 0
            model1_10_real_output = float(data.loc[0, 'ITEM037'])
            predict_state1_10 = 'Not working'
            different1_10 = 0
        elif int(data.loc[0,['ITEM128']]) != 0:  
            model1_10_real_output = data.loc[0,['ITEM037']]
            model1_10_input1 = data.loc[1,['ITEM037']]
            model1_10_input = pd.concat([model1_input, model1_10_input1])
            model1_10_input_numpy = np.reshape(model1_10_input,(1, 5))

            model1_10_input_scaler = scaler1_10.transform(model1_10_input_numpy)
            model1_10_input_scaler = pd.DataFrame(model1_10_input_scaler, columns=['ITEM125', 'ITEM126', 'ITEM127', 'ITEM128', 'alpha'])
            dtest = xgb.DMatrix(model1_10_input_scaler)
            model1_10_pre_output = model1_10.predict(dtest)
            model1_10_pre_output = model1_10_pre_output.item()
            different1_10 = model1_10_real_output - model1_10_pre_output
            
            different1_10 = different1_10.item()

            threshold1_10 = model1_10_real_output*0.1
            if different1_10.item() > threshold1_10.item():
                predict_state1_10 = 'error'
            else:
                predict_state1_10 = 'Good'
            model1_10_real_output = float(model1_10_real_output)
    #===========================================================================
    #컨베이어벨트
        model3_input = data.loc[0,['ITEM101', 'ITEM102', 'ITEM103', 'ITEM104', 'ITEM105', 'ITEM106', 'ITEM107', 'ITEM108']]
        model3_input = model3_input.to_numpy()
        model3_input=np.reshape(model3_input,(1, 8))

        if int(data.loc[0,['ITEM101']]) == 0:
            model3_real_output = data.loc[0,['ITEM081']]
            model3_real_output = np.array(model3_real_output).reshape(len(model3_real_output),1)
            model3_pre_output = 0
            different3 = 0
            predict_state3 = 'Not working'
            model3_real_output = float(model3_real_output)
        elif int(data.loc[0,['ITEM101']]) != 0:
            model3_real_output = data.loc[0,['ITEM081']]
            model3_real_output = np.array(model3_real_output).reshape(len(model3_real_output),1)
            model3_input_scaler = scaler3.transform(model3_input)
            model3_input_pca = pca3.transform(model3_input_scaler)
            model3_input_pca = np.reshape(model3_input_pca,(1, 3, ))
            model3_pre_output = model3.predict(model3_input_pca)
            model3_pre_output = np.reshape(model3_pre_output, (1, ))
            model3_pre_output = model3_pre_output.item()

            different3 = abs(model3_real_output - model3_pre_output)
            different3 = different3.item()            
            
            predict_data = model3_pre_output
            predict_data1 = predict_data*0.394
            predict_data2 = predict_data*0.634
            predict_data3 = predict_data*1
            predict_data4 = predict_data*1.577
            predict_data5 = predict_data*2.535
            predict_data6 = predict_data*3.944
            predict_data7 = predict_data*6.338
            predict_data8 = predict_data*10
            predict_data9 = predict_data*15.775
            
            if model3_real_output < predict_data1:
                predict_state3 = '100%'
            elif model3_real_output >= predict_data1 and model3_real_output < predict_data2:
                predict_state3 = '90%'  
            elif model3_real_output >= predict_data2 and model3_real_output < predict_data3:
                predict_state3 = "80%"
            elif model3_real_output >= predict_data3 and model3_real_output < predict_data4:
                predict_state3 = "70%"
            elif model3_real_output >= predict_data4 and model3_real_output < predict_data5:
                predict_state3 = "60%"
            elif model3_real_output >= predict_data5 and model3_real_output < predict_data6:
                predict_state3 = "50%"
            elif model3_real_output >= predict_data6 and model3_real_output < predict_data7:
                predict_state3 = "40%"
            elif model3_real_output >= predict_data7 and model3_real_output < predict_data8:
                predict_state3 = "30%"
            elif model3_real_output >= predict_data8 and model3_real_output < predict_data9:
                predict_state3 = "20%"
            elif model3_real_output >= predict_data9:
                predict_state3 = "10%"
            model3_real_output = float(model3_real_output)
            
            
    #예측 데이터 저장
        conn = pymssql.connect(
        host='211.197.86.57',
        port=1433,
        user='sa',
        password='mk12#$',
        database='MG_IOT_ENERGY',
        charset='utf8')
        
        # 커서 생성
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO MG_PREDICT (
            TIME, ITEM001, ITEM002, ITEM003, ITEM004, ITEM005, ITEM006, ITEM007,
            ITEM008, ITEM009, ITEM010, ITEM011, ITEM012, ITEM013, ITEM014, ITEM015,
            ITEM016, ITEM017, ITEM018, ITEM019, ITEM020, ITEM021, ITEM022, ITEM023,
            ITEM024, ITEM025, ITEM026, ITEM027, ITEM028, ITEM029, ITEM030, ITEM031,
            ITEM032, ITEM033, ITEM034, ITEM035, ITEM036, ITEM037, ITEM038, ITEM039,
            ITEM040, ITEM081, ITEM082, ITEM083, ITEM084
        ) VALUES (
            %(TIME)s, %(ITEM001)s, %(ITEM002)s, %(ITEM003)s, %(ITEM004)s, %(ITEM005)s, %(ITEM006)s, %(ITEM007)s,
            %(ITEM008)s, %(ITEM009)s, %(ITEM010)s, %(ITEM011)s, %(ITEM012)s, %(ITEM013)s, %(ITEM014)s, %(ITEM015)s,
            %(ITEM016)s, %(ITEM017)s, %(ITEM018)s, %(ITEM019)s, %(ITEM020)s, %(ITEM021)s, %(ITEM022)s, %(ITEM023)s,
            %(ITEM024)s, %(ITEM025)s, %(ITEM026)s, %(ITEM027)s, %(ITEM028)s, %(ITEM029)s, %(ITEM030)s, %(ITEM031)s,
            %(ITEM032)s, %(ITEM033)s, %(ITEM034)s, %(ITEM035)s, %(ITEM036)s, %(ITEM037)s, %(ITEM038)s, %(ITEM039)s,
            %(ITEM040)s, %(ITEM081)s, %(ITEM082)s, %(ITEM083)s, %(ITEM084)s
        )
        """
        
        # 데이터 바인딩
        # 데이터 바인딩
        data = {
            'TIME': Time,
            'ITEM001': model1_1_real_output,
            'ITEM002': model1_1_pre_output,
            'ITEM003': different1_1,
            'ITEM004': predict_state1_1,
            'ITEM005': model1_2_real_output,
            'ITEM006': model1_2_pre_output,
            'ITEM007': different1_2,
            'ITEM008': predict_state1_2,
            'ITEM009': model1_3_real_output,
            'ITEM010': model1_3_pre_output,
            'ITEM011': different1_3,
            'ITEM012': predict_state1_3,
            'ITEM013': model1_4_real_output,
            'ITEM014': model1_4_pre_output,
            'ITEM015': different1_4,
            'ITEM016': predict_state1_4,
            'ITEM017': model1_5_real_output,
            'ITEM018': model1_5_pre_output,
            'ITEM019': different1_5,
            'ITEM020': predict_state1_5,
            'ITEM021': model1_6_real_output,
            'ITEM022': model1_6_pre_output,
            'ITEM023': different1_6,
            'ITEM024': predict_state1_6,
            'ITEM025': model1_7_real_output,
            'ITEM026': model1_7_pre_output,
            'ITEM027': different1_7,
            'ITEM028': predict_state1_7,
            'ITEM029': model1_8_real_output,
            'ITEM030': model1_8_pre_output,
            'ITEM031': different1_8,
            'ITEM032': predict_state1_8,
            'ITEM033': model1_9_real_output,
            'ITEM034': model1_9_pre_output,
            'ITEM035': different1_9,
            'ITEM036': predict_state1_9,
            'ITEM037': model1_10_real_output,
            'ITEM038': model1_10_pre_output,
            'ITEM039': different1_10,
            'ITEM040': predict_state1_10,
            'ITEM081': model3_real_output,
            'ITEM082': model3_pre_output,
            'ITEM083': different3,
            'ITEM084': predict_state3
        }

        # 데이터 삽입 실행
        cursor.execute(insert_query, data)

        # 변경 사항을 DB에 반영
        conn.commit()

        # 연결 종료
        conn.close()
        

        







#===================================================================
#When error occurs, the program will run again automatically
while True:
    try:
        time.sleep(5)
        main_run()
    except Exception as e:
        # 모든 예외 처리
        print(e)

# %%
