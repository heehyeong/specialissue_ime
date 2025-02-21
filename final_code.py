# %% [markdown]
# ## 모듈 불러오기

# %%
!pip install requests
!pip install folium
!pip install pulp
!pip install gurobipy

# %%
import requests
import pandas as pd
import numpy as np
import itertools

import json
import re

import folium

import copy

from pulp import LpProblem, LpMaximize, LpVariable, LpInteger, LpBinary, lpSum, value
from sklearn.preprocessing import RobustScaler

from gurobipy import Model, GRB, quicksum
import gurobipy as gp

import matplotlib.pyplot as plt

# %% [markdown]
# ## 데이터 로드 및 수정

# %% [markdown]
# ### 데이터 관련 고민

# %%
# 데이터 로드
file_path = '202404_버스노선별_정류장별_시간대별_승하차_인원_정보.csv'
data = pd.read_csv(file_path, encoding='cp949')

# %%
# '버스정류장ARS번호' 열을 문자열로 변환 (NaN 값 포함)
data['버스정류장ARS번호'] = data['버스정류장ARS번호'].astype(str)

# 정수로 변환할 수 없는 값 확인 (NaN 값 처리)
invalid_values = data[~data['버스정류장ARS번호'].str.isnumeric()]

print(invalid_values['버스정류장ARS번호'].unique())
print(invalid_values['버스정류장ARS번호'].info())

# %%
file_path = '202404_버스노선별_정류장별_시간대별_승하차_인원_정보.csv'
data = pd.read_csv(file_path, encoding='cp949')

data = data[ data['버스정류장ARS번호'] != '~' ]

import copy

data_to_str = copy.deepcopy(data)
data_to_int = copy.deepcopy(data)

# %%
data_to_str['버스정류장ARS번호'] = data_to_str['버스정류장ARS번호'].astype(str)

data_to_str = data_to_str[ data_to_str['버스정류장ARS번호'].str.startswith('8') ]

set_2_str = set(data_to_str['버스정류장ARS번호'].unique())

len(set_2_str), len(data_to_str)

# %%
data_to_int['버스정류장ARS번호'] = data_to_int['버스정류장ARS번호'].astype(int)

data_to_int = data_to_int[ (data_to_int['버스정류장ARS번호']>=8000) & (data_to_int['버스정류장ARS번호']<9000) ]

set_2_int = set(data_to_int['버스정류장ARS번호'].astype(str).unique())

len(set_2_int), len(data_to_int)

# %%
# 데이터 로드
file_path = '20240507 서울시 버스 정류소 위치 정보.xlsx'
data = pd.read_excel(file_path)

data = data[ (data['ARS_ID']>=8000) & (data['ARS_ID']<9000) ]
set_1 = set(data['ARS_ID'].astype(str).unique())

len(set_1), len(data)

# %%
only_1 = set_1 - set_2_str
only_2 = set_2_str - set_1

display(only_1, only_2)

# %%
only_1 = set_1 - set_2_int
only_2 = set_2_int - set_1

display(only_1, only_2)

# %% [markdown]
# ### 진짜 데이터 로드 및 수정

# %%
# 따릉이 대여소 위치 데이터 로드
file_path = '20240522 서울시 따릉이 대여소 마스터 정보.csv'
data = pd.read_csv(file_path, encoding='cp949')

# '주소1' 열에서 '성북구'가 포함된 행만 필터링
bike = data[data['주소1'].str.contains('성북')]

# 위도와 경도 모두 0이 아닌 데이터만 사용
bike = bike[ (bike['위도'] != 0) & (bike['경도'] != 0) ].reset_index(drop=True)

# 경도와 위도 열 위치 변경 / '주소2'열 삭제
bike.drop(columns=['주소2'], inplace=True)

# 필터링된 데이터 확인
bike

# %%
# 성북구 버스 정류장 데이터 로드
file_path = '성북구 버스정류장 정보.xlsx'
data = pd.read_excel(file_path)

# 순위 기록한 행 삭제
data.dropna(how='any', inplace=True)

# 열 이름 변경
data.rename(columns={'버스정류장ARS번호':'ARS_ID','X좌표':'경도','Y좌표':'위도'}, inplace=True)

# candidate location 결정할 때, 필요없는 정보 삭제
to_drop_columns = {f'{hour}시 승하차 승객수' for hour in range(0,24,1)}
bus_location = data.drop(columns=to_drop_columns, inplace=False)

# 열 순서 바꾸기
bus_location = bus_location[ ['ARS_ID', '역명', '위도', '경도'] ]

bus_location

# %% [markdown]
# ## Multi-Period Maximal Covering

# %% [markdown]
# ### Tmap api 이용

# %%
# 도보 시간을 계산하는 함수
def get_walking_time(start_lat, start_lon, end_lat, end_lon):
    url = f"https://apis.openapi.sk.com/tmap/routes/pedestrian?version=1&format=json"
    headers = {
        "appKey": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "startX": start_lon,
        "startY": start_lat,
        "endX": end_lon,
        "endY": end_lat,
        "reqCoordType": "WGS84GEO",
        "resCoordType": "WGS84GEO",
        "startName": "start",
        "endName": "end"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # HTTP 오류를 예외로 발생시킴

    # 응답 데이터를 텍스트로 저장하여 확인
    with open("response_raw.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    
    # 제어 문자 제거
    clean_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response.text)

    result = json.loads(clean_text)
    
    if "features" in result:
        time = result["features"][0]["properties"]["totalTime"]
        time_minutes = time / 60
        return time_minutes
    else:
        print(f"Error in response: {result}")
        return None

# %%
# Tmap app key 입력
# API_KEY = 'goO4WLtMpN4ut5PUpOsAe5yBeJsPYyim2lmdlKNs'
API_KEY = '3nNLd08n7V597uyNQiQgh2yrabhJmgqJ5pNd88Ft'
# API_KEY = 'zrD4S8WEeB3a9FEaMR4Sb7RmEJpadv1I4qJ15gIy'
# API_KEY = 'INgo9JZIfo2OONP73afaH2Cd5ezouh3y5D4xRwax'

# 계산 횟수 counting
counts = 1

# %% [markdown]
# ### 대여소와 버스 정류장 사이 도보 시간 계산 -> candidate location 설정

# %%
# start_index 지정
start_index = 476

# %%
# 후보지 탈락의 기준 : 도보 소요 시간 time_threshold 분 이내에 대여소가 존재
time_threshold = 10

# 탈락된 후보지가 담길 리스트 / 체크 완료된 후보지가 담길 리스트
not_candidate_list = []
checked_candidate_list = []

# 각 버스 정류장를 기점으로 주위에 도보 거리 time_threshold분 미만의 대여소가 있다면 후보지 탈락. 
for i, bus_record in bus_location.iterrows() :

    # 이미 계산된 버스 정류장은 skip
    if i < start_index : continue

    # 버스 정류장의 위도와 경도 지정
    start_lat = bus_record['위도']
    start_long = bus_record['경도']

    try :
        
        for j, bike_record in bike.iterrows() :
           
            # 따릉이 대여소의 위도와 경도 지정
            end_lat = bike_record['위도']
            end_long = bike_record['경도']
            
            # 도보 거리 계산
            time = get_walking_time(start_lat, start_long, end_lat, end_long)

            # 결과 출력 및 계산 횟수 update
            print(f"{counts}. {bus_record['ARS_ID']} -> {bike_record['주소1']} : {time} 분")
            counts += 1
            
            if time < time_threshold : 
                not_candidate_list.append(bus_record['ARS_ID'])
                break

    except Exception as e :

        # 에러 메세지 출력
        print(f"에러가 발생했습니다: {e}")
        
        # candidate location 계산
        candidate_list = list( set(checked_candidate_list) - set(not_candidate_list) )

        print(f'체크 완료된 버스 정류장 리스트 : {checked_candidate_list}')
        print(f'\n체크된 버스 정류장 중 candidate location : {candidate_list}')
        break
        

    # 체크 완료된 버스 정류장 리스트 업데이트
    checked_candidate_list.append(bus_record['ARS_ID'])
    print(f"\n>>> 버스 정류장 {bus_record['ARS_ID']} 체크 완료\n")

# %%
# candidate location 계산
candidate_list = list( set(checked_candidate_list) - set(not_candidate_list) )

print(f'체크 완료된 버스 정류장 리스트 : {checked_candidate_list}')
print(f'\n체크된 버스 정류장 중 candidate location : {candidate_list}')

# %% [markdown]
# ### Maximal Covering 문제에서의 N_i 계산

# %%
# 위 계산을 통해 결정된 candidate location 정리
candidate_location_IDs = [
    8107, 8160, 8344, 8347, 8351, 8352, 8355, 8359, 8367, 8368, 
    8369, 8370, 8372, 8449, 8455, 8459, 8460, 8462, 8474, 8475, 
    8487, 8491, 8522, 8807, 8828, 8855, 8856, 8942, 8943, 8950,
    8953, 8954, 8957, 8960, 8966, 8970, 8987, 8988, 8989, 8990, 8991
]

# candidate location 정보 추출
candidate_location_df = bus_location[ bus_location['ARS_ID'].isin(candidate_location_IDs) ].reset_index(drop=True)

candidate_location_df.head(5)

# %%
# 지도의 중심 : 8988의 위도와 경도 값을 얻는 코드
target_id = 8988
target_row = candidate_location_df[candidate_location_df['ARS_ID'] == target_id]
latitude = target_row['위도'].values[0]
longitude = target_row['경도'].values[0]

# 초기 지도 설정
# location : 초기 지도의 중심 위치 / zoom_start : 초기 줌 레벨을 설정, 값이 클수록 더 확대 
map = folium.Map(location=[latitude, longitude], zoom_start=13)

# 데이터프레임의 각 행에 대해 위에 형성된 지도에 마커 추가
# location : 마커의 위치 / popup : 지도를 클릭했을 때의 텍스트 설정 / tooltip : 마커 위에 마우스를 올렸을 때의 텍스트 설정
for idx, row in candidate_location_df.iterrows():
    folium.Marker(
        location = [row['위도'], row['경도']],
        popup = row['역명'],
        tooltip = row['ARS_ID'],
        icon = folium.Icon(color='red')
    ).add_to(map)

# 지도 출력 (Jupyter Notebook에서만 동작)
map

# %%
# 모든 지점 쌍에 대해 도보 시간을 계산하여 데이터프레임 생성
results = []

# .combinations : 주어진 iterable에서 지정된 길이의 조합 생성
# .combinations(df.iterrows(), 2) : df의 모든 가능한 두 행의 조합을 생성 / 각 조합은 ((index1, row1), (index2, row2)) 형태
for (i, row1), (j, row2) in itertools.combinations(candidate_location_df.iterrows(), 2):
    
    start_lat, start_lon = row1['위도'], row1['경도']
    end_lat, end_lon = row2['위도'], row2['경도']
    
    # 도보 거리 계산
    time = get_walking_time(start_lat, start_lon, end_lat, end_lon)

    # 결과 출력 및 계산 횟수 업데이트
    print(f"{counts}. {row1['ARS_ID']} -> {row2['ARS_ID']} : {time}분")
    counts += 1
    
    if time is not None :
        results.append({
            '출발 지점': row1['ARS_ID'],
            '도착 지점': row2['ARS_ID'],
            '예상 이동 시간 (분)': time
        })

results_df = pd.DataFrame(results)

display(results_df)

# %%
# 결과 데이터프레임 파일로 저장
file_path = 'candidate location 사이의 도보 시간.csv'
results_df.to_csv(file_path)

# %%
# candidate location 사이의 도보 거리 데이터 로드
file_path = 'candidate location 사이의 도보 시간.csv'
time_between_candidates = pd.read_csv(file_path, index_col=0)

# '출발 지점', '도착 지점' 열은 int로 만들기
time_between_candidates[['출발 지점', '도착 지점']] = time_between_candidates[['출발 지점', '도착 지점']].astype(int)

time_between_candidates

# %%
# 예상 이동 시간이 10분 이내인 쌍만 추출
N_I_data = time_between_candidates[time_between_candidates['예상 이동 시간 (분)'] <= 10]

# '출발 지점'에 대해서 그룹화 / '도착 지점' 데이터를 리스트화 / 결과물을 딕셔너리로 출력
N_I_dict = N_I_data.groupby('출발 지점')['도착 지점'].apply(list).to_dict()

display(N_I_dict)

# %% [markdown]
# ## TSP

# %% [markdown]
# ### 성북구 도시관리공단 좌표 얻기

# %%
# 네이버 클라우드 플랫폼에서 발급받은 클라이언트 ID와 클라이언트 Secret
client_id = "7v7m52epdt"
client_secret = "CoOrZK45a2M5KfU99L8goEEDJRO9owszFp58suR9"
# client_id = 'w305h8ecqn'
# client_secret = 'kROPapgH82lrjGjiIClG1SpM3jswSUeK8cxP5qgR'

counts = 1

# %%
def get_lat_lng(address, client_id, client_secret):
    base_url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret
    }
    params = {
        "query": address
    }
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['addresses']:
            location = data['addresses'][0]
            return location['y'], location['x']
        else:
            return None, None
    else:
        return None, None

# %%
# 성북구 도시관리공단 주소 입력
address = "서울특별시 성북구 화랑로 18자길 13"

# 위도와 경도 계산
lat, lng = get_lat_lng(address, client_id, client_secret)

# 정보 저장
manage_corp_dict = {'ID':'0000', '위치 정보':address, '위도':lat, '경도':lng}

manage_corp_dict

# %% [markdown]
# ### 도시관리공단, candidate location, 기존 대여소 정보 병합

# %%
# 위 계산을 통해 결정된 candidate location 정리
candidate_location_IDs = [
    8107, 8160, 8344, 8347, 8351, 8352, 8355, 8359, 8367, 8368, 
    8369, 8370, 8372, 8449, 8455, 8459, 8460, 8462, 8474, 8475, 
    8487, 8491, 8522, 8807, 8828, 8855, 8856, 8942, 8943, 8950,
    8953, 8954, 8957, 8960, 8966, 8970, 8987, 8988, 8989, 8990, 8991
]

# candidate location 정보 추출
candidate_location_df = bus_location[ bus_location['ARS_ID'].isin(candidate_location_IDs) ].reset_index(drop=True)

candidate_location_df

# %%
# 병합을 위한 데이터프레임 생성 / 도시관리공단 정보 먼저 입력
manage_corp_df = pd.DataFrame([manage_corp_dict])

# 정보 결합을 위해 열 이름 일치시키기
bike.columns = manage_corp_df.columns
candidate_location_df.columns = manage_corp_df.columns

# 도시관리공단, candidate_location, 기존 따릉이 대여소 정보 결합
final_bike = pd.concat([manage_corp_df, bike, candidate_location_df], axis=0, ignore_index=True)

final_bike

# %%
# 파일로 다운 받기
file_path = 'demand_and_candidate location 정보.csv'
final_bike.to_csv(file_path)

# %%
# ST-597의 위도와 경도 값을 얻는 코드
target_id = 'ST-597'
target_row = final_bike[final_bike['ID'] == target_id]
latitude = target_row['위도'].values[0]
longitude = target_row['경도'].values[0]

# 초기 지도 설정
map = folium.Map(location=[latitude, longitude], zoom_start=13)

# 데이터프레임의 각 행에 대해 위에 형성된 지도에 마커 추가
for idx, row in final_bike.iterrows():

    # 마커 컬러 설정
    if str(row['ID']).startswith('S') :
        marker_color = 'blue'
    elif str(row['ID']).startswith('0') : 
        marker_color = 'green'
    else :
        marker_color = 'red'
        
    folium.Marker(
        location = [row['위도'], row['경도']],
        popup = row['위치 정보'],
        tooltip = row['ID'],
        icon = folium.Icon(color=marker_color)
    ).add_to(map)

# 지도 출력 (Jupyter Notebook에서만 동작)
map

# %% [markdown]
# ### 각 Node 사이의 차량 이동 시간 계산

# %%
def calculate_travel_time(start_lat, start_lon, end_lat, end_lon, client_id, client_secret):
    # 네이버 지도 API의 URL
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    
    # 요청 헤더에 Client ID와 Secret 추가
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret
    }
    
    # 요청 파라미터 설정
    params = {
        "start": f"{start_lon},{start_lat}",
        "goal": f"{end_lon},{end_lat}",
        "option": "trafast"  # 가장 빠른 경로 옵션
    }
    
    # API 호출
    response = requests.get(url, headers=headers, params=params)
    
    # 응답이 성공적인지 확인
    if response.status_code == 200:
        data = response.json()
        
        if "route" in data and "trafast" in data["route"]:
            # 거리(m)와 시간(밀리초) 추출
            distance = data["route"]["trafast"][0]["summary"]["distance"]
            duration = data["route"]["trafast"][0]["summary"]["duration"]
            
            # 시간(밀리초)을 분으로 변환
            duration_minutes = duration / (1000 * 60)
            
            return distance, duration_minutes
        else:
            raise Exception(f"Error in response: {data['message']}")
    else:
        raise Exception(f"HTTP error: {response.status_code}")

# %%
# 코드 돌린 시간 : 금요일 9시 40분 ~ 토요일 03시 10분
# 모든 지점 쌍에 대해 거리와 시간을 계산하여 데이터프레임 생성
start_index = 89
results = []

for (i, row1), (j, row2) in itertools.combinations(final_bike.iterrows(), 2):

    if i < start_index : continue
    if i > start_index : break
        
    start_lat, start_lon = row1['위도'], row1['경도']
    end_lat, end_lon = row2['위도'], row2['경도']
    
    distance, duration_minutes = calculate_travel_time(start_lat, start_lon, end_lat, end_lon, client_id, client_secret)

    print(f"{counts}. {row1['위치 정보']} -> {row2['위치 정보']} : {duration_minutes}분")
    counts += 1
    
    if distance is not None and duration_minutes is not None:
        results.append({
            '출발 지점': row1['ID'],
            '출발 위치 정보' : row1['위치 정보'],
            '도착 지점': row2['ID'],
            '도착 위치 정보' : row2['위치 정보'],
            '거리 (km)': distance / 1000,
            '예상 이동 시간 (분)': duration_minutes
        })

# %%
# 현재 계산된 부분 데이터프레임화
results_df = pd.DataFrame(results)

# 제외할 부분 제외
# results_df = results_df[ results_df['출발 지점'] != 'ST-2235' ]

display(results_df)

# %%
# 지금까지 저장된 데이터 로드
file_path = '따릉이 대여소 및 candidate location 사이의 차량 이동 시간_1.csv'
temp_df = pd.read_csv(file_path, index_col=0)

# 기존 데이터와 새로 계산된 데이터 병합
final_df = pd.concat([temp_df, results_df], axis=0, ignore_index=True)

# 결과 데이터프레임 파일로 저장
file_path = '따릉이 대여소 및 candidate location 사이의 차량 이동 시간.csv'
final_df.to_csv(file_path)

# %% [markdown]
# ## 최종 Modeling

# %%
# demand이자 candidate location 파악
file_path = 'demand_and_candidate location 정보.csv'
existing_and_candidate = pd.read_csv(file_path, index_col=0)

def transform_id(id_value):
    try:
        # float 값일 경우 int로 변환
        return int(float(id_value))
    except ValueError:
        # 'ST-'로 시작하는 경우 'ST-'를 제거하고 int로 변환
        if id_value.startswith('ST-'):
            return int(id_value[3:])
        else:
            raise ValueError(f"Unexpected ID format: {id_value}")

# ID 열 변환 <- 기존 + candidate
existing_and_candidate['ID'] = existing_and_candidate['ID'].apply(transform_id)

# 기존 따릉이는 제외
only_candidate = existing_and_candidate[ existing_and_candidate['ID'] > 8000 ].reset_index(drop=True)

display(existing_and_candidate)
display(only_candidate)
len(existing_and_candidate['ID'].unique()), len(only_candidate['ID'].unique())

# %%
# N_I 파악

# candidate location 사이의 도보 거리 데이터 로드
file_path = 'candidate location 사이의 도보 시간.csv'
time_between_candidates = pd.read_csv(file_path, index_col=0)

# '출발 지점', '도착 지점' 열은 int로 만들기
time_between_candidates[['출발 지점', '도착 지점']] = time_between_candidates[['출발 지점', '도착 지점']].astype(int)
display(time_between_candidates.head(5))

# 예상 이동 시간이 10분 이내인 쌍만 추출
N_I_data = time_between_candidates[time_between_candidates['예상 이동 시간 (분)'] <= 10]

# '출발 지점'에 대해서 그룹화 / '도착 지점' 데이터를 리스트화 / 결과물을 딕셔너리로 변환
N_I_dict = N_I_data.groupby('출발 지점')['도착 지점'].apply(list).to_dict()

# 자신만 cover하는 candidate location에 대해 데이터 추가
for key in set(only_candidate['ID'].values) - set(N_I_dict.keys()) :
    N_I_dict[key] = []

# 자신 또한 자신의 demand -> key 값을 value에 포함시키기
for key in N_I_dict.keys() : 
    N_I_dict[key].append(key)

display(len(N_I_dict.keys()))
display(N_I_dict)

# %%
# demand node마다 시간별 demand 파악

# 성북구 버스 정류장 데이터 로드
file_path = '성북구 버스정류장 정보.xlsx'
data = pd.read_excel(file_path)

# 순위 기록한 행 삭제
data.dropna(how='any', inplace=True)

# '버스정류장ARS번호'열과 시간별 demand가 기록된 열을 제외하고 나머지 열 모두 삭제
data.drop(columns=['역명','X좌표','Y좌표'], inplace=True)

# ARS_ID와 모든 demand 값들을 int로 변경
data = data.astype(int)

# candidate location이 아닌 버스 정류장은 제외
data = data[ data['버스정류장ARS번호'].isin(only_candidate['ID']) ].reset_index(drop=True)

# index를 ARS_ID로 설정 / transpose / 딕셔너리로 변환
demand_by_time_dict = data.set_index('버스정류장ARS번호').T.to_dict('list')

# 버스 정류장 demand를 시간에 관계없이 통합
demand_dict = {}
for key, values in demand_by_time_dict.items():
    demand_dict[key] = sum(values)

# Boxplot 생성
demand_df = pd.DataFrame(list(demand_dict.items()), columns=['ID', 'Demand'])
plt.figure(figsize=(10, 3))
plt.boxplot(demand_df['Demand'], vert=False, patch_artist=True, meanline=True, showmeans=True)
plt.xlabel('Demand')
plt.yticks([])
plt.title('Boxplot of Demand Data')
plt.show()

# Robust Scaler를 적용한 후, 변환 후 데이터의 최솟값을 0으로 만들어주기
values = np.array(list(demand_dict.values())).reshape(-1, 1)
scaler = RobustScaler()
scaled_values = scaler.fit_transform(values)

min_scaled_value = np.min(scaled_values)
normalized_scaled_values = scaled_values - min_scaled_value
scaled_demand_dict = {key: float(normalized_scaled_values[i][0]) for i, key in enumerate(demand_dict.keys())}

# 결과 출력
scaled_demand_dict

# %%
# candidate location 및 대여소 사이의 거리 계산

# candidate location 및 대여소 사이의 거리 데이터 로드
file_path = '따릉이 대여소 및 candidate location 사이의 차량 이동 시간.csv'
data = pd.read_csv(file_path, index_col=0)

# 필요없는 열 삭제
data.drop(columns=['출발 위치 정보', '도착 위치 정보', '예상 이동 시간 (분)'], inplace=True)

# '출발 지점', '도착 지점'열 int로 변환
data['출발 지점'] = data['출발 지점'].apply(transform_id)
data['도착 지점'] = data['도착 지점'].apply(transform_id)

# (출발 지점, 도착 지점) : 거리(km) <- 이렇게 생긴 dictionary 생성
distance_dict = { (row["출발 지점"], row["도착 지점"]) : row["거리 (km)"] for _, row in data.iterrows() }

# 거리를 대칭적으로 만듭니다
for (i, j) in list(distance_dict.keys()):
    distance_dict[(j, i)] = distance_dict[(i, j)]

# Boxplot 생성
distance_df = pd.DataFrame(list(distance_dict.items()), columns=['pair', 'Distance'])
plt.figure(figsize=(10, 3))
plt.boxplot(distance_df['Distance'], vert=False, patch_artist=True, meanline=True, showmeans=True)
plt.xlabel('Distance (km)')
plt.yticks([])
plt.title('Boxplot of Distance Data')
plt.show()

# Robust Scaler를 적용한 후, 변환 후 데이터의 최솟값을 0으로 만들어주기
values = np.array(list(distance_dict.values())).reshape(-1, 1)
scaler = RobustScaler()
scaled_values = scaler.fit_transform(values)

min_scaled_value = np.min(scaled_values)
normalized_scaled_values = scaled_values - min_scaled_value
scaled_distance_dict = {key: float(normalized_scaled_values[i][0]) for i, key in enumerate(distance_dict.keys())}

# 결과 출력
scaled_distance_dict

# %%
# 데이터 정의

# candidate location index = 실험을 통과한 버스 정류장
I = only_candidate['ID'].unique().tolist()

# demand and candidate location index = depot & 기존 따릉이 대여소 & 실험을 통과한 버스 정류장 
A = existing_and_candidate['ID'].unique().tolist()

# N_I 정의 -> demand node i를 cover하는 candidate location j의 집합(i도 포함)
N = copy.deepcopy(N_I_dict)

# candidate location의 demand <- 시간별로 통합했음.
h = copy.deepcopy(scaled_demand_dict)

# candidate location 및 대여소 사이의 거리
d = copy.deepcopy(scaled_distance_dict)

# 임의로 설정한 값
P = 7

# %%
def validate_and_convert_data():
    # ID 값 검증 및 변환
    I_int = [int(i) for i in I]
    A_int = [int(i) for i in A]

    # N 딕셔너리의 키와 값을 정수로 변환
    N_int = {int(key): [int(v) for v in value] for key, value in N.items()}

    # h 딕셔너리의 키는 정수, 값은 정수로 변환
    h_int = {int(key): float(value) for key, value in h.items()}

    # d 딕셔너리의 키를 정수 튜플로 변환하고 값을 정수 또는 실수로 변환
    d_int = {(int(key[0]), int(key[1])): float(value) for key, value in d.items()}

    # 데이터 타입 및 범위 확인
    for i in I_int:
        assert isinstance(i, int), f"ID {i} is not an integer"
    for i in A_int:
        assert isinstance(i, int), f"ID {i} is not an integer"
    for key, value in N_int.items():
        assert isinstance(key, int), f"Key {key} in N is not an integer"
        assert all(isinstance(v, int) for v in value), f"Values in N[{key}] are not all integers"
    for key, value in h_int.items():
        assert isinstance(key, int), f"Key {key} in h is not an integer"
        assert isinstance(value, (int, float)) , f"Values in h[{key}] is not float"
    for key, value in d_int.items():
        assert isinstance(key, tuple) and len(key) == 2, f"Key {key} in d is not a tuple of length 2"
        assert isinstance(key[0], int) , f"Key {key}의 0번째 component is not an integer"
        assert isinstance(key[1], int) , f"Key {key}의 1번째 component is not an integer"
        assert isinstance(value, (int, float)), f"Value {value} in d is not a number"
    
    print("All data validated and converted successfully!")
    return I_int, A_int, N_int, h_int, d_int

# 원본 데이터를 변환된 데이터로 업데이트
I, A, N, h, d = validate_and_convert_data()

# %%
sets = d
sets, len(sets)

# %%
from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import copy

# Create an environment with your WLS license
params = {
    "WLSACCESSID": '4697706d-8805-46b0-a28c-abe0d3d2eb8a',
    "WLSSECRET": '22b50eb0-f60e-4d69-a65d-129377f201c8',
    "LICENSEID": 2526067,
}
env = gp.Env(params=params)

def solve_multiobjective_problem(weight_coverage, weight_distance):
    
    # Create a new model
    model = Model("Multi_Objective_Optimization", env=env)

    """ Decision Variables 정의 """
    # Z_i : demand node i가 cover되면 1 아니면 0
    Z = model.addVars(I, vtype=GRB.BINARY, name="Z")

    # O_j : candidate location i에 따릉이 대여소가 건설되면 1 아니면 0 / 기존 따릉이 대여소에 대해서는 1의 값을 가짐
    O = model.addVars(A, vtype=GRB.BINARY, name="O")

    # x_a_b : TSP 결과 demand node a와 demand node b가 연결되어 있으면 1 아니면 0
    x = model.addVars([(a, b) for a in A for b in A if a != b], vtype=GRB.BINARY, name="x")

    # u_a : 방문 순서를 의미 / n은 TSP 문제에서 들러야할 node 수 -> 시작점 + node ( A에는 depot도 포함되어있음. )
    n = len(A) - len(I) + P
    u = model.addVars(A, vtype=GRB.INTEGER, lb=0, ub=n, name="u")

    """ Objective Function 정의 """
    # Objective Function 정의
    location_objective = quicksum(h[i] * Z[i] for i in I)
    TSP_objective = quicksum(d[a, b] * x[a, b] for a in A for b in A if a != b)

    # 목적 함수 추가
    model.setObjectiveN(-TSP_objective, 0, priority=0, name="TSP")
    model.setObjectiveN(location_objective, 1, priority=1, name="MaximalCovering")


    """ Constraints 정의 """
    # candidate location 중 건설해야하는 따릉이 대여소는 총 P개
    model.addConstr(quicksum(O[i] for i in I) == P, "P_docks")

    # 특정 거리 안에 새로운 대여소가 위치하면 해당 demand는 cover됨.
    for i in I:
        model.addConstr(Z[i] <= quicksum(O[j] for j in N[i]), f"Cover_{i}")

    # 기존 따릉이 대여소 + depot은 원래부터 건설되어 있었던 것임.
    for i in set(A) - set(I):
        model.addConstr(O[i] == 1, f"Existing_{i}")

    # 만약 node에 대여소가 있다면, 특정 node로 들어오는 arc는 1개
    for b in A:
        model.addConstr(quicksum(x[a, b] for a in A if a != b) == O[b], f"Inbound_{b}")

    # 만약 node에 대여소가 있다면, 특정 node에서 나가는 arc는 1개
    for a in A:
        model.addConstr(quicksum(x[a, b] for b in A if b != a) == O[a], f"Outbound_{a}")

    # 출발점(=성북구 도시관리공단) 지정
    model.addConstr(u[0] == 1, "Depot")

    # 나머지 node들은 모두 1번보다 후순위 / node에 대여소 및 depot이 존재한다면 양수, 아니면 0으로 설정
    for a in set(A)-set([0]):
        model.addConstr(u[a] <= n * O[a], f"Rank_{a}")
        model.addConstr(u[a] >= 2 * O[a], f"Rank_{a}")

    # subtour elimination constraint
    for a in set(A)-set([0]) :
        for b in set(A)-set([0]) :
            if b != a :
                model.addConstr(u[a] - u[b] + 1 <= n * (1 - x[a, b]), f"Subtour_{a}_{b}")

    # Optimize the model
    model.optimize()
    
    # Extract objective values
    coverage_value = location_objective.getValue()  # Objective value
    tsp_distance_value = TSP_objective.getValue()  # Secondary objective

    # Extract variable values
    O_values = {i: O[i].X for i in A}
    u_values = {i: u[i].X for i in A}

    return coverage_value, tsp_distance_value, O_values, u_values

w = 0.5
coverage, distance, O_values, u_values = solve_multiobjective_problem(w, 1 - w)

coverage, distance, O_values, u_values

# %%
p=7
n = len(A)-len(I)+p

# Gurobi 모델 생성
model = gp.Model("TSP_Maximal_Covering",env=env)

# TSP 변수
X_ij = model.addVars(A, A, vtype=GRB.BINARY, name="x_ij")
U_i = model.addVars(A, vtype=GRB.INTEGER, name="u_i")

# Maximal Covering 변수
Z_i = model.addVars(I, vtype=GRB.BINARY, name="z_i")
O_j = model.addVars(A, vtype=GRB.BINARY, name="o_j")

model.ModelSense = GRB.MINIMIZE
model.setParam('MIPGap', 0.03)

# 목적 함수 1: TSP 최소화
model.setObjectiveN(gp.quicksum(d[i,j] * X_ij[i, j] for i in A for j in A), index=0, priority=1, name="TSP")

# 목적 함수 2: 최대 커버링
model.setObjectiveN(-gp.quicksum(h[j] * Z_i[j] for j in I), index=1, priority=0, name="MaximalCovering")

# 커버링 제약 조건
model.addConstrs((Z_i[i] <= gp.quicksum(O_j[j] for j in N[i])) for i in I)
model.addConstr(gp.quicksum(O_j[j] for j in I) == p)
model.addConstrs(O_j[j]==1 for j in AmB)

# 제약 조건: 각 도시를 한 번 방문
model.addConstrs(gp.quicksum(X_ij[i, j] for i in A if i != j) == O_j[j] for j in A)
model.addConstrs(gp.quicksum(X_ij[i, j] for j in A if j != i) == O_j[i] for i in A)

# 서브투어 제거 제약 조건
model.addConstr(U_i[0]==1)
model.addConstrs((U_i[i] - U_i[j] + n * X_ij[i, j] <= n - 1) for i in range(1,n) for j in range(1,n) )
model.addConstrs((U_i[i]<=n*O_j[i] for i in range(1,n)))
model.addConstrs(U_i[i]>=2*O_j[i] for i in range(1,n))


# 모델 최적화
model.optimize()


# %%
# coverage 값과 distance 값에 다른 weight를 부여하여 하나로 결합된 목적식을 최적화하는 함수

def solve_multiobjective_problem(weight_coverage, weight_distance):
    
    # Model definition
    model = LpProblem("Multi_Objective_Optimization", LpMaximize)

    """ Decision Variables 정의 """
    # Z_i : demand node i가 cover되면 1 아니면 0
    Z = LpVariable.dicts("Z", I, 0, 1, LpBinary)
    
    # O_j : candidate location i에 따릉이 대여소가 건설되면 1 아니면 0 / 기존 따릉이 대여소에 대해서는 1의 값을 가짐 
    O = LpVariable.dicts("O", A, 0, 1, LpBinary)

    # x_a_b : TSP 결과 demand node a와 demand node b가 연결되어 있으면 1 아니면 0
    x = LpVariable.dicts("x", [(a, b) for a in A for b in A if a != b], 0, 1, LpBinary)

    # u_a : 방문 순서를 의미 / n은 TSP 문제에서 들러야할 node 수 -> 시작점 + node ( A에는 depot도 포함되어있음. )
    n = len(A) - len(I) + P
    u = LpVariable.dicts("u", A, 0, n, LpInteger)
    
    
    """ Objective Function 정의 """
    # Multi-period Maximal Covering 문제 목적 함수
    location_objective = lpSum(h[i] * Z[i] for i in I)

    # TSP 문제 목적함수
    TSP_objective = lpSum(d[a, b] * x[a, b] for a in A for b in A if a != b)

    # combined objective function
    model += weight_coverage * location_objective - weight_distance * TSP_objective, "Combined_Objective"

    
    """ Constraints 정의 """
    # candidate location 중 건설해야하는 따릉이 대여소는 총 P개
    model += lpSum(O[i] for i in I) == P

    # 특정 거리 안에 새로운 대여소가 위치하면 해당 demand는 cover됨.
    for i in I :
        model += Z[i] <= lpSum(O[j] for j in N[i])

    # 기존 따릉이 대여소 + depot은 원래부터 건설되어 있었던 것임.
    for i in set(A) - set(I) :
        model += O[i] == 1

    # 만약 node에 대여소가 있다면, 특정 node로 들어오는 arc는 1개
    for b in A :
        model += lpSum(x[a, b] for a in A if a != b) == O[b]

    # 만약 node에 대여소가 있다면, 특정 node에서 나가는 arc는 1개
    for a in A :
        model += lpSum(x[a, b] for b in A if b != a) == O[a]

    # 출발점(=성북구 도시관리공단) 지정
    model += u[0] == 1

    # 나머지 node들은 모두 1번보다 후순위 / node에 대여소 및 depot이 존재한다면 양수, 아니면 0으로 설정
    for a in A :
        model += u[a] <= n * O[a]

    # subtour elimination constraint
    for a in A :
        for b in A :
            if b != a :
                model += u[a] - u[b] + 1 <= n * (1 - x[a, b])

    # Solve the model
    model.solve()

    # Extract objective values
    coverage_value = value(location_objective)
    tsp_distance_value = value(TSP_objective)

    # Extract variable values
    O_values = {i: value(O[i]) for i in A}
    u_values = {i: value(u[i]) for i in A}
    
    return coverage_value, tsp_distance_value, O_values, u_values

w = 0.5
coverage, distance, O_values, u_values = solve_multiobjective_problem(w, 1 - w)

coverage, distance, O_values, u_values

# %%
# # Generate Pareto front by varying weights
# results = []
# for w in np.linspace(0, 1, 20):
#     coverage, distance = solve_multiobjective_problem(w, 1 - w)
#     results.append((distance, coverage))

w = 0.5
coverage, distance, O_values, u_values = solve_multiobjective_problem(w, 1 - w)

coverage, distance, O_values, u_values
# # Sort results to plot the Pareto frontier
# results.sort()

# # Extract distances and coverages
# distances, coverages = zip(*results)

# # Plotting the Pareto front
# plt.figure(figsize=(10, 6))
# plt.scatter(distances, coverages, label='Solutions')
# plt.plot(distances, coverages, linestyle='--', color='r', label='Pareto Frontier')
# plt.xlabel('Total Distance')
# plt.ylabel('Total Coverage')
# plt.title('Pareto Front')
# plt.legend()
# plt.show()

# %%



