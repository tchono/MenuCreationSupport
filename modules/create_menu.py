from datetime import datetime, timedelta
import calendar
import jpholiday
import numpy as np
import pandas as pd
import pulp
import random


import streamlit as st


df_ayame_ryori_list = None
df_ayame_ryori = None

df_dates = None
df_monthly_menu = None

data = {
    'カテゴリ': ['主食', '汁物', '主菜', '副菜', '漬物', '主食', '主菜', '副菜', '副々菜or漬物', '果物', '乳製品', '主食', '主菜', '副菜', '副々菜or漬物'],
    '食事': ['朝', '朝', '朝', '朝', '朝', '昼', '昼', '昼', '昼', '昼', '昼', '夕', '夕', '夕', '夕'],
    '使用': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df_melted = pd.DataFrame(data)

columns = ['Ryori_正規化後', '食_区分CD', '区分', '料理_区分CD', '料理_区分', 'Kashokuryo', 'Energy', 'Protein', 'Fat', 'Carbohydrate', 'Calcium', 'Potassium', 'DietaryFiber', 'Sodium', 'TotalCost']

def create_dates_with_weekday_and_holiday(year, month, base_weekday):
    start_date = datetime.date(year, month, 1)
    next_month = month % 12 + 1
    next_month_year = year + month // 12
    end_date = datetime.date(next_month_year, next_month, 1)
    dates = pd.date_range(start_date, end_date - datetime.timedelta(days=1))

    weekday_map = {0: "月曜日", 1: "火曜日", 2: "水曜日", 3: "木曜日", 4: "金曜日", 5: "土曜日", 6: "日曜日"}
    date_info = []
    for date in dates:
        weekday_code = (date.weekday() - base_weekday) % 7  # Adjusting with base_weekday
        weekday_name = weekday_map[date.weekday()]  # Monday is 0 and Sunday is 6
        is_holiday = jpholiday.is_holiday(date)
        date_info.append({"日付": date, "曜日": weekday_name, "曜日コード": weekday_code, "祝日": is_holiday})

    return pd.DataFrame(date_info)

def set_initial_values(year, month, ryori_list, ryori, target_nutrition):
    global df_ayame_ryori_list
    df_ayame_ryori_list = ryori_list
    df_ayame_ryori = ryori

    base_weekday = 5  # 月曜日を基準とする (月曜日を0とする)

    # 日付の作成（曜日と祝日判定付き）
    df_dates = create_dates_with_weekday_and_holiday(year, month, base_weekday)

    # df_meltedと日付を組み合わせる
    df_monthly_menu = df_dates.assign(key=1).merge(df_melted.assign(key=1), on="key").drop("key", axis=1)

    columns = ['Ryori_正規化後', '食_区分CD', '区分', '料理_区分CD', '料理_区分', 'Kashokuryo', 'Energy', 'Protein', 'Fat', 'Carbohydrate', 'Calcium', 'Potassium', 'DietaryFiber', 'Sodium', 'TotalCost']
    for col in columns:
        df_monthly_menu[f"{col}"] = ''

    df_recipes_data = df_ayame_ryori_list

    year = 2024
    month = 1

    # 祝日を除外する
    temp = df_monthly_menu[(df_monthly_menu['曜日コード'] <= 1) | (df_monthly_menu['祝日']==True)]['日付'].unique().tolist()
    banned_dates = [date.strftime('%Y-%m-%d') for date in temp]

    ryori_cd = [1]  # パン
    weekday = 4  # 金曜 = 4
    n_weeks = 1  # 提供間隔（何週間おきか）（0 = 毎週）
    provide_date_bread = get_provide_start_date(ryori_cd, year, month, weekday, n_weeks)

    banned_dates.extend(provide_date_bread)

    ryori_cd = [2]  # 麺
    weekday = 2  # 水曜 = 2
    n_weeks = 0  # 毎週（何週間おきか）（0 = 毎週）
    provide_date_noodles = get_provide_start_date(ryori_cd, year, month, weekday, n_weeks)

    banned_dates.extend(provide_date_noodles)
    banned_dates.sort()

    ryori_cd = [3]  # 丼
    interval = 20
    provide_date_ricerbowl = get_provide_candidate_date(ryori_cd, year, month, interval, 1, banned_dates)

    banned_dates.extend(provide_date_ricerbowl)

    ryori_cd = [4]  # カレー
    interval = 20
    provide_date_curry = get_provide_candidate_date(ryori_cd, year, month, interval, 1, banned_dates)

    banned_dates.extend(provide_date_curry)

    ryori_cd = [7, 8] # ピラフ、チャーハン
    interval = 20
    provide_date_pilaffriedrice = get_provide_candidate_date(ryori_cd, year, month, interval, 1, banned_dates)

    banned_dates.extend(provide_date_pilaffriedrice)

    ryori_cd = [10, 11] # 混ぜ込みごはん（平均5回くらい提供？）
    interval = 0
    provide_date_mixedrice = get_provide_candidate_date(ryori_cd, year, month, interval, 5, banned_dates)

    banned_dates.extend(provide_date_mixedrice)
    banned_dates.sort()

    """## 初期値設定"""
    # ShokuKubun = ['朝', '昼', '夕']

    # 提供日をデータフレームに反映
    for day in df_dates['日付']:
        formatted_month = day.month
        formatted_day = day.strftime('%Y-%m-%d')
        if formatted_day in provide_date_bread:
            matching_recipes = df_recipes_data[df_recipes_data['料理_区分CD'] == 1][columns].sample(n=1).values[0]
            df_monthly_menu.loc[(df_monthly_menu['日付']==formatted_day) & (df_monthly_menu['カテゴリ']=='主食') & (df_monthly_menu['食事']=='昼'), columns] = matching_recipes
        if formatted_day in provide_date_noodles:
            matching_recipes = df_recipes_data[df_recipes_data['料理_区分CD'] == 2][columns].sample(n=1).values[0]
            df_monthly_menu.loc[(df_monthly_menu['日付']==formatted_day) & (df_monthly_menu['カテゴリ']=='主食') & (df_monthly_menu['食事']=='昼'), columns] = matching_recipes
        if formatted_day in provide_date_ricerbowl:
            matching_recipes = df_recipes_data[df_recipes_data['料理_区分CD'] == 3][columns].sample(n=1).values[0]
            df_monthly_menu.loc[(df_monthly_menu['日付']==formatted_day) & (df_monthly_menu['カテゴリ']=='主食') & (df_monthly_menu['食事']=='昼'), columns] = matching_recipes
        if formatted_day in provide_date_curry:
            matching_recipes = df_recipes_data[df_recipes_data['料理_区分CD'] == 4][columns].sample(n=1).values[0]
            df_monthly_menu.loc[(df_monthly_menu['日付']==formatted_day) & (df_monthly_menu['カテゴリ']=='主食') & (df_monthly_menu['食事']=='昼'), columns] = matching_recipes
        if formatted_day in provide_date_pilaffriedrice:
            if formatted_month % 2 == 0:
                matching_recipes = df_recipes_data[df_recipes_data['料理_区分CD'] == 8][columns].sample(n=1).values[0]
            else:
                matching_recipes = df_recipes_data[df_recipes_data['料理_区分CD'] == 7][columns].sample(n=1).values[0]
            df_monthly_menu.loc[(df_monthly_menu['日付']==formatted_day) & (df_monthly_menu['カテゴリ']=='主食') & (df_monthly_menu['食事']=='昼'), columns] = matching_recipes
        if formatted_day in provide_date_mixedrice:
            matching_recipes = df_recipes_data[(df_recipes_data['料理_区分CD'] == 10) | (df_recipes_data['料理_区分CD'] == 10)][columns].sample(n=1).values[0]
            # 昼食か夕食か
            if random.random() < 0.5:
                df_monthly_menu.loc[(df_monthly_menu['日付']==formatted_day) & (df_monthly_menu['カテゴリ']=='主食') & (df_monthly_menu['食事']=='昼'), columns] = matching_recipes
            else:
                df_monthly_menu.loc[(df_monthly_menu['日付']==formatted_day) & (df_monthly_menu['カテゴリ']=='主食') & (df_monthly_menu['食事']=='夕'), columns] = matching_recipes

        # 過去料理一覧から、朝食汁物を取得（確率的にサンプル）
        df_monthly_menu.loc[(df_monthly_menu['日付']==formatted_day) & (df_monthly_menu['カテゴリ']=='汁物') & (df_monthly_menu['食事']=='朝'), columns] = df_ayame_ryori[(df_ayame_ryori['ShokuKubunCode'] == 1) & (df_ayame_ryori['区分'] == '汁物')][columns].sample(n=1).values[0]

    staple_food = df_recipes_data[(df_recipes_data['Ryori_正規化後'] == 'ご飯') & (df_recipes_data['Kashokuryo'] == 78)][columns].values[0]
    df_monthly_menu.loc[(df_monthly_menu['カテゴリ'] == '主食') & (df_monthly_menu["Ryori_正規化後"] == ''), columns] = staple_food

    milk = df_recipes_data[(df_recipes_data['Ryori_正規化後'] == '牛乳200cc') & (df_recipes_data['Kashokuryo'] == 200)][columns].values[0]
    df_monthly_menu.loc[(df_monthly_menu['カテゴリ'] == '乳製品') & (df_monthly_menu["Ryori_正規化後"] == ''), columns] = milk

    columns_to_sum = ['Kashokuryo', 'Energy', 'Protein', 'Fat', 'Carbohydrate', 'Potassium', 'Calcium', 'DietaryFiber', 'Sodium', 'TotalCost']

    # 合計する列に文字列が混ざってる場合は数値に変換
    for column in columns_to_sum:
        df_monthly_menu[column] = pd.to_numeric(df_monthly_menu[column], errors='coerce')

    # 合計する列にNaN値があるかチェック
    for column in columns_to_sum:
        df_monthly_menu[column].fillna(0, inplace=True)

    # グループ化して合計
    total_per_date = df_monthly_menu.groupby('日付')[columns_to_sum].sum().reset_index()
    total_per_date_shoku = df_monthly_menu.groupby(['日付', '食事'])[columns_to_sum].sum().reset_index()

    """## 関数"""

    # 目標原価率
    target_cost_rate = 0.602

    df_monthly_menu_temp = df_monthly_menu.copy()

    meal_type = '昼'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    meal_type = '朝'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    meal_type = '夕'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    meal_type = '昼'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    meal_type = '朝'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    meal_type = '夕'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    return df_monthly_menu_temp

def create_dates_with_weekday_and_holiday(year, month, base_weekday):
    start_date = datetime(year, month, 1)
    next_month = month % 12 + 1
    next_month_year = year + month // 12
    end_date = datetime(next_month_year, next_month, 1)
    dates = pd.date_range(start_date, end_date - timedelta(days=1))

    weekday_map = {0: "月曜日", 1: "火曜日", 2: "水曜日", 3: "木曜日", 4: "金曜日", 5: "土曜日", 6: "日曜日"}
    date_info = []
    for date in dates:
        weekday_code = (date.weekday() - base_weekday) % 7  # Adjusting with base_weekday
        weekday_name = weekday_map[date.weekday()]  # Monday is 0 and Sunday is 6
        is_holiday = jpholiday.is_holiday(date)
        date_info.append({"日付": date, "曜日": weekday_name, "曜日コード": weekday_code, "祝日": is_holiday})

    return pd.DataFrame(date_info)

# 提供日付を取得する関数群


# 最終提供日を取得
def get_latest_provide_date(ryori_cd_list):
    latest_date = None
    for ryori_cd in ryori_cd_list:
        filtered_df = df_ayame_ryori_list[df_ayame_ryori_list['料理_区分CD'] == ryori_cd]
        current_latest_date = filtered_df['最終提供日'].max()
        if latest_date is None or (current_latest_date is not None and current_latest_date > latest_date):
            latest_date = current_latest_date
    return datetime.strptime(latest_date, '%Y-%m-%d')

# 指定条件を基に料理を取得（現在はランダム）
def get_row_by_ryori_cd(ryori_cd):
    filtered_df = df_ayame_ryori_list[df_ayame_ryori_list['料理_区分CD'] == ryori_cd]
    random_row = filtered_df.sample(1)
    return random_row

# 指定曜日で最初の日付を取得
def get_first_weekday(year, month, weekday):
    first_day_of_month = datetime(year, month, 1)
    first_day_weekday = first_day_of_month.weekday()
    days_to_first_weekday = (weekday - first_day_weekday) % 7
    first_weekday_date = first_day_of_month + timedelta(days=days_to_first_weekday)

    return first_weekday_date

# 日付間の間隔（週）
def between_dates(date1, date2):
    if isinstance(date1, str):
        date1 = datetime.strptime(date1, '%Y-%m-%d')
    if isinstance(date2, str):
        date2 = datetime.strptime(date2, '%Y-%m-%d')
    delta = abs((date2 - date1).days)
    return delta

def calculate_provide_start_date(ryori_cd, year, month, weekday, n_weeks):
    first_weekday_date = get_first_weekday(year, month, weekday)
    if n_weeks <= 0:
        return first_weekday_date
    latest_date = get_latest_provide_date(ryori_cd)
    date_difference = between_dates(first_weekday_date, latest_date)

    # 前回提供が何週間前（付近（四捨五入））か。
    add_weeks = n_weeks + 1 - round(date_difference / 7)

    if add_weeks <= 0:
        return first_weekday_date
    else:
        return first_weekday_date + timedelta(weeks=(add_weeks))

def get_provide_start_date(ryori_cd, year, month, weekday, n_weeks):
    start_date = calculate_provide_start_date(ryori_cd, year, month, weekday, n_weeks)

    dates = []
    current_date = start_date
    while current_date.month == month:
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(weeks=n_weeks + 1)

    return dates

def random_date(year, month, s_date, num_dates, banned_dates=[]):
    # 指定された月の日付を全部生成
    days_in_month = calendar.monthrange(year, month)[1]
    dates_in_month = [datetime(year, month, day) for day in range(s_date, days_in_month + 1)]
    # 禁止日を除外する
    dates_in_month = [date.strftime('%Y-%m-%d') for date in dates_in_month if date not in banned_dates]
    if dates_in_month == []:
        return None
    # ランダムに指定された個数だけ選択
    if len(dates_in_month) > num_dates:
        random_dates = random.sample(dates_in_month, num_dates)
    else:
        random_dates = dates_in_month  # 候補が少ない場合は全て返す

    return random_dates

def calculate_provide_candidate_dates(ryori_cd, year, month, interval, banned_dates=[]):
    latest_date = get_latest_provide_date(ryori_cd)
    start_date = latest_date + timedelta(days=interval)

    start_of_month = datetime(year, month, 1)
    days_in_month = calendar.monthrange(year, month)[1]
    end_of_month = datetime(year, month, days_in_month)

    if start_date <= end_of_month:
        if start_date < start_of_month:
            start_date = start_of_month
        candidate_dates = [start_date + timedelta(days=i) for i in range((end_of_month - start_date).days + 1)]
        # 禁止日を除外する
        candidate_dates = [date.strftime('%Y-%m-%d') for date in candidate_dates]
        candidate_dates = [date for date in candidate_dates if date not in banned_dates]
        return candidate_dates
    else:
        return []

def get_provide_candidate_date(ryori_cd, year, month, interval, num_dates, banned_dates=[]):
    candidate_dates = calculate_provide_candidate_dates(ryori_cd, year, month, interval, banned_dates)

    # ランダムに指定された個数だけ選択
    if len(candidate_dates) > num_dates:
        random_dates = random.sample(candidate_dates, num_dates)
    else:
        random_dates = candidate_dates  # 候補が少ない場合は全て返す
    random_dates.sort()

    return random_dates

def get_meal_plan_optimization_df(temp_df, categories, target_nutrition):
    # 最適化問題を設定
    prob = pulp.LpProblem("Meal_Plan_Optimization", pulp.LpMinimize)

    # 選択変数の枠を定義
    num_recipes = len(temp_df)
    num_categories = len(categories)

    # 選択変数の枠を定義
    recipe_vars = np.zeros((num_recipes, num_categories), dtype=object)
    for i in range(num_recipes):
        for j, cate in enumerate(categories):
            if "or" in cate and temp_df.loc[i, '区分'] in [c.strip() for c in cate.split("or")]:
                recipe_vars[i][j] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpBinary)
            elif temp_df.loc[i, '区分'] == cate:
                recipe_vars[i][j] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpBinary)
            else:
                recipe_vars[i][j] = 0

    # 各カテゴリごとに一意のレシピが選ばれる制約を追加
    for j in range(num_categories):
        prob += pulp.lpSum([recipe_vars[i][j] for i in range(num_recipes) if recipe_vars[i][j] != 0]) == 1

    # 栄養価の制約を追加
    for nutrient, (min_val, max_val) in target_nutrition.items():
        nutrition_constraint = pulp.lpSum([recipe_vars[i][j] * temp_df.loc[i, nutrient] for i in range(num_recipes) for j in range(num_categories) if recipe_vars[i][j] != 0])
        if min_val is not None:
            prob += nutrition_constraint >= min_val
        if max_val is not None:
            prob += nutrition_constraint <= max_val

    if prob.solve() == -1:
        raise Exception("エラー：問題が解けなかった")

    # 結果の出力
    selected_recipes_df = pd.DataFrame(columns=temp_df.columns)

    for j in range(num_categories):
        for i in range(num_recipes):
            if recipe_vars[i][j] != 0 and pulp.value(recipe_vars[i][j]) == 1:
                new_row = temp_df.loc[i].tolist()
                selected_recipes_df.loc[len(selected_recipes_df)] = new_row

    return selected_recipes_df


# 栄養価の合計値を計算
def calculate_total_nutrition(selected_recipes_df, target_nutrition):
    total_nutrition = {}
    for nutrient in target_nutrition.keys():
        total_nutrition[nutrient] = selected_recipes_df[nutrient].sum()
    return total_nutrition

def get_ryori_df(categories, num=300):
    df = pd.DataFrame()
    for cate in categories:
        if "or" in cate:
            for c in cate.split("or"):
                #temp_df = pd.concat([temp_df, df_ayame_ryori_list[df_ayame_ryori_list['区分'] == c]])
                df = pd.concat([df, df_ayame_ryori_list[df_ayame_ryori_list['区分'] == c].sample(min(num, len(df_ayame_ryori_list[df_ayame_ryori_list['区分'] == c])))])
        else:
            #temp_df = pd.concat([temp_df, df_ayame_ryori_list[df_ayame_ryori_list['区分'] == cate]])
            #temp_df = pd.concat([temp_df, df_ayame_ryori_list[df_ayame_ryori_list['区分'] == cate].head(100)])
            df = pd.concat([df, df_ayame_ryori_list[df_ayame_ryori_list['区分'] == cate].sample(min(num, len(df_ayame_ryori_list[df_ayame_ryori_list['区分'] == cate])))])

    # temp_df = temp_df[temp_df['最終提供日'] <= 日付]
    df = df.reset_index()

    return df

def optimize_meal_plan(total_per_date, df_monthly_menu, target_nutrition, total_per_date_shoku, meal_type):
    # TotalCostが高い順にソート
    total_per_date = total_per_date.sort_values('TotalCost', ascending=False)

    for index, row in total_per_date.iterrows():
        formatted_day = row['日付'].strftime('%Y-%m-%d')
        categories = df_monthly_menu[(df_monthly_menu['日付']==formatted_day) & (df_monthly_menu['食事']==meal_type) & (df_monthly_menu['Ryori_正規化後']=='')]['カテゴリ'].tolist()
        if categories == []:
            break
        print(formatted_day)
        total_per_date_shoku_day = total_per_date_shoku[(total_per_date_shoku['日付']==formatted_day) & (total_per_date_shoku['食事']==meal_type)].iloc[0]

        tn = adjust_target_nutrition(target_nutrition, total_per_date_shoku_day, 3)

        max_attempts =5
        attempts = 0
        while attempts < max_attempts:
            try:
                temp_df = get_ryori_df(categories, 100)
                selected_recipes_df = get_meal_plan_optimization_df(temp_df, categories, tn)
                break
            except:
                attempts += 1
                if attempts <= 3:
                    tn = adjust_target_nutrition(target_nutrition, total_per_date_shoku_day, 3, 1 - attempts * 0.1)
                print(f"Attempt {attempts} failed. Retrying...")

        if max_attempts == attempts:
            print(f"error...")
            break

        for ndex2, row2 in selected_recipes_df.iterrows():
            matching_recipes = [row2[columns]]
            df_monthly_menu.loc[(df_monthly_menu['日付']==formatted_day) & (df_monthly_menu['カテゴリ'].str.contains(row2['区分'])) & (df_monthly_menu['食事']==meal_type), columns] = matching_recipes
    return df_monthly_menu


# 1食あたりの目標栄養価
def adjust_target_nutrition(target_nutrition, total_per_date_shoku, serving_size=3, multiplier=0):
    tn = target_nutrition.copy()
    for key in tn.keys():
        if tn[key][0] is not None:
            tn[key] = (tn[key][0] * (1 - multiplier), tn[key][1])
        if tn[key][1] is not None:
            tn[key] = (tn[key][0], tn[key][1] * (1 + multiplier))

    for key in tn.keys():
        if tn[key][0] is not None:
            tn[key] = (tn[key][0] / serving_size, tn[key][1])
        if tn[key][1] is not None:
            tn[key] = (tn[key][0], tn[key][1] / serving_size)

    for key in tn.keys():
        if tn[key][0] is not None:
            tn[key] = (tn[key][0] - total_per_date_shoku[key], tn[key][1])
        if tn[key][1] is not None:
            tn[key] = (tn[key][0], tn[key][1] - total_per_date_shoku[key])
    return tn

"""## 献立生成"""
def create():
    # 目標原価率
    target_cost_rate = 0.602

    df_monthly_menu_temp = df_monthly_menu.copy()

    meal_type = '昼'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    meal_type = '朝'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    meal_type = '夕'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    meal_type = '昼'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    meal_type = '朝'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    meal_type = '夕'
    df_monthly_menu_temp = optimize_meal_plan(total_per_date, df_monthly_menu_temp, target_nutrition, total_per_date_shoku, meal_type)

    return df_monthly_menu_temp