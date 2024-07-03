import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
from datetime import datetime, timedelta
import streamlit_calendar as st_calendar
import csv
import io

### 自作のモジュールをインポートする ###
from modules import create_menu

### 各種フラグなどを初期化するセクション ###
if 'df' not in st.session_state:
    st.session_state['df'] = None
    st.session_state['year'] = None
    st.session_state['month'] = None

# CSSの設定をする関数
def set_font_style():

    st.markdown(
        """
        <style>
        textarea {
            font-size: 1.2rem !important;
            font-family:  monospace !important;
        }

        code {
            font-size: 1.2rem !important;
            font-family:  monospace !important;
        }

        div.stButton > button:first-child  {
            margin: 0 auto;
            max-width: 240px;
            padding: 10px 10px;
            background: #6bb6ff;
            color: #FFF;
            transition: 0.3s ease-in-out;
            font-weight: 600;
            border-radius: 100px;
            box-shadow: 0 5px 0px #4f96f6, 0 10px 15px #4f96f6;            
            border: none            
        }

        div.stButton > button:first-child:hover  {
            color: #FFF;
            background:#FF2F2F;
            box-shadow: 0 5px 0px #B73434,0 7px 30px #FF2F2F;
            border: none            
          }

        div.stButton > button:first-child:focus {
            color: #FFF;
            background: #6bb6ff;
            box-shadow: 0 5px 0px #4f96f6, 0 10px 15px #4f96f6;
            border: none            
          }


        button[title="View fullscreen"]{
            visibility: hidden;}

        </style>
        """,
        unsafe_allow_html=True,
    )

#----------------------------------------------#
# シミュレーター画面の表示の処理群
#----------------------------------------------#
def view_mockup():
    # スライドバー
    # 年月を指定するためのウィジェット
    st.sidebar.write("献立作成年月を指定")
    year = st.sidebar.selectbox('年', range(2024, 2031))
    month = st.sidebar.selectbox('月', range(1, 13))

    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader("CSVファイル「過去料理データ一覧」をアップロードしてください", type=["csv"])
    uploaded_file2 = st.sidebar.file_uploader("CSVファイル「過去献立データ一覧」をアップロードしてください", type=["csv"])

    st.sidebar.markdown("---")

    st.title('献立作成支援AI')
    # 画像のパス
    image_path = "assets/image/MenuOptimization.png"
    # 画像を表示
    st.image(image_path, use_column_width=True)

    labels = ['エネルギー', 'タンパク質', '脂質', '食塩', '価格']
    labels_en = ['Energy', 'Protein', 'Fat', 'Sodium', 'TotalCost']
    num_cols = len(labels)
    min_values = [0, 0, 0, 0, 0]
    max_values = [2000, 100, 100, 50, 1000]
    initial_values = [(1600, 1800), (60, 100), (0, 100), (0, 8), (0, 500)]
    colsA = st.columns(num_cols)
    goal_data = [0] * num_cols

    for col_index in range(num_cols):
        with colsA[col_index]:
            goal_data[col_index] = st.slider(labels[col_index], min_values[col_index], max_values[col_index], initial_values[col_index])

    button_pressed = st.button('✔献立表作成')

    st.markdown("---")

    if uploaded_file is not None and uploaded_file2 is not None:
        df = pd.read_csv(uploaded_file, encoding='CP932')
        df2 = pd.read_csv(uploaded_file2, encoding='CP932')

        if button_pressed:
            target_nutrition = {
                "Energy": (None, None),
                "Protein": (None, None),
                "Fat": (None, None),
                "Sodium": (None, None),
                "TotalCost": (None, None)
            }

            for i, label in enumerate(labels):
                a = [None, None]
                if goal_data[i][0] != min_values[i]:
                    a[0] = goal_data[i][0]
                if goal_data[i][1] != max_values[i]:
                    a[1] = goal_data[i][1]
                target_nutrition[labels_en[i]] = tuple(a)

            st.session_state['df'] = create_menu.set_initial_values(year, month, df, df2, target_nutrition)
            st.session_state['year'] = year
            st.session_state['month'] = month

        if st.session_state['df'] is not None:
            # st.write(st.session_state['df'])
            temp_df = st.session_state['df'].copy()
            temp_df = temp_df.sort_values(by=["日付", "食_区分CD"])

            option = st.radio('表示する献立を選択', ['朝', '昼', '夕'])
            events = []
            second_offset = 0
            current_date = temp_df["日付"][0]
            for idx, row in temp_df.iterrows():
                if current_date != row["日付"]:
                    current_date = row["日付"]
                    second_offset = 0
                title = f"{row['Ryori_正規化後']}"
                if row['食事'] == '朝':
                    start = row["日付"].replace(hour=9, minute=0, second=second_offset).strftime("%Y-%m-%dT%H:%M:%S")
                elif row['食事'] == '昼':
                    start = row["日付"].replace(hour=12, minute=0, second=second_offset).strftime("%Y-%m-%dT%H:%M:%S")
                elif row['食事'] == '夕':
                    start = row["日付"].replace(hour=17, minute=0, second=second_offset).strftime("%Y-%m-%dT%H:%M:%S")
                end = start
                event = {
                    "title": title,
                    "start": start,
                    "end": end,
                }
                if option == row['食事']:
                    events.append(event)
                    second_offset += 1


            calendar_options = {
                "initialDate": f"{st.session_state['year']}-{st.session_state['month']:02d}-01",
                "headerToolbar": {
                    "left": "prev,next today",
                    "center": "title",
                    "right": "dayGridMonth"
                },
                "editable": True,
                "selectable": True,
                "events": events,
    }
            st_calendar.calendar(options=calendar_options)

            # Create a CSV buffer with CP932 encoding
            csv_buffer = io.StringIO(newline='')
            csv_buffer.write(u'\ufeff')  # BOM for UTF-8
            csv_writer = csv.writer(csv_buffer)

            # Write the header (column names)
            csv_writer.writerow(st.session_state['df'].columns.tolist())

            # Write the data to the CSV buffer
            csv_writer.writerows(st.session_state['df'].values.tolist())

            # Create a download button
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name="data.csv",
                mime="text/csv"
            )



#----------------------------------------------#
# メイン関数
#----------------------------------------------#
def main():

    ### 各種フラグなどを初期化する関数をコール ###
    ### init_parameters()

    # フォントスタイルの設定
    set_font_style()

    # メニューを表示する
    view_mockup()

if __name__ == "__main__":
    main()