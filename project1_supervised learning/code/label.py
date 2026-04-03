import pandas as pd

def get_binary_sentiment(text):
    neg_keywords = ['不適合', '噁心', '怕', '扣分', '囂張', '爛', '恐怖', '輸', '世襲', '咆哮', '騙',"黑金",'貪汙']
    pos_keywords = ['支持', '加油', '讚', '優秀', '強', '贏', '希望', '理性', '唯一']
    if any(k in text for k in neg_keywords): return 'Negative'
    if any(k in text for k in pos_keywords): return 'Positive'
    return 'Positive'

# 讀取原始資料
df_raw = pd.read_csv('fb_storm_boost.csv')
processed_list = []

for _, row in df_raw.iterrows():
    text = str(row['Comment']).replace('\n', ' ').strip()
    sentiment = get_binary_sentiment(text)
    
    # 偵測陣營
    has_kmt = any(k in text for k in ['李四川', '川', '國民黨', '藍',"川伯"])
    has_tpp = any(k in text for k in ['黃國昌', '國昌', '民眾黨', '白', '老師',"國蔥","戰神"])
    has_dpp = any(k in text for k in ['蘇巧慧', '蘇貞昌', '民進黨', '綠', '巧慧',"賴清德","蘇市長","阿慧","爸"])

    #  核心邏輯：藍白合處理 
    if has_kmt and has_tpp:
        # 同時產生兩筆，分別給李和黃
        processed_list.append({'Content': text, 'Label': 'KMT_Lee', 'Sentiment': sentiment})
        processed_list.append({'Content': text, 'Label': 'TPP_Huang', 'Sentiment': sentiment})
    elif has_dpp:
        processed_list.append({'Content': text, 'Label': 'DPP_Su', 'Sentiment': sentiment})
    elif has_kmt:
        processed_list.append({'Content': text, 'Label': 'KMT_Lee', 'Sentiment': sentiment})
    elif has_tpp:
        processed_list.append({'Content': text, 'Label': 'TPP_Huang', 'Sentiment': sentiment})
    else:
        # 其他不具備明確人名的，暫標為 Other
        processed_list.append({'Content': text, 'Label': 'Other', 'Sentiment': sentiment})

final_df = pd.DataFrame(processed_list)
# 重新編排 ID 並存檔
final_df.insert(0, 'ID', range(1, len(final_df) + 1))
final_df.to_csv('fb_label.csv', index=False, encoding='utf-8-sig')

print(f"處理完成！總樣本數增加至：{len(final_df)} 筆")
print("提示：因為處理了藍白合重複標記，樣本數會比原本的 477 筆多。")