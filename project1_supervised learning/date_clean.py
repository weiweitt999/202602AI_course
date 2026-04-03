import pandas as pd

# 讀取你剛剛人工校對完、有空格的 CSV 檔
try:
    df = pd.read_csv("yt_label.csv", encoding='utf-8-sig')
except FileNotFoundError:
    print("找不到檔案，請確認 Final_Manual_Check.csv 是否在同一個資料夾內")

# 核心清理：刪除 Content 或 Label 為空的橫列
# 這會幫你把截圖中那些只有 ID 沒有內容的空白列全部移掉
df_cleaned = df.dropna(subset=['Content', 'Label', 'Sentiment']).copy()

# 重新排序 ID：讓 ID 變成漂亮的 1, 2, 3... 連續數字
df_cleaned['ID'] = range(1, len(df_cleaned) + 1)

# 儲存成最終乾淨的版本
output_filename = 'yt_label_clean.csv'
df_cleaned.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"清理完成！")
print(f"原本筆數: {len(df)}")
print(f"清理後有效樣本數: {len(df_cleaned)}")
print(f"最終檔案已儲存為: {output_filename}")