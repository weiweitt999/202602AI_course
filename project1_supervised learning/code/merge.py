import pandas as pd

# 讀取兩份 CSV 檔案
# 請確保檔名與你電腦中的實際檔名一致
try:
    df_yt = pd.read_csv('fb_label_clean.csv', encoding='utf-8-sig')
    df_fb = pd.read_csv("yt_label_clean.csv", encoding='utf-8-sig') # 或是你存放 FB 留言的檔名
    
    print(f"讀取成功！")
    print(f"原本 YouTube 樣本數: {len(df_yt)}")
    print(f"新增 Facebook 樣本數: {len(df_fb)}")

    # 合併資料 (上下堆疊)
    # ignore_index=True 會自動忽略舊的索引，方便我們後面重排 ID
    df_combined = pd.concat([df_yt, df_fb], ignore_index=True)

    # 統一欄位名稱 (預防萬一：將 'Comment' 統一改名為 'Content')
    # 如果你的 FB 檔案欄位叫 Comment，請執行這行
    if 'Comment' in df_combined.columns:
        df_combined = df_combined.rename(columns={'Comment': 'Content'})

    # 重新編號 ID (從 1 開始到最後一筆)
    df_combined['ID'] = range(1, len(df_combined) + 1)

    # 儲存成最終最強大的資料集
    output_name = 'Final_Dataset_Combined.csv'
    df_combined.to_csv(output_name, index=False, encoding='utf-8-sig')

    print(f"\n{'='*30}")
    print(f"合併完成！")
    print(f"最終總樣本數：{len(df_combined)} 筆")
    print(f"儲存檔名：{output_name}")
    print(f"{'='*30}")

except FileNotFoundError as e:
    print(f"讀取失敗，請確認檔案是否存在：{e}")