import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
driver = webdriver.Chrome(options=options)

try:
    driver.get("https://www.youtube.com/watch?v=3xAmShCdkOo")
    wait = WebDriverWait(driver, 15)    # 設定最長等待 15 秒，避免網路太慢時程式直接報錯

    print("正在嘗試觸發留言區...")

    #觸發留言載入 (Scrolling)
    for _ in range(20): 
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)   #模擬真人操作 視窗本體 按下end
        time.sleep(3) 

    print("開始抓取留言內容...")


    # 抓取留言內容標籤
    comment_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text")))

    comments_list = []  #comment_elements是抓取的物件
    
    print(f"成功！抓到了 {len(comment_elements)} 則留言。")
    
    for index, el in enumerate(comment_elements):  #把抓到的那一整疊留言「編號」並「排好隊」，一個一個拿出來處理。
        text = el.text.strip()
        if text:
            comments_list.append({     #把抓取內文附加在新的list
                "ID": index + 1,
                "Comment": text
            })

    # 將抓取到的資料存成 DataFrame 並導出 CSV
    df = pd.DataFrame(comments_list)
    # 儲存為原始數據集檔案
    df.to_csv('raw_comments.csv', index=False, encoding='utf-8-sig')
    
    print(f"已成功將 {len(df)} 筆原始留言存至 yt_original_comments.csv")



except Exception as e:
    print(f"錯誤原因: {e}")
    driver.save_screenshot("debug_screen.png") 

finally:
    driver.quit()