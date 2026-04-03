from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd

chrome_options = Options()
chrome_options.add_argument("--disable-notifications")
chrome_options.add_argument("--window-size=1200,900")

service = Service(r'chromedriver.exe') 
driver = webdriver.Chrome(service=service, options=chrome_options)

url = "https://www.facebook.com/stormmedia/posts/pfbid0tQdRwoGwWwrR2qQmTtZqpsy66WULMTezwjeARR5SUsMQar4MTi8tJ26MSrT73XUEl"

try:
    driver.get(url)
    print("【手動操作階段】請在 20 秒內完成以下動作：")
    print("1. 如果有登入彈窗，請手動點掉它")
    print("2. 確保你能看到留言列表")
    time.sleep(20) # 給你足夠時間清理視窗

    print(" 程式開始強力捲動抓取...")
    
    # 強力捲動
    for i in range(30):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(4) # FB 載入很慢，給它 4 秒
        print(f"正在捲動第 {i+1} 次...")

    # 抓取所有內容
    # FB 的留言特徵：dir="auto" 且在 span 或 div 內
    elements = driver.find_elements(By.XPATH, '//div[@dir="auto"] | //span[@dir="auto"]')
    
    comments_raw = [el.text.strip() for el in elements if len(el.text.strip()) > 2]
    
    # 過濾噪音：去掉重複、去掉官方連結、去掉太短的
    clean_comments = []
    seen = set()
    for c in comments_raw:
        if c not in seen and "http" not in c and "風傳媒" not in c and "相關" not in c:
            clean_comments.append(c)
            seen.add(c)

    df = pd.DataFrame({"Comment": clean_comments})
    df.insert(0, 'ID', range(1, len(df) + 1))
    df.to_csv('fb_original_comments.csv', index=False, encoding='utf-8-sig')
    
    print(f"突破成功！抓取到 {len(df)} 筆不重複留言。")

except Exception as e:
    print(f"發生錯誤: {e}")
finally:
    driver.quit()