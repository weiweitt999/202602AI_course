import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options=Options
driver=webdriver.Chrome(options=options)

url=""
driver.get(url)
wait=WebDriverWait(driver,15)

for _ in range(20):
    driver.find_element(BY.TAG_NAME,"BODY").send_keys(keys.END)
    time.sleep(3)

comments=wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text")))
commentslist=[]
for index,el in enumeratr(comments):
    text=el.text.strip()
    if test:
        commentslist.append({
            "ID":index+1
            "txet":text
        })

df=pd.DataFrame(comments_list)
df.to_csv("")