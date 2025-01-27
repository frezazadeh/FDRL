import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Chrome("C:/Users/nle5230/Downloads/chromedriver_win32/chromedriver.exe")
driver.get("https://5g-tools.com/5g-nr-tbs-transport-block-size-calculator/")
tbs_values = []

content = driver.page_source
soup = BeautifulSoup(content, features="html.parser")

# Reset MCS Slider
slider = driver.find_element("xpath",  '//*[@id="fieldname48_1_slider"]')
# slider = driver.find_element("xpath",  '/html/body/div[1]/div/main/article/div/div[2]/form/div[1]/div/div[2]/div/div[3]/div[1]/div[1]/span')

move = ActionChains(driver)
move.click_and_hold(slider).move_by_offset(-500, 0).release().perform()
time.sleep(1)

# Initialize empty matrix
MAX_MCS = 32
MAX_PRBS = 273
tbs_mcs_matrix = np.zeros((MAX_MCS ,MAX_PRBS))
df = pd.DataFrame(tbs_mcs_matrix)

# Populate the matrix
for mcs in range(14, MAX_MCS):
    print("mcs " + str(mcs))
    move = ActionChains(driver)
    move.click_and_hold(slider).move_by_offset(-241 + 15*mcs, 0).release().perform() ##Scroll 1 element right over the slider
    time.sleep(1)
    MCS_value = driver.find_element("xpath", '//*[@id="fieldname48_1_caption"]').text   #Collect mcs value
    print(MCS_value)

    manual_offset = 0
    if mcs != int(MCS_value):
        while mcs != int(MCS_value):
            move.click_and_hold(slider).move_by_offset(-240 + 15 * mcs + manual_offset,  0).release().perform()  ##Scroll 1 element right over the slider
            time.sleep(1)
            MCS_value = driver.find_element("xpath", '//*[@id="fieldname48_1_caption"]').text   #Collect mcs value
            manual_offset = manual_offset + 1

    for prb in range(0, MAX_PRBS):
        inputElement = driver.find_element("xpath", '//*[@id="fieldname35_1"]')
        inputElement.clear()
        inputElement.send_keys(str(prb))   #Set PRB value
        inputElement.send_keys(Keys.ENTER)
        time.sleep(0.5)

        # TBS_value = driver.find_element("xpath", ' //*[@id="fieldname54_1"]').text
        TBS_value = driver.find_element("xpath",      '/html/body/div[1]/div/main/article/div/div[2]/form/div[1]/div/div[2]/div/div[41]/div/div[1]/span[2]').text
        df.iloc[mcs, prb] = TBS_value #Collect TBS value

        print("     " + str(TBS_value))

    df.to_csv('tbs_mcs.csv', index=True)

print("Done")
