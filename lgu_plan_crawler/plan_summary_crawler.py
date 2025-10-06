# LG U+ 모바일 요금제 요약 크롤러
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# 최종적으로 크롤링한 데이터를 저장할 리스트
plan_data_list = []

print("🚀 크롤링을 시작합니다...")

try:
    # 1. Selenium 설정
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    URL = "https://www.lguplus.com/mobile/plan/mplan/plan-all"
    driver.get(URL)
    time.sleep(3)

    # 2. '요금제 더보기' 버튼 클릭
    print("📜 '요금제 더보기' 버튼을 클릭하여 모든 요금제를 로드합니다...")
    while True:
        try:
            load_more_button = driver.find_element(By.CSS_SELECTOR, "button.btn-more")
            driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
            time.sleep(0.5)
            load_more_button.click()
            print("... '더보기' 클릭")
            time.sleep(1)
        except Exception:
            print("✅ '더보기' 버튼을 모두 클릭했거나, 더 이상 없습니다.")
            break

    # 3. 모든 정보가 로드된 최종 페이지의 HTML 소스를 가져옵니다.
    print("📄 모든 요금제가 로드된 페이지의 HTML을 가져옵니다.")
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # 4. BeautifulSoup으로 각 요금제 정보 파싱하기
    plan_cards = soup.select("ul.plan-list > li")

    print(f"📊 총 {len(plan_cards)}개의 요금제를 찾았습니다. 개별 정보 추출을 시작합니다.")

    for i, card in enumerate(plan_cards):
        plan_name, price, data_summary = None, None, None

        try:
            # ✅ [수정 완료] 이름, 가격, 데이터를 각각 추출하고 실패 시 None으로 처리

            # 요금제 이름 찾기
            name_tag = card.select_one("button.btn-plan")
            if name_tag:
                plan_name = name_tag.text.strip()

            # 월정액 찾기
            price_tag = card.select_one("div.plan-price strong")
            if price_tag:
                price_text = price_tag.text.strip()
                price = int(price_text.replace(',', ''))

            # 데이터 요약 정보 찾기
            summary_tags = card.select("p.plan-info, dl.benefit-area dd")
            if summary_tags:
                summary_texts = [tag.text.strip().replace('\n', ' ') for tag in summary_tags]
                data_summary = ' / '.join(summary_texts)

            # 필수 정보인 이름과 가격이 모두 있을 때만 리스트에 추가
            if plan_name and price is not None:
                plan_info = {
                    'plan_name': plan_name,
                    'monthly_price': price,
                    'data_summary': data_summary or "정보 없음",  # 데이터 정보가 없는 경우 대비
                }
                plan_data_list.append(plan_info)
            else:
                print(f"⚠️ {i + 1}번째 카드에서 필수 정보(이름 또는 가격)를 찾지 못해 건너뜁니다.")

        except Exception as e:
            # 예상치 못한 오류 발생 시 로그 출력
            print(f"🚨 {i + 1}번째 카드 처리 중 예상치 못한 오류 발생: {e}")
            continue

finally:
    # 5. 작업이 끝나면 브라우저를 닫습니다.
    if 'driver' in locals() and driver:
        driver.quit()
        print("🚪 브라우저를 닫았습니다.")

# 6. Pandas DataFrame으로 변환 후 CSV 파일로 저장
if plan_data_list:
    df = pd.DataFrame(plan_data_list)
    df_sorted = df.sort_values(by='monthly_price').reset_index(drop=True)

    file_name = 'lgu_all_plans_final.csv'
    df_sorted.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"\n✅ 총 {len(df_sorted)}개의 요금제 정보를 '{file_name}' 파일로 성공적으로 저장했습니다.")
    print("\n---------- 미리보기 ----------")
    print(df_sorted.head())
    print("--------------------------")
else:
    print("\n❌ 크롤링된 데이터가 없습니다. 사이트 구조가 변경되었거나 선택자가 여전히 일치하지 않을 수 있습니다.")