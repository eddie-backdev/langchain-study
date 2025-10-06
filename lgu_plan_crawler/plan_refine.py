# 수집된 LG U+ 모바일 요금제 정제 기능

import pandas as pd
import re


def refine_plan_data(input_file='lgu_all_plans_final.csv', output_file='lgu_plans_refined.csv'):
    """
    크롤링된 LG U+ 요금제 CSV 파일을 읽어와 데이터를 정제하고 강화합니다.
    """
    print(f"🔄 '{input_file}' 파일을 불러와 데이터 정제를 시작합니다.")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ 파일 '{input_file}'을 찾을 수 없습니다. 파일이 현재 폴더에 있는지 확인해주세요.")
        return

    # 1. 새로운 컬럼들을 미리 생성하고 기본값으로 초기화합니다.
    df['data_gb'] = 0
    df['data_type'] = '기본제공'
    df['data_speed_limit'] = '제한없음'
    df['sharing_data'] = '제공안함'
    df['voice_call'] = '기본제공'
    df['sms'] = '기본제공'
    df['tags'] = ''

    # 2. 각 요금제(행)를 순회하며 data_summary 컬럼을 분석하고 정보 추출
    for index, row in df.iterrows():
        summary = row['data_summary']
        plan_name = row['plan_name']

        # --- 데이터 제공량 (GB) 추출 ---
        # "데이터 무제한" 처리
        if '데이터 무제한' in summary:
            df.loc[index, 'data_gb'] = 9999  # 무제한을 나타내는 임의의 큰 숫자
            df.loc[index, 'data_type'] = '무제한'
        else:
            # 정규표현식을 사용하여 '숫자GB' 또는 '숫자MB' 패턴을 찾습니다.
            gb_match = re.search(r'데이터 (\d+\.?\d*)\s?GB', summary)
            mb_match = re.search(r'데이터 (\d+\.?\d*)\s?MB', summary)
            if gb_match:
                df.loc[index, 'data_gb'] = float(gb_match.group(1))
            elif mb_match:
                df.loc[index, 'data_gb'] = float(mb_match.group(1)) / 1024  # MB를 GB로 변환

        # --- 소진 후 속도 제한(Mbps/Kbps) 추출 ---
        speed_match = re.search(r'최대 (\d+)\s?(Mbps|Kbps)', summary)
        if speed_match:
            df.loc[index, 'data_speed_limit'] = f"{speed_match.group(1)}{speed_match.group(2)}"
            df.loc[index, 'data_type'] = '기본제공후속도제어'

        # --- 공유 데이터(테더링/쉐어링) 추출 ---
        sharing_match = re.search(r'(테더링|쉐어링)\s?\+?\s?(\d+GB)', summary)
        if sharing_match:
            df.loc[index, 'sharing_data'] = sharing_match.group(2)

        # --- 음성통화 정보 추출 ---
        if '집/이동전화 무제한' in summary:
            df.loc[index, 'voice_call'] = '무제한'
        else:
            voice_match = re.search(r'(\d+)분', summary)
            if voice_match:
                df.loc[index, 'voice_call'] = f"{voice_match.group(1)}분"

        # --- 문자 정보 추출 ---
        if '기본제공' in summary:
            df.loc[index, 'sms'] = '기본제공'
        else:
            sms_match = re.search(r'(\d+)건', summary)
            if sms_match:
                df.loc[index, 'sms'] = f"{sms_match.group(1)}건"

        # 3. 추출된 데이터를 기반으로 태그(Tags) 생성
        tags = []
        if '5G' in plan_name:
            tags.append('5G')
        if 'LTE' in plan_name:
            tags.append('LTE')
        if '청소년' in plan_name or '키즈' in plan_name or '주니어' in plan_name:
            tags.append('청소년/키즈')
        if '시니어' in plan_name:
            tags.append('시니어')

        if df.loc[index, 'data_type'] == '무제한':
            tags.append('데이터무제한')
        elif df.loc[index, 'data_gb'] >= 100:
            tags.append('데이터많이')

        if row['monthly_price'] <= 40000:
            tags.append('알뜰/가성비')

        df.loc[index, 'tags'] = ','.join(tags)

    # 4. 원본 data_summary 컬럼은 삭제하고 필요한 컬럼만 선택
    df_refined = df[
        ['plan_name', 'monthly_price', 'data_gb', 'data_type', 'data_speed_limit', 'sharing_data', 'voice_call', 'sms',
         'tags']]

    # 5. 정제된 데이터를 새로운 CSV 파일로 저장
    df_refined.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 데이터 정제 완료! '{output_file}' 파일로 저장했습니다.")
    print("\n---------- 정제된 데이터 미리보기 ----------")
    print(df_refined.head())
    print("------------------------------------------")


# --- 메인 코드 실행 ---
if __name__ == "__main__":
    refine_plan_data()