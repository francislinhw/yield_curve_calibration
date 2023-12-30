import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

# 以下提及「各步驟拆開」皆指先前拆開各步驟估計的過程，而本程式碼的目的在於將過去拆的步驟全部串起來
# 為了維持兩邊(此程式碼與各步驟拆開的程式碼)一致性，兩邊程式碼基本上都會是相同的
# 當時寫此程式碼就是依據各步驟拆開的程式碼，一步步整理而來，但難免會有細節上的調整，以下遇到時都有說明
# 也可以依據以下的各Step去比對各步驟拆開的程式碼

# 兩邊主要差異來自於讀入資料的部分，讀入資料主要有兩種：
# 第一種：給定原始Excel檔案
# 第二種：各步驟完成後的估計結果

# 在整理過程中有發現，把各步驟完成結果存成csv後，再讀入給後續使用時，會有浮點數的問題
# 例如：程式原先結果是0.123456789，存成csv後再讀入程式，有出現數值變成可能0.123456788的現象(僅為舉例)
# 也就是小數點前幾位值是相同的，但是後幾位後就開始變得不同，基本上差異沒有到很大
# 但可能會導致估計的結果不同，一些在Calibration要minimize error時可能會導致差異
# 且估計步驟繁瑣，單一個環節的差異在後續步驟中可能會被逐步放大
# 以本質而言，原始程式計算出來的值才是最正確的，因此除了將各步驟的估計結果存成csv外
# 同時也會在程式內部建立一個copy，此copy的命名都會在尾巴加上'_original'
# 目的在於後續步驟若需要使用到前面步驟的估計結果時，不必透過讀入csv的方式獲取資料
# 而是直接以本身就在程式內的估計結果，避免可能因為浮點數差異而導致的一些問題

# 而針對本來就給定的原始Excel檔，在各步驟中就會直接讀入。由於過去各步驟拆開是獨立的，資料都需要在一開始時讀入
# 為了維持兩邊一致性，因此此程式中可能會出現重複讀入同樣類型的資料，並不會因為前述步驟已出現過就不執行
# 但這並不會影響結果而會直接蓋掉，但蓋掉的檔案與原先的其實相同

# 辨別上述兩種類型的方法相當容易，只要是直接讀入Excel的就會是第一種，若是指定xxx_original方式的，就會是第二種

# 指定參數
k = float(input('請輸入 k: '))
theta = float(input('請輸入 theta: '))
rho = float(input('請輸入 rho: '))
Fed_fund_rate_fitted_result_used_model = input(
    '請輸入 fit Fed Fund rate時使用的model \n(可選項目：Cubic_spline, NSS, NS): '
)



# Step 1 - 透過NSS model對Fed Fund rates給定的T和Zero rates進行擬合，接著估計出指定T的Zero rates及P(0, T)
# 讀入資料
# 讀入要fit的Fed Fund rate
Fed_Fund_rate = pd.read_excel(
    'data/Fed Fund rate.xlsx'
)

# 讀入要估的Require Tenor
required_Tenor = pd.read_excel(
    'data/Required Tenor.xlsx'
)

# 依據Required Tenor估計Zero rates及P(0, T)，嘗試不同model
# Cubic Spline
CubicSpline_model = CubicSpline(x=Fed_Fund_rate.Tenor, y=Fed_Fund_rate.Zero)


# Nelson-Siegel Approach
def function_NS(t, B_0, B_1, B_2, theta_NS):
    return (
        B_0 +
        (B_1 + B_2)*((1 - np.exp(-t/theta_NS)) / (t/theta_NS)) - 
        B_2*(np.exp(-t/theta_NS))
    )

popt_NS, pcov_NS = curve_fit(
    f=function_NS,
    xdata=Fed_Fund_rate.Tenor,
    ydata=Fed_Fund_rate.Zero
)

# Nelson-Siegel-Svensson Approach
def function_NSS(t, B_0, B_1, B_2, B_3, theta_NSS, vega):
    return (
        B_0 +
        B_1 * ((1 - np.exp(-t/theta_NSS)) / (t/theta_NSS)) +
        B_2 * (((1 - np.exp(-t/theta_NSS)) / (t/theta_NSS)) - np.exp(-t/theta_NSS)) +
        B_3 * (((1 - np.exp(-t/vega)) / (t/vega)) - np.exp(-t/vega))
    )

popt_NSS, pcov_NSS = curve_fit(
    f=function_NSS,
    xdata=Fed_Fund_rate.Tenor,
    ydata=Fed_Fund_rate.Zero
)

# 當時發現Cubic Spline fit最好，再來是NS, 最後則為NSS
# 和老師討論後，老師說後續步驟使用Cubic Spline，但同時儲存NS結果備用，也將NSS的結果一併儲存
# 把上方結果放入Dataframe
Fed_fund_rate_fitted_result = pd.DataFrame({
    'Required Tenor': required_Tenor['Required Tenor'],
    'Zero_Rate_Cubic_spline': CubicSpline_model(required_Tenor['Required Tenor']),
    'Zero_Rate_NS': function_NS(required_Tenor['Required Tenor'], *popt_NS),
    'Zero_Rate_NSS': function_NSS(required_Tenor['Required Tenor'], *popt_NSS)
})

# 計算P(0, T)
Fed_fund_rate_fitted_result['Discount_Factor_Cubic_spline'] = Fed_fund_rate_fitted_result.apply(
    lambda x: np.exp(-x['Zero_Rate_Cubic_spline'] * x['Required Tenor']),
    axis=1
)

Fed_fund_rate_fitted_result['Discount_Factor_NS'] = Fed_fund_rate_fitted_result.apply(
    lambda x: np.exp(-x['Zero_Rate_NS'] * x['Required Tenor']),
    axis=1
)

Fed_fund_rate_fitted_result['Discount_Factor_NSS'] = Fed_fund_rate_fitted_result.apply(
    lambda x: np.exp(-x['Zero_Rate_NSS'] * x['Required Tenor']),
    axis=1
)

# 整理欄位順序
Fed_fund_rate_fitted_result = Fed_fund_rate_fitted_result[[
    'Required Tenor',
    'Zero_Rate_Cubic_spline', 'Discount_Factor_Cubic_spline',
    'Zero_Rate_NS', 'Discount_Factor_NS',
    'Zero_Rate_NSS', 'Discount_Factor_NSS'
]]

# 把結果存到csv
Fed_fund_rate_fitted_result.to_csv(
    'result per step/Step 1 - Fed_fund_rate_fitted_result.csv',
    index=False
)

# 同時儲存一個original的copy
Fed_fund_rate_fitted_result_original = Fed_fund_rate_fitted_result.copy()
print('Step 1 已完成！')
print('')



# Step 2 - 估計phi(T1, T2)
# 讀入資料
Fed_fund_rate_fitted_result = Fed_fund_rate_fitted_result_original

sigma_r_square_collection = pd.read_excel(
    'data/Calibration_result_from Leo-Numerix(arranged).xlsx'
)

# 「Matched_sigma_r_square_Tenor.xlsx」為估計phi(T1, T2)時，
# 當要選擇sigma_r_square時會遇到T1, T2跨區間的問題，
# 與老師討論後選擇佔比大的作為依據，因此直接先比對好，
# 之後程式就不用另外判斷，直接取對應Tenor的sigma_r_square的值即可
Matched_sigma_r_square_Tenor = pd.read_excel(
    'data/Matched_sigma_r_square_Tenor.xlsx'
)

# 整理資料
# 整理sigma_r_square_collection
# 重新命名column
sigma_r_square_collection.rename(columns={'Unnamed: 0': 'Maturity'}, inplace=True)

# 把Maturity欄位存到index，目的是之後要變成dict，方便後續步驟拿值
sigma_r_square_collection.index = sigma_r_square_collection.Maturity

# 把Maturity欄位drop掉
sigma_r_square_collection.drop(columns='Maturity', inplace=True)

# 把Matched_sigma_r_square_Tenor與Fed_fund_rate_fitted_result併在一起
# 原本想要依據Required Tenor將兩張表merge在一起，但是發現因為浮點數關係會無法串在一起
# 因此在知道兩張表的Required Tenor長度與排序一樣情況下可以直接用join接起來
Fed_fund_rate_fitted_result = Fed_fund_rate_fitted_result.join(
    Matched_sigma_r_square_Tenor['Macthed sigma r square Tenor']
)

# 指定相關變數
# 之後要從Fed_fund_rate_fitted_result選擇欄位的名稱
# 與上方Fed_fund_rate_fitted_result_used_model變數連動
select_column = 'Discount_Factor_' + Fed_fund_rate_fitted_result_used_model

# 用dict形式方便讀取值
discount_factor_dict = dict(zip(
    Fed_fund_rate_fitted_result['Required Tenor'],
    Fed_fund_rate_fitted_result[select_column]
))

# 因為會需要0，可是原始資料中沒有，因此自己加上，T=0的折現為Discount factor為1
discount_factor_dict[0] = 1

# 建立給定Tenor對應sigma_r_square值的dict
# example: {
#     '1m': 0.00623488457694546,
#     '3m': 0.00533050349235285,
#     .....
# }
sigma_r_square_series_for_match = dict(
    sigma_r_square_collection[theta]
)

# 在Fed_fund_rate_fitted_result中建立T1, T2，給for loop跑
Fed_fund_rate_fitted_result['T2'] = Fed_fund_rate_fitted_result['Required Tenor']
Fed_fund_rate_fitted_result['T1'] = Fed_fund_rate_fitted_result['T2'].shift()
Fed_fund_rate_fitted_result.T1.fillna(0, inplace=True)

# 建立給定Required_Tenor對應Tenor的dict
# example: {
#     0.011111111111111113: '1m',
#     0.08333333333333333: '1m',
#      ......
# }
sigma_r_square_matching_Tenor = dict(zip(
    Fed_fund_rate_fitted_result['T2'],
    Fed_fund_rate_fitted_result['Macthed sigma r square Tenor']
))

# 建立function接受到T後透過sigma_r_square_matching_Tenor找到對應Tenor
# 再去向sigma_r_square_series_for_match找對應的sigma_r_square
def select_sigma_r_square(T):
    return sigma_r_square_series_for_match[
        sigma_r_square_matching_Tenor[T]
    ]

# 估計phi(T1, T2)
# 注意：本處的變數phi_T與各步驟拆開的phi_T為同一個東西
# 先前拆開各步驟時沒注意到因此命名為phi_T，而此處pipline有統一更正為大T
def estimate_phi_T(T1, T2):
    # print出來方便檢查
#     print('T1: {}'.format(T1))
#     print('T2: {}'.format(T2))
#     print('P_T1: {}'.format(discount_factor_dict[T1]))
#     print('P_T2: {}'.format(discount_factor_dict[T2]))
#     print('sigma_r_square: {}'.format(select_sigma_r_square(T2)))
#     print('')
    
    return (
        (
            discount_factor_dict[T1] / discount_factor_dict[T2]
        ) *
        np.exp(
            (
                0.5 * select_sigma_r_square(T2) / (k**2)
            ) * 
            (
                (T2 - T1) -
                (2/k) * (1-np.exp(-k*(T2 - T1))) +
                (1/(2*k)) * (1-np.exp(-2*k*(T2 - T1)))
            )
        )
    )

# 執行上述function
phi_T_collection = pd.DataFrame()

for i in range(len(Fed_fund_rate_fitted_result)):
    T_1 = Fed_fund_rate_fitted_result['T1'].values[i]
    T_2 = Fed_fund_rate_fitted_result['T2'].values[i]
    current_phi_T = estimate_phi_T(T_1, T_2)
    used_sigma_r_square = select_sigma_r_square(T_2)
    
    temp_phi_T_collection = pd.DataFrame([[
        T_1, T_2, used_sigma_r_square, current_phi_T
    ]], columns=[
        'T_1', 'T_2', 'used_sigma_r_square', 'phi_T'
    ])
    
    phi_T_collection = pd.concat([phi_T_collection, temp_phi_T_collection])

# 由於實際上要的值是phi_T取完log後的值，因此最後還要取log才是對的
phi_T_collection['phi_T_after_log'] = phi_T_collection['phi_T'].apply(lambda x: np.log(x))

# 把結果存成csv
# 去除不必要的欄位
phi_T_collection.drop(columns=['T_1', 'used_sigma_r_square', 'phi_T'], inplace=True)

# 將T_2(原先的Required Tenor)命名回Required Tenor
# 將phi_T_after_log(最後要的結果)命名為phi(T1, T2)
phi_T_collection.rename(
    columns={
        'T_2': 'Required Tenor',
        'phi_T_after_log': 'phi(T1, T2)'
    },
    inplace=True
)

# 儲存結果
phi_T_collection.to_csv('result per step/Step 2 - phi(T1, T2).csv', index=False)

# 同時儲存一個original的copy
phi_T_collection_original = phi_T_collection.copy()

print('Step 2 已完成！')
print('')



# Step 3 - 估計sigma_s_square
# 讀入資料
# 「T1_T2_sigma_r_square_SOFR_rate.xlsx」為1M, 3M SOFR Futures對應的T與rates，
# 同時間也有對應該使用的sigma_r_square(此sigma_r_square延用之前老師先幫忙配好的)
T1_T2_sigma_r_square = pd.read_excel(
    'data/T1_T2_sigma_r_square_SOFR_rate.xlsx'
)

# sigma_r_square_collection
sigma_r_square_collection = pd.read_excel(
    'data/Calibration_result_from Leo-Numerix(arranged).xlsx'
)

phi_T_collection = phi_T_collection_original

# 整理資料
# 整理T1_T2_sigma_r_square
# 把SOFR futures rate由百分比轉換成小數點
T1_T2_sigma_r_square['SOFR futures rate'] = T1_T2_sigma_r_square['SOFR futures rate'] / 100

# 由於Required Tenor前段是依據SOFR future 1M, 3M的T組成
# 而小於一年的部分是採用1M的T，1年以後則採用3M
# 但讀入的資料同時含有3M小於一年的部分，所以要drop掉
# 否則phi(T1, T2)小於一年的部分的T也只能對應到1M，這樣處理到3M會有問題
# 以下找出屬於3M的且T小於1年的index然後drop掉
drop_index = T1_T2_sigma_r_square[T1_T2_sigma_r_square['Type of futures'] == 3][
    T1_T2_sigma_r_square[T1_T2_sigma_r_square['Type of futures'] == 3].T2 < 1
].index

T1_T2_sigma_r_square.drop(index=drop_index, inplace=True)

# 重新設定index
T1_T2_sigma_r_square.reset_index(inplace=True, drop=True)

# 整理sigma_r_square_collection
sigma_r_square_collection.rename(columns={'Unnamed: 0': 'Maturity'}, inplace=True)
sigma_r_square_collection.index = sigma_r_square_collection.Maturity
sigma_r_square_collection.drop(columns='Maturity', inplace=True)

# 把第一個row drop掉，因為是0~0.011111，這部分不會被用到
phi_T_collection = phi_T_collection.iloc[1:]

# phi(T1, T2)中的1.141667代表的是phi(1.016667, 1.141667)這段區間
# 這段區間剛好是1M, 3M SOFR Futures的交接期間(3M一年以前的已被drop掉)
# 因此並不會有對應的SOFR futures rate，所以這一段要drop掉
# 下面才能將phi_T_collection併回去T1_T2_sigma_r_square
# 以下做法是直接選擇Tenor不等於1.141667的row，等同於把等於1.141667的欄位drop掉
phi_T_collection = phi_T_collection[
    round(phi_T_collection['Required Tenor'], 3) != round(1.141667, 3)
]

# 然後要reset_index
phi_T_collection.reset_index(inplace=True, drop=True)

# 把phi_T_collection併回去T1_T2_sigma_r_square
# 原本想要依據Tenor用merge，但會有浮點數的問題導致無法對起來
# 所以在確認好對應位置後，直接合併起來
T1_T2_sigma_r_square = T1_T2_sigma_r_square.join(
    phi_T_collection
)

# 建立隨時間區間會有不同變數的sigma_r_square
sigma_r_square_series_for_match = dict(
    sigma_r_square_collection[theta]
)

# 估計參數
def target_function(parameters, sigma_s_square):
    T_1, T_2, sigma_r_square, phi_T = parameters
#     print('T_1: {}'.format(T_1))
#     print('T_2: {}'.format(T_2))
#     print('sigma_r_square: {}'.format(sigma_r_square))
#     print('phi_T: {}'.format(phi_T))
#     print('')
    
    return (
        np.exp(
            -theta * (
                (T_2 - T_1) - ((1 - np.exp(-k*(T_2 - T_1))) / k)
            )
        ) * np.exp(
            -phi_T
        ) * np.exp(
            0.5*(
                (sigma_s_square * ((T_2 - T_1)**3) / 3) +
                (sigma_r_square / (k**2)) * (
                    (T_2 - T_1) -
                    (1 - np.exp(-k*(T_2 - T_1))) / k -
                    (1 - 2 * np.exp(-k * (T_2 - T_1)) + np.exp(-2*k * (T_2 - T_1))) / (2*k)
                ) +
                (2 * rho * (sigma_s_square**0.5) * (sigma_r_square**0.5)) / k *(
                    ((T_2 - T_1)**2) / 2 -
                    ((1/k + (T_2 - T_1)) * (1/k) * (1 - np.exp(-k * (T_2 - T_1)))) +
                    ((1/k) * (T_2 - T_1))
                )
            )
        )
    )

estimated_sigma_s_square_collection = pd.DataFrame()

for i in range(len(T1_T2_sigma_r_square)):
    T_1 = T1_T2_sigma_r_square.iloc[i, ]['T1']
    T_2 = T1_T2_sigma_r_square.iloc[i, ]['T2']
    # 依據對應的Tenor去找sigma_r_square
    sigma_r_square_for_match = T1_T2_sigma_r_square.iloc[i, ]['sigma_r_square_for_match']
    sigma_r_square = sigma_r_square_series_for_match[sigma_r_square_for_match]
    phi_T = T1_T2_sigma_r_square.iloc[i, ]['phi(T1, T2)']

    SOFR_futures_rate = T1_T2_sigma_r_square.iloc[i, ]['SOFR futures rate']
    current_target = 1 / (1 + SOFR_futures_rate * (T_2 - T_1))
    
    popt, pcov = curve_fit(
        target_function,
        (T_1, T_2, sigma_r_square, phi_T),
        current_target
    )
    
    estimated_sigma_s_square = pd.DataFrame([popt], columns=['estimated_sigma_s_square'])
    estimated_sigma_s_square_collection = pd.concat([
        estimated_sigma_s_square_collection,
        estimated_sigma_s_square
    ])
#     print('')
#     print('=======================================')
#     print('')
    
# 把結果並回去原本資料
T1_T2_sigma_r_square['estimated_sigma_s_square'] = estimated_sigma_s_square_collection[
    'estimated_sigma_s_square'
].values

# 把結果存成csv
# 注意：各步驟拆開中原先變數名稱命名為result
# 這樣命名不太好，因為在所有步驟和在一起時並無法知道result指的是什麼，當時沒注意到
# 因此以下將result改為sigma_s_square
# 只保留要的欄位
sigma_s_square = T1_T2_sigma_r_square[['T2', 'estimated_sigma_s_square']]

# 重新命名column
sigma_s_square = sigma_s_square.rename(columns={'T2': 'Required Tenor'})

# 存結果
# 原先各步驟拆開中的檔案名稱為'Sigma s square.csv，s為大寫，以下改為小s
# 不影響結果，只是為了維持一致性
sigma_s_square.to_csv('result per step/Step 3 - sigma s square.csv', index=False)

# 同時儲存一個original的copy
sigma_s_square_original = sigma_s_square.copy()

print('Step 3 已完成！')
print('')



# Step 4 - 對sigma_r_square, sigma_s_square進行bootstrap
# 讀入資料
Fed_Fund_rate_collection = Fed_fund_rate_fitted_result_original

sigma_r_square_collection = pd.read_excel(
    'data/Calibration_result_from Leo-Numerix(arranged).xlsx'
)

phi_T_collection = phi_T_collection_original

sigma_s_square_collection = sigma_s_square_original

# T1_T2_SOFR_rate.xlsx」來自Step 3中的「T1_T2_sigma_r_square_SOFR_rate.xlsx」
# 只是把“sigma_r_square_for_match”欄位刪除
# 目的是用於讓到時bootstrap sigma_s_square時可以對應到應該對應的SOFR futures rate
T1_T2_SOFR_rate = T1_T2_sigma_r_square = pd.read_excel(
    'data/T1_T2_sigma_r_square_SOFR_rate.xlsx'
)
T1_T2_SOFR_rate = T1_T2_SOFR_rate.drop(columns='sigma_r_square_for_match')

# 整理資料
# 整理sigma_r_square_collection
sigma_r_square_collection.rename(columns={'Unnamed: 0': 'Maturity'}, inplace=True)
sigma_r_square_collection.index = sigma_r_square_collection.Maturity
sigma_r_square_collection.drop(columns='Maturity', inplace=True)
# 只會用到第一個值，因為是從0開始，其他後面的要用bootstrap算出來

# 把T1_T2_SOFR_rate和sigma_s_square_collection依據Required Tenor合併起來
# T1_T2_SOFR_rate含有3M小於1年的部分，而sigma_s_square_collection已經是整理過的Required Tenor
# 所以依據sigma_s_square_collection的Required Tenor去merge就不會有問題
# 這裡因為會有浮點數的問題，所以先建立一個column當作merge的key，直接對要merge的Tenor四捨五入到小數點後5位
# 照樣就可以確保兩邊會對的起來，不會有浮點數問題
T1_T2_SOFR_rate['key_for_merge'] = round(
    T1_T2_SOFR_rate['T2'], 5
)

sigma_s_square_collection['key_for_merge'] = round(
    sigma_s_square_collection['Required Tenor'], 5
)

sigma_s_square_collection = sigma_s_square_collection.merge(
    T1_T2_SOFR_rate,
    on='key_for_merge'
)

# 去除剛剛用來merge的key
sigma_s_square_collection.drop(columns=['key_for_merge'], inplace=True)

# 將SOFR futures rate由百分比轉成小數點
sigma_s_square_collection['SOFR futures rate'] = (
    sigma_s_square_collection['SOFR futures rate'] / 100
)

# 第一個row是0.011111其實等同於0的概念，不需計算，也無法有對應的其他估計值可以計算，所以drop掉
Fed_Fund_rate_collection = Fed_Fund_rate_collection.iloc[1:]

# 依據給定的model去選出欄位
Fed_Fund_rate_collection = Fed_Fund_rate_collection[[
    'Required Tenor',
    'Discount_Factor_{}'.format(Fed_fund_rate_fitted_result_used_model),
    'Zero_Rate_{}'.format(Fed_fund_rate_fitted_result_used_model)
]]

# 重新命名欄位，方便下面for loop讀值
Fed_Fund_rate_collection = Fed_Fund_rate_collection.rename(columns={
    'Discount_Factor_{}'.format(Fed_fund_rate_fitted_result_used_model): 'Discount factor'
})

# 重新設定index
Fed_Fund_rate_collection.reset_index(inplace=True, drop=True)

# 建立隨時間區間會有不同變數的sigma_r_square
sigma_r_square_series_for_match = dict(
    sigma_r_square_collection[theta]
)

# 給定初始值
initial_sigma_r_square = sigma_r_square_series_for_match['1m']

# 給定初始值
initial_sigma_s_square = sigma_s_square_collection.iloc[0]['estimated_sigma_s_square']

# 估計sigma_r_square與sigma_s_square時使用的方法不同，詳請可以看各步驟拆開的最底部有詳細敘述
# 對sigma_r_square進行bootstrap
bootstrapped_sigma_r_square_collection = {}
bootstrapped_sigma_r_square_collection[(1/12)] = initial_sigma_r_square

def sigma_r_square_bootstrap_function(sigma_r_square):
#     print('short_T: {}'.format(short_T))
#     print('long_T: {}'.format(long_T))
#     print('current_phi_T: {}'.format(current_phi_T))
#     print('P_short_T: {}'.format(P_short_T))
#     print('P_long_T: {}'.format(P_long_T))
#     print('sigma_r_square_short_T: {}'.format(
#         bootstrapped_sigma_r_square_collection[short_T]
#     ))
#     print('')
    
    return (
        np.exp(-current_phi_T) * (
            (
                np.exp(
                    (0.5 * sigma_r_square / (k**2)) * (
                        long_T - 
                        (2/k) * (1 - np.exp(-k * long_T)) +
                        (1/(2*k)) * (1 - np.exp(-2 * k * long_T))
                    )
                )
            ) / (
                np.exp(
                    (0.5 * bootstrapped_sigma_r_square_collection[short_T] / (k**2)) * (
                        short_T - 
                        (2/k) * (1 - np.exp(-k * short_T)) +
                        (1/(2*k)) * (1 - np.exp(-2 * k * short_T))
                    )
                )
            )
        ) * (
            P_short_T / P_long_T
        )
    ) -1

def validate_bootstrapped_sigma_r_square(bootstrapped_sigma_r_square):
    return (
        np.exp(-current_phi_T) * (
            (
                np.exp(
                    (0.5 * bootstrapped_sigma_r_square / k**2) * (
                        long_T - 
                        (2/k) * (1 - np.exp(-k * long_T)) +
                        (1/(2*k)) * (1 - np.exp(-2 * k * long_T))
                    )
                )
            ) / (
                np.exp(
                    (0.5 * bootstrapped_sigma_r_square_collection[short_T] / k**2) * (
                        short_T - 
                        (2/k) * (1 - np.exp(-k * short_T)) +
                        (1/(2*k)) * (1 - np.exp(-2 * k * short_T))
                    )
                )
            )
        ) 
    )

for i in range(len(Fed_Fund_rate_collection)-1):
    # 指定好T，之後都會依據此T去抓點對應的值
    short_T = Fed_Fund_rate_collection['Required Tenor'][i]
    long_T = Fed_Fund_rate_collection['Required Tenor'][i+1]

    # 找出對應的phi_T
    current_phi_T = phi_T_collection[
        round(phi_T_collection['Required Tenor'], 5) == round(long_T, 5)
    ]['phi(T1, T2)'].values[0]

    # 找出對應的P
    P_short_T = Fed_Fund_rate_collection[
        Fed_Fund_rate_collection['Required Tenor'] == short_T
    ]['Discount factor'].values[0]

    P_long_T = Fed_Fund_rate_collection[
        Fed_Fund_rate_collection['Required Tenor'] == long_T
    ]['Discount factor'].values[0]

    bootstrapped_sigma_r_square = fsolve(sigma_r_square_bootstrap_function, 0)

    bootstrapped_sigma_r_square_collection[long_T] = bootstrapped_sigma_r_square[0]
    
#     print('Difference: {}'.format((
#         validate_bootstrapped_sigma_r_square(bootstrapped_sigma_r_square) -
#         (P_long_T / P_short_T )
#     )[0]))
#     print('')
#     print('=========================================')
#     print('')

# 把結果存成另外一個Dataframe，因為後面估sigma_s_square還是需要用到dict
bootstrapped_sigma_r_square_result = pd.DataFrame([bootstrapped_sigma_r_square_collection]).T
bootstrapped_sigma_r_square_result.reset_index(inplace=True)
bootstrapped_sigma_r_square_result.columns = ['T', 'sigma_r_square']

# 對sigma_s_square進行bootstrap
# 由於日期只有1M的期貨對的起來，因此目前只先針對1M的去估，後面的值再用NSS估出來
bootstrapped_sigma_s_square_collection = {}
bootstrapped_sigma_s_square_collection[(1/12)] = initial_sigma_s_square

# 找出1M, 3M SOFR futures有資料的對應區間，選至5.205600年(前29個值)，且要drop掉1.141667年(index=12的地方)
Fed_Fund_rate_collection_for_sigma_s_square_estimation = Fed_Fund_rate_collection[:29].drop(
    index=12
)

# 把sigma_s_square_collection中要使用到的值合併回去Fed_Fund_rate_collection_for_sigma_s_square_estimation
# 這裡確認Fed_Fund_rate_collection_for_sigma_s_square_estimation的Tenor已經與Fed_Fund_rate_collection一致
# 所以就沒有用merge的方式了，而是直接寫入
Fed_Fund_rate_collection_for_sigma_s_square_estimation = Fed_Fund_rate_collection_for_sigma_s_square_estimation.copy()
for column_name in ['T1', 'T2', 'Type of futures', 'SOFR futures rate']:
    Fed_Fund_rate_collection_for_sigma_s_square_estimation[
        column_name
    ] = sigma_s_square_collection[column_name].values
# 最後要reset_index下方for loop才不會有問題，因為會依據index拿值  
Fed_Fund_rate_collection_for_sigma_s_square_estimation.reset_index(inplace=True, drop=True)

def B(T):
    return (1 - np.exp(-k * T)) / k

def sigma_s_square_bootstrap_function(
    X,
    sigma_s_square
):
    short_T, long_T, current_phi_T, current_SORF_rate = X
#     print('short_T: {}'.format(short_T))
#     print('long_T: {}'.format(long_T))
#     print('current_phi_T: {}'.format(current_phi_T))
#     print('current_SORF_rate: {}'.format(current_SORF_rate))
#     print('sigma_s_square_short_T: {}'.format(
#         bootstrapped_sigma_s_square_collection[short_T]
#     ))
#     print('')
    
    return (
        np.exp(
            -current_phi_T
        ) *
        np.exp(
            -theta * (
                (long_T - short_T) + (np.exp(-k * long_T) - np.exp(-k * short_T))/k
            )
        ) * np.exp(
            0.5*(
                (
                    (sigma_s_square * (long_T ** 3) / 3) - 
                    (bootstrapped_sigma_s_square_collection[short_T] * (short_T ** 3) / 3)
                ) +
                (
                   (bootstrapped_sigma_r_square_collection[long_T] / (k**2)) * (
                       long_T - B(long_T) - (k * (B(long_T)**2) / 2)
                   ) -
                   (bootstrapped_sigma_r_square_collection[short_T] / (k**2)) * (
                       short_T - B(short_T) - (k * (B(short_T)**2) / 2)
                   )
                ) +
                (
                    (2 * rho / k) * (
                        (sigma_s_square ** 0.5) * 
                        (bootstrapped_sigma_r_square_collection[long_T] ** 0.5) *
                        (
                            ((long_T ** 2) / 2) -
                            (1/k + long_T) * B(long_T) +
                            (long_T / k)
                        ) - 
                        (bootstrapped_sigma_s_square_collection[short_T] ** 0.5) * 
                        (bootstrapped_sigma_r_square_collection[short_T] ** 0.5) *
                        (
                            ((short_T ** 2) / 2) -
                            (1/k + short_T) * B(short_T) +
                            (short_T / k)
                        )
                    )
                )
            )
        ) * (
            1 + current_SORF_rate * (long_T - short_T)
        )
    ) - 1

def validate_bootstrapped_sigma_s_square(bootstrapped_sigma_s_square):
    return (
        np.exp(
            -current_phi_T
        ) *
        np.exp(
            -theta * (
                (long_T - short_T) + (np.exp(-k * long_T) - np.exp(-k * short_T))/k
            )
        ) * np.exp(
            0.5*(
                (
                    (bootstrapped_sigma_s_square * (long_T ** 3) / 3) - 
                    (bootstrapped_sigma_s_square_collection[short_T] * (short_T ** 3) / 3)
                ) +
                (
                   (bootstrapped_sigma_r_square_collection[long_T] / (k**2)) * (
                       long_T - B(long_T) - (k * (B(long_T)**2) / 2)
                   ) -
                   (bootstrapped_sigma_r_square_collection[short_T] / (k**2)) * (
                       short_T - B(short_T) - (k * (B(short_T)**2) / 2)
                   )
                ) +
                (
                    (2 * rho / k) * (
                        (bootstrapped_sigma_s_square ** 0.5) * 
                        (bootstrapped_sigma_r_square_collection[long_T] ** 0.5) *
                        (
                            (long_T ** 2) / 2 -
                            (1/k + long_T) * B(long_T) +
                            (long_T / k)
                        ) - 
                        (bootstrapped_sigma_s_square_collection[short_T] ** 0.5) * 
                        (bootstrapped_sigma_r_square_collection[short_T] ** 0.5) *
                        (
                            (short_T ** 2) / 2 -
                            (1/k + short_T) * B(short_T) +
                            (short_T / k)
                        )
                    )
                )
            )
        )
    )

for i in range(len(Fed_Fund_rate_collection_for_sigma_s_square_estimation)-1):
    # 指定好T，之後都會依據此T去抓點對應的值
    short_T = Fed_Fund_rate_collection_for_sigma_s_square_estimation['Required Tenor'][i]
    long_T = Fed_Fund_rate_collection_for_sigma_s_square_estimation['Required Tenor'][i+1]

    # 找出對應的phi_T
    current_phi_T = phi_T_collection[
        round(phi_T_collection['Required Tenor'], 3) == round(long_T, 3)
    ]['phi(T1, T2)'].values[0]

    # 指定對應的k star
    current_SORF_rate = Fed_Fund_rate_collection_for_sigma_s_square_estimation[
        Fed_Fund_rate_collection_for_sigma_s_square_estimation['Required Tenor'] == long_T
    ]['SOFR futures rate'].values[0]
    
    
    popt, pcov = curve_fit(
        sigma_s_square_bootstrap_function,
        [short_T, long_T, current_phi_T, current_SORF_rate],
        0,
        maxfev=500000
    )

    bootstrapped_sigma_s_square_collection[long_T] = popt[0]
    
#     print('Difference: {}'.format((
#         validate_bootstrapped_sigma_s_square(popt[0]) -
#         (1 / (1 + current_SORF_rate * (long_T - short_T)))
#     )))
    
#     print('')
#     print('=========================================')
#     print('')

# 把結果存成另外一個Dataframe，因為後面估sigma_s_square還是需要用到dict
bootstrapped_sigma_s_square_result = pd.DataFrame([bootstrapped_sigma_s_square_collection]).T
bootstrapped_sigma_s_square_result.reset_index(inplace=True)
bootstrapped_sigma_s_square_result.columns = ['T', 'sigma_s_square']

# 之前有嘗試將sigma_s_square透過NSS把五年後的值也fit出來
# 但是當時有發現NSS出來的sigma_s_square會有小於0的情況
# 帶入R(0, T)裡面會有sigma_s_square需要開根號的時候就會出現na
# 和老師討論之後，目前不透過NSS把sigma_s_square後五年的值fit出來，到時會改成自己帶入
# (且之前也有發現用fit出來的sigma_s_square會有讓R(0, T)發散的問題)

# 先把所有所需的參數都串起來
# 先將Fed_Fund_rate, phi_T, bootstrapped_sigma_r_square,都建立一個key_for_merge避免浮點數的問題
Fed_Fund_rate_collection['key_for_merge'] = round(
    Fed_Fund_rate_collection['Required Tenor'], 5
)

# 積分值需要衡量0~T_2的區間，先前都是一段一段的，因此這裡要累加才會是0~T_2的積分值
# 這裡針對phi(T1, T2)先進行cumsum，因為之後merge完一開始0.0111年的部分會被drop掉
# 如果在此步驟之後才cumsum會有問題，會漏掉0.111年的結果，因此要先在merge前就cumsum
phi_T_collection['key_for_merge'] = round(
    phi_T_collection['Required Tenor'], 5
)
phi_T_collection['phi(T1, T2)'] = phi_T_collection['phi(T1, T2)'].cumsum()

bootstrapped_sigma_r_square_result['key_for_merge'] = round(
    bootstrapped_sigma_r_square_result['T'], 5
)

# 接著以Fed_Fund_rate為主將其他表串起來
# 雖然之前在Tenor: 1.141667會有需要drop掉的問題，但那是在估計sigma_s_square時才會有的問題(因為沒有對應的k*)
# 且之前估計sigma_s_square都沒有直接將原始Fed_Fund_rate的Tenor: 1.141667drop掉
# 而是另外建立一個複製的Fed_Fund_rate才drop掉，因此Fed_Fund_rate原始檔案還是有Tenor: 1.141667

# 先看一下原始Fed_Fund_rate有多少筆資料，用以判斷merge後是否有掉資料問題(浮點數問題)
# print(len(Fed_Fund_rate_collection))

# 合併三個資料，並將其指定為parameters_collection
parameters_collection = Fed_Fund_rate_collection.merge(
    phi_T_collection, on='key_for_merge'
).merge(
    bootstrapped_sigma_r_square_result, on='key_for_merge'
)

# 檢查merge後的資料筆數，確認沒有掉資料，與merge前相同
# print(len(parameters_collection))

# 看一下合併後的結果
# Required Tenor_x, Required Tenor_y，後面出現x, y代表在merge時資料本身同時都有Required Tenor此欄位
# 而merge是以key_for_megre為key進行，所以其他重複欄位名稱會自動被加上_x, _y，底下會再把重複欄位drop掉
# 到了這一步驟才將sigma_s_square merge進來，原本要透過NSS model直接取值，但現在直接用bootstrap的結果
bootstrapped_sigma_s_square_result['key_for_merge'] = round(
    bootstrapped_sigma_s_square_result['T'], 5
)

# 併入上述parameters_collection，這裡改採用outer的方式，之前預設都是inner
# 主要差異在於之前的表格Tenor都是完整的所以對的起來，用inner也不會有掉值的問題
# (只有phi(T1, T2)中會有0.011111的Tenor會自動被drop掉，但因為本來就不需要此Tenor所以不影響)
# 而sigma_s_square由於只有前5年的值，所以要用outer才不會使其他表格資料被drop掉
parameters_collection = parameters_collection.merge(
    bootstrapped_sigma_s_square_result,
    on='key_for_merge',
    how='outer'
)

# drop重複的欄位
parameters_collection.drop(
    columns=['key_for_merge', 'Required Tenor_y', 'T_x', 'T_y'], inplace=True
)

# 重新命名
parameters_collection.rename(columns={'Required Tenor_x': 'Required Tenor'}, inplace=True)

# 先建立空值等等裝結果
# 注意：各步驟拆開中到這ㄧ步驟是先建立空的欄位要裝R(0, T)
# 但此步驟其實Excel也可以計算，所以可以當成驗證的欄位
# 之後可以看看Excel算的和python是否有差異
parameters_collection['R_T'] = None

def R_T(T, sigma_r_square, sigma_s_square, phi_T):
#     print('T: {}'.format(T))
#     print('sigma_r_square: {}'.format(sigma_r_square))
#     print('sigma_s_square: {}'.format(sigma_s_square))
#     print('phi_T: {}'.format(phi_T))
    
    return (1/T) * (
        (theta * (T - B(T))) +
        phi_T -
        0.5 * (
            (sigma_s_square * (T**3) / 3) +
            (
                (sigma_r_square / (k**2)) *
                (T - B(T) - (k/2) * (B(T)**2))
            ) +
            (2 * rho * (sigma_s_square ** 0.5) * (sigma_r_square ** 0.5) / k) * (
                (T**2) / 2 -
                ((1/k) + T) * B(T) +
                (T/k)
            )
        )
    )

for i in range(len(parameters_collection)):
    T = parameters_collection.iloc[i]['Required Tenor']
    sigma_r_square = parameters_collection.iloc[i]['sigma_r_square']
    sigma_s_square = parameters_collection.iloc[i]['sigma_s_square']
    phi_T = parameters_collection.iloc[i]['phi(T1, T2)']
    
    current_R_T = R_T(
        T, sigma_r_square, sigma_s_square, phi_T
    )
    
    parameters_collection.loc[i, 'R_T'] = current_R_T 
#     print('R_T: {}'.format(current_R_T ))
    
    
#     print('')
#     print('=========================================')
#     print('')

# 避免混淆針對以上幾個欄位重新命名
# phi(T1, T2)已經經過累積加總所以會是phi(0, T)
# sigma_r_square, sigma_s_square是bootstrapped結果，所以前面加上bootstrapped
# 同時儲存結果，這樣就不會改到原先檔案欄位，以下code就不必更改
parameters_collection.rename(
    columns={
        'phi(T1, T2)': 'phi(0, T)',
        'sigma_r_square': 'bootstrapped_sigma_r_square',
        'sigma_s_square': 'bootstrapped_sigma_s_square'
    }
).to_csv(
    'result per step/Step 4 - bootstrapped sigma_r_square & sigma_s_square.csv',
    index=False
)

print('Step 4 已完成！')
print('')



# Final Step - 把最終要送入Excel的結果整理好
# 把parameters_collection的R_T drop掉
# 原先想說可以當成驗證功能去看Excel算的和python算的是否一致
# 但3M的sigma s square可能會改成用自己指定，所以值就會當然會不同了，便失去比較意義，所以先不包含此欄位
# 但是之後如果需要驗證還是可以從each step的資料夾去找來比對
final_result = parameters_collection.drop(columns=['R_T'])

name_of_paramaters = 'k={}, theta={}, rho={}, Fed_fund_rate_fitting_used_model={}'.format(
    k, theta, rho, Fed_fund_rate_fitted_result_used_model
)

# 建立給定參數的dataframe，將要併入final_result中
# 因為之後Excel在R(0, T)的sheet中會需要顯示給定參數為何，因此多增加此欄位資訊
given_parameters = pd.DataFrame(
    [
        ['k', 'theta', 'rho', 'ed_fund_rate_fitting_used_model', None, None],
        [k, theta, rho, Fed_fund_rate_fitted_result_used_model, None, None]
    ],
    columns=final_result.columns
)

final_result = pd.concat([final_result, given_parameters])

# 把結果存成csv
final_result.to_csv(
    'parameters from python/{}.csv'.format(
        name_of_paramaters
    ),
    index=False
)

print('Final step 已完成！')
print('')



print('給定參數的估計結果已完成')
print('當前參數組合為：{}'.format(name_of_paramaters))