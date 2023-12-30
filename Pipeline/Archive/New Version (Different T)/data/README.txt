1. 「Fed Fund rate.xlsx」的資料由「Project Final Representation-20200429.xlsx」的Main Sheet裡column E, H而來。

2. 「Required Tenor.xlsx」的資料由「Project Final Representation-20200429.xlsx」的Main Sheet裡column J而來。

3. 「Fed_fund_rate_fitted_result.csv」為Step 1的估計結果，「Project Final Representation-20200429.xlsx」的Main Sheet裡column K, L來自此估計結果，可以選擇要用哪一個model的估計值，目前預設使用Cubic Spline。

4. 「phi(T1, T2).xlsx」為Step 2的估計結果，「Project Final Representation-20200429.xlsx」的Main Sheet裡column M來自此估計結果，估計phi(T1, T2)與以下三項變數有關：
   a. 給定k值
   b. theta(與選擇的sigma_r_square有關)
   c. 先前fit Fed Fund rate使用的model(目前預設為Cubic Spline)
以此其實之後的Main Sheet表格並不能任意換k或theta，因為對應的phi(T1, T2)結果都會變，若要動則要全部連動重估

5. 「Sigma s square.csv」為Step 3的估計結果，「Project Final Representation-20200429.xlsx」的Main Sheet裡column N來自此估計結果，column N中Tenor 0.0111沒有值是因為SOFR futures 1M並沒有(0, 0.0111)此區間的報價因此無法估計；而Tenor 1.1417為(1.016667, 1.141667)此區間，此區間為1M, 3M SOFR futures的切換點，沒有此區間的報價因此無法估計；而Tenor 6.0219之後則是因為3M futures只到前5年所以沒有報價無法估計。

6. 「bootstrapped sigma_r_square & sigma_s_square.csv」為Step 4的估計結果，「Project Final Representation-20200429.xlsx」的Main Sheet裡column O, P來自此估計結果，column P中Tenor 0.0111沒有值是因為SOFR futures 1M並沒有(0, 0.0111)此區間的報價因此無法估計；而Tenor 1.1417為(1.016667, 1.141667)此區間，此區間為1M, 3M SOFR futures的切換點，沒有此區間的報價因此無法估計；而Tenor 6.0219之後則是因為3M futures只到前5年所以沒有報價無法估計。



註：
1. 「Project Final Representation-20200429-原始資料對照用.xlsx」為原始資料，目的在於看哪些columns原先是空白的，便於比對之後新加入了哪些。

2. 「透過NSS model fit R(0, T)」資料夾中「NSS fitted result from Excel.csv」是「Project Final Representation-20200429.xlsx」裡sheet-R(0, T) with NSS的結果，給定使用1M期貨bootstrapped出的sigma_s_square，1M之後對應Tenor使用的bootstrapped sigma_s_square則給0；「透過NSS model fit R(0, T).ipynb」則是將上述結果透過python fit NSS模型，目的在於與Excel比較NSS模型fit的好壞，有將python估計的參數與結果放在sheet-R(0, T) with NSS中，發現python fit的結果與Excel差不多(以MSE而言)，但參數卻有所差異；另外，有發現Excel給不通初始值對於fit出的結果會有影響。