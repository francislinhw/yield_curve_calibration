1. 估計phi(T1, T2)與以下三項變數有關：
   A. 給定k值
   b. theta(與選擇的sigma_r_square有關)
   c. 先前fit Fed Fund rate使用的model(目前預設為Cubic Spline)
以此其實之後的Main Sheet表格並不能任意換k或theta，因為對應的phi(T1, T2)結果都會變，若要動則要全部連動重估

2. 「Matched_sigma_r_square_Tenor.xlsx」為估計phi(T1, T2)時當要選擇sigma_r_square會遇到T1, T2跨區間的問題，與老師討論後選擇佔比大的作為依據，因此直接先比對好，之後程式就不用另外判斷，直接取對應Tenor的sigma_r_square的值即可