目前sigma_s_square會用到1M, 3M的futures，因此會bootstrap到5年內的資料，後續再透過NSS curve去fit出一條for sigma_s_square的線，接著當要計算R(0, T)時，再把對應的T值帶入for sigma_s_square的NSS curve找出sigma_s_square值帶入R(0, T)


問題在於，原先估計出的sigma_s_square都是正值，但是當我將對應的T帶入for sigma_s_square的NSS curve時，得到的部分sigma_s_square卻會變成負值，而這在帶入R(0, T)會有問題，因為R(0, T)裡面的公式會有sigma_s_square的開根號，負值開根號他會是nan，因此無法計算對應的R(0, T)


先前和老師討論後，可能5年之後的sigma_s_square會採用0或是第五年可估計出的最後一個值，因此我在想是不是不需要再把sigma_s_saure透過NSS curve fit一條線後再帶入，而是直接帶入原始bootstrap出的sigma_s_square即可呢？因為5年後如果sigma_s_square要使用0或是第五年可估計出的最後一個值，其實NSS for sigma_s_square就沒派上用場了，反而是在平滑前五年sigma_s_square值的感覺

但如果直接使用原始bootstrap出的sigma_s_square的話，可能對於5年後要使用0或是第五年可估計出的最後一個值就會需要解釋，因為我們是已經發現for sigma_s_square的NSS curve會導致R(0, T)有發散問題所以才採取替代方案，但若使用NSS for sigma_s_square的方法就會有sigma_s_square為負值的問題，
且目前使用1, 3M去估的sigma_s_square再套入NSS後，其5年後的值會發散更嚴重，如此帶入R(0, Ｔ)應該也會有更發散的問題




和老師討論之後，目前不透過NSS把sigma_s_square後五年的值fit出來，到時會改成自己帶入
(且之前也有發現用fit出來的sigma_s_square會有讓R(0, T)發散的問題)。