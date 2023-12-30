1. 執行Pipeline後，參數結果會放在parameters from python資料夾中，同時給定參數下的每個步驟結果皆會被存放於result per step中
2. parameters from python中的檔案會依據參數的不同而有不同的檔名，但result per step裡的檔案在每次執行完Pipeline後就會被蓋掉，因此留存的檔案會是最後一次執行時，給定參數的結果
3. 之後再到parameters from python資料夾中，選取要使用的參數結果，把值貼到Final result中去跑後續分析
4. Archive中存放的New Version (Different T)資料夾為過去各步驟拆開估計的過程；Pipeline.ipynb則為Pipeline.py的Jupyter notebook檔案