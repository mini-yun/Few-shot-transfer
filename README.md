# Few-shot-transfer

Data:   new.csv-龙井茶叶数据    multi_brand3000.csv-杂牌茶叶数据

model：baseline、Textcnn、LSTM。分为train和迁移,train之后保存模型再在迁移加载

词向量：word.vector-192维词向量  word60.vector-300维词向量

更改标签：更改load_data函数中的y值
