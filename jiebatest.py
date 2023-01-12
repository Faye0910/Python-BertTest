import jieba
from snownlp import SnowNLP
import pynlpir
print('==================jieba=========================')
text = '疫苗選擇權成了台灣民眾關切的問題，中央疫情指揮中心指揮官陳時中日前鬆口表示：正在規劃開放接種民眾自選廠牌，但需要考慮疫苗供應量的問題。政府拍胸保證，你將能夠挑選AZ、BNT' \
       '（如果郭台銘、台積電和慈濟有買到）、莫德納、高端、聯亞等各家廠牌，現階段孕婦率先受惠，但前提是「疫苗要足夠」。'

print('預設:', '|'.join(jieba.cut(text, cut_all=False, HMM=True)))

jieba.load_userdict('tt.txt')
print('-----------新增完字詞後-------------')
print('預設:', '|'.join(jieba.cut(text, cut_all=False, HMM=True)))



print('==================SnowNLP=========================')

snow_test=SnowNLP(text).words
print(snow_test)

