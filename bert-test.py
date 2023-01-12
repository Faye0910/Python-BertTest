from bert_serving.client import BertClient
import numpy as np

def cos_similar(sen_a_vec, sen_b_vec):
    '''
    計算兩個句子的余弦相似度
    '''
    vector_a = np.mat(sen_a_vec)
    vector_b = np.mat(sen_b_vec)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos

def main():
    bc = BertClient()
    doc_vecs = bc.encode(['今天天空很藍，陽光明媚', '今天天氣好晴朗', '現在天氣如何', '自然語言處理', '機器學習任務'])

    print(doc_vecs)
    similarity=cos_similar(doc_vecs[0],doc_vecs[4])
    print(similarity,'??')

if __name__ == '__main__':
    main()