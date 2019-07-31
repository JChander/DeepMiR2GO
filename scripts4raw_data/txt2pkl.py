import sys
import numpy as np
import pandas as pd


def main():
	merge_emb()

def merge_emb():
	query = list()
	proteins = list()
	embeddings = list()
	
	with open('/ifs/data1/wangjiacheng/step4_data/miRNA_exists_miRInteraction.txt','r') as n:
		k = 1
		for lines in n:
			pro = lines.strip().split('\t')
			if(not pro[0]):
				print("Line %d is error"%k)
			query.append(pro[0])
			k = k + 1
		
	#gos = list()
	with open('/ifs/data1/wangjiacheng/2scripts/LINE-master/linux/global_LINE_s100_n10_d50.embeddings', 'r') as f:
		i = 1
		for line in f:
			if(i > 20731):
				break
			items = line.split(' ')
			
			mir = items.pop(0)               #LINE
			non = items.pop()
			if(mir not in query):
				continue
			emb_list = np.zeros((50,), dtype='float32')
			proteins.append(mir)
			j = 0
			for item in items:
				emb_list[j] = item
				j = j + 1
			embeddings.append(emb_list)
			i = i + 1
	df = pd.DataFrame({
        'proteins': proteins,
        'embeddings': embeddings})
	print(len(df))
	df.to_pickle('/ifs/data1/wangjiacheng/2step3/train_data/Line_miRNA_embeddings_s100_n10_50.pkl')

if __name__ == '__main__':
    main()