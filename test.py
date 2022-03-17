"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/3/14
@description: null
"""
from pprint import pformat

for a, b, c, d in zip(list(range(10)), list(range(10)), list(range(10)), list(range(10, 20))):
    print(a, b, c, d)
text = {'avg_answers_count': 4.6535449089614485, 'queries_count': 29493, 'test': {'avg_answers_count': 4.904689248296206, 'queries_count': 9831}, 'train': {'avg_answers_count': 4.403214322042518, 'queries_count': 9831}, 'valid': {'avg_answers_count': 4.652731156545621, 'queries_count': 9831}}
print(pformat(text, indent=2, depth=4))
