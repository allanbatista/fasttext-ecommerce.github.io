## FASTTEXT E-COMMERCE

Este é um modelo (FastText) para a transformação de palavras em vetores especializado com os dados do e-commerce nacional.

* dimensão: 128
* epochs: 50
* wordNgrams: 3
* min caracteres grams:	2
* max caracteres grams: 7
* quantidade de palavras no dicionário:	5000000
* método de treinamento: SKIP-GRAM

### Como usar

Um exemplo de como utilizar pode ser visto neste [notebook](https://github.com/allanbatista/fasttext-ecommerce.github.io/blob/gh-pages/notebooks/FastText_E_Commerce.ipynb)

### Download

* [FastText Model 7,18GB](https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/model.bin)
* [Vetores das Palavras 5,01GB](https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/model.vec)

**Corpus**

Neste corpus já foi aplicado a regra de normalização de texto que pode ser visto na próxima sessão.

* Quantidade de Textos: 68,825,382
* Tamanho total: 18,75GB

```
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000000.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000001.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000002.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000003.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000004.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000005.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000006.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000007.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000008.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000009.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000010.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000011.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000012.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000013.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000014.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000015.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000016.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000017.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000018.txt
https://storage.googleapis.com/black-magic-us-west1/fasttext/2020-07-16T00-00-00/000000000019.txt
```

### Cleaner

o modelo foi treinado usando uma função de normalização do texto à fim de simplificar o seu formato removendo caracters desnecessários.

```python
import re
 
re_html = re.compile(r"<[^>]*>")
re_especial = re.compile(r'^[\-\.\,]+|\s[\-\.\,]+|[\-\.\,]+\s|[\-\.\,]+$')
re_chars = re.compile(r'[^a-z0-9àáâãçéêíóôõúü+\-\"\s\.\,]')
re_lines = re.compile(r'\n')
re_spaces = re.compile(r'\s+')
 
tokens_norm = dict(
   zip(
       'æ,œ,á,è,ì,ò,ù,ä,ë,ï,ö,ü,ÿ,â,ê,î,ô,û,å,ø,Ø,ñ'.split(","),
       'ae,oe,a,e,i,o,u,a,e,i,o,u,y,a,e,i,o,u,a,o,O,n'.split(",")
   )
)
 
 
def _accent2latin(text):
   result = []
 
   for char in text.lower():
       newchar = tokens_norm.get(char)
 
       if newchar:
           result.append(newchar)
       else:
           result.append(char)
 
   return "".join(result)
 
 
def clear(text):
   text = str(text).lower()
   text = _accent2latin(text)
   text = re_html.sub(' ', text)
   text = re_chars.sub(' ', text)
   text = re_lines.sub(' ', text)
   text = re_especial.sub(' ', text)
   text = re_spaces.sub(' ', text)
   return text.strip()
   
def text_to_vector(text):
    return np.array([ft.get_word_vector(word) for word in clear(text).split(" ")])
 
```
