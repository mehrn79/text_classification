---
jupyter:
  colab:
    name: Copy of generating_datase.ipynb
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

# **ساختن فایل csv**

## **نصب کتابخانه هضم**
``` {.python}
pip install hazm
```

## **وارد کردن کتابخانه های مورد نیاز**
``` {.python}
from hazm import *
import string
from string import digits
```
## **آماده کردن دیتا**
### **دانلود دیتا همشهری به صورت مستقیم**

``` {.python}
! gdown --id 1D3yt99D0GcCRCbdKbUQGxbqjkeh91hTg
```

### **unrar folder**

``` {.python}
!pip install unrar
```

``` {.python}
!unrar x /content/hamshahri.rar
```

### فایل متن ها و stop words را از حالت فشرده خارج می کنیم

``` {.python}
!unzip /content/hamshahriold/Corpus/Hamshahri-Corpus.zip -d /content/corpus
!unzip /content/hamshahriold/Corpus/PersianStopWords.zip -d /content/corpus/stopWords
```

## **عملیات پیش پردازش بر روی دیتا**
### در این قسمت دو فایل خبرها و stopwords را خوانده و بر اساس DID جدا سازی می کنیم و آنها را در یک آرایه قرار می دهیم زیرا هر خبر دارای یک DID می باشد که در ابتدای آن قرار دارد بنابراین می توانیم اخبار را توسط این عنوان از هم جدا کنیم.
``` {.python}
with open('/content/corpus/Hamshahri-Corpus.txt') as f:
  data = f.read()

with open('/content/corpus/stopWords/PersianStopWords.txt') as file:
  stopLines = file.readlines()
  stopWord = [item.replace('\n',"") for item in stopLines]

data=data.split('.DID')
data.pop(0)
```
### این تابع کلمات stop words را از متن جدا می کند
``` {.python}
def cleaning(sentance) :
  words= []
  cleanSent=[]
  stemmer = Stemmer()

  words=word_tokenize(sentance)

  for wordSent in words :
    if wordSent not in stopWord :
      stermmering=stemmer.stem(wordSent)
      cleanSent.append(stermmering)
  
  sent =' '.join(map(str, cleanSent))

  return sent
```
### این تابع اعداد را از متن جدا می کند
``` {.python}
def remove_digits(sent):
  remove_digits = str.maketrans('', '', digits)
  sentance =' '.join([w.translate(remove_digits) for w in sent.split(' ')]) 
  return sentance
```

###  در این قسمت ستون های جدول را می سازیم و در ابتدا برای هر ستون یک آرایه در نظر می گیریم و هر خبر را بر اساس DID,Date,Cat,text,clean_text تقسیم می کنیم. 

``` {.python}
DID=[]
Date=[]
Cat=[]
Text=[]
clean_text=[]
for item in data :
  text=item.split('\n')
  text[3]=' '.join(text[3:])
  DID.append(text[0].replace('\t',''))
  Date.append(text[1].replace('.Date\t7',''))
  Cat.append(text[2].replace('.Cat\t',''))
  Text.append(text[3])

  total_text=[]
  for elm in text[4:]:
    sent=cleaning(remove_digits(elm))
    total_text.append(sent.strip())
  clean_text.append(' '.join(total_text))
  
```

## **ذخیره اصلاعات در فایل فرمت csv**

``` {.python}
import pandas as pd

chart=pd.DataFrame(columns=['DID','Date','Cat','text','clean_text'])

chart['DID']=DID
chart['Date']=Date
chart['Cat']=Cat
chart['text']=Text
chart['clean_text']=clean_text
  
chart.to_csv('chart.csv', index=False)

``` {.python}
chart_data = pd.read_csv('chart.csv')
chart_data.head()
```

## **دخیره فایل ساخته شده در google drive**

``` {.python}
from google.colab import drive
drive.mount('/content/drive')

!cp -r "/content/chart.csv" "/content/drive/MyDrive"
MyDrive"
```

## **توجه:**
در این بخش فایل csv مطابق خواسته مسئله ساخته شد و برای مشاهده و دانلود این فایل به لینک زیر مراجعه کنید.

https://drive.google.com/file/d/1vaX5SO_wGoiucO9P-bF--J7WNj-qUJhx/view?usp=sharing
