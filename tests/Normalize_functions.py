import re 
import pandas as pd 

#clen_text
def clean_text(text):
    if pd.isnull(text):
        return text
    # Перевести в нижний регистр
    text = text.lower()
    # Удалить пунктуацию
    text = re.sub(r'[^\w\s]', '', text)
    # Сжать пробелы
    text = re.sub(r'\s+', ' ', text)
    # Удалить пробелы по краям
    return text.strip()

#normalize_phone
def normalize_phone(phone, length=10):
    if pd.isnull(phone):
        return None
    # Оставить только цифры
    digits = re.sub(r'\D', '', str(phone))
    
    # Если слишком короткий — дополняем нулями слева
    if len(digits) < length:
        digits = digits.zfill(length)
    # Если слишком длинный — обрезаем справа
    elif len(digits) > length:
        digits = digits[:length]
    
    return digits

#normalize_zip
def normalize_zip(zip, length=5):
 if pd.isnull(zip):
  return None 
 digits = re.sub(r'\D', '', str(zip))
 
 return digits


#clean_email
def clean_email(email):
    if pd.isnull(email):
        return None
    email = str(email).lower().strip()
    email = re.sub(r'[^\w@.\-]', '', email)  # сохраняем @, точку и дефис
    return email


# def is_valid_email(email):
#  if pd.isnull(email):
#   return False
#  pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
#  return re.match(pattern, str(email)) is not None
# df["valid_email"] = df["email"].apply(is_valid_email)