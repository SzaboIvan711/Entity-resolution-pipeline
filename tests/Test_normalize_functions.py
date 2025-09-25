import pandas as pd
import pytest
import re 
from Normalize_functions import clean_text, clean_email, normalize_phone, normalize_zip

def test_basic_text():
    assert clean_text("Hello World!") == "hello world"

def test_multiple_spaces():
    assert clean_text("Hello     World") == "hello world"

def test_punctuation_removal():
    assert clean_text("Hi, Ivan! How's it going?") == "hi ivan hows it going"

def test_uppercase_conversion():
    assert clean_text("THIS IS TEXT") == "this is text"

def test_leading_trailing_spaces():
    assert clean_text("   spaced out   ") == "spaced out"

def test_numbers_and_text():
    assert clean_text("Room 101: Ready!") == "room 101 ready"

def test_empty_string():
    assert clean_text("") == ""

def test_only_punctuation():
    assert clean_text("!!!???") == ""

def test_null_value():
    assert pd.isna(clean_text(pd.NA))

def test_mixed_characters():
    assert clean_text(" 123_ABC!@# ") == "123_abc"


#////////////

def test_basic_email():
    assert clean_email("Ivan.Petrov@Gmail.com") == "ivan.petrov@gmail.com"

def test_email_with_spaces():
    assert clean_email("  ivan.petrov@gmail.com  ") == "ivan.petrov@gmail.com"

def test_email_with_extra_chars():
    assert clean_email("ivan.petrov+test@gmail.com") == "ivan.petrovtest@gmail.com"

def test_email_with_uppercase():
    assert clean_email("IVAN@EXAMPLE.COM") == "ivan@example.com"

def test_email_with_unicode():
    assert clean_email("иван@почта.рф") == "иван@почта.рф"

def test_email_with_newline():
    assert clean_email("ivan\n@gmail.com") == "ivan@gmail.com"

def test_email_with_tab():
    assert clean_email("ivan\t@gmail.com") == "ivan@gmail.com"

def test_email_with_special_symbols():
    assert clean_email("ivan!@gmail.com") == "ivan@gmail.com"

def test_empty_string():
    assert clean_email("") == ""

def test_null_value():
    assert clean_email(pd.NA) is None

#///////////////////

def test_standard_format():
    assert normalize_phone("+7 (912) 345-67-89") == "7912345678"

def test_short_number():
    assert normalize_phone("12345") == "0000012345"

def test_long_number():
    assert normalize_phone("1234567890123") == "1234567890"

def test_only_digits():
    assert normalize_phone("9876543210") == "9876543210"

def test_with_spaces():
    assert normalize_phone(" 987 654 3210 ") == "9876543210"

def test_with_letters():
    assert normalize_phone("abc123def456") == "0000123456"

def test_empty_string():
    assert normalize_phone("") == "0000000000"

def test_none_input():
    assert normalize_phone(None) is None

def test_custom_length():
    assert normalize_phone("123456", length=8) == "00123456"

def test_zero_padding():
    assert normalize_phone("1") == "0000000001"

#/////////////////////


def normalize_zip(zip, length=5):
    if pd.isnull(zip):
        return None 
    digits = re.sub(r'\D', '', str(zip))
    return digits

@pytest.mark.parametrize("input_val, expected", [
    ("12345", "12345"),
    ("abc123", "123"),
    (None, None),
    (pd.NA, None),
])
def test_normalize_zip(input_val, expected):
    assert normalize_zip(input_val) == expected