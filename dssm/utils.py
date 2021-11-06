# -*- coding: utf-8 -*-
import re


ZH_PATTERN = re.compile(r'[\u4e00-\u9fa5]+')
EN_PATTERN = re.compile(r'[a-zA-Z]')


def contain_zh(text):
    global ZH_PATTERN
    match = re.search(ZH_PATTERN, text)
    if match:
        return True
    return False


def contain_en(text):
    match = re.search(EN_PATTERN, text)
    if match:
        return True
    else:
        return False


def clean_text(raw):
    regex = re.compile(r'[^0-9a-zA-Z\u4e00-\u9fa5]+')
    return regex.sub(' ', raw)


def tokenize4zh(text):
    """
    Bilingual word tokenizer: Chinese is split by character,
    English is split by word, and numbers are split by space
    """

    regex1 = re.compile(r'[\W]*')  # special characters(non-letters, non-digits, non-Chinese characters, and non-underscores)
    regex2 = re.compile(r'([\u4e00-\u9fa5])')  # chinese characters

    texts = regex1.split(text.lower())  # split with special characters
    tokens = []
    for s in texts:
        ret = regex2.split(s)
        if ret is None:
            tokens.append(s)
        else:
            for ch in ret:
                tokens.append(ch)

    tokens = [w for w in tokens if len(w.strip()) > 0]  # remove blank characters

    return tokens


if __name__ == '__main__':
    s = "China's Legend Holdings will split its several business arms to go public on stock markets,  the group's president Zhu Linan said on Tuesday.该集团总裁朱利安周二表示，中国联想控股将分拆其多个业务部门在股市上市。"
    tokens = tokenize4zh(s)
    print(tokens)
