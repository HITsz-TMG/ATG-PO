import re
from nltk import sent_tokenize

def process_sentence(sentence, flag):
    punctuation = (".", ",", "!", "?")
    if sentence.endswith(punctuation):
        return re.sub(r'([.,?!])$', flag + r'\1', sentence)
    else:
        return sentence + flag

def move_period(text):
    sentences = sent_tokenize(text)
    output_text = []
    for sentence in sentences:
        quotes = re.findall(r'\[(\d+(?:\]\[\d+)*)\]', sentence)
        quotes = sorted(list(set(quotes)))
        sentence = re.sub(r'\[(\d+(?:\]\[\d+)*)\]', "", sentence)
        sentence = process_sentence(sentence, ''.join(['[{}]'.format(i) for i in quotes]))
        output_text.append(sentence.strip())
    output_text = ' '.join(output_text)
    return output_text.strip()

def convert_forward(text):
    text = re.sub(r'\[No document\]', '', text)
    text = re.sub(r'\[Use document (\d+)\]', r'[\1]', text)
    text = re.sub(r'\[Use document (\w+(?:,\w+)*)\]', lambda x: '[' + ','.join(x.group(1).split(',')) + ']', text)
    text = re.sub(r',\s(\d)', r',\1', text)
    text = text.replace("] [", "][")
    text = re.sub(r'\[(.*?)\]', lambda x: "[" + "][".join(x.group(1).split(',')) + "]", text)
    text = move_period(text)
    return text

def convert_backward(text):
    def replace_ref(match):
        refs = match.group(1).split('][')
        return f'[Use document {",".join(refs)}]'
    s = re.sub(r'\[(\d+(?:\]\[\d+)*)\]', replace_ref, text)
    s = re.sub(r'\. \[Use document (\d+(?:,\d+)*)\]', r'. [Use document \1]', s)
    sentences = []
    s = sent_tokenize(s.replace('].', ']. '))
    for sentence in s:
        quotes = re.findall(r"\[Use document (\d+(?:,\d+)*)\]", sentence)
        quotes = sorted(list(set(quotes)))
        sentence = re.sub(r"\[Use document (\d+(?:,\d+)*)\]", '', sentence)
        if len(quotes) != 0:
            sentence = sentence + "[Use document {}]".format(",".join(quotes))
        if not '[Use document' in sentence:
            sentences.append("[No document]" + sentence.strip())
        else:
            sentences.append(re.sub(r'(.+?)(\[Use document (\d+(?:,\d+)*)\])', r'\2\1', sentence.strip()))
    s = ' '.join(sentences)
    s = s.replace(' .', '.')
    s = s.rstrip('[No document]').strip()
    if s[0] != '[':
        s = '[No document]' + s
    return s


if __name__ == '__main__':
    input_str = "[Use document 2,3,4,53] According to doc2 and doc 3, I know the Fact."
    print(convert_forward(input_str))
    print(convert_backward(convert_forward(input_str)))