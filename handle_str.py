import re
import os
import docx
import json

def read_word_file(file_path='C:\\Users\\Administrator\\Desktop\\报价单 (1)'):
    """
    读取word文件，返回字符串
    """
    str_list = []
    for file in os.listdir(file_path):
        file_full_path = os.path.join(file_path, file)
        doc = docx.Document(file_full_path)
        for para in doc.paragraphs:
            str_list.append(para.text)
    return '\n'.join(str_list)


def handle_price(pro_name):
    """
    处理价格
    """
    if '$' in pro_name:
        pattern1 = r'\$\s*(\d+\.?\d*)/?t?'
    elif '/unit' in pro_name:
        pattern1 = r'(\d+\.?\d*)/unit'
    elif 'u' in pro_name:
        pattern1 = r'(\d+\.?\d*)u/?t?'
    else:
        return pro_name
    
    pattern2 = r'(\d+)t'
    unit_price_match = re.search(pattern1, pro_name)
    unit_price = float(unit_price_match.group(1)) if unit_price_match else 0.0
    hash_rate_match = re.search(pattern2, pro_name)
    hash_rate = float(hash_rate_match.group(1)) if hash_rate_match else 0.0
    if unit_price < 300:
        pro_name = f"{pro_name} ${round(unit_price*hash_rate, 2)}"
    elif '/unit' in pro_name or 'u' in pro_name:
        pro_name = f"{pro_name} ${unit_price}"
    
    return pro_name
    

def handle_str(str):
    """
    处理字符串，返回列表
    """
    str_list = str.splitlines() # 按行分割

    pro_list = []
    pattern = r"[^a-z0-9./+$￥＄\s]"
    for i in str_list:
        i = ' '.join(i.lower().split()).replace('＄', '$').replace('-', ' ')
        match = re.sub(pattern, "", i)
        if match and re.search('[u$￥]', match):
            # 处理价格
            pro_name = handle_price(re.sub(r'[♡|⭐|💫]', '', match))
            pro_list.append(pro_name)
    
    # print(pro_list, len(pro_list))

    return pro_list


def save_jsonl(filename, pro_list):
    """
    保存列表到jsonl文件
    """
    if type(pro_list[0]) is dict:
        with open(filename, 'w', encoding='utf-8') as f:
            for pro in pro_list:
                f.write(json.dumps(pro, ensure_ascii=False) + '\n')
    else:
        pro_list = list(set(pro_list))
        with open(filename, 'w', encoding='utf-8') as f:
            for pro in pro_list:
                pro_json = {
                    'input': pro,
                    'output': '',
                }
                f.write(json.dumps(pro_json, ensure_ascii=False) + '\n')
        




def handle_output():
    """
    处理jsonl文件，返回列表
    """
    models = {
        # Goldshell 系列
        'minidogeiii': 'Goldshell Mini Doge III',
        'minidoge3+': 'Goldshell Mini Doge III+',
        'aebox': 'Goldshell AE BOX',
        'aeboxpro': 'Goldshell AE Box Pro',
        'aebox2': 'Goldshell AE Box II',
        'al box': 'Goldshell AL Box',
        'dg max': 'Goldshell DG Max',
        'dgmax': 'Goldshell DG Max',
        'almax': 'Goldshell AL Max',
        'al0': 'Goldshell AL Box',
        'al2': 'Goldshell AL2',
        'al3': 'Goldshell AL3',
        'mini3': 'Goldshell Mini3',
        'dg home1': 'Goldshell DG Home1',
        'dghome 1': 'Goldshell DG Home1',
        'ks5': 'Bitmain Antminer KS5',
        'ks7': 'Bitmain Antminer KS7',
        'ks5pro': 'Bitmain Antminer KS5 Pro',
        'nano3': 'Avalon Nano3',
        'dg1': 'Elphapex DG1',
        'dg1+': 'Elphapex DG1+',
        'd1 mini': 'Volc Miner D1 Mini',
        'd1': 'Volc Miner D1',
        's19xp': 'Bitmain Antminer S19 XP',
        'fluminer l1': 'FluMiner Fluminer L1',
        's19xphyd': 'Bitmain Antminer S19 XP Hyd',
        's21+ hyd': 'Bitmain Antminer S21+ Hyd',
        'l9': 'Bitmain Antminer L9',
        's21xp+hyd': 'Bitmain Antminer S21+ XP Hyd',
        'l7': 'Bitmain Antminer L7',
        's19kpro': 'Bitmain Bitcoin Miner S19K Pro',
        's19j xp': 'Bitmain Bitcoin Miner S19j XP',
        's19': 'Bitmain Antminer S19',
        's19xp': 'Bitmain Antminer S19 XP',
        's19pro': 'Bitmain Antminer S19 Pro',
        's21': 'Bitmain Antminer S21',
        's21pro': 'Bitmain Antminer S21 Pro',
        's21+': 'Bitmain Antminer S21+',
        's21+pro': 'Bitmain Antminer S21+ Pro',
        's21xp': 'Bitmain Antminer S21 XP',
        's21xp+hyd': 'Bitmain Antminer S21+ XP Hyd',
        's21+hyd': 'Bitmain Antminer S21+ HYD',
        't21': 'Bitmain Antminer T21',
        '21': 'Bitmain Antminer S21',
        's19kpro': 'Bitmain Bitcoin Miner S19K Pro',
        'e9pro': 'Bitmain Antminer E9 Pro',
        'e9': 'Bitmain Antminer E9',
        'e9+': 'Bitmain Antminer E9+',
        'e9+pro': 'Bitmain Antminer E9+ Pro',
        'e9+hyd': 'Bitmain Antminer E9+ Hyd',
        'e9+hyd pro': 'Bitmain Antminer E9+ Hyd Pro',
        'd10+': 'Bitmain Antminer D10+',
        'k7': 'Bitmain Antminer K7',
        'ka3': 'Bitmain Antminer KA3',
        'x16p': 'Jasminer X16-P',
        'x16q': 'Jasminer X16-Q',
        'x16 q': 'Jasminer X16-Q',
        'mini3': 'Canaan Avalon Miner Mini 3',

        # Goldshell 系列
        'dg home': 'Goldshell DG Home',
        'dghome1': 'Goldshell DG Home 1',
        'dghome1+': 'Goldshell DG Home 1+',
        'dghome1+pro': 'Goldshell DG Home 1+ Pro',
        'dghome1+hyd': 'Goldshell DG Home 1+ Hyd',
        'dghome1+hyd pro': 'Goldshell DG Home1+ Hyd Pro',
        'mini doge iii': 'Goldshell Mini Doge III',
        'mini doge': 'Goldshell Mini Doge',
        'ae': 'Goldshell AE BOX',
        'minidoge': 'Goldshell Mini Doge',
        'minidoge+': 'Goldshell Mini Doge+',
        
        'ez100': 'Bombax Miner EZ100-C',
        '1346': 'Canaan Avalon Miner A1346',
        '1246': 'Canaan Avalon Miner A1246',
        '1466': 'Canaan Avalon Miner A1466',
        'a1566': 'Canaan Avalon Miner A1566',
        'ae0': 'Iceriver AE0',
        'aeo': 'Iceriver AEO',
        'a2': 'Bitdeer SealMiner A2',
        'e11': 'Ebang Ebit E11',
        # MicroBT 系列
        'm50s': 'MicroBT Whatsminer M50S',
        'm50': 'MicroBT Whatsminer M50',
        'm60s': 'MicroBT Whatsminer M60S',
        'm60': 'MicroBT Whatsminer M60',
        'm61': 'MicroBT Whatsminer M61',
        'm30s+': 'MicroBT Whatsminer M30S+',
    }
    
    train_list = []
    with open(r'datasets\train_data\train.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            train_json = json.loads(line)
            input_str = train_json['input']
            hash_pattern = r'(\d+\.?\d*)([mgt])'

            # 匹配hashrate
            hash_match = re.search(hash_pattern, input_str)
            hash_rate = float(hash_match.group(1)) if hash_match else 0

             # 定义单位升级映射
            unit_upgrade = {'k': 'm', 'm': 'g', 'g': 't', 't': 'p', 'p': 'e'}
            # 当数值大于1000时,进行单位升级
            hashrate_unit = hash_match.group(2) if hash_match else ''
            while hash_rate >= 1000 and hashrate_unit in unit_upgrade:
                hash_rate /= 1000
                hashrate_unit = unit_upgrade[hashrate_unit]
            
            # 提取价格
            if '$' in input_str:
                price_pattern = r'\$\s*(\d+\.?\d*)'
            elif 'u' in input_str:
                price_pattern = r'(\d+\.?\d*)\s*u'
            else:
                price_pattern = r'￥?\s*(\d+\.?\d*)\s*￥?'

            price_match = re.findall(price_pattern, input_str)
            price = float(price_match[-1]) if price_match else 0.0
            
            key_len_list = {}
            for key, value in models.items():
                if key in input_str:
                    key_len_list[key] = len(key)
            # print(max_len_key, key_len_list)
            if key_len_list:
                max_len_key = max(key_len_list, key=key_len_list.get)
                if price > 0 and hash_rate > 0 and '￥' not in input_str:
                    train_json['output'] = f"{models[max_len_key]} - {hash_rate}{hashrate_unit}h/t ${price}"
                elif '￥' in input_str:
                    train_json['output'] = f"{models[max_len_key]} - {hash_rate}{hashrate_unit}h/t ￥{price}"
            train_list.append(train_json)
    save_jsonl(r'datasets\train_data\train_v1.jsonl', remove_place_output(train_list))


def remove_place_output(json_list):
    """
    删除输出中的地点
    """
    pro_list = []
    for line in json_list:
        if line['output'] != "":
            pro_list.append(line)
    return pro_list

if __name__ == "__main__":
    # str = read_word_file()
    # pro_list = handle_str(str)
    # save_jsonl('./datasets/train_data/train.jsonl', pro_list)
    # handle_output()
    pass
