# python generate_dataset.py --size 50000 --version 1 --output_path ./datasets

import random
import pandas as pd
from tqdm import tqdm
import math
import argparse

def define_argparse():
    parser = argparse.ArgumentParser(description='Generate numerical dataset')
    parser.add_argument('--size', type=int, default=1000, help='Size of the dataset')
    parser.add_argument('--max_precision', type=int, default=4, help='Maximum precision of the generated numbers')
    parser.add_argument('--output_path', type=str, default="./datasets", help='Path to save the generated dataset')
    parser.add_argument('--version', type=int, default=1, help='Version of the dataset to generate')
    parser.add_argument('--max_value', type=int, default=99999, help='Maximum value of the generated numbers')
    
    args = parser.parse_args()
    
    args.output_fn = f"numerical_dataset_v{args.version}_size{args.size}.xlsx"
    
    return args

def generate_numeric_dataset_v1(size, value_to_multiply_by_unit, max_value=99999, max_precision=5):
    
    target_units = list(value_to_multiply_by_unit.keys())
    df = pd.DataFrame([], columns=["value1", "unit1", "operation", "value2", "unit2"])
    trials = 0
    
    with tqdm(total=size, unit='rows', desc='Generating dataset') as pbar:
        while len(df) < size:
            value = round(random.uniform(0, max_value), max_precision)
            unit1, unit2 = random.sample(target_units, 2)
            
            if unit1 == unit2:
                continue
            
            value1 = value * value_to_multiply_by_unit[unit1]
            value2 = value * value_to_multiply_by_unit[unit2]
            
            try:
                value1_cnt, value2_cnt = int(math.log10(value1)), int(math.log10(value2))
            except:
                continue
            
            if value1_cnt > max_precision or value2_cnt > max_precision:
                continue

            try:
                value1_decimal_count = len(str(value1).split(".")[1])
                value2_decimal_count = len(str(value2).split(".")[1])
            except:
                continue
            
            if value1_decimal_count > max_precision or value2_decimal_count > max_precision:
                continue
            
            temp = [
                [value1, unit1, "=", value2, unit2],
            ]
            
            df = pd.concat([df, pd.DataFrame(temp, columns=["value1", "unit1", "operation", "value2", "unit2"])])
            df = df.drop_duplicates()
            df = df.reset_index(drop=True)
            trials += 1
            pbar.update(len(df) - pbar.n)
    return df
            
def generate_numeric_dataset_v2(size, value_to_multiply_by_unit, max_value=99999, max_precision=4):
    
    target_units = list(value_to_multiply_by_unit.keys())
    df = pd.DataFrame([], columns=["value1", "unit1", "operation", "value2", "unit2"])
    trials = 0
    
    with tqdm(total=size, unit='rows', desc='Generating dataset') as pbar:
        while len(df) < size:
            value = round(random.uniform(0, max_value), max_precision)
            unit1, unit2 = random.sample(target_units, 2)
            
            if unit1 == unit2:
                continue
            
            value1 = value * value_to_multiply_by_unit[unit1]
            value2 = value * value_to_multiply_by_unit[unit2]
            
            try:
                value1_cnt, value2_cnt = int(math.log10(value1)), int(math.log10(value2))
            except:
                continue
            
            if value1_cnt > max_precision or value2_cnt > max_precision:
                continue

            try:
                value1_decimal_count = len(str(value1).split(".")[1])
                value2_decimal_count = len(str(value2).split(".")[1])
            except:
                continue
            
            if value1_decimal_count > max_precision or value2_decimal_count > max_precision:
                continue
            
            diff = round(random.uniform(0, value*0.2), max_precision)
            diff1 = round(diff * value_to_multiply_by_unit[unit1], max_precision)
            diff2 = round(diff * value_to_multiply_by_unit[unit2], max_precision)
            
            temp_equal = [
                [value1, unit1, "=", value2, unit2],
            ]
            
            temp_inequal = [
                [value1, unit1, "<", value2+diff2, unit2],
                [value1, unit1, ">", value2-diff2, unit2],
                [value1, unit1, "<", value1+diff1, unit1],
                [value1, unit1, ">", value1-diff1, unit1],
                [value1+diff1, unit1, ">", value2, unit2],
                [value1-diff1, unit1, "<", value2, unit2],
                [value2, unit2, "<", value2+diff2, unit2],
                [value2, unit2, ">", value2-diff2, unit2],
            ]
            
            temp_inequal = random.sample(temp_inequal, len(temp_equal))
            temp = temp_equal + temp_inequal
            
            df = pd.concat([df, pd.DataFrame(temp, columns=["value1", "unit1", "operation", "value2", "unit2"])])
            df = df.drop_duplicates()
            df = df.reset_index(drop=True)
            trials += 1
            pbar.update(len(df) - pbar.n)
    pbar.close()
    
    return df
            
def generate_numeric_dataset_v3(size, value_to_multiply_by_unit, max_value=99999, max_precision=4):
    
    target_units = list(value_to_multiply_by_unit.keys())
    df = pd.DataFrame([], columns=["value1", "unit1", "operation", "diff_value", "diff_unit", "value2", "unit2"])
    trials = 0
    
    with tqdm(total=size, unit='rows', desc='Generating dataset') as pbar:
        while len(df) < size:
            value = round(random.uniform(0, max_value), max_precision)
            unit1, unit2 = random.sample(target_units, 2)
            
            if unit1 == unit2:
                continue
            
            value1 = value * value_to_multiply_by_unit[unit1]
            value2 = value * value_to_multiply_by_unit[unit2]
            
            try:
                value1_cnt, value2_cnt = int(math.log10(value1)), int(math.log10(value2))
            except:
                continue
            
            if value1_cnt > max_precision or value2_cnt > max_precision:
                continue

            try:
                value1_decimal_count = len(str(value1).split(".")[1])
                value2_decimal_count = len(str(value2).split(".")[1])
            except:
                continue
            
            if value1_decimal_count > max_precision or value2_decimal_count > max_precision:
                continue
            
            diff = round(random.uniform(0, value*0.2), max_precision)
            diff1 = round(diff * value_to_multiply_by_unit[unit1], max_precision)
            diff2 = round(diff * value_to_multiply_by_unit[unit2], max_precision)
            
            temp_equal = [
                [value1, unit1, "+", 0.0, unit1, value2, unit2],
                [value1, unit1, "-", 0.0, unit1, value2, unit2],
                [value1, unit1, "+", 0.0, unit2, value2, unit2],
                [value1, unit1, "-", 0.0, unit2, value2, unit2],
            ]
            
            temp_inequal = [
                [value1, unit1, "+", diff1, unit1, value2+diff2, unit2],
                [value1, unit1, "-", diff1, unit1, value2-diff2, unit2],
                [value1, unit1, "+", diff2, unit2, value2+diff2, unit2],
                [value1, unit1, "-", diff2, unit2, value2-diff2, unit2],
                [value1, unit1, "+", diff1, unit1, value1+diff1, unit1],
                [value1, unit1, "-", diff1, unit1, value1-diff1, unit1],
                [value1, unit1, "+", diff2, unit2, value1+diff1, unit1],
                [value1, unit1, "-", diff2, unit2, value1-diff1, unit1],
            ]
            
            temp_inequal = random.sample(temp_inequal, len(temp_equal))
            temp = temp_equal + temp_inequal
            
            df = pd.concat([df, pd.DataFrame(temp, columns=["value1", "unit1", "operation", "diff_value", "diff_unit", "value2", "unit2"])])
            df = df.drop_duplicates()
            df = df.reset_index(drop=True)
            trials += 1
            pbar.update(len(df) - pbar.n)
    pbar.close()
    
    return df

length_value_to_multiply_by_unit = {
    "km": 0.001,
    "m": 1,
    "cm": 100,
    "mm": 1000,
}

weight_value_to_multiply_by_unit = {
    "kg": 0.001,
    "g": 1,
    "mg": 1000,
}

volumne_value_to_multiply_by_unit = {
    "l": 1,
    "ml": 1000,
}

def main(args):
    
    length_size = int(args.size * 0.4)
    weight_size = int(args.size * 0.3)
    volume_size = args.size - length_size - weight_size
    
    if args.version == 1:
        generate_fn = generate_numeric_dataset_v1
    elif args.version == 2:
        generate_fn = generate_numeric_dataset_v2
    elif args.version == 3:
        generate_fn = generate_numeric_dataset_v3
    
    length_df = generate_fn(length_size, length_value_to_multiply_by_unit, max_value=args.max_value, max_precision=args.max_precision)
    weight_df = generate_fn(weight_size, weight_value_to_multiply_by_unit, max_value=args.max_value, max_precision=args.max_precision)
    volume_df = generate_fn(volume_size, volumne_value_to_multiply_by_unit, max_value=args.max_value, max_precision=args.max_precision)
    
    total_data = pd.concat([length_df, weight_df, volume_df])
    total_data = total_data.sample(n=args.size, ignore_index=True)

    total_data.to_excel(f"{args.output_path}/{args.output_fn}", index=False, engine="openpyxl")
    print(f"Dataset has been saved to {args.output_path}/{args.output_fn}")

if __name__ == "__main__":
    args = define_argparse()
    main(args)