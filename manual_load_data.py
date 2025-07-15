"""
transtab load_data の手動実装
arffファイルまたはOpenMLデータを使用してload_dataと同じ出力形式を生成
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_data_manual(dataname="credit-g", arff_file_path=None, test_size=0.2, val_size=0.2, random_state=42):
    """
    transtab.load_data() と同じ形式でデータを読み込み・整形
    
    Parameters:
    -----------
    dataname : str
        データセット名（"credit-g"など）
    arff_file_path : str or None
        arffファイルのパス（Noneの場合はOpenMLを使用）
    test_size : float
        テストデータの割合
    val_size : float
        検証データの割合（残りの訓練データに対する割合）
    random_state : int
        ランダムシード
    
    Returns:
    --------
    allset : tuple (X_all, y_all)
        全データセット
    trainset : tuple (X_train, y_train)
        訓練データセット
    valset : tuple (X_val, y_val)
        検証データセット
    testset : tuple (X_test, y_test)
        テストデータセット
    cat_cols : list
        カテゴリ変数のインデックスリスト
    num_cols : list
        数値変数のインデックスリスト
    bin_cols : list
        バイナリ変数のインデックスリスト
    """
    
    print(f"=== Manual load_data for {dataname} ===")
    print("########################################")
    print(f"load data from {dataname}")
    
    # 1. データ読み込み
    if arff_file_path and os.path.exists(arff_file_path):
        print(f"Loading from ARFF file: {arff_file_path}")
        # ARFFファイルから読み込み
        try:
            from scipy.io import arff
            data, meta = arff.loadarff(arff_file_path)
            df = pd.DataFrame(data)
            
            # バイト文字列をデコード
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except:
                        pass
            
            # ターゲット列を分離（通常は最後の列）
            target_col = df.columns[-1]
            X = df.iloc[:, :-1]
            y = df[target_col]
            
        except ImportError:
            print("scipy not available for ARFF, falling back to OpenML")
            arff_file_path = None
    
    if not arff_file_path:
        print("Loading from OpenML...")
        # OpenMLから読み込み
        if dataname == "credit-g":
            credit = fetch_openml(data_id=31, as_frame=True, parser='auto')
            X = credit.data
            y = credit.target
        else:
            raise ValueError(f"Dataset {dataname} not supported")
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # 2. 変数タイプの分類
    cat_cols = []
    num_cols = []
    bin_cols = []
    
    print("\\nAnalyzing feature types...")
    for i, col in enumerate(X.columns):
        col_data = X[col]
        unique_count = col_data.nunique()
        dtype = col_data.dtype
        
        # transtabのロジックを模倣
        if dtype == 'object' or dtype.name == 'category':
            if unique_count == 2:
                bin_cols.append(i)
            else:
                cat_cols.append(i)
        else:
            if unique_count == 2:
                bin_cols.append(i)
            elif unique_count <= 10:  # 離散的な数値変数はカテゴリとして扱う
                cat_cols.append(i)
            else:
                num_cols.append(i)
    
    print(f"Categorical columns: {len(cat_cols)}")
    print(f"Numerical columns: {len(num_cols)}")
    print(f"Binary columns: {len(bin_cols)}")
    
    # 3. ターゲット変数のエンコーディング
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(f"Target classes: {le.classes_}")
        print(f"Target mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    else:
        y_encoded = y.values
    
    # 4. 特徴量のエンコーディング
    X_processed = X.copy()
    
    # カテゴリ変数とバイナリ変数をエンコード
    categorical_indices = cat_cols + bin_cols
    for col_idx in categorical_indices:
        col_name = X.columns[col_idx]
        if X_processed[col_name].dtype == 'object' or X_processed[col_name].dtype.name == 'category':
            le = LabelEncoder()
            X_processed[col_name] = le.fit_transform(X_processed[col_name])
    
    # numpy配列に変換
    X_numpy = X_processed.values.astype(np.float64)
    y_numpy = y_encoded.astype(np.int64)
    
    # 5. データ分割
    # まず全体をtrain+val と test に分割
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_numpy, y_numpy, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_numpy
    )
    
    # train+val を train と val に分割
    val_size_adjusted = val_size / (1 - test_size)  # 残りのデータに対する割合に調整
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size_adjusted, 
        random_state=random_state, 
        stratify=y_temp
    )
    
    print(f"\\nData split:")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 6. transtab形式の出力を作成
    allset = (X_numpy, y_numpy)
    trainset = (X_train, y_train)
    valset = (X_val, y_val)
    testset = (X_test, y_test)
    
    print(f"\\nColumn type assignments:")
    print(f"cat_cols: {cat_cols}")
    print(f"num_cols: {num_cols}")
    print(f"bin_cols: {bin_cols}")
    
    return allset, trainset, valset, testset, cat_cols, num_cols, bin_cols

def show_load_data_output(dataname="credit-g", arff_file_path=None):
    """load_data_manualの出力を詳細表示"""
    
    # データ読み込み
    results = load_data_manual(dataname=dataname, arff_file_path=arff_file_path)
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = results
    
    print("\\n" + "="*60)
    print("TRANSTAB LOAD_DATA FORMAT OUTPUT")
    print("="*60)
    
    # 各データセットの詳細
    datasets = {
        "allset": allset,
        "trainset": trainset, 
        "valset": valset,
        "testset": testset
    }
    
    for name, (X, y) in datasets.items():
        print(f"\\n{name.upper()}:")
        print(f"  X shape: {X.shape}, dtype: {X.dtype}")
        print(f"  y shape: {y.shape}, dtype: {y.dtype}")
        print(f"  X sample (first 5 rows, first 5 cols):")
        print(f"    {X[:5, :5]}")
        print(f"  y sample (first 10): {y[:10]}")
    
    # 変数タイプ情報
    print(f"\\nVARIABLE TYPES:")
    print(f"  cat_cols ({len(cat_cols)}): {cat_cols}")
    print(f"  num_cols ({len(num_cols)}): {num_cols}")
    print(f"  bin_cols ({len(bin_cols)}): {bin_cols}")
    
    # 元の列名情報（参考用）
    if dataname == "credit-g":
        try:
            credit = fetch_openml(data_id=31, as_frame=True, parser='auto')
            column_names = credit.data.columns.tolist()
            print(f"\\nCOLUMN NAMES (for reference):")
            for i, name in enumerate(column_names):
                var_type = "cat" if i in cat_cols else "num" if i in num_cols else "bin"
                print(f"  {i:2d}: {name:25s} ({var_type})")
        except:
            print("\\nCould not fetch column names")
    
    return results

if __name__ == "__main__":
    # credit-gデータセットで実行
    results = show_load_data_output("credit-g")
    
    print("\\n" + "="*60)
    print("USAGE EXAMPLE:")
    print("="*60)
    print("# これでtranstab.load_data()と同じ出力が得られます")
    print("allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_data_manual('credit-g')")
    print("X_all, y_all = allset")
    print("X_train, y_train = trainset")
    print("# ... etc")
