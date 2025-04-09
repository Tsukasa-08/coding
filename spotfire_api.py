
import clr
clr.AddReference("System.Windows.Forms") 

# --- ここから元のスクリプト ---
from System.Windows.Forms import OpenFileDialog, DialogResult
from Spotfire.Dxp.Application.Properties import PropertyValue

# ドキュメントプロパティ名を指定
propName = "selectedFilePath" 

# ファイル選択ダイアログの設定
dialog = OpenFileDialog()
dialog.Title = "データファイルを選択してください"
# dialog.Filter = "CSVファイル (*.csv)|*.csv|すべてのファイル (*.*)|*.*" # 必要に応じてフィルタを設定
dialog.FilterIndex = 1
dialog.RestoreDirectory = True

# ダイアログを表示し、ユーザーがOKをクリックしたか確認
if dialog.ShowDialog() == DialogResult.OK:
    # 選択されたファイルパスを取得
    filePath = dialog.FileName
    
    # ドキュメントプロパティにファイルパスを設定
    prop = Document.Properties.GetProperty(PropertyValue.QualifiedName(propName))
    if prop.IsNone():
       print("エラー: ドキュメントプロパティ '" + propName + "' が見つかりません。")
    else:
       Document.Properties.SetProperty(prop.Name, filePath)
       print("ファイルパスが設定されました: " + filePath)
       # ここでデータ関数を明示的に実行することも可能
       # dataFunction = Document.Data.DataFunctions["あなたのデータ関数名"]
       # if dataFunction is not None:
       #    dataFunction.Execute()
else:
    print("ファイル選択がキャンセルされました。")

解説:
 * import clr: Common Language Runtimeライブラリをインポートします。これは.NETアセンブリを操作するために必要です。
 * clr.AddReference("System.Windows.Forms"): System.Windows.Forms.dllというアセンブリへの参照をスクリプト環境に追加します。これにより、スクリプト内でSystem.Windows.Forms名前空間とその中のクラス（OpenFileDialogなど）が利用可能になります。
この修正により、System.Windows.Formsが見つからないというエラーは解消されるはずです。
再確認: このスクリプトはSpotfire Analystクライアントで実行することを前提としています。Web Player（ブラウザ経由での利用）では、セキュリティ上の理由からローカルファイルの選択ダイアログを表示する機能は基本的に動作しません。
