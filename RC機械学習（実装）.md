# 機械学習レポート（実装）
下記の要件の通り、区分ごとに単元レポートを作成、ご提出ください。
<br> １）各章につき100文字以上で要点をまとめ、実装演習結果、確認テストについての自身の考察等を取り入れたレポートとする。
<br> 単元毎に①〜④（各１点）を組み合わせ科目の基準点を超えるようにする。（②実装は応用数学以外は必須）
<br> ①単元毎の要点まとめ
<br> ②実装演習キャプチャー（応用数学はなし）
<br> ③「確認テスト」など自身の考察結果（応用数学・機械学習はなし）
<br> ④演習問題や参考図書、修了課題など関連記事レポートによる加点
<br> ２）各科目の基準点が足りない場合、実装演習が不足する場合は差し戻しとする。
<br> ※各章は講義動画および講義資料（PDF）でご確認ください。

## 準備
Google ColaboratoryではなくVisual Studio Codeを使用して実装した。
<br> メモ：実行エラーが出たため、scikit-learnをインストールし実行した。

## 線形回帰モデル
<img width="768" alt="線形回帰" src="https://user-images.githubusercontent.com/52492098/145054055-26dd740b-b044-4503-8834-f7499f4c98ca.png">
<img width="768" alt="線形回帰２" src="https://user-images.githubusercontent.com/52492098/145165054-6f28d0d3-9f59-47b4-8896-41207ed2c0b3.png">

<img width="793" alt="スクリーンショット 2021-12-08 18 30 46" src="https://user-images.githubusercontent.com/52492098/145184247-6a68ad53-0b8e-4e50-976f-cd37a80d3768.png">
まず、sklernから今回使用するdatasetを取り出す。
<br> 二次元配列を扱うため、PandasからDataframeライブラリを呼び出す。
<br> 数値計算のnumpyも呼び出す。
<br> データをbostonに入れる。
<br> （結果略）
<img width="692" alt="スクリーンショット 2021-12-08 18 18 32" src="https://user-images.githubusercontent.com/52492098/145184266-226d69a1-c3f2-46a4-bd27-b3aaa3de0f64.png">
インポートしたデータはprintで確認できる。
<br> （結果略）
<img width="692" alt="スクリーンショット 2021-12-08 18 29 05" src="https://user-images.githubusercontent.com/52492098/145184275-838157f3-c866-4bea-a7da-328401f2984b.png">
データの変数の説明を確認することもできる。
<br> （結果略）
<img width="692" alt="スクリーンショット 2021-12-08 18 29 22" src="https://user-images.githubusercontent.com/52492098/145184296-baaa4401-7bbc-4d45-b1ed-7a4de7220224.png">
分析に利用できる変数の名前特徴量(feature_names)を確認する。
<img width="692" alt="スクリーンショット 2021-12-08 18 29 32" src="https://user-images.githubusercontent.com/52492098/145184318-836add1f-c31e-42c3-a8cf-e13d4c09c432.png">
説明変数(data)の中身を確認する。
<img width="692" alt="スクリーンショット 2021-12-08 18 29 48" src="https://user-images.githubusercontent.com/52492098/145184326-74358d6d-236c-48b0-b494-4764241e4448.png">
目的変数(target)の中身を確認する。今回の場合、住宅価格(price)が入っている。
<br> ここまでは、今回使用するデータの確認。
<img width="793" alt="スクリーンショット 2021-12-08 18 30 22" src="https://user-images.githubusercontent.com/52492098/145184333-a3d5d36f-f942-439c-9bba-e6b665afd3e5.png">
説明変数をデータフレームに変換。（タイトルはfeature_names,中身はdata）
<br> 目的変数をデータフレームに追加。(住宅価格も先ほどのデータに並べる)
<br> 作成したデータフレームを確認。
<img width="512" alt="スクリーンショット 2021-12-08 18 31 19" src="https://user-images.githubusercontent.com/52492098/145184341-09466a8e-db4b-43b3-bb3b-5e25384df19f.png">
平均部屋数(RM)のカラムを確認。
<br> loc[:, ['RM']]はRMの全てを取り出す。それをdataとして設定。
<br> dataの表示。
<img width="734" alt="スクリーンショット 2021-12-08 18 31 35" src="https://user-images.githubusercontent.com/52492098/145184354-57aaabbb-1690-458d-9a71-a03389d05617.png">
目的変数(target)も同様に設定し、確認する。
<br> sklearnモジュールからLinearRegressionをインポートし、modelという名前をつける。
<br> fit関数でパラメータ推定すると、以下のようにpredictで予測値を算出できる。
<br> 今回の場合、部屋数（RM）で住宅価格（PRICE）を単回帰で予測。
<br> 結果は部屋数（RM）1で住宅価格（PRICE）は-25.5685118と負の数となってしまった。
<br> 部屋数のみでは、十分な予測とは言えなさそうなので、以下で重回帰も行う。
<img width="734" alt="スクリーンショット 2021-12-08 18 31 50" src="https://user-images.githubusercontent.com/52492098/145184369-04de5ad9-0b71-4d86-8110-9c3702ca9c91.png">
単回帰と同様にデータ確認。
<img width="734" alt="スクリーンショット 2021-12-08 18 32 03" src="https://user-images.githubusercontent.com/52492098/145184375-d9adda01-ecfc-46ca-9fe5-5c413e85aa6b.png">
data2として、部屋数（RM）に加えて犯罪率(CRIM)を追加し同様に線形回帰すると、
<br> 犯罪率(CRIM)0.3、部屋数（RM）4の時、重回帰により4.24007956　(千ドル）と予測される。
<br> 
<br> 感想
<br> まず、大前提として利用データは整理されており、前処理の必要がないものであったことに注意する。
<br> 本来は前処理ステップが発生する。
<br> また、今回はパラメータを限定して算出している。より正確な予測値を算出してみるためには色々と試行錯誤する必要がありそう。
<br> データ分析は簡単ではない。

## 非線形回帰モデル
課題なし

## ロジスティック回帰モデル
<img width="768" alt="ロジスティック回帰" src="https://user-images.githubusercontent.com/52492098/145106591-e67d1816-372c-4491-a81d-c9e2ef2d3ffb.png">
未

## 主成分分析
<img width="768" alt="主成分分析" src="https://user-images.githubusercontent.com/52492098/145106609-0edcbc10-15d0-4fb5-8fda-e01d4440a2fe.png">
未

## サポートベクターマシン
課題なし
