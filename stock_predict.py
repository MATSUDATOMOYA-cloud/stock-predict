# ======================================================================
#  AIを使った株価の予想（stock_predict.py）
#  実行: 株価予測.bat をダブルクリック
#        または  python stock_predict.py [ティッカー] [開始日] [終了日]
#   - yfinance で株価データを取得
#   - 終値の日次変化率（%）を入力に、ランダムフォレスト回帰（決定木100本）で
#     次の5日間の「最新終値からの変化率」を予測し、価格に戻す
#   - 予測値の前日比（±0.5%）から Buy / Sell / Hold の売買シグナルを判定
#   - 下落後のハンマー型ローソク足を特徴量に加えた場合と加えない場合で、
#     翌日終値（変化率で予測）の精度を比較（評価用）
# ======================================================================
import datetime as dt
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- 設定 ----------------
DEFAULT_TICKER = "9984.T"       # ソフトバンクグループ
DEFAULT_START = "2020-01-01"
FORECAST_DAYS = 5               # 何日先まで予測するか
LOOKBACK = 60                   # 予測の入力に使う過去の営業日数
N_ESTIMATORS = 100              # 決定木の本数
TRAIN_RATIO = 0.8               # 訓練データ 80% / テストデータ 20%
SIGNAL_THRESHOLD_PCT = 0.5      # 売買シグナルのしきい値（前日比 %）
HAMMER_LOOKBACK = 10            # ハンマー分析で入力に使う過去の営業日数
DOWNTREND_DAYS = 5              # 「下落が続いた後」とみなす期間（営業日）
HISTORY_DAYS = 30               # 予測グラフに表示する過去の営業日数
RANDOM_STATE = 42


# ---------------- 入力 ----------------
def ask(prompt, default):
    try:
        s = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        s = ""
    return s or default


def normalize_ticker(t):
    t = t.replace("﻿", "").strip().upper()
    if t.isdigit() and len(t) == 4:   # 「9984」だけ入力されたら東証銘柄とみなす
        t += ".T"
    return t


def get_inputs():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    today = dt.date.today().isoformat()
    if args:
        ticker = args[0]
        start = args[1] if len(args) > 1 else DEFAULT_START
        end = args[2] if len(args) > 2 else today
    else:
        print("ティッカーコードを入力してください（例: 9984.T ソフトバンクG / 7203.T トヨタ / AAPL アップル）")
        ticker = ask("ティッカーコード", DEFAULT_TICKER)
        start = ask("開始日 (YYYY-MM-DD)", DEFAULT_START)
        end = ask("終了日 (YYYY-MM-DD)", today)
    return normalize_ticker(ticker), start, end


# ---------------- データ取得・前処理 ----------------
def fetch_data(ticker, start, end):
    print(f"\n{ticker} の株価データを取得しています（{start} ～ {end}）...")
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, end=end_excl, auto_adjust=False, actions=False)
    if df is None or df.empty:
        raise RuntimeError(
            "株価データを取得できませんでした。ティッカーコード・期間・インターネット接続を確認してください。\n"
            "（Yahoo Finance 側の仕様変更の場合は「株価予測.bat --update」で yfinance を更新してください）"
        )
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    # 欠損値の補完（前日の値で埋める。先頭で埋められない行は削除し、未来の値は使わない）
    df = df.ffill().dropna()
    try:
        currency = (tk.history_metadata or {}).get("currency") or "JPY"
    except Exception:
        currency = "JPY"
    print(f"取得件数: {len(df)} 日分（{df.index[0].date()} ～ {df.index[-1].date()}）")
    return df, currency


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, color="#b0b0b0")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper right")


# ---------------- ローソク足チャート ----------------
def save_candlestick(df, ticker, currency, out_dir):
    path = os.path.join(out_dir, f"{ticker}_candlestick_chart.png")
    mpf.plot(
        df, type="candle", style="charles",
        title=f"{ticker} Candlestick Chart", ylabel=f"Price ({currency})",
        figsize=(14, 6), savefig=dict(fname=path, dpi=100),
    )
    plt.close("all")
    return path


# ---------------- 5日間の株価予測 ----------------
def make_signals(last_close, prices):
    """予測値の前日比（%）から売買シグナルを判定する。
    1日目は最新の実際の終値と、2日目以降は前日の予測値と比べる。
    前日比 +1% 以上 → Buy、-1% 以下 → Sell、それ以外 → Hold"""
    signals, changes = [], []
    prev = last_close
    for p in prices:
        pct = (p - prev) / prev * 100
        changes.append(round(pct, 2))
        if pct >= SIGNAL_THRESHOLD_PCT:
            signals.append("Buy")
        elif pct <= -SIGNAL_THRESHOLD_PCT:
            signals.append("Sell")
        else:
            signals.append("Hold")
        prev = p
    return signals, changes


def forecast(df, ticker, currency, out_dir):
    close = df["Close"].values.astype(float)
    ret = np.diff(close) / close[:-1] * 100   # ret[j]: close[j] → close[j+1] の変化率（%）

    # 入力: 基準日 t までの過去 LOOKBACK 日分の日次変化率
    # 正解: 基準日 t の終値から見た、1～FORECAST_DAYS 日先の終値の変化率（%）
    X, y, base = [], [], []
    for t in range(LOOKBACK, len(close) - FORECAST_DAYS):
        X.append(ret[t - LOOKBACK:t])
        y.append((close[t + 1:t + 1 + FORECAST_DAYS] / close[t] - 1) * 100)
        base.append(close[t])
    X, y, base = np.array(X), np.array(y), np.array(base)
    if len(X) < 20:
        raise RuntimeError("データが少なすぎます。開始日をもっと前にしてください（目安: 1年以上）。")

    # 学習とテストの間に (FORECAST_DAYS - 1) 件の空きを入れ、
    # 学習データの正解（5日分）がテスト期間と重ならないようにする
    split = int(len(X) * TRAIN_RATIO)
    gap = FORECAST_DAYS - 1
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X[:split - gap], y[:split - gap])

    # テストデータでの評価（変化率を価格に戻して比べる）
    to_price = lambda pct, b: b[:, None] * (1 + pct / 100)
    y_test = to_price(y[split:], base[split:])
    y_pred = to_price(model.predict(X[split:]), base[split:])
    horizons = []
    for k in range(FORECAST_DAYS):
        a, p = y_test[:, k], y_pred[:, k]
        horizons.append(dict(day=k + 1, mae=mean_absolute_error(a, p), r2=r2_score(a, p),
                             mape=float(np.mean(np.abs((a - p) / a)) * 100)))
    yt, yp = y_test.ravel(), y_pred.ravel()
    mse = mean_squared_error(yt, yp)
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100)

    # 全データで学習し直して、直近 LOOKBACK 日の変化率から次の5日間を予測
    model.fit(X, y)
    pred_pct = model.predict(ret[-LOOKBACK:].reshape(1, -1))[0]
    future = [round(float(close[-1] * (1 + v / 100)), 2) for v in pred_pct]
    last_date = df.index[-1]
    last_close = float(df["Close"].iloc[-1])
    signals, changes = make_signals(last_close, future)
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)

    # JSON 保存
    json_path = os.path.join(out_dir, f"{ticker}_prediction_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "last_close": round(last_close, 2),
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "predicted_prices": future,
            "change_pct": changes,
            "signals": signals,
        }, f, indent=4)

    # グラフ保存：左に直近の実績、右に予測（最新の実績からつなげる）
    hist = df["Close"].iloc[-HISTORY_DAYS:]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist.index, hist.values, color="blue", linewidth=2, label="Actual Price")
    ax.plot([last_date, *dates], [last_close, *future], color="red", linestyle="--",
            linewidth=2, marker="o", markersize=4, label="Predicted Price")
    ax.axvline(last_date, color="gray", linestyle=":", linewidth=1)
    for name, color, marker in (("Buy", "green", "^"), ("Sell", "orange", "v")):
        idx = [i for i, s in enumerate(signals) if s == name]
        if idx:
            ax.scatter([dates[i] for i in idx], [future[i] for i in idx],
                       color=color, marker=marker, s=80, zorder=5, label=f"{name} Signal")
    style_axes(ax, f"{ticker} Forecast for Next {FORECAST_DAYS} Days with Trading Signals (Random Forest)",
               "Date", f"Price ({currency})")
    fig.autofmt_xdate()
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{ticker}_stock_prediction.png")
    fig.savefig(png_path, dpi=100)
    plt.close(fig)

    return dict(future=future, signals=signals, changes=changes, dates=dates, horizons=horizons,
                mse=mse, mae=mae, r2=r2, mape=mape, json_path=json_path, png_path=png_path)


# ---------------- ハンマーパターンを使った予測（評価用） ----------------
def hammer_analysis(df, ticker, currency, out_dir):
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    body = np.abs(c - o)
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    # ハンマーの形：下ヒゲが実体の2倍以上 かつ 上ヒゲが実体未満
    shape = (lower >= 2 * body) & (upper < body)
    # 下落が続いた後：前日の終値が、その DOWNTREND_DAYS 営業日前の終値より低い
    down = np.zeros(len(c), dtype=bool)
    down[DOWNTREND_DAYS + 1:] = c[DOWNTREND_DAYS:-1] < c[:-DOWNTREND_DAYS - 1]
    hammer = (shape & down).astype(int)

    # 入力: 基準日 t までの過去 HAMMER_LOOKBACK 日分の日次変化率（＋ t日のハンマー有無）
    # 正解: 基準日 t の終値から翌営業日の終値への変化率（%）
    ret = np.diff(c) / c[:-1] * 100   # ret[j]: c[j] → c[j+1] の変化率（%）
    X_ret, X_ham, y, base, days = [], [], [], [], []
    for t in range(HAMMER_LOOKBACK, len(c) - 1):
        X_ret.append(ret[t - HAMMER_LOOKBACK:t])
        X_ham.append(hammer[t])
        y.append(ret[t])
        base.append(c[t])
        days.append(t + 1)            # 予測対象の日
    X_ret, X_ham, y, base = np.array(X_ret), np.array(X_ham), np.array(y), np.array(base)
    X_with = np.column_stack([X_ret, X_ham])

    split = int(len(y) * TRAIN_RATIO)
    actual = base[split:] * (1 + y[split:] / 100)
    test_ham = X_ham[split:] == 1     # テスト期間のうち、前日がハンマーだった日

    # 「ハンマーなし」と「ハンマーあり」を同じ条件で学習・評価して比べる
    results = {}
    for key, X in (("without", X_ret), ("with", X_with)):
        model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X[:split], y[:split])
        pred = base[split:] * (1 + model.predict(X[split:]) / 100)
        results[key] = dict(
            pred=pred,
            mae=mean_absolute_error(actual, pred),
            r2=r2_score(actual, pred),
            mae_hammer=(mean_absolute_error(actual[test_ham], pred[test_ham])
                        if test_ham.any() else None),
        )

    # 参考：下落後のハンマーの翌日に上昇した割合（全期間）
    up_ratio = float(np.mean(y[X_ham == 1] > 0) * 100) if (X_ham == 1).any() else None

    # グラフ：テスト期間の実績と2つのモデルの予測、ハンマー翌日に印
    dates = df.index[days][split:]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, actual, color="blue", linewidth=2, label="Actual Price")
    ax.plot(dates, results["without"]["pred"], color="gray", linewidth=1.5, linestyle="--",
            label="Predicted (without Hammer)")
    ax.plot(dates, results["with"]["pred"], color="red", linewidth=1.5, label="Predicted (with Hammer)")
    if test_ham.any():
        ax.scatter(dates[test_ham], actual[test_ham], color="green", marker="^", s=80, zorder=5,
                   label="Day after Hammer")
    style_axes(ax, f"{ticker} Next-Day Prediction: with vs without Hammer Pattern (Random Forest)",
               "Date", f"Price ({currency})")
    fig.autofmt_xdate()
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{ticker}_hammer_pattern_stock_prediction.png")
    fig.savefig(png_path, dpi=100)
    plt.close(fig)

    for r in results.values():
        r.pop("pred")
    return dict(count_shape=int(shape.sum()), count=int(hammer.sum()), test_count=int(test_ham.sum()),
                up_ratio=up_ratio, without=results["without"], with_=results["with"], png_path=png_path)


# ---------------- メイン ----------------
def main():
    print("=" * 60)
    print(" AIを使った株価の予想（ランダムフォレスト回帰）")
    print("=" * 60)
    ticker, start, end = get_inputs()
    out_dir = os.environ.get("STOCK_OUT_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "出力結果")
    os.makedirs(out_dir, exist_ok=True)

    df, currency = fetch_data(ticker, start, end)

    print("ローソク足チャートを作成しています...")
    candle_path = save_candlestick(df, ticker, currency, out_dir)

    print("ランダムフォレストで次の5日間を予測しています...")
    fc = forecast(df, ticker, currency, out_dir)

    print("ハンマーパターンを使った予測を評価しています...")
    hm = hammer_analysis(df, ticker, currency, out_dir)

    unit = "円" if currency == "JPY" else f" {currency}"
    print("\n" + "=" * 60)
    print(f" {ticker} の予測結果（最終取得日 {df.index[-1].date()} の終値: {df['Close'].iloc[-1]:,.2f}{unit}）")
    print("=" * 60)
    print(" 日付(目安)    予測終値           前日比    売買シグナル")
    for d, p, c, s in zip(fc["dates"], fc["future"], fc["changes"], fc["signals"]):
        print(f" {d.date()}  {p:>14,.2f}{unit}  {c:>+7.2f}%  {s}")
    print(f" ※ 前日比 +{SIGNAL_THRESHOLD_PCT}% 以上 → Buy、-{SIGNAL_THRESHOLD_PCT}% 以下 → Sell、それ以外 → Hold"
          "（1日目は最新の実際の終値と比較）")
    print("\n[5日間予測モデルのテスト評価（1～5日先をまとめて）]")
    print(f"  MSE : {fc['mse']:,.2f}")
    print(f"  MAE : {fc['mae']:,.2f}")
    print(f"  R²  : {fc['r2']:.4f}")
    print(f"  平均誤差率 : {fc['mape']:.2f} %")
    print("\n[何日先かごとの誤差]")
    print("  日数         MAE        R²   平均誤差率")
    for h in fc["horizons"]:
        print(f"  {h['day']}日先  {h['mae']:>10,.2f}  {h['r2']:>8.4f}  {h['mape']:>7.2f} %")
    print("\n[ハンマーパターンの評価（翌日終値・変化率で予測）]")
    print(f"  ハンマー型の足 : {hm['count_shape']} 本"
          f"（うち{DOWNTREND_DAYS}営業日の下落後: {hm['count']} 本、テスト期間内: {hm['test_count']} 本）")
    if hm["up_ratio"] is not None:
        print(f"  下落後のハンマーの翌日に上昇した割合 : {hm['up_ratio']:.1f} %")
    print("  モデル              MAE        R²   MAE(ハンマー翌日のみ)")
    for label, r in (("ハンマーなし", hm["without"]), ("ハンマーあり", hm["with_"])):
        mh = "-" if r["mae_hammer"] is None else f"{r['mae_hammer']:,.2f}"
        print(f"  {label}  {r['mae']:>12,.2f}  {r['r2']:>8.4f}  {mh:>14}")
    print("  ※ ハンマーありの MAE が小さいほど、ハンマーが予測に役立っています")
    print("\n[出力ファイル]  保存先: " + out_dir)
    for p in (candle_path, fc["png_path"], fc["json_path"], hm["png_path"]):
        print("  " + os.path.basename(p))
    print("\n※ 予測は学習用の結果であり、投資判断を保証するものではありません。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断しました。")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        sys.exit(1)
