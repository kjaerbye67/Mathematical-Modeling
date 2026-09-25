import numpy as np
# data
year=np.array([1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])
price=np.array([767, 895, 995, 1117, 1261, 1437, 1640, 1957, 2244, 2489, 2801, 3096, 3500])
gdp=np.array([3540, 3783, 3916, 4239, 4922, 5560, 6399, 7842, 9116, 10879, 13475, 16737, 18745])
income=np.array([5156, 5138, 6526, 7434, 8475, 9688, 10703, 11384, 12343, 13630, 15558, 18472, 19820])

n = len(year)

# ========== Linear Regression: Price = b0 + b1*GDP + b2*Income ==========
X = np.column_stack([np.ones(n), gdp, income])
beta = np.linalg.inv(X.T @ X) @ X.T @ price
y_pred = X @ beta
residuals = price - y_pred
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((price - np.mean(price)) ** 2)
r2 = 1 - ss_res / ss_tot

print("=" * 55)
print("Price = b0 + b1*GDP + b2*Income")
print("=" * 55)
print(f"b0 = {beta[0]:.2f}")
print(f"b1 = {beta[1]:.4f}")
print(f"b2 = {beta[2]:.4f}")
print(f"R^2 = {r2:.4f}")

# ========== LOOCV ==========
mape_list = []
for i in range(n):
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(price, i)
    b = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    pred = X[i] @ b
    mape = abs(pred - price[i]) / price[i] * 100
    mape_list.append(mape)
print(f"LOOCV MAPE = {np.mean(mape_list):.2f}%")

# ========== Fitting detail ==========
print(f"\n{'Year':<6} {'Actual':>8} {'Pred':>8} {'Resid':>8} {'Err%':>7}")
print("-" * 40)
for i in range(n):
    err_pct = abs(residuals[i]) / price[i] * 100
    print(f"{year[i]:<6} {price[i]:>8} {y_pred[i]:>8.0f} {residuals[i]:>8.0f} {err_pct:>6.1f}%")

# ========== Predict 2010-2013 ==========
gdp_rate = (gdp[-1] / gdp[-4]) ** (1/3)
inc_rate = (income[-1] / income[-4]) ** (1/3)

print(f"\nAvg GDP growth (last 3yr) = {gdp_rate*100-100:.1f}%")
print(f"Avg Income growth (last 3yr) = {inc_rate*100-100:.1f}%")

print(f"\n{'Year':<6} {'GDP_est':>10} {'Inc_est':>10} {'Price_est':>10}")
print("-" * 42)
for yr in [2010, 2011, 2012, 2013]:
    g = gdp[-1] * (gdp_rate ** (yr - 2009))
    inc = income[-1] * (inc_rate ** (yr - 2009))
    x_new = np.array([1, g, inc])
    p = x_new @ beta
    print(f"{yr:<6} {g:>10.0f} {inc:>10.0f} {p:>10.0f}")

# ========== Correlation ==========
corr_gdp = np.corrcoef(gdp, price)[0, 1]
corr_inc = np.corrcoef(income, price)[0, 1]
print(f"\nCorr(Price, GDP) = {corr_gdp:.4f}")
print(f"Corr(Price, Income) = {corr_inc:.4f}")
