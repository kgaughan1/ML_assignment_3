


model.fit(X)

X_train_pca = model.transform(X)
X_projected = model.inverse_transform(X_train_pca)

loss = ((X - X_projected) ** 2).mean()