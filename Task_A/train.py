# Transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# Datasets and loaders
train_ds = ImageFolder('/content/drive/MyDrive/comys/Comys_Hackathon5/Task_A/train', transform)
val_ds   = ImageFolder('/content/drive/MyDrive/comys/Comys_Hackathon5/Task_A/val',   transform)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)
class_names = train_ds.classes
num_classes = len(class_names)
print("Classes:", class_names)

# Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training loop
num_epochs = 50
history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

start_time = time.time()
for epoch in range(num_epochs):
    # ⚙️ Training
    model.train()
    tl, ta, cnt = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        tl += loss.item()
        ta += (out.argmax(1)==labels).sum().item()
        cnt += len(labels)
    train_loss, train_acc = tl/len(train_loader), ta/cnt

    # Validation
    model.eval()
    vl, va, vc = 0,0,0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            vl += criterion(out,labels).item()
            va += (out.argmax(1)==labels).sum().item()
            vc += len(labels)
    val_loss, val_acc = vl/len(val_loader), va/vc

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | Train L/A: {history['train_loss'][-1]:.4f}/{history['train_acc'][-1]:.4f} "
              f"| Val L/A: {history['val_loss'][-1]:.4f}/{history['val_acc'][-1]:.4f}")

total_time = time.time() - start_time
print(f"Total training time: {total_time/60:.1f} mins")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

#pretrained path
torch.save(model.state_dict(), 'mobilenetv3_se_face_recognition.pth')
